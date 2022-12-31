import pickle
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pyccl as ccl
import sacc

from .wigner_transform import bin_cov, WignerTransform
from . import tools
from .covariance_io import CovarianceIO


class CovarianceBuilder(ABC):
    """Base class in charge of building the full covariance.

    Class in charge of building the full covariance needed for the sacc
    file from individual covariance blocks. This is meant to be used as a
    parent class and the child classes would actually implement the actual
    computation of the blocks.
    """

    def __init__(self, config):
        """Init the base class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        self.io = CovarianceIO(config)
        config = self.config = self.io.config

        use_mpi = self.config["tjpcov"].get("use_mpi", False)

        if use_mpi:
            try:
                import mpi4py.MPI
            except ImportError:
                raise ValueError("MPI option requires mpi4py to be installed")

            self.comm = mpi4py.MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = None
            self.size = None

        self.cosmo = None

        # TODO: Think a better place to put this
        self.bias_lens = {
            k.replace("bias_", ""): v
            for k, v in self.config["tjpcov"].items()
            if "bias_" in k
        }

        self.IA = self.config["tjpcov"].get("IA")

        d2r = np.pi / 180
        self.Ngal = {
            k.replace("Ngal_", ""): v * 3600 / d2r**2
            for k, v in self.config["tjpcov"].items()
            if "Ngal" in k
        }
        self.sigma_e = {
            k.replace("sigma_e_", ""): v
            for k, v in self.config["tjpcov"].items()
            if "sigma_e" in k
        }

        self.cov = None

        self.nbpw = None

        # TODO: Move this somewhere else. They shouldn't be needed for fsky
        # approx.
        # We leave the masks in the base class because they are needed for most
        # methods. Even for fsky we can use them to estimate the fsky if not
        # provided. However, we might want to move it to a different class
        self.mask_files = config["tjpcov"].get("mask_file")
        self.mask_names = config["tjpcov"].get("mask_names")

        # nside is needed if mask_files contain hdf5 files
        self.nside = config["tjpcov"].get("nside")

    def _split_tasks_by_rank(self, tasks):
        """Yield the tasks corresponding to the given process.

        Iterate through a list of items, yielding the ones this process is
        responsible for. The tasks are allocated in a round-robin way.

        Args:
            tasks (iterable): Tasks to split up

        Returns:
            :obj:`generator`: Tasks associated to this process
        """
        # Copied from https://github.com/LSSTDESC/ceci/blob/7043ae5776d9b2c210a26dde6f84bcc2366c56e7/ceci/stage.py#L586  # noqa: E501

        for i, task in enumerate(tasks):
            if self.rank is None:
                yield task
            elif i % self.size == self.rank:
                yield task

    @property
    def _tracer_types(self):
        """Tuple with the tracer types (e.g. ("cl", "cl")).

        This is used to decide if the block covariance should be computed or is
        zero. For instance, if the class is meant to produce the covariance for
        Cells and the tracer types are clusters, the class should return 0.
        """
        pass

    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        """Build full matrix from blocks.

        Args:
            blocks (list): List of blocks
            tracers_cov (list): List of tracer combinations corresponding to
                each block in blocks. They must have the same order

        Returns:
            array: Covariance matrix for all combinations in the sacc file.
        """
        blocks = iter(blocks)

        s = self.io.get_sacc_file()
        ndim = s.mean.size

        cov_full = -1 * np.ones((ndim, ndim))

        print("Building the covariance: placing blocks in their place")
        for tracer_comb1, tracer_comb2 in tracers_cov:
            # We do not need to do the reshape here because the blocks have
            # been build looping over the data types present in the sacc file
            print(tracer_comb1, tracer_comb2)

            cov_ij = next(blocks)
            ix1 = s.indices(tracers=tracer_comb1)
            ix2 = s.indices(tracers=tracer_comb2)

            cov_full[np.ix_(ix1, ix2)] = cov_ij
            cov_full[np.ix_(ix2, ix1)] = cov_ij.T

        if np.any(cov_full == -1):
            raise Exception(
                "Something went wrong. Probably related to the data types"
            )

        return cov_full

    def _compute_all_blocks(self, **kwargs):
        """Compute all the independent covariance blocks.

        Args:
            **kwargs: Arguments to pass to the get_covariance_block method.
                These will depend on the covariance type requested.

        Returns:
            tuple:
                - blocks (list): List of all the independent super sample
                  covariance blocks.
                - tracer_blocks (list): List of all tracer combinations in
                  order as the blocks.
        """
        # Make a list of all independent tracer combinations
        tracers_cov = self.get_list_of_tracers_for_cov()

        # Save blocks and the corresponding tracers, as comm.gather does not
        # return the blocks in the original order.
        blocks = []
        tracers_blocks = []
        print("Computing independent covariance blocks")
        tasks_per_rank = self._split_tasks_by_rank(tracers_cov)
        for tracer_comb1, tracer_comb2 in tasks_per_rank:
            print(tracer_comb1, tracer_comb2)
            # TODO: Options to compute the covariance block should be defined
            # at initialization and/or through kwargs?
            cov = self.get_covariance_block_for_sacc(
                tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, **kwargs
            )
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))

        return blocks, tracers_blocks

    def get_cosmology(self):
        """Return a CCL Cosmology instance.

        The Cosmology is generated with the information passed in the
        configuration file in the "cosmo" section of "tjpcov". This can be a
        file path to a yaml, pickle object or "set" to read the parameters from
        the "parameters" section.

        Returns:
            :obj:`pyccl.Cosmology` instance
        """
        if self.cosmo is None:
            cosmo = self.config["tjpcov"].get("cosmo")

            if cosmo is None or cosmo == "set":
                self.cosmo = ccl.Cosmology(**self.config["parameters"])
            elif isinstance(cosmo, ccl.core.Cosmology):
                self.cosmo = cosmo
            elif isinstance(cosmo, str):
                ext = cosmo.split(".")[-1]
                if ext in ["yaml", "yml"]:
                    self.cosmo = ccl.Cosmology.read_yaml(cosmo)
                elif ext == "pkl":
                    with open(cosmo, "rb") as ccl_cosmo_file:
                        self.cosmo = pickle.load(ccl_cosmo_file)
                else:
                    raise ValueError(
                        "Cosmology path extension must be one "
                        f"of 'yaml', 'yml' or 'pkl'. Found {ext}."
                    )
            else:
                raise ValueError(
                    "cosmo entry looks wrong. It has to be one"
                    "of ['set', ccl.core.Cosmology instance, "
                    "a yaml file or a pickle"
                )

        return self.cosmo

    @abstractmethod
    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """Return the covariance block for the two pair of tracers.

        This is what you would get from an external code and could contain
        data types not present in the sacc file. For Fourier space covariances,
        we assume the same order as in NaMaster.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            **kwargs: extra possible arguments

        Returns:
            array:  Covariance block
        """
        # This function returns the covariance block with all elements (e.g.
        # EE-EB-BE-BB in the case of shear-shear), which might not be all
        # present in the sacc file
        # TODO: Consider leaving it only for the Fourier covariances since we
        # only really need the _for_sacc function for this class to work
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Return the covariance block as needed for the sacc file.

        For instance, if the sacc file does not contain B-modes, the covariance
        block would only have the EE-EE component). Furthermore, the ordering
        must be the same as in the sacc file.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            **kwargs: Arguments accepted by get_covariance_block

        Returns:
            array:  Covariance block
        """
        # This function returns the covariance block with the elements in the
        # sacc file (e.g. only EE in the case of shear-shear)
        raise NotImplementedError("Not implemented")

    def get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Return the covariance block as needed for the sacc file.

        For instance, if the sacc file does not contain B-modes, the covariance
        block would only have the EE-EE component). Furthermore, the ordering
        must be the same as in the sacc file.

        This function returns 0 if the covariance class is not meant to produce
        covariances for the given tracer combination.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            **kwargs: Arguments accepted by get_covariance_block

        Returns:
            array:  Covariance block
        """
        # If the tracers are of a data type not supported by the instantiated
        # class, return 0's. Elsewise, call _get_covariance_block_for_sacc

        # Let's use only one of the data types since all of them will have to
        # be e.g. cl, xi for the same tracer combination
        dtypes1 = self.get_tracer_comb_data_types(tracer_comb1)[0]
        dtypes2 = self.get_tracer_comb_data_types(tracer_comb2)[0]

        s = self.io.get_sacc_file()

        # Check in the given order
        correct_dtypes = (self._tracer_types[0] in dtypes1) and (
            self._tracer_types[1] in dtypes2
        )
        if not correct_dtypes:
            # Check in the opposite order
            correct_dtypes = (self._tracer_types[1] in dtypes1) and (
                self._tracer_types[0] in dtypes2
            )

        # If still the dtypes are not of the class, return 0's
        if not correct_dtypes:
            ix1 = s.indices(tracers=tracer_comb1)
            ix2 = s.indices(tracers=tracer_comb2)
            return np.zeros((ix1.size, ix2.size))

        return self._get_covariance_block_for_sacc(
            tracer_comb1, tracer_comb2, **kwargs
        )

    def get_covariance(self, **kwargs):
        """Return the full covariance.

        Args:
            **kwargs: Arguments to pass to _compute_all_blocks.

        Returns:
            array: Full covariance
        """
        if self.cov is None:
            blocks, tracers_cov = self._compute_all_blocks(**kwargs)

            if self.comm is not None:
                blocks = self.comm.gather(blocks, root=0)
                tracers_cov = self.comm.gather(tracers_cov, root=0)

                if self.rank == 0:
                    blocks = sum(blocks, [])
                    tracers_cov = sum(tracers_cov, [])

            if (self.comm is None) or (self.rank == 0):
                cov = self._build_matrix_from_blocks(blocks, tracers_cov)

                if not np.any(cov):
                    # You'll get a covariance full of 0's if none of the data
                    # types in the sacc file are compatible with the class
                    # _tracer_types
                    raise ValueError(
                        "The covariance is all 0's. This very likely "
                        "means that the sacc file does not contain "
                        "any tracer data type compatible with this "
                        "covariance class tracers: "
                        f"{self._tracer_types}."
                    )
                self.cov = cov

            # Broadcast the covariance to the other processes so all of them
            # have self.cov not None and avoid recomputing blocks
            if self.comm is not None:
                self.cov = self.comm.bcast(self.cov, root=0)

        return self.cov

    def get_list_of_tracers_for_cov(self):
        """Return the covariance independent tracers combinations.

        Returns:
            list of str: List of independent tracers combinations.
        """
        sacc_file = self.io.get_sacc_file()
        tracers = sacc_file.get_tracer_combinations()

        tracers_out = []
        for i, trs1 in enumerate(tracers):
            for trs2 in tracers[i:]:
                tracers_out.append((trs1, trs2))

        return tracers_out

    def get_mask_names_dict(self, tracer_names):
        """Return a dictionary with the mask names for the given tracers.

        Args:
            tracer_names (dict):  Dictionary of the tracer names of the same
                form as mask_name. It has to be given as {1: name1, 2: name2,
                3: name3, 4: name4}, where 12 and 34 are the pair of tracers
                that go into the first and second Cell you are computing the
                covariance for; i.e. <Cell^12 Cell^34>.

        Returns:
            dict:  Dictionary with the mask names assotiated to the fields to
            be correlated.
        """
        mask_names = self.mask_names
        mn = {}
        for i in [1, 2, 3, 4]:
            mn[i] = mask_names[tracer_names[i]]
        return mn

    def get_masks_dict(self, tracer_names, cache=None):
        """Return a dictionary with the masks assotiated to the given tracers.

        Args:
            tracer_names (dict):  Dictionary of the tracer names of the same
                form as mask_name. It has to be given as {1: name1, 2: name2,
                3: name3, 4: name4}, where 12 and 34 are the pair of tracers
                that go into the first and second Cell you are computing the
                covariance for; i.e. <Cell^12 Cell^34>.
            cache (dict, optional): Dictionary with cached variables. It will
                use the cached masks if found. The keys must be 'm1', 'm2',
                'm3' or 'm4' and the values the loaded maps.

        Returns:
            dict: Dictionary with the masks assotiated to the fields to be
            correlated.
        """
        mask_files = self.mask_files
        mask_names = self.get_mask_names_dict(tracer_names)
        nside = self.nside
        mask = {}
        mask_by_mask_name = {}
        if cache is None:
            cache = {}
        for i in [1, 2, 3, 4]:
            # Mask
            key = f"m{i}"
            if key in cache:
                mask[i] = cache[key]
            else:
                k = mask_names[i]
                if k not in mask_by_mask_name:
                    if type(mask_files) is str:
                        mf = mask_files
                    else:
                        mf = mask_files[tracer_names[i]]
                    if isinstance(mf, np.ndarray):
                        mask_by_mask_name[k] = mf
                    else:
                        mask_by_mask_name[k] = tools.read_map(mf, k, nside)
                mask[i] = mask_by_mask_name[k]

        return mask

    def get_nbpw(self):
        """Return the number of bandpowers in which the data has been binned.

        Returns:
            int: Number of bandpowers; i.e. ell_effective.size
        """
        if self.nbpw is None:
            sacc_file = self.io.get_sacc_file()
            dtype = sacc_file.get_data_types()[0]
            tracers = sacc_file.get_tracer_combinations(data_type=dtype)[0]
            ix = sacc_file.indices(data_type=dtype, tracers=tracers)
            self.nbpw = ix.size

        return self.nbpw

    def get_tracers_spin_dict(self, tracer_names):
        """Return a dictionary with the spins assotiated to the given tracers.

        Args:
            tracer_names (dict):  Dictionary of the tracer names of the same
                form as mask_name. It has to be given as {1: name1, 2: name2,
                3: name3, 4: name4}, where 12 and 34 are the pair of tracers
                that go into the first and second Cell you are computing the
                covariance for; i.e. <Cell^12 Cell^34>.

        Returns:
            dict: Dictionary with the spins assotiated to the fields to be
            correlated.
        """
        s = {}
        for i, tni in tracer_names.items():
            s[i] = self.get_tracer_spin(tni)
        return s

    def get_tracer_comb_spin(self, tracer_comb):
        """Return the spins of a pair of tracers.

        Args:
            tracer_comb (tuple):  List or tuple of a pair of tracer names

        Returns:
            tuple:
                - s1 (int):  Spin of the first tracer
                - s2 (int):  Spin of the second tracer

        """
        s1 = self.get_tracer_spin(tracer_comb[0])
        s2 = self.get_tracer_spin(tracer_comb[1])

        return s1, s2

    def get_tracer_spin(self, tracer):
        """Return the spin of a given tracer.

        Args:
            sacc_data (Sacc):  Data Sacc instance
            tracer (str):  Tracer name

        Returns:
            int:  Spin of the given tracer
        """
        sacc_file = self.io.get_sacc_file()
        tr = sacc_file.get_tracer(tracer)
        if (tr.quantity in ["cmb_convergence", "galaxy_density"]) or (
            "lens" in tracer
        ):
            return 0
        elif (
            (tr.quantity == "galaxy_shear")
            or ("source" in tracer)
            or ("src" in tracer)
        ):
            return 2
        else:
            raise NotImplementedError(
                f"tracer.quantity {tr.quantity} not implemented."
            )

    def get_tracer_nmaps(self, tracer):
        """Return the number of maps assotiated to the given tracer.

        Args:
            sacc_data (Sacc):  Data Sacc instance
            tracer (str):  Tracer name

        Returns:
            int:  Number of maps assotiated to the tracer.
        """
        s = self.get_tracer_spin(tracer)
        if s == 0:
            return 1
        else:
            return 2

    def get_tracer_comb_data_types(self, tracer_comb):
        """Return the tracer data types associated to the given tracers.

        Args:
            tracer_comb (list): List of a pair of tracer names in the sacc file

        Returns:
            list: List of data types associated to the given tracer pair.
        """
        s = self.io.get_sacc_file()
        data_types = s.get_data_types()

        dt_output = []
        for dt in data_types:
            if len(s.indices(data_type=dt, tracers=tracer_comb)) != 0:
                dt_output.append(dt)

        return dt_output


class CovarianceFourier(CovarianceBuilder):
    """Parent class for Cell x Cell covariances in Fourier space.

    This has all the methods common to all the Fourier covariance calculations.
    The child classes will actually implement the actual computation of the
    blocks.
    """

    space_type = "Fourier"
    _tracer_types = ("cl", "cl")

    def __init__(self, config):
        """Init the base class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)
        self.ccl_tracers = None
        self.tracer_Noise = None
        self.tracer_Noise_coupled = None

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Return the covariance block as needed for the sacc file.

        For instance, if the sacc file does not contain B-modes, the covariance
        block would only have the EE-EE component). Furthermore, the ordering
        must be the same as in the sacc file.

        Args:
            tracer_comb 1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb 2 (list): List of the pair of tracer names of C_ell^2
            **kwargs: Arguments accepted by get_covariance_block

        Returns:
            array: Covariance matrix for a pair of C_ell for the data
            types considered in the sacc file.
        """
        nbpw = self.get_nbpw()

        ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
        dtypes1 = self.get_datatypes_from_ncell(ncell1)

        ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
        dtypes2 = self.get_datatypes_from_ncell(ncell2)

        # The reshape below assumes that the covariances from
        # get_covariance_block follow the NaMaster ordering. This is because
        # NaMaster is the main code at the moment. If in the future we have new
        # ways of comuting the covariance that follow a different ordering, eg.
        # Cell[:, None] * Cell[None, :], as in sacc, we could modify this and
        # make this a NaMaster specific method.
        cov = self.get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)
        cov = cov.reshape(
            (nbpw, ncell1, nbpw, ncell2),
        )

        # Keep only elements in the sacc file
        s = self.get_sacc_with_concise_dtypes()

        # To avoid having to modify the indices for the block. This is a waste
        # of time though
        cov_full = -1 * np.ones((s.mean.size, s.mean.size))

        for i, dt1 in enumerate(dtypes1):
            ix1 = s.indices(tracers=tracer_comb1, data_type=dt1)
            if len(ix1) == 0:
                continue
            for j, dt2 in enumerate(dtypes2):
                ix2 = s.indices(tracers=tracer_comb2, data_type=dt2)
                if len(ix2) == 0:
                    continue
                cov_full[np.ix_(ix1, ix2)] = cov[:, i, :, j]

        ix1 = s.indices(tracers=tracer_comb1)
        ix2 = s.indices(tracers=tracer_comb2)

        return cov_full[ix1][:, ix2]

    def get_datatypes_from_ncell(self, ncell):
        """Return the datatypes (e.g cl_00) for a the number of cells.

        Args:
            ncell (int):  Number of Cell for a pair of tracers

        Returns:
            list: List of data types assotiated to the given degrees of freedom
        """
        # Copied from https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L202-L212  # noqa: E501
        if ncell == 1:
            cl_types = ["cl_00"]
        elif ncell == 2:
            cl_types = ["cl_0e", "cl_0b"]
        elif ncell == 4:
            cl_types = ["cl_ee", "cl_eb", "cl_be", "cl_bb"]
        else:
            raise ValueError("ncell does not match 1, 2, or 4.")

        return cl_types

    def get_ell_eff(self):
        """Return the effective ell in the sacc file.

        It assume that all of them have the same effective ell (true with
        current TXPipe implementation).

        Args:
            sacc_data (:obj:`sacc.sacc.Sacc`): Data Sacc instance

        Returns:
            array: Array with the effective ell in the sacc file.
        """
        sacc_file = self.io.get_sacc_file()
        dtype = sacc_file.get_data_types()[0]
        tracers = sacc_file.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = sacc_file.get_ell_cl(dtype, *tracers)

        return ell

    def get_sacc_with_concise_dtypes(self):
        """Return a copy of the sacc file with concise data types.

        Returns:
            :obj:`sacc.sacc.Sacc`: Data Sacc instance with concise data types
        """
        s = self.io.get_sacc_file().copy()
        dtypes = s.get_data_types()

        dt_long = []
        for dt in dtypes:
            if len(dt.split("_")) > 2:
                dt_long.append(dt)

        for dt in dt_long:
            pd = sacc.parse_data_type_name(dt)

            if pd.statistic != "cl":
                raise ValueError(
                    f"data_type {dt} not recognized. Is it a Cell?"
                )

            if pd.subtype is None:
                dc = "cl_00"
            elif pd.subtype == "e":
                dc = "cl_0e"
            elif pd.subtype == "b":
                dc = "cl_0b"
            elif pd.subtype == "ee":
                dc = "cl_ee"
            elif pd.subtype == "bb":
                dc = "cl_bb"
            elif pd.subtype == "eb":
                dc = "cl_eb"
            elif pd.subtype == "be":
                dc = "cl_be"
            else:
                raise ValueError(f"Data subtype {pd.subtype} not recognized")

            # Change the data_type to its concise versio
            for dp in s.get_data_points(dt):
                dp.data_type = dc

        return s

    def get_tracer_comb_ncell(self, tracer_comb, independent=False):
        """Return the number of Cell for a pair of tracers.

        For instance, for shear-shear, ncell = 4: EE, EB, BE, BB.

        Args:
            sacc_data ( Sacc):  Data Sacc instance
            tracer_comb (tuple):  List or tuple of a pair of tracer names
            independent (bool, optional): If True, just return the number of
                independent Cell.

        Returns:
            int:  Number of Cell for the pair of tracers given
        """
        nmaps1 = self.get_tracer_nmaps(tracer_comb[0])
        nmaps2 = self.get_tracer_nmaps(tracer_comb[1])

        ncell = nmaps1 * nmaps2

        if independent and (tracer_comb[0] == tracer_comb[1]) and (ncell == 4):
            # Remove BE, because it will be the same as EB if tr1 == tr2
            ncell = 3

        return ncell

    def get_tracer_info(self, return_noise_coupled=False):
        """Returns CCL tracer objects and the noise for all the tracers.

        This is done based on the specifications given in the configuration
        file.

        Args:
            return_noise_coupled (bool, optional): If True, also return
                tracers_Noise_coupled. Default False.

        Returns:
            tuple:
                - ccl_tracers (:obj:`pyccl.tracers.Tracer`): CCL tracers used
                  to compute the theory vector
                - tracer_Noise (:obj:`dict`): shot (shape) noise for lens
                  (sources) tracers
                - tracer_Noise_coupled (:obj:`dict`): coupled shot (shape)
                  noise for lens (sources). Returned if retrun_noise_coupled is
                  True.
        """
        if self.ccl_tracers is None:
            cosmo = self.get_cosmology()
            ccl_tracers = {}
            tracer_Noise = {}
            tracer_Noise_coupled = {}

            sacc_file = self.io.get_sacc_file()

            for tracer in sacc_file.tracers:
                tracer_dat = sacc_file.get_tracer(tracer)

                # Coupled noise read from tracer metadata in the sacc file
                tracer_Noise_coupled[tracer] = tracer_dat.metadata.get(
                    "n_ell_coupled", None
                )

                if (
                    (tracer_dat.quantity == "galaxy_shear")
                    or ("src" in tracer)
                    or ("source" in tracer)
                ):
                    # CCL Tracer
                    z = tracer_dat.z
                    dNdz = tracer_dat.nz
                    if self.IA is None:
                        ia_bias = None
                    else:
                        IA_bin = self.IA * np.ones(len(z))
                        ia_bias = (z, IA_bin)
                    ccl_tracers[tracer] = ccl.WeakLensingTracer(
                        cosmo, dndz=(z, dNdz), ia_bias=ia_bias
                    )
                    # Noise
                    if tracer in self.sigma_e:
                        tracer_Noise[tracer] = (
                            self.sigma_e[tracer] ** 2 / self.Ngal[tracer]
                        )
                    else:
                        tracer_Noise[tracer] = None

                elif (tracer_dat.quantity == "galaxy_density") or (
                    "lens" in tracer
                ):
                    # CCL Tracer
                    z = tracer_dat.z
                    dNdz = tracer_dat.nz
                    b = self.bias_lens[tracer] * np.ones(len(z))
                    ccl_tracers[tracer] = ccl.NumberCountsTracer(
                        cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b)
                    )

                    # Noise
                    if tracer in self.Ngal:
                        tracer_Noise[tracer] = 1.0 / self.Ngal[tracer]
                    else:
                        tracer_Noise[tracer] = None

                elif tracer_dat.quantity == "cmb_convergence":
                    # CCL Tracer
                    ccl_tracers[tracer] = ccl.CMBLensingTracer(
                        cosmo, z_source=1100
                    )

            if None in list(tracer_Noise.values()):
                warnings.warn(
                    "Missing noise for some tracers in file. "
                    "You will have to pass it with the cache"
                )

            if None in list(tracer_Noise_coupled.values()):
                warnings.warn(
                    "Missing n_ell_coupled info for some tracers in "
                    "the sacc file. You will have to pass it with"
                    "the cache"
                )

            self.ccl_tracers = ccl_tracers
            self.tracer_Noise = tracer_Noise
            self.tracer_Noise_coupled = tracer_Noise_coupled

        if return_noise_coupled:
            return (
                self.ccl_tracers,
                self.tracer_Noise,
                self.tracer_Noise_coupled,
            )

        return self.ccl_tracers, self.tracer_Noise


class CovarianceReal(CovarianceBuilder):
    """Parent class for xi x xi covariances in Real space.

    This has all the methods common to all the Real covariance calculations.
    The child classes will actually implement the actual computation of the
    blocks.
    """

    space_type = "Real"
    _tracer_types = ("xi", "xi")

    def get_theta_eff(self):
        """Return the effective theta in the sacc file.

        It assume that all of them have the same effective theta (true with
        current TXPipe implementation).

        Args:
            sacc_data (:obj:`sacc.sacc.Sacc`):  Data Sacc instance

        Returns:
            array: Array with the effective theta in the sacc file.
        """
        sacc_file = self.io.get_sacc_file()
        dtype = sacc_file.get_data_types()[0]
        tracers = sacc_file.get_tracer_combinations()[0]
        theta, _ = sacc_file.get_theta_xi(dtype, *tracers)

        return theta


class CovarianceProjectedReal(CovarianceReal):
    # TODO: The transforms here should be generalized to handle EB-BE-BB modes.
    # For now we will only consider the EE contribution, which should be
    # dominant
    """Parent class for xi x xi covariances in Real space.

    This has all the methods common to all the Real covariance calculations
    that are computed by projecting the Fourier space ones. The child classes
    will only have to call a CovarianceFourier child to get the covariance in
    Fourier space.
    """

    def __init__(self, config):
        """Init the base class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)
        self.WT = None
        self.lmax = self.config["ProjectedReal"].get("lmax")
        if self.lmax is None:
            raise ValueError(
                "You need to specify the lmax you want to "
                "compute the Fourier covariance up to"
            )

    @property
    def fourier(self):
        pass

    def get_binning_info(self, binning="log", in_radians=True):
        """Get the theta for bins given the sacc object.

        Args:
            binning (str): Binning type.
            in_radians (bool): If the angles must be given in radians. Needed
                for the Wigner transforms.

        Returns:
            tuple:
                - theta (array): All the thetas covered
                - theta_eff (array): The effective thetas
                - theta_edges (array): The bandpower edges
        """
        # TODO: This should be obtained from the sacc file or the input
        # configuration. Check how it is done in TXPipe:
        # https://github.com/LSSTDESC/TXPipe/blob/a9dfdb7809ac7ed6c162fd3930c643a67afcd881/txpipe/covariance.py#L23

        theta_eff = self.get_theta_eff()
        nbpw = theta_eff.size

        thetab_min, thetab_max = theta_eff.min(), theta_eff.max()
        if binning == "log":
            # assuming constant log bins
            del_logtheta = np.log10(theta_eff[1:] / theta_eff[:-1]).mean()
            theta_min = 2 * thetab_min / (10**del_logtheta + 1)
            theta_max = 2 * thetab_max / (1 + 10 ** (-del_logtheta))

            th_min = theta_min
            th_max = theta_max
            theta_edges = np.logspace(
                np.log10(th_min), np.log10(th_max), nbpw + 1
            )
            th = np.logspace(np.log10(th_min * 0.98), np.log10(1), nbpw * 30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max * 1.02, nbpw * 30)

            theta = np.unique(np.sort(np.append(th, th2)))
        else:
            raise NotImplementedError(f"Binning {binning} not implemented yet")

        if in_radians:
            arcmin_rad = np.pi / 180 / 60
            theta *= arcmin_rad
            theta_eff *= arcmin_rad
            theta_edges *= arcmin_rad

        return theta, theta_eff, theta_edges

    def get_cov_WT_spin(self, tracer_comb):
        """Get the Wigner transform factors.

        Args:
            tracer_comb (list of two str): tracer combination in sacc format

        Returns:
            WT_factors
        """
        WT_factors = {}
        WT_factors["lens", "source"] = (0, 2)
        WT_factors["source", "lens"] = (2, 0)  # same as (0,2)
        WT_factors["source", "source"] = {"plus": (2, 2), "minus": (2, -2)}
        WT_factors["lens", "lens"] = (0, 0)

        tracers = []
        for i in tracer_comb:
            if "lens" in i:
                tracers += ["lens"]
            elif ("src" in i) or ("source" in i):
                tracers += ["source"]
            else:
                raise NotImplementedError(
                    "This functions requires your "
                    "tracers to be called 'lens', "
                    f"'src' or 'source', given {i}"
                )
        return WT_factors[tuple(tracers)]

    def get_Wigner_transform(self):
        """Return an instance of the wigner_transform class.

        Returns:
            :obj:`~tjpcov.wigner_transform.WignerTransform` instance
        """
        if self.WT is None:
            # Removing ell <= 1 (following original implementation)
            ell = np.arange(2, self.lmax + 1)
            theta, _, _ = self.get_binning_info(in_radians=True)

            WT_kwargs = {
                "ell": ell,
                "theta": theta,
                "s1_s2": [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)],
            }

            self.WT = WignerTransform(**WT_kwargs)

        return self.WT

    @abstractmethod
    def _get_fourier_block(self, tracer_comb1, tracer_comb2):
        raise NotImplementedError("Not yet implemented")

    def get_covariance_block(
        self,
        tracer_comb1,
        tracer_comb2,
        xi_plus_minus1="plus",
        xi_plus_minus2="plus",
        binned=True,
    ):
        """Compute a single covariance matrix for a given pair of xi.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            xi_plus_minus1 (str): 'plus' if one wants the covariance for the
                xi+ component or 'minus' for the xi-. This is ignored if
                tracer_comb1 is not a spin 2 (e.g. shear) field.
            xi_plus_minus2 (str): As xi_plus_minus1 for tracer_comb2.

        Returns:
            array: Covariance matrix
        """
        # For now we just use the EE block which should be dominant over the
        # EB, BE and BB pieces
        cov = self._get_fourier_block(tracer_comb1, tracer_comb2)

        WT = self.get_Wigner_transform()

        s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
        s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1[xi_plus_minus1]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2[xi_plus_minus2]
        # Remove ell <= 1 for WT (following original implementation)
        ell = np.arange(2, self.lmax + 1)
        cov = cov[2:][:, 2:]
        th, cov = WT.projected_covariance(
            ell_cl=ell, s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2, cl_cov=cov
        )
        if binned:
            theta, _, theta_edges = self.get_binning_info(in_radians=False)
            thb, cov = bin_cov(r=theta, r_bins=theta_edges, cov=cov)

        return cov

    def _get_covariance_block_for_sacc(self, tracer_comb1, tracer_comb2):
        """Compute a the covariance matrix for a given pair of C_ell or xi.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2

        Returns:
            array: Covariance matrix
        """
        data_types1 = self.get_tracer_comb_data_types(tracer_comb1)
        data_types2 = self.get_tracer_comb_data_types(tracer_comb2)

        nbpw = self.get_nbpw()

        cov = np.zeros((nbpw, len(data_types1), nbpw, len(data_types2)))

        auto = tracer_comb1 == tracer_comb2

        for i, dt1 in enumerate(data_types1):
            xi_plus_minus1 = "plus" if "plus" in dt1 else "minus"
            start_ix = i if auto else 0
            for j, dt2 in enumerate(data_types2[start_ix:]):
                xi_plus_minus2 = "plus" if "plus" in dt2 else "minus"
                cov[:, i, :, j] = self.get_covariance_block(
                    tracer_comb1, tracer_comb2, xi_plus_minus1, xi_plus_minus2
                )
                if auto:
                    cov[:, j, :, i] = cov[:, i, :, j].T

        cov = cov.reshape((nbpw * len(data_types1), nbpw * len(data_types2)))
        return cov
