from . import tools
from . import wigner_transform
from .covariance_io import CovarianceIO
from abc import ABC, abstractmethod
import pyccl as ccl
import pickle
import sacc
import numpy as np
import warnings


class CovarianceBuilder(ABC):
    def __init__(self, config):
        """
        Covariance Calculator object for TJPCov.

        Parameters
        ----------
        config (dict or str):
        """
        self.io = CovarianceIO(config)
        config = self.config = self.io.config

        use_mpi = self.config['tjpcov'].get('use_mpi', False)

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
        self.bias_lens = {k.replace('bias_',''):v for k,v in
                          self.config['tjpcov'].items() if 'bias_' in k}

        self.IA = self.config['tjpcov'].get('IA')

        d2r = np.pi/180
        self.Ngal = {k.replace('Ngal_',''):v*3600/d2r**2 for k, v in
                     self.config['tjpcov'].items() if 'Ngal' in k}
        self.sigma_e = {k.replace('sigma_e_',''):v for k, v in
                        self.config['tjpcov'].items() if 'sigma_e' in k}

        self.cov = None

        self.nbpw = None

        # We leave the masks in the base class because they are needed for most
        # methods. Even for fsky we can use them to estimate the fsky if not
        # provided. However, we might want to move it to a different class
        self.mask_files = config['tjpcov'].get('mask_file')
        self.mask_names = config['tjpcov'].get('mask_names')

        # nside is needed if mask_files contain hdf5 files
        self.nside = config['tjpcov'].get('nside')

    def _split_tasks_by_rank(self, tasks):
        """
        Iterate through a list of items, yielding ones this process is responsible for/
        Tasks are allocated in a round-robin way.
        Parameters
        ----------
        tasks: iterable
            Tasks to split up
        """
        # Copied from https://github.com/LSSTDESC/ceci/blob/7043ae5776d9b2c210a26dde6f84bcc2366c56e7/ceci/stage.py#L586

        for i, task in enumerate(tasks):
            if self.rank is None:
                yield task
            elif i % self.size == self.rank:
                yield task

    @abstractmethod
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        raise NotImplementedError("Not implemented")

    def _compute_all_blocks(self, **kwargs):
        """
        Compute all the independent covariance blocks.

        Parameters:
        -----------
        kwargs: Parameters to pass to the `get_covariance_block` method. These
        will depend on the covariance type requested

        Returns:
        --------
        blocks (list): List of all the independent super sample covariance
        blocks.
        """

        # Make a list of all independent tracer combinations
        tracers_cov = self.get_list_of_tracers_for_cov()

        # Save blocks and the corresponding tracers, as comm.gather does not
        # return the blocks in the original order.
        blocks = []
        tracers_blocks = []
        print('Computing independent covariance blocks')
        tasks_per_rank = self._split_tasks_by_rank(tracers_cov)
        for tracer_comb1, tracer_comb2 in tasks_per_rank:
            print(tracer_comb1, tracer_comb2)
            # TODO: Options to compute the covariance block should be defined
            # at initialization and/or through kwargs?
            cov = self.get_covariance_block(tracer_comb1=tracer_comb1,
                                            tracer_comb2=tracer_comb2,
                                            **kwargs)
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))


        return blocks, tracers_blocks

    def get_cosmology(self):
        if self.cosmo is None:
            cosmo = self.config['tjpcov'].get('cosmo')

            if cosmo is None or cosmo == 'set':
                self.cosmo = ccl.Cosmology(**self.config['parameters'])
            elif isinstance(cosmo, ccl.core.Cosmology):
                self.cosmo = cosmo
            elif isinstance(cosmo, str):
                ext = cosmo.split('.')[-1]
                if ext in ['yaml', 'yml']:
                    self.cosmo = ccl.Cosmology.read_yaml(cosmo)
                elif ext  == 'pkl':
                    with open(cosmo, 'rb') as ccl_cosmo_file:
                        self.cosmo = pickle.load(ccl_cosmo_file)
                else:
                    raise ValueError('Cosmology path extension must be one ' +
                                     "of 'yaml', 'yml' or 'pkl'. " +
                                     f"Found {ext}.")
            else:
                raise ValueError("cosmo entry looks wrong. It has to be one" +
                                 "of ['set', ccl.core.Cosmology instance, " +
                                 "a yaml file or a pickle")

        return self.cosmo

    @abstractmethod
    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        raise NotImplementedError("Not implemented")

    def get_covariance(self, **kwargs):
        if self.cov is None:
            blocks, tracers_cov = self._compute_all_blocks(**kwargs)

            if self.comm is not None:
                blocks = self.comm.gather(blocks, root=0)
                tracers_cov = self.comm.gather(tracers_cov, root=0)

                if self.rank == 0:
                    blocks = sum(blocks, [])
                    tracers_cov = sum(tracers_cov, [])
                else:
                    return

            self.cov = self._build_matrix_from_blocks(blocks, tracers_cov)

        return self.cov

    def get_list_of_tracers_for_cov(self):
        """
        Return the covariance independent tracers combinations.

        Parameters:
        -----------

        Returns:
        --------
            tracers (list of str): List of independent tracers combinations.
        """
        sacc_file = self.io.get_sacc_file()
        tracers = sacc_file.get_tracer_combinations()

        tracers_out = []
        for i, trs1 in enumerate(tracers):
            for trs2 in tracers[i:]:
                tracers_out.append((trs1, trs2))

        return tracers_out

    def get_mask_names_dict(self, tracer_names):
        """
        Return a dictionary with the mask names assotiated to the fields to be
        correlated

        Parameters:
        -----------
            tracer_names (dict):  Dictionary of the tracer names of the same form
            as mask_name. It has to be given as {1: name1, 2: name2, 3:
            name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
            the first and second Cell you are computing the covariance for; i.e.
            <Cell^12 Cell^34>.

        Returns:
        --------
            masks_names_dict (dict):  Dictionary with the mask names assotiated to the
            fields to be correlated.

        """
        mask_names = self.mask_names
        mn  = {}
        for i in [1, 2, 3, 4]:
            mn[i] = mask_names[tracer_names[i]]
        return mn

    def get_masks_dict(self, tracer_names, cache=None):
        """
        Return a dictionary with the masks assotiated to the fields to be
        correlated

        Parameters:
        -----------
            tracer_names (dict):  Dictionary of the tracer names of the same form
            as mask_name. It has to be given as {1: name1, 2: name2, 3:
            name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
            the first and second Cell you are computing the covariance for; i.e.
            <Cell^12 Cell^34>.
            cache (dict): Dictionary with cached variables. It will use the cached
            masks if found. The keys must be 'm1', 'm2', 'm3' or 'm4' and the
            values the loaded maps.

        Returns:
        --------
            masks_dict (dict):  Dictionary with the masks assotiated to the fields
            to be correlated.

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
            key = f'm{i}'
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
        """
        Return the number of bandpowers in which the data has been binned

        Parameters:
        -----------

        Returns:
        --------
            nbpw (int): Number of bandpowers; i.e. ell_effective.size
        """
        if self.nbpw is None:
            sacc_file = self.io.get_sacc_file()
            dtype = sacc_file.get_data_types()[0]
            tracers = sacc_file.get_tracer_combinations(data_type=dtype)[0]
            ix = sacc_file.indices(data_type=dtype, tracers=tracers)
            self.nbpw = ix.size

        return self.nbpw

    def get_tracers_spin_dict(self, tracer_names):
        """
        Return a dictionary with the masks assotiated to the fields to be
        correlated

        Parameters:
        -----------
            tracer_names (dict):  Dictionary of the tracer names of the same form
            as mask_name. It has to be given as {1: name1, 2: name2, 3:
            name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
            the first and second Cell you are computing the covariance for; i.e.
            <Cell^12 Cell^34>.

        Returns:
        --------
            spins_dict (dict):  Dictionary with the spins assotiated to the
            fields to be correlated.
        """
        s = {}
        for i, tni in tracer_names.items():
            s[i] = self.get_tracer_spin(tni)
        return s

    def get_tracer_comb_spin(self, tracer_comb):
        """
        Return the spins of a pair of tracers

        Parameters:
        -----------
            tracer_comb (tuple):  List or tuple of a pair of tracer names

        Returns:
        --------
            s1 (int):  Spin of the first tracer
            s2 (int):  Spin of the second tracer

        """
        s1 = self.get_tracer_spin(tracer_comb[0])
        s2 = self.get_tracer_spin(tracer_comb[1])

        return s1, s2

    def get_tracer_spin(self, tracer):
        """
        Return the spin of a given tracer

        Parameters:
        -----------
            sacc_data (Sacc):  Data Sacc instance
            tracer (str):  Tracer name

        Returns:
        --------
            spin (int):  Spin of the given tracer

        """
        sacc_file = self.io.get_sacc_file()
        tr = sacc_file.get_tracer(tracer)
        if (tr.quantity in ['cmb_convergence', 'galaxy_density']) or \
           ('lens' in tracer):
            return 0
        elif (tr.quantity == 'galaxy_shear') or ('source' in tracer) or \
            ('src' in tracer):
            return 2
        else:
            raise NotImplementedError(f'tracer.quantity {tr.quantity} not implemented.')

    def get_tracer_nmaps(self, tracer):
        """
        Return the number of maps assotiated to the given tracer

        Parameters:
        -----------
            sacc_data (Sacc):  Data Sacc instance
            tracer (str):  Tracer name

        Returns:
        --------
            nmaps (int):  Number of maps assotiated to the tracer.

        """
        s = self.get_tracer_spin(tracer)
        if s == 0:
            return 1
        else:
            return 2


class CovarianceFourier(CovarianceBuilder):
    # TODO: Move Fourier specific methods here
    space_type = 'Fourier'

    def __init__(self, config):
        super().__init__(config)
        self.ccl_tracers = None
        self.tracer_Noise = None
        self.tracer_Noise_coupled = None

    @property
    def _reshape_order(self):
        """
        order (str) : {'C', 'F', 'A'}, optional. The order option to pass to
        numpy.reshape when reshaping the blocks to `(nbpw, ncell1, nbpw,
        ncell2)`. If you are using NaMaster blocks, 'C' should be used. If the
        blocks are as in the sacc file, 'F' should be used.
        """
        pass

    def _build_matrix_from_blocks(self, blocks, tracers_cov,
                                  only_independent=False):
        """
        Build full matrix from blocks.

        Parameters:
        -----------
        blocks (list): List of blocks
        tracers_cov (list): List of tracer combinations corresponding to each
        block in blocks. They must have the same order
        only_independent (bool): If True, the blocks only contain the
        covariance for the independent Cells. E.g. for wl-wl, Cell EB = BE. If
        True, BE will not be considered.

        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations.
            Npt = (number of bins ) * (number of combinations)
        """
        # TODO: Genearlize this for both real and Fourier space and move it to
        # CovarianceBuilder
        blocks = iter(blocks)

        # Covariance construction based on
        # https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L86-L123
        s = self.io.get_sacc_with_concise_dtypes()
        nbpw = self.get_nbpw()
        #
        ndim = s.mean.size
        cl_tracers = s.get_tracer_combinations()

        cov_full = -1 * np.ones((ndim, ndim))

        print('Building the covariance: placing blocks in their place')
        for tracer_comb1, tracer_comb2 in tracers_cov:
            print(tracer_comb1, tracer_comb2)
            # Although these two variables do not vary as ncell2 and dtypes2,
            # it is cleaner to tho the loop this way
            ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
            dtypes1 = self.get_datatypes_from_ncell(ncell1)

            ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
            dtypes2 = self.get_datatypes_from_ncell(ncell2)

            if only_independent:
                if (tracer_comb1[0] == tracer_comb1[1]) and (ncell1 == 4):
                    ncell1 -= 1
                    dtypes1.remove('cl_be')

                if (tracer_comb2[0] == tracer_comb2[1]) and (ncell2 == 4):
                    ncell2 -= 1
                    dtypes2.remove('cl_be')

            cov_ij = next(blocks)
            # The reshape works for the NaMaster ordering with order 'C'
            # If the blocks are ordered as in the sacc file, you need order 'F'
            cov_ij = cov_ij.reshape((nbpw, ncell1, nbpw, ncell2),
                                    order=self._reshape_order)

            for i, dt1 in enumerate(dtypes1):
                ix1 = s.indices(tracers=tracer_comb1, data_type=dt1)
                if len(ix1) == 0:
                    continue
                for j, dt2 in enumerate(dtypes2):
                    ix2 = s.indices(tracers=tracer_comb2, data_type=dt2)
                    if len(ix2) == 0:
                        continue
                    covi = cov_ij[:, i, :, j]
                    cov_full[np.ix_(ix1, ix2)] = covi
                    cov_full[np.ix_(ix2, ix1)] = covi.T

        if np.any(cov_full == -1):
            raise Exception('Something went wrong. Probably related to the ' +
                            'data types')

        return cov_full

    def get_datatypes_from_ncell(self, ncell):
        """
        Return the possible datatypes (cl_00, cl_0e, cl_0b, etc.) given a number
        of cells for a pair of tracers

        Parameters:
        -----------
            ncell (int):  Number of Cell for a pair of tracers
        Returns:
        --------
            datatypes (list):  List of data types assotiated to the given degrees
            of freedom

        """
        # Copied from https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L202-L212
        # TODO: Can this be genearlized for real space and promoted to the
        # CovarianceBuilder parent class?
        if ncell == 1:
            cl_types = ['cl_00']
        elif ncell == 2:
            cl_types = ['cl_0e', 'cl_0b']
        elif ncell == 4:
            cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        else:
            raise ValueError('ncell does not match 1, 2, or 4.')

        return cl_types

    def get_ell_eff(self):
        """
        Return the effective ell in the sacc file. It assume that all of them have
        the same effective ell (true with current TXPipe implementation).

        Parameters:
        -----------
            sacc_data (Sacc):  Data Sacc instance

        Returns:
        --------
            ell (array): Array with the effective ell in the sacc file.
        """
        sacc_file = self.io.get_sacc_file()
        dtype = sacc_file.get_data_types()[0]
        tracers = sacc_file.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = sacc_file.get_ell_cl(dtype, *tracers)

        return ell

    def get_tracer_comb_ncell(self, tracer_comb, independent=False):
        """
        Return the number of Cell for a pair of tracers (e.g. for shear-shear,
        ncell = 4: EE, EB, BE, BB)

        Parameters:
        -----------
            sacc_data (Sacc):  Data Sacc instance
            tracer_comb (tuple):  List or tuple of a pair of tracer names
            independent (bool): If True, just return the number of independent
            Cell.

        Returns:
        --------
            ncell (int):  Number of Cell for the pair of tracers given

        """
        nmaps1 = self.get_tracer_nmaps(tracer_comb[0])
        nmaps2 = self.get_tracer_nmaps(tracer_comb[1])

        ncell = nmaps1 * nmaps2

        if independent and (tracer_comb[0] == tracer_comb[1]) and (ncell == 4):
            # Remove BE, because it will be the same as EB if tr1 == tr2
            ncell = 3

        return ncell

    def get_tracer_info(self, return_noise_coupled=False):
        """
        Creates CCL tracer objects and computes the noise for all the tracers
        based on the specifications given in the configuration file.

        Parameters:
        -----------
            return_noise_coupled (bool): If True, also return
            tracers_Noise_coupled. Default False.

        Returns:
        --------
            ccl_tracers: dict, ccl obj
                ccl.WeakLensingTracer or ccl.NumberCountsTracer
            tracer_Noise ({dict: float}):
                shot (shape) noise for lens (sources)
            tracer_Noise_coupled ({dict: float}):
                coupled shot (shape) noise for lens (sources). Returned if
                retrun_noise_coupled is True.

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
                tracer_Noise_coupled[tracer] = tracer_dat.metadata.get('n_ell_coupled', None)

                if (tracer_dat.quantity == 'galaxy_shear') or ('src' in tracer) \
                        or ('source' in tracer):
                    # CCL Tracer
                    z = tracer_dat.z
                    dNdz = tracer_dat.nz
                    if self.IA is None:
                        ia_bias = None
                    else:
                        IA_bin = self.IA*np.ones(len(z))
                        ia_bias = (z, IA_bin)
                    ccl_tracers[tracer] = ccl.WeakLensingTracer(cosmo,
                                                                dndz=(z, dNdz),
                                                                ia_bias=ia_bias)
                    # Noise
                    if tracer in self.sigma_e:
                        tracer_Noise[tracer] = self.sigma_e[tracer]**2/self.Ngal[tracer]
                    else:
                        tracer_Noise[tracer] = None

                elif (tracer_dat.quantity == 'galaxy_density') or \
                    ('lens' in tracer):
                    # CCL Tracer
                    z = tracer_dat.z
                    dNdz = tracer_dat.nz
                    b = self.bias_lens[tracer] * np.ones(len(z))
                    ccl_tracers[tracer] = ccl.NumberCountsTracer(
                        cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))

                    # Noise
                    if tracer in self.Ngal:
                        tracer_Noise[tracer] = 1./self.Ngal[tracer]
                    else:
                        tracer_Noise[tracer] = None

                elif tracer_dat.quantity == 'cmb_convergence':
                    # CCL Tracer
                    ccl_tracers[tracer] = ccl.CMBLensingTracer(cosmo,
                                                               z_source=1100)

            if None in list(tracer_Noise.values()):
                warnings.warn('Missing noise for some tracers in file. You will ' +
                              'have to pass it with the cache')

            if None in list(tracer_Noise_coupled.values()):
                warnings.warn('Missing n_ell_coupled info for some tracers in '
                              + 'the sacc file. You will have to pass it with'
                              + 'the cache')

            self.ccl_tracers = ccl_tracers
            self.tracer_Noise = tracer_Noise
            self.tracer_Noise_coupled = tracer_Noise_coupled

        if return_noise_coupled:
            return self.ccl_tracers, self.tracer_Noise, self.tracer_Noise_coupled

        return self.ccl_tracers, self.tracer_Noise


class CovarianceReal(CovarianceBuilder):
    # TODO: Move Real space specific methods here and check WT for general case
    space_type = 'Real'

    def get_theta_eff(self):
        """
        Return the effective theta in the sacc file. It assume that all of them
        have the same effective theta (true with current TXPipe
        implementation).

        Parameters:
        -----------
            sacc_data (Sacc):  Data Sacc instance

        Returns:
        --------
            ell (array): Array with the effective theta in the sacc file.
        """
        # TODO: Consider moving this method to CovarianceBuilder and merge it
        # with the one for Fourier space
        sacc_file = self.io.get_sacc_file()
        dtype = sacc_file.get_data_types()[0]
        tracers = sacc_file.get_tracer_combinations()[0]
        theta, _ = sacc_file.get_theta_xi(dtype, *tracers)

        return theta


class CovarianceProjectedReal(CovarianceReal):
    # TODO: The transforms here should be generalized to handle EB-BE-BB modes
    """
    Real covariance class for the cases we compute the covariance in Fourier
    space and then we project to real space.
    """
    def __init__(self, config):
        super().__init__(config)
        self.WT = None
        self.lmax = self.config['ProjectedReal'].get('lmax')
        if self.lmax is None:
            raise ValueError('You need to specify the lmax you want to ' +
                             'compute the Fourier covariance up to')

    @property
    def fourier(self):
        pass

    def get_binning_info(self, binning='log', in_radians=True):
        """
        Get the theta for bins given the sacc object

        Parameters:
        -----------
        binning (str): Binning type.
        in_radians (bool): If the angles must be given in radians. Needed for
        the Wigner transforms.

        Returns:
        --------
        theta (array): All the thetas covered
        theta_eff (array): The effective thetas
        theta_edges (array): The bandpower edges
        """
        # TODO: This should be obtained from the sacc file or the input
        # configuration. Check how it is done in TXPipe:
        # https://github.com/LSSTDESC/TXPipe/blob/a9dfdb7809ac7ed6c162fd3930c643a67afcd881/txpipe/covariance.py#L23

        theta_eff = self.get_theta_eff()
        nbpw = theta_eff.size

        thetab_min, thetab_max = theta_eff.min(), theta_eff.max()
        if binning == 'log':
            # assuming constant log bins
            del_logtheta = np.log10(theta_eff[1:]/theta_eff[:-1]).mean()
            theta_min = 2 * thetab_min  / (10**del_logtheta + 1)
            theta_max = 2 * thetab_max  / (1 + 10**(-del_logtheta))

            th_min = theta_min
            th_max = theta_max
            theta_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                      nbpw+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), nbpw*30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, nbpw*30)

            theta = np.unique(np.sort(np.append(th, th2)))
        else:
            raise NotImplementedError(f'Binning {binning} not implemented yet')

        if in_radians:
            arcmin_rad = np.pi / 180 / 60
            theta *= arcmin_rad
            theta_eff *= arcmin_rad
            theta_edges *= arcmin_rad

        return theta, theta_eff, theta_edges

    def get_cov_WT_spin(self, tracer_comb=None):
        """
        Get the Wigner transform factors

        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------
        WT_factors:

        """
        WT_factors = {}
        WT_factors['lens', 'source'] = (0, 2)
        WT_factors['source', 'lens'] = (2, 0)  # same as (0,2)
        WT_factors['source', 'source'] = {'plus': (2, 2), 'minus': (2, -2)}
        WT_factors['lens', 'lens'] = (0, 0)

        tracers = []
        for i in tracer_comb:
            if 'lens' in i:
                tracers += ['lens']
            if ('src' in i) or ('source' in i):
                tracers += ['source']
        return WT_factors[tuple(tracers)]

    def get_Wigner_transform(self):
        """
        Return the wigner_transform class

        Returns:
        --------
        wigner_transform class
        """
        if self.WT is None:
            # Removing ell <= 1 (following original implementation)
            ell = np.arange(2, self.lmax + 1)
            theta, _, _= self.get_binning_info(in_radians=True)

            WT_kwargs = {'l': ell,
                         'theta': theta,
                         's1_s2': [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)]}

            self.WT = wigner_transform(**WT_kwargs)

        return self.WT
