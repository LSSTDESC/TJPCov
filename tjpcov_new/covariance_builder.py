from .covariance_io import CovarianceIO
import pyccl as ccl
import pickle
import sacc
import numpy as np
import warnings


class CovarianceBuilder(CovarianceIO):
    def __init__(self, config):
        """
        Covariance Calculator object for TJPCov.

        Parameters
        ----------
        config (dict or str):
        """
        super().__init__(config)

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

        two_point_data = self.get_sacc_file()
        ccl_tracers, _ = self.get_tracer_info()

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
                                            ccl_tracers=ccl_tracers, **kwargs)
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))


        return blocks, tracers_blocks

    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        """
        Build full matrix from blocks.

        Parameters:
        -----------
        blocks (list): List of blocks
        tracers_cov (list): List of tracer combinations corresponding to each
        block in blocks. They should have the same order

        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations.
            Npt = (number of bins ) * (number of combinations)
        """
        blocks = iter(blocks)

        # Covariance construction based on
        # https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L86-L123
        s = self.get_sacc_with_concise_dtypes()
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

            cov_ij = next(blocks)
            cov_ij = cov_ij.reshape((nbpw, ncell1, nbpw, ncell2))

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

    def get_covariance_block(self, **kwargs):
        raise NotImplementedError("Do not use the base class directly")

    def get_covariance(self, **kwargs):
        if self.cov is not None:
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
        sacc_file = self.get_sacc_file()
        dtype = sacc_file.get_data_types()[0]
        tracers = sacc_file.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = sacc_file.get_ell_cl(dtype, *tracers)

        return ell

    def get_list_of_tracers_for_cov(self):
        """
        Return the covariance independent tracers combinations.

        Parameters:
        -----------

        Returns:
        --------
            tracers (list of str): List of independent tracers combinations.
        """

        tracers = self.sacc_file.get_tracer_combinations()

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
            as mask_name.

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

    def get_masks_dict(self, mask_names, tracer_names, cache):
        """
        Return a dictionary with the masks assotiated to the fields to be
        correlated

        Parameters:
        -----------
            mask_names (dict):  Dictionary of the masks names assotiated to the
            fields to be correlated. It has to be given as {1: name1, 2: name2, 3:
            name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
            the first and second Cell you are computing the covariance for; i.e.
            <Cell^12 Cell^34>. In fact, the tjpcov.mask_names.
            tracer_names (dict):  Dictionary of the tracer names of the same form
            as mask_name.
            cache (dict): Dictionary with cached variables. It will use the cached
            masks if found. The keys must be 'm1', 'm2', 'm3' or 'm4' and the
            values the loaded maps.

        Returns:
        --------
            masks_dict (dict):  Dictionary with the masks assotiated to the fields
            to be correlated.

        """
        mask_files = self.mask_files
        nside = self.nside
        mask = {}
        mask_by_mask_name = {}
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
        dtype = self.sacc_file.get_data_types()[0]
        tracers = self.sacc_file.get_tracer_combinations(data_type=dtype)[0]
        ix = self.sacc_file.indices(data_type=dtype, tracers=tracers)
        nbpw = ix.size

        return nbpw

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
        Check usage: Can we call all the tracer at once?

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
        ccl_tracers = {}
        tracer_Noise = {}
        tracer_Noise_coupled = {}

        for tracer in self.sacc_file.tracers:
            tracer_dat = self.sacc_file.get_tracer(tracer)
            tracer_Noise_coupled[tracer] = tracer_dat.metadata.get('n_ell_coupled', None)

            if (tracer_dat.quantity == 'galaxy_shear') or ('src' in tracer) \
                    or ('source' in tracer):
                z = tracer_dat.z
                dNdz = tracer_dat.nz
                if self.IA is None:
                    ia_bias = None
                else:
                    IA_bin = self.IA*np.ones(len(z)) # fao: refactor this
                    ia_bias = (z, IA_bin)
                ccl_tracers[tracer] = ccl.WeakLensingTracer(self.cosmo,
                                                            dndz=(z, dNdz),
                                                            ia_bias=ia_bias)
                # CCL automatically normalizes dNdz
                if tracer in self.sigma_e:
                    tracer_Noise[tracer] = self.sigma_e[tracer]**2/self.Ngal[tracer]
                else:
                    tracer_Noise[tracer] = None

            elif (tracer_dat.quantity == 'galaxy_density') or \
                ('lens' in tracer):
                z = tracer_dat.z
                dNdz = tracer_dat.nz
                # import pdb; pdb.set_trace()
                b = self.bias_lens[tracer] * np.ones(len(z))
                ccl_tracers[tracer] = ccl.NumberCountsTracer(
                    self.cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))
                if tracer in self.Ngal:
                    tracer_Noise[tracer] = 1./self.Ngal[tracer]
                else:
                    tracer_Noise[tracer] = None
            elif tracer_dat.quantity == 'cmb_convergence':
                ccl_tracers[tracer] = ccl.CMBLensingTracer(self.cosmo,
                                                           z_source=1100)

        if not np.all(list(tracer_Noise.values())):
            warnings.warn('Missing noise for some tracers in file. You will ' +
                          'have to pass it with the cache')

        if return_noise_coupled:
            vals = list(tracer_Noise_coupled.values())
            if not np.all(vals):
                tracer_Noise_coupled = None
            elif not np.all(vals):
                warnings.warn('Missing n_ell_coupled info for some tracers in '
                              + 'the sacc file. You will have to pass it with'
                              + 'the cache')
            return ccl_tracers, tracer_Noise, tracer_Noise_coupled

        return ccl_tracers, tracer_Noise

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
        tr = self.sacc_file.get_tracer(tracer)
        if (tr.quantity in ['cmb_convergence', 'galaxy_density']) or \
           ('lens' in tracer):
            return 0
        elif (tr.quantity == 'galaxy_shear') or ('source' in tracer):
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
    pass


class CovarianceReal(CovarianceBuilder):
    # TODO: Move Real space specific methods here
    space_type = 'Real'
    pass


