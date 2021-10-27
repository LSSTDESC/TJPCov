# import pdb
from . import wigner_transform, bin_cov, parse
from . import nmt_tools
import healpy as hp
import numpy as np
import sacc
import pyccl as ccl
import sys
import os
import pymaster as nmt
import warnings

cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd)+"/tjpcov")


d2r = np.pi/180


class CovarianceCalculator():
    def __init__(self,
                 tjpcov_cfg=None):
        """
        Covariance Calculator object for TJPCov.

        .. note::
            - the cosmo_fn parameter is always necessary
            - theta input in arcmin
            - Is reading all config from a single yaml a good option?
            - sacc passing values after scale cuts and no angbin_edges
            - angbin_edges necessary for bin averages
            - Check firecrown's way to handle survey features

        Parameters
        ----------
        tjpcov_cfg (str):
            filename and path to the TJPCov configuration yaml
            Check minimal example at: tests/data/conf_tjpcov_minimal.yaml

            This file MUST have
                - a sacc path
                - a xi_fn OR cl_fn
                - ccl.Cosmology object (in pickle format)  OR cosmology.yaml file generated by CCL
                - ...
                it contains info from the deprecated files:

                cosmo_fn(pyccl.object or str):
                    Receives the cosmo object or a the yaml filename
                    WARNING CCL Cosmo write_yaml seems to not pass
                            the transfer_function
                sacc_fn_xi/cl (None, str):
                    path to sacc file yaml
                    sacc object containing the tracers and binning to be used in covariance calculation
                window (None, dict, float):
                    If None, used window function specified in the sacc file
                    if dict, the keys are sacc tracer names and values are either HealSparse inverse variance
                    maps
                    if float it is assumed to be f_sky value

            TODO: cov_type = gauss
                  params
                  bias
                  fsky/mask
        """
        if isinstance(tjpcov_cfg, dict):
            config = tjpcov_cfg
        else:
            config, _ = parse(tjpcov_cfg)

        self.do_xi = config['tjpcov'].get('do_xi')

        if not isinstance(self.do_xi, bool):
            raise Exception("Err: check if you set do_xi: False (Harmonic Space) "
                            + "or do_xi: True in 'tjpcov' field of your yaml")

        print("Starting TJPCov covariance calculator for", end=' ')
        print("Configuration space" if self.do_xi else "Harmonic Space")

        if self.do_xi:
            xi_fn = config['tjpcov'].get('xi_file')
        else:
            cl_fn = config['tjpcov'].get('cl_file')

        cosmo_fn = config['tjpcov'].get('cosmo')
        # sacc_fn  = config['tjpcov'].get('sacc_file')

        # biases
        # reading values w/o passing the number of tomographic bins
        # import pdb; pdb.set_trace()
        self.bias_lens = {k.replace('bias_',''):v for k,v in config['tjpcov'].items()
                            if 'bias_' in k}
        self.IA = config['tjpcov'].get('IA')
        self.Ngal = {k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items()
                            if 'Ngal' in k}
        # self.Ngal_src = {k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items()
        #                     if 'Ngal_src' in k}
        self.sigma_e = {k.replace('sigma_e_',''):v for k, v in config['tjpcov'].items()
                            if 'sigma_e' in k}


        # Treating fsky = 1 if no input is given
        self.fsky = config['tjpcov'].get('fsky')
        if self.fsky is None:
            print("No input for fsky. Assuming ", end='')
            self.fsky=1

        print(f"fsky={self.fsky}")


        if cosmo_fn is None or cosmo_fn == 'set':
            self.cosmo = self.set_ccl_cosmo(config)
        elif isinstance(cosmo_fn, ccl.core.Cosmology):
            self.cosmo = cosmo_fn

        elif cosmo_fn.split('.')[-1] == 'yaml':
            self.cosmo = ccl.Cosmology.read_yaml(cosmo_fn)
            # TODO: remove this hot fix of ccl
            self.cosmo.config.transfer_function_method = 1

        elif cosmo_fn.split('.')[-1]  == 'pkl':
            import pickle
            with open(cosmo_fn, 'rb') as ccl_cosmo_file:
                self.cosmo = pickle.load(ccl_cosmo_file)


        else:
            raise Exception(
                "Err: File for cosmo field in input not recognized")

        # TO DO: remove this hotfix
        self.xi_data, self.cl_data = None, None

        if self.do_xi:
            self.xi_data = sacc.Sacc.load_fits(
                config['tjpcov'].get('sacc_file'))

        # TO DO: remove this dependence here
        #elif not do_xi:
        cl_data = config['tjpcov'].get('cl_file')
        if isinstance(cl_data, sacc.Sacc):
            self.cl_data = cl_data
        else:
            self.cl_data = sacc.Sacc.load_fits(cl_data)

        self.binning_info = config['tjpcov'].get('binning_info', None)
        if self.binning_info is None:
            # TO DO: remove this dependence here
            trcomb = self.cl_data.get_tracer_combinations()[0]
            ell_list = self.get_ell_theta(self.cl_data,  # fix this
                                          'galaxy_density_cl',
                                          trcomb,
                                          'linear', do_xi=False)
            # fao Set this inside get_ell_theta ?
            # ell, ell_bins, ell_edges = None, None, None
            theta, theta_bins, theta_edges = None, None, None

            # fix this for getting from the sacc file:
            th_list = self.set_ell_theta(2.5, 250., 20, do_xi=True)

            self.theta,  self.theta_bins, self.theta_edges,  = th_list


            # ell is the value for WT
            self.ell, self.ell_bins, self.ell_edges = ell_list
        elif not isinstance(self.binning_info, nmt.NmtBin):
            raise ValueError('If passed, binning_info has to be a NmtBin ' +
                             'instance')

        self.mask_fn = config['tjpcov'].get('mask_file')  # windown handler TBD
        self.mask_names = config['tjpcov'].get('mask_names')


        # Calling WT in method, only if do_xi
        self.WT = None

        # Output directory where to save all the time consuming calculations
        self.outdir = config['tjpcov'].get('outdir', None)
        if self.outdir and not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        self.nmt_conf = config.get('NaMaster', {})
        for k in ['f', 'w', 'cw']:
            if k not in self.nmt_conf:
                self.nmt_conf[k] = {}

        return

    def print_setup(self, output=None):
        """
        Placeholder of function to return setup
        TODO: Check the current setup for TJPCovs
        """
        cosmo = self.cosmo
        ell = self.ell
        if self.do_xi:
            bins = self.theta_bins
        else:
            bins = self.ell_bins
        run_configuration = {
        'do_xi': self.do_xi,
        'bins': bins
        }
        # TODO: save as yaml output
        if isinstance(output, str):
            with open(output, 'w') as ff:
                ff.write('....txt')


    def set_ccl_cosmo(self, config):
        """
        set the ccl cosmo from paramters in config file

        """
        print("Setting up cosmology...")

        cosmo_param_names = ['Omega_c', 'Omega_b', 'h',
                             'sigma8', 'n_s', 'transfer_function']
        cosmo_params = {name: config['parameters'][name]
                        for name in cosmo_param_names}
        cosmo = ccl.Cosmology(**cosmo_params)
        return cosmo


    def set_ell_theta(self, ang_min, ang_max, n_ang,
                      ang_scale='linear', do_xi=False):
        """
        Utility for return custom theta/ell bins (outside sacc)

        Parameters:
        -----------
        ang_min (int, float):
            if do_xi, ang is assumed to be theta (arcmin)
            if do_xi == False,  ang is assumed to be ell
        Returns:
        --------
            (theta, theta_edges ) (degrees):
        """
        # FIXME:
        # Use sacc is passing this
        if not do_xi:
            ang_delta = (ang_max-ang_min)//n_ang
            ang_edges = np.arange(ang_min, ang_max+1, ang_delta)
            ang = np.arange(ang_min, ang_max + ang_delta - 2)

        if do_xi:
            th_min = ang_min/60  # in degrees
            th_max = ang_max/60
            n_th_bins = n_ang
            ang_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                    n_th_bins+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), n_th_bins*30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, n_th_bins*30)

            ang = np.unique(np.sort(np.append(th, th2)))
            ang_bins = 0.5 * (ang_edges[1:] + ang_edges[:-1])

            return ang, ang_bins, ang_edges  # TODO FIXIT

        return ang, ang_edges


    def get_ell_theta(self, two_point_data, data_type, tracer_comb, ang_scale,
                      do_xi=False):
        """
        Get ell or theta for bins given the sacc object
        For now, presuming only log and linear bins

        Parameters:
        -----------

        Returns:
        --------
        """
        ang_name = "ell" if not do_xi else 'theta'

        # assuming same ell for all bins:
        data_types = two_point_data.get_data_types()
        ang_bins = two_point_data.get_tag(ang_name, data_type=data_types[0],
                                          tracers=tracer_comb)

        ang_bins = np.array(ang_bins)

        angb_min, angb_max = ang_bins.min(), ang_bins.max()
        if ang_name == 'theta':
            # assuming log bins
            del_ang = (ang_bins[1:]/ang_bins[:-1]).mean()
            ang_scale = 'log'
            assert 1 == 1

        elif ang_name == 'ell':
            # assuming linear bins
            del_ang = (ang_bins[1:] - ang_bins[:-1])[0]
            ang_scale = 'linear'

        ang, ang_edges = self.set_ell_theta(angb_min-del_ang/2,
                                            angb_max+del_ang/2,
                                            len(ang_bins),
                                            ang_scale=ang_scale, do_xi=do_xi)
        # Sanity check
        if ang_scale == 'linear':
            assert np.allclose((ang_edges[1:]+ang_edges[:-1])/2, ang_bins), \
                "differences in produced ell/theta"
        return ang, ang_bins, ang_edges


    def wt_setup(self, ell, theta):
        """
        Set this up once before the covariance evaluations

        Parameters:
        -----------
        ell (array): array of multipoles
        theta ( array): array of theta in degrees

        Returns:
        --------
        """
        # ell = two_point_data.metadata['ell']
        # theta_rad = two_point_data.metadata['th']*d2r
        # get_ell_theta()

        WT_factors = {}
        WT_factors['lens', 'source'] = (0, 2)
        WT_factors['source', 'lens'] = (2, 0)  # same as (0,2)
        WT_factors['source', 'source'] = {'plus': (2, 2), 'minus': (2, -2)}
        WT_factors['lens', 'lens'] = (0, 0)

        self.WT_factors = WT_factors

        ell = np.array(ell)
        if not np.alltrue(ell > 1):
            # fao check warnings in WT for ell < 2
            print("Removing ell=1 for Wigner Transformation")
            ell = ell[(ell > 1)]

        WT_kwargs = {'l': ell,
                     'theta': theta*d2r,
                     's1_s2': [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)]}

        WT = wigner_transform(**WT_kwargs)
        return WT


    def get_cov_WT_spin(self, tracer_comb=None):
        """
        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------
        WT_factors:

        """
    #     tracers=tuple(i.split('_')[0] for i in tracer_comb)
        tracers = []
        for i in tracer_comb:
            if 'lens' in i:
                tracers += ['lens']
            if 'src' in i:
                tracers += ['source']
        return self.WT_factors[tuple(tracers)]


    def get_tracer_info(self, two_point_data={}):
        """
        Creates CCL tracer objects and computes the noise for all the tracers
        Check usage: Can we call all the tracer at once?

        Parameters:
        -----------
            two_point_data (sacc obj):

        Returns:
        --------
            ccl_tracers: dict, ccl obj
                ccl.WeakLensingTracer or ccl.NumberCountsTracer
            tracer_Noise ({dict: float}):
                shot (shape) noise for lens (sources)
        """
        ccl_tracers = {}
        tracer_Noise = {}
        # b = { l:bi*np.ones(len(z)) for l, bi in self.lens_bias.items()}

        for tracer in two_point_data.tracers:
            tracer_dat = two_point_data.get_tracer(tracer)
            # z = tracer_dat.z

            # FIXME: Following should be read from sacc dataset.--------------
            #Ngal = 26.  # arc_min^2
            #sigma_e = .26
            #b = 1.5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
            # AI = .5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
            #Ngal = Ngal*3600/d2r**2
            # ---------------------------------------------------------------

            # dNdz = tracer_dat.nz
            # dNdz /= (dNdz*np.gradient(z)).sum()
            # dNdz *= self.Ngal[tracer]
            #FAO  this should be called by tomographic bin
            if (tracer_dat.quantity == 'galaxy_shear') or ('src' in tracer) \
                    or ('source' in tracer):
                z = tracer_dat.z
                dNdz = tracer_dat.nz
                if self.IA is None:
                    ia_bias = None
                else:
                    IA_bin = self.IA*np.ones(len(z)) # fao: refactor this
                    ia_bias = (z, IA_bin)
                ccl_tracers[tracer] = ccl.WeakLensingTracer(
                    self.cosmo, dndz=(z, dNdz), ia_bias=ia_bias)
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
                tracer_Noise[tracer] = 1./self.Ngal[tracer]
                ccl_tracers[tracer] = ccl.NumberCountsTracer(
                    self.cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))
            elif tracer_dat.quantity == 'cmb_convergence':
                ccl_tracers[tracer] = ccl.CMBLensingTracer(self.cosmo,
                                                           z_source=1100)

        return ccl_tracers, tracer_Noise

    def nmt_gaussian_cov(self, tracer_comb1=None, tracer_comb2=None,
                        ccl_tracers=None, tracer_Noise=None,
                        tracer_Noise_coupled=None, coupled=False, cache=None):
        """
        Compute a single covariance matrix for a given pair of C_ell

        Parameters:
        -----------
            tracer_comb 1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb 2 (list): List of the pair of tracer names of C_ell^2
            ccl_tracers (dict): Dictionary with necessary ccl_tracers with keys
            the tracer names
            tracer_Noise (dict): Dictionary with necessary (uncoupled) noise
            with keys the tracer names. The values must be a float or int, not
            an array
            tracer_Noise_coupled (dict): As tracer_Noise but with coupled
            noise.
            coupled (bool): True to return the coupled Gaussian covariance
            (default False)
            cache (dict): Dictionary with the necessary workspaces and
            covariance workspaces. It accept masks (keys: 'm1', 'm2', 'm3',
            'm4'), fields (keys: 'f1', 'f2', 'f3', 'f4'), workspaces (keys:
            'w13', 'w23', 'w14', 'w24', 'w12', 'w34'), the covariance
            workspace (key: 'cw') and a NmtBin (key: 'bins').

        Returns:
        --------
            cov (dict):  Gaussian covariance matrix for a pair of C_ell. keys
            are 'final' and 'final_b'. The covariance stored is the same in
            both cases.

        """
        if (tracer_Noise is not None) and (tracer_Noise_coupled is not None):
            raise ValueError('Only one tracer_Noise or tracer_Noise_coupled ' +
                             'can be given')
        if coupled:
            raise ValueError('Computing coupled covariance matrix not ' +
                             'implemented yet')

        if cache is None:
            cache = {}

        if  'bins' in cache:
            warnings.warn('Reading binning from cache. You will ignore the one'
                          + 'passed through binning_info at initialization')
            bins = cache['bins']
        elif self.binning_info is not None:
            bins = self.binning_info
        else:
            raise ValueError('You must pass a NmtBin instance through the ' +
                             'cache or at initialization')

        ell = np.arange(bins.lmax + 1)
        ell_eff = bins.get_effective_ells()

        if 'cosmo' in cache:
            cosmo = cache['cosmo']
        else:
            cosmo = self.cosmo

        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2

        ncell = {}
        ncell[12] = nmt_tools.get_tracer_comb_ncell(self.cl_data, tracer_comb1)
        ncell[34] = nmt_tools.get_tracer_comb_ncell(self.cl_data, tracer_comb2)
        ncell[13] = nmt_tools.get_tracer_comb_ncell(self.cl_data, (tr[1], tr[3]))
        ncell[24] = nmt_tools.get_tracer_comb_ncell(self.cl_data, (tr[2], tr[4]))
        ncell[14] = nmt_tools.get_tracer_comb_ncell(self.cl_data, (tr[1], tr[4]))
        ncell[23] = nmt_tools.get_tracer_comb_ncell(self.cl_data, (tr[2], tr[3]))

        s = {}
        s[1], s[2] = nmt_tools.get_tracer_comb_spin(self.cl_data, tracer_comb1)
        s[3], s[4] = nmt_tools.get_tracer_comb_spin(self.cl_data, tracer_comb2)


        # Fiducial cl
        cl = {}
        # Noise (coupled or not)
        SN = {'coupled': tracer_Noise is None}

        if SN['coupled'] is False:
            warnings.warn("Computing the coupled noise from the uncoupled " +
                          "noise. This assumes the noise is white")

        for i in [13, 24, 14, 23]:
            # Fiducial cl
            i1, i2 = [int(j) for j in str(i)]
            key = f'cl{i}'
            if key in cache:
                cl[i] = cache[key]
            else:
                cl[i] = np.zeros((ncell[i], ell.size))
                cl[i][0] = ccl.angular_cl(cosmo, ccl_tracers[tr[i1]],
                                          ccl_tracers[tr[i2]], ell)

            # Noise
            auto = tr[i1] == tr[i2]
            key = f'SN{i}'
            if key in cache:
                SN[i] = cache[key]
            else:
                SN[i] = np.zeros((ncell[i], ell.size))
                SN[i][0] = SN[i][-1] = np.ones_like(ell)
                if SN['coupled']:
                    SN[i] *= tracer_Noise_coupled[tr[i1]] if auto else 0
                else:
                    SN[i] *= tracer_Noise[tr[i1]] if auto else 0
                if s[i1] == 2:
                    SN[i][0, :2] = SN[i][-1, :2] = 0


        if np.any(cl[13]) or np.any(cl[24]) or np.any(cl[14]) or \
                np.any(cl[23]):


            # TODO: Modify depending on how TXPipe caches things
            # Mask, mask_names, field and workspaces dictionaries
            mn = nmt_tools.get_mask_names_dict(self.mask_names, tr)
            m = nmt_tools.get_masks_dict(self.mask_fn, mn, tr, cache)
            f = nmt_tools.get_fields_dict(m, s, mn, tr, self.nmt_conf['f'],
                                          cache)
            w = nmt_tools.get_workspaces_dict(f, m, mn, bins, self.outdir,
                                              self.nmt_conf['w'], cache)

            # TODO; Allow input options as output folder, if recompute, etc.
            if 'cw' in cache:
                cw = cache['cw']
            else:
                cw = nmt_tools.get_covariance_workspace(f[1], f[2], f[3], f[4],
                                                        mn[1], mn[2], mn[3],
                                                        mn[4], self.outdir,
                                                        **self.nmt_conf['cw'])

            cl_cov = {}
            cl_cov[13] = nmt_tools.get_cl_for_cov(cl[13], SN[13], m[1], m[3],
                                                  w[13], nl_is_cp=SN['coupled'])
            cl_cov[23] = nmt_tools.get_cl_for_cov(cl[23], SN[23], m[2], m[3],
                                                  w[23], nl_is_cp=SN['coupled'])
            cl_cov[14] = nmt_tools.get_cl_for_cov(cl[14], SN[14], m[1], m[4],
                                                  w[14], nl_is_cp=SN['coupled'])
            cl_cov[24] = nmt_tools.get_cl_for_cov(cl[24], SN[24], m[2], m[4],
                                                  w[24], nl_is_cp=SN['coupled'])

            cov = nmt.gaussian_covariance(cw, s[1], s[2], s[3], s[4],
                                          cl_cov[13], cl_cov[14], cl_cov[23],
                                          cl_cov[24], w[12], w[34], coupled)
        else:
            size1 = ncell[12] * ell_eff.size
            size2 = ncell[34] * ell_eff.size
            cov = np.zeros((size1, size2))

        return {'final': cov, 'final_b': cov}


    def cl_gaussian_cov(self, tracer_comb1=None, tracer_comb2=None,
                        ccl_tracers=None, tracer_Noise=None,
                        two_point_data=None, do_xi=False,
                        xi_plus_minus1='plus', xi_plus_minus2='plus'):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Returns:
        --------
            final:  unbinned covariance for C_ell
            final_b : binned covariance
        """
        # fsky should be read from the sacc
        # tracers 1,2,3,4=tracer_comb1[0],tracer_comb1[1],tracer_comb2[0],tracer_comb2[1]
        # ell=two_point_data.metadata['ell']
        # fao to discuss: indices
        cosmo = self.cosmo
        #do_xi=self.do_xi

        if not do_xi:
            ell = self.ell
        else:
            # FIXME:  check the max_ell here in the case of only xi
            ell = self.ell

        cl = {}
        cl[13] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell)
        cl[24] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell)
        cl[14] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell)
        cl[23] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell)

        SN = {}
        SN[13] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[0] else 0
        SN[24] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[1] else 0
        SN[14] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[1] else 0
        SN[23] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[0] else 0

        if do_xi:
            norm = np.pi*4* self.fsky #two_point_data.metadata['fsky']
        else:  # do c_ell
            norm = (2*ell+1)*np.gradient(ell)* self.fsky #two_point_data.metadata['fsky']

        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell))  # placeholder
        coupling_mat[1423] = np.eye(len(ell))  # placeholder

        cov = {}
        cov[1324] = np.outer(cl[13]+SN[13], cl[24]+SN[24])*coupling_mat[1324]
        cov[1423] = np.outer(cl[14]+SN[14], cl[23]+SN[23])*coupling_mat[1423]

        cov['final'] = cov[1423]+cov[1324]

        if do_xi:
            if self.WT is None:  # class modifier of WT initialization
                print("Preparing WT...")
                self.WT = self.wt_setup(self.ell, self.theta)
                print("Done!")

            # Fixme: SET A CUSTOM ELL FOR do_xi case, in order to use
            # a single sacc input filefile
            ell = self.ell
            s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
            s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
            if isinstance(s1_s2_1, dict):
                s1_s2_1 = s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2, dict):
                s1_s2_2 = s1_s2_2[xi_plus_minus2]
            th, cov['final'] = self.WT.projected_covariance2(l_cl=ell, s1_s2=s1_s2_1,
                                                             s1_s2_cross=s1_s2_2,
                                                             cl_cov=cov['final'])

        cov['final'] /= norm

        if do_xi:
            thb, cov['final_b'] = bin_cov(
                r=th/d2r, r_bins=self.theta_edges, cov=cov['final'])
            # r=th/d2r, r_bins=two_point_data.metadata['th_bins'], cov=cov['final'])
        else:
            # if two_point_data.metadata['ell_bins'] is not None:
            if self.ell_edges is not None:
                lb, cov['final_b'] = bin_cov(
                    r=self.ell, r_bins=self.ell_edges, cov=cov['final'])
                # r=ell, r_bins=two_point_data.metadata['ell_bins'], cov=cov['final'])

    #     cov[1324]=None #if want to save memory
    #     cov[1423]=None #if want to save memory
        return cov

    def get_all_cov(self, do_xi=False, use_nmt=False, **kwargs):
        """
        Compute all the covariances and then combine them into one single giant matrix
        Parameters:
        -----------
        two_point_data (sacc obj): sacc object containg two_point data
        **kwargs: The arguments to pass to your chosen covariance estimation
        method.

        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations.
            Npt = (number of bins ) * (number of combinations)

        """
        # FIXME: Only input needed should be two_point_data,
        # which is the sacc data file. Other parameters should be
        # included within sacc and read from there."""
        if use_nmt:
            raise ValueError('This function does not work with the NaMaster' +
                             'wrapper at the moment. Use get_all_cov_nmt.')

        two_point_data = self.xi_data if do_xi else self.cl_data

        ccl_tracers, tracer_Noise = self.get_tracer_info(
            two_point_data=two_point_data)

        # we will loop over all these
        tracer_combs = two_point_data.get_tracer_combinations()
        N2pt = len(tracer_combs)

        N_data = len(two_point_data.indices())
        print(f"Producing covariance with {N_data}x{N_data} points", end=" ")
        print(f"({N2pt} combinations of tracers)")

        # if two_point_data.metadata['ell_bins'] is not None:
        #     Nell_bins = len(two_point_data.metadata['ell_bins'])-1
        # else:
        #     Nell_bins = len(two_point_data.metadata['ell'])

        # if do_xi:
        #     Nell_bins = len(two_point_data.metadata['th_bins'])-1

        # cov_full = np.zeros((Nell_bins*N2pt, Nell_bins*N2pt))

        cov_full = np.zeros((N_data, N_data))

        # Fix this loop for uneven scale cuts (different N_ell)
        for i in np.arange(N2pt):
            print("{}/{}".format(i+1, N2pt))
            tracer_comb1 = tracer_combs[i]
            # solution for non-equal number of ell in bins
            Nell_bins_i = len(two_point_data.indices(tracers=tracer_comb1))
            indx_i = i*Nell_bins_i
            for j in np.arange(i, N2pt):
                tracer_comb2 = tracer_combs[j]
                Nell_bins_j = len(two_point_data.indices(tracers=tracer_comb2))
                indx_j = j*Nell_bins_j
                if use_nmt:
                    cov_ij = self.nmt_gaussian_cov(tracer_comb1=tracer_comb1,
                                                   tracer_comb2=tracer_comb2,
                                                   ccl_tracers=ccl_tracers,
                                                   tracer_Noise=tracer_Noise,
                                                   **kwargs)
                else:
                    cov_ij = self.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                                  tracer_comb2=tracer_comb2,
                                                  ccl_tracers=ccl_tracers,
                                                  tracer_Noise=tracer_Noise,
                                                  do_xi=do_xi,
                                                  two_point_data=two_point_data)

                # if do_xi or two_point_data.metadata['ell_bins'] is not None:
                # check
                if do_xi or self.ell_bins is not None:
                    cov_ij = cov_ij['final_b']
                else:
                    cov_ij = cov_ij['final']

                cov_full[indx_i:indx_i+Nell_bins_i,
                         indx_j:indx_j+Nell_bins_j] = cov_ij
                cov_full[indx_j:indx_j+Nell_bins_i,
                         indx_i:indx_i+Nell_bins_j] = cov_ij.T
        return cov_full

    def get_all_cov_nmt(self, tracer_noise=None, tracer_noise_coupled=None,
                        **kwargs):
        """
        Compute all the covariances and then combine them into one single giant matrix
        Parameters:
        -----------
        tracer_noise (dict): Dictionary with necessary (uncoupled) noise
        with keys the tracer names. The values must be a float or int, not
        an array
        tracer_noise_coupled (dict): As tracer_Noise but with coupled
        noise.
        **kwargs: The arguments to pass to your chosen covariance estimation
        method.

        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations.
            Npt = (number of bins ) * (number of combinations)
        """

        if (tracer_noise is not None) and (tracer_noise_coupled is not None):
            raise ValueError('Only one of tracer_nose or ' +
                             'tracer_noise_coupled can be given')

        two_point_data = self.cl_data

        ccl_tracers, tracer_Noise = self.get_tracer_info(
            two_point_data=two_point_data)

        if tracer_noise_coupled is not None:
            tracer_Noise_coupled = tracer_Noise.copy()
            tracer_Noise = None
        else:
            tracer_Noise_coupled = None

        # Circunvent the impossibility of inputting noise by hand
        for tracer in ccl_tracers:
            if tracer_noise and tracer in tracer_noise:
                tracer_Noise[tracer] = tracer_noise[tracer]
            elif tracer_noise_coupled and tracer in tracer_noise_coupled:
                tracer_Noise_coupled[tracer] = tracer_noise_coupled[tracer]

        # Covariance construction based on
        # https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L86-L123
        s = nmt_tools.get_sacc_with_concise_dtypes(two_point_data)
        dtype = s.get_data_types()[0]
        tracers = s.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = s.get_ell_cl(dtype, *tracers)
        nbpw = ell.size
        #
        ndim = s.mean.size
        cl_tracers = s.get_tracer_combinations()

        cov_full = -1 * np.ones((ndim, ndim))

        for i, tracer_comb1 in enumerate(cl_tracers):
            ncell1 = nmt_tools.get_tracer_comb_ncell(s, tracer_comb1)
            dtypes1 = nmt_tools.get_datatypes_from_ncell(ncell1)
            for tracer_comb2 in cl_tracers[i:]:
                ncell2 = nmt_tools.get_tracer_comb_ncell(s, tracer_comb2)
                dtypes2 = nmt_tools.get_datatypes_from_ncell(ncell2)
                print(tracer_comb1, tracer_comb2)
                cov_ij = self.nmt_gaussian_cov(tracer_comb1=tracer_comb1,
                                               tracer_comb2=tracer_comb2,
                                               ccl_tracers=ccl_tracers,
                                               tracer_Noise=tracer_Noise,
                                               tracer_Noise_coupled=tracer_Noise_coupled,
                                               **kwargs)
                cov_ij = cov_ij['final']

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

    def create_sacc_cov(output, do_xi=False):
        """ Write created cov to a new sacc object

        Parameters:
        ----------
        output (str): filename output
        do_xi (bool): do_xi=True for real space, do_xi=False for harmonic
            space

        Returns:
        -------
        None

        """
        print("Placeholder...")
        if do_xi:
            print(f"Saving xi covariance as \n{output}")
        else:
            print(f"Saving xi covariance as \n{output}")
        pass


if __name__ == "__main__":
    import tjpcov.main as cv
    import pickle
    import sys
    import os

    cwd = os.getcwd()
    sys.path.append(os.path.dirname(cwd)+"/tjpcov")
    # reference:
    with open("./tests/data/tjpcov_cl.pkl", "rb") as ff:
        cov0cl = pickle.load(ff)


    tjp0 = cv.CovarianceCalculator(tjpcov_cfg="tests/data/conf_tjpcov_minimal.yaml")

    ccl_tracers, tracer_Noise = tjp0.get_tracer_info(tjp0.cl_data)
    trcs = tjp0.cl_data.get_tracer_combinations()

    gcov_cl_0 = tjp0.cl_gaussian_cov(tracer_comb1=('lens0', 'lens0'),
                                     tracer_comb2=('lens0', 'lens0'),
                                     ccl_tracers=ccl_tracers,
                                     tracer_Noise=tracer_Noise,
                                     two_point_data=tjp0.cl_data,
                                     )


    if np.array_equal(gcov_cl_0['final_b'].diagonal()[:], cov0cl.diagonal()[:24]):
        print("Cov (diagonal):\n", gcov_cl_0['final_b'].diagonal()[:])
    else:
        print(gcov_cl_0['final_b'].diagonal()[:], cov0cl.diagonal()[:24])


