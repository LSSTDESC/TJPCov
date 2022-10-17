from collections import namedtuple
from .covariance_builder import CovarianceBuilder
import healpy as hp
import numpy as np
import os
import pyccl as ccl
import pyccl.halos.hmfunc as hmf
import time
import warnings
# Replace with CCL functions
from scipy.integrate import quad, romb
from scipy.special import gamma 
from scipy.interpolate import interp1d



class CovarianceClusters(CovarianceBuilder):
    """ Covariance of Cluster + 3x2pt
        version 1 , date: - ?? ?? 2019
        Covariance matrix for a LSST-like case, using CCL packages
        Evaluate here:
        - NxN & NxCls (gg, gk, kk) 
        - Assuming full sky for now
        - Included shot noise (1-halo term)
        - Added full matrix in the end 
        TODO: verify the other limber auxiliary functions and new
        shot noise
        """
    
    c = 299792.458          # km/s
    bias_fft = 1.4165
    ko = 1e-4
    kmax = 3                # TODO check if this is 3 or 4
    N = 1024
    overdensity_delta = 200 

    def __init__(self, config, ovdelta=200, survey_area=4*np.pi):
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if 'clusters' not in str(sacc_file.tracers.keys()):
            print('Clusters are not within the SACC file tracers. Not performing cluster covariances.')
            return

        self.cosmo = self.get_cosmology()
        mass_def = ccl.halos.MassDef200m()
        self.mass_func = ccl.halos.MassFuncTinker08(self.cosmo, mass_def=mass_def)

        # Read from SACC file relevant quantities
        self.num_z_bins = sacc_file.metadata['nbins_cluster_redshift']
        self.num_richness_bins = sacc_file.metadata['nbins_cluster_richness']
        min_mass = sacc_file.metadata['min_mass']
        # survey_area = sacc_file.metadata['survey_area']

        min_redshifts = [sacc_file.tracers[x].metadata['z_min'] for x in sacc_file.tracers if x.__contains__('clusters')]
        max_redshifts = [sacc_file.tracers[x].metadata['z_max'] for x in sacc_file.tracers if x.__contains__('clusters')]
        min_richness = [sacc_file.tracers[x].metadata['Mproxy_min'] for x in sacc_file.tracers if x.__contains__('clusters')]
        max_richness = [sacc_file.tracers[x].metadata['Mproxy_max'] for x in sacc_file.tracers if x.__contains__('clusters')]

        # Setup Richness Bins
        self.min_richness = min(min_richness)
        self.max_richness = max(max_richness)
        self.richness_bins = np.round(np.logspace(np.log10(self.min_richness),
                                             np.log10(self.max_richness), 
                                             self.num_richness_bins+1), 2)

        # Define arrays for bins for Photometric z and z grid
        self.z_max = max(max_redshifts)
        self.z_min = min(min_redshifts)
        self.z_bins = np.round(np.linspace(self.z_min, self.z_max, self.num_z_bins+1), 2)

        #   minimum log mass in solar masses; 
        self.min_mass = np.log(min_mass)
        #   maximum log mass in solar masses; above this HMF < 10^-10
        self.max_mass = np.log(1e16)

        # Cosmology
        self.survey_area = survey_area
        self.ovdelta = ovdelta
        self.h0 = self.cosmo.h

        # FFT parameters:
        ko = self.ko
        kmax = self.kmax
        self.ro = 1/kmax
        self.rmax = 1/ko
        self.G = np.log(kmax/ko)
        self.L = 2*np.pi*self.N/self.G
        self.k_vec = np.logspace(np.log(self.ko), np.log(self.kmax), self.N, base=np.exp(1))
        self.r_vec = np.logspace(np.log(self.ro), np.log(self.rmax), self.N, base=np.exp(1))

        # minimum z_true for the integrals. I am assuming z_true>0.02 
        self.z_lower_limit = max(0.02, self.z_bins[0]-4*self.z_bin_range) 
        # maximum z_true for the integrals, assuming 40% larger than max z, so we dont need to go till infinity
        self.z_upper_limit = self.z_bins[-1]+6*self.z_bin_range 
        
        # TODO: optimize these ranges like Nelson did interpolation limits for double_bessel_integral
        # zmin & zmax drawn from Z_true_vec
        self.radial_lower_limit = self.radial_distance(self.z_lower_limit)
        self.radial_upper_limit = self.radial_distance(self.z_upper_limit)
        self.imin = self.get_min_radial_idx()
        self.imax = self.get_max_radial_idx()       

        # Do it ONCE
        self.pk_vec = ccl.linear_matter_power(self.cosmo, self.k_vec, 1)
        self.fk_vec = (self.k_vec/self.ko)**(3. - self.bias_fft)*self.pk_vec
        self.Phi_vec = np.conjugate(np.fft.rfft(self.fk_vec))/self.L

        self.z_true_vec = np.linspace(self.z_bins[0], self.z_bins[self.num_z_bins], self.n_z)
        self.sigma_vec = self.eval_sigma_vec(self.n_z)

    def radial_distance(self,z):
        return ccl.comoving_radial_distance(self.cosmo, 1/(1+ z))

    def photoz(self, z_true, z_i, sigma_0=0.05):
        """ 
        Evaluation of Photometric redshift (Photo-z),given true redshift 
        z_true and photometric bin z_i
        
        Note: Z_bins & z_i is a bad choice of variables!
                check how one use the photoz function and adapt it !
        Note: I am truncating the pdf, so as to absorb the negative redshifts 
            into the positive part of the pdf
        Args:
            z_true ()
            z_i
            sigma_0 (float): set as 0.05
        Returns: 
            (array)
        """

        sigma_z = sigma_0*(1+z_true)

        def integrand(z_phot): 
            return np.exp(-(z_phot - z_true)**2. / (2.*sigma_z**2.)) / (np.sqrt(2.*np.pi) * sigma_z)
        
        integral = (
            quad(integrand, self.z_bins[z_i], self.z_bins[z_i+1])[0] / (1.-quad(integrand, -np.inf, 0.)[0])
        )

        return integral


    def dV(self, z_true, z_i):
        ''' Evaluates the comoving volume per steridian as function of 
        z_true for a photometric redshift bin in units of Mpc**3
        Returns:
            dv(z) = dz*dr/dz(z)*(r(z)**2)*photoz(z, bin z_i)
        '''

        dV = self.c * (ccl.comoving_radial_distance(self.cosmo, 1/(1+z_true))**2) \
                    / (100*self.h0*ccl.h_over_h0(self.cosmo, 1/(1+z_true))) * (self.photoz(z_true, z_i))
        return dV


    def mass_richness(self, ln_true_mass, lbd_i):
        """ returns the probability that the true mass ln(M_true) is observed within 
        the bins lambda_i and lambda_i + 1
        """
        richness_bin = self.richness_bins[lbd_i]
        richness_bin_next = self.richness_bins[lbd_i+1]
        return MassRichnessRelation.MurataCostanzi(ln_true_mass, richness_bin, richness_bin_next, self.h0)

    def integral_mass(self, z, lbd_i):
        """ z is the redshift; i is the lambda bin,with lambda from 
        Lambda_bins[i] to Lambda_bins[i+1]
        note: ccl.function returns dn/dlog10m, I am changing integrand below 
        to d(lnM)
        """

        f = lambda ln_m: (1/np.log(10.)) \
            * self.mass_func.get_mass_function(self.cosmo, np.exp(ln_m), 1/(1+z)) \
            * ccl.halo_bias(self.cosmo, np.exp(ln_m), 1/(1+z), overdensity=self.overdensity_delta) \
            * self.mass_richness( ln_m, lbd_i)

        return quad(f, self.min_mass, self.max_mass)[0]


    def integral_mass_no_bias (self, z,lbd_i):
        """ Integral mass for shot noise function
        """
        f = lambda ln_m:(1/np.log(10)) \
            * self.mass_func.get_mass_function(self.cosmo, np.exp(ln_m), 1/(1+z)) \
            * self.mass_richness(ln_m, lbd_i)
        # Remember ccl.function returns dn/dlog10m, I am changing integrand to d(lnM)
        return quad(f, self.min_mass, self.max_mass)[0]


    def Limber(self, z):
        """ Calculating Limber approximation for double Bessel 
        integral for l equal zero
        """
        return ccl.linear_matter_power(self.cosmo, 0.5/ccl.comoving_radial_distance(self.cosmo, 1/(1+z)), 1)/(4*np.pi)


    def cov_Limber(self, z_i, z_j, lbd_i, lbd_j):
        """Calculating the covariance of diagonal terms using Limber (the delta 
        transforms the double redshift integral into a single redshift integral)
        CAUTION: hard-wired ovdelta and survey_area!
        """

        def integrand(z_true): 
            return self.dV(self.cosmo, z_true, z_i) \
            * (ccl.growth_factor(self.cosmo, 1/(1+z_true))**2) \
            * self.photoz(z_true, z_j)\
            * self.integral_mass(self.cosmo, z_true, lbd_i, self.min_mass, self.max_mass, self.ovdelta) \
            * self.integral_mass(self.cosmo, z_true, lbd_j, self.min_mass, self.max_mass, self.ovdelta) \
            * self.Limber(self.cosmo, z_true)

        return (self.survey_area**2) * quad(integrand, self.z_lower_limit, self.z_upper_limit)[0]


    def shot_noise(self, z_i, lbd_i):
        """Evaluates the Shot Noise term
        """

        def integrand(z): 
            return self.c*(ccl.comoving_radial_distance(self.cosmo, 1/(1+z))**2) \
            / (100*self.h0*ccl.h_over_h0(self.cosmo, 1/(1+z))) \
            * self.integral_mass_no_bias(z, lbd_i) \
            * self.photoz(z, z_i)  # TODO remove the bias!

        result = quad(integrand, self.z_lower_limit, self.z_upper_limit)
        return self.survey_area * result[0]


    def I_ell(self, m, R):
        """Calculating the function M_0_0
        the formula below only valid for R <=1, l = 0,  
        formula B2 ASZ and 31 from 2-fast paper 
        """

        t_m = 2*np.pi*m/self.G
        alpha_m = self.bias_fft-1.j*t_m
        pre_factor = (self.ko*self.ro)**(-alpha_m)

        if R < 1:
            iell = pre_factor*0.5*np.cos(np.pi*alpha_m/2)*gamma(alpha_m-2) \
                * (1/R)*((1+R)**(2-alpha_m)-(1-R)**(2-alpha_m))
            
        elif R == 1:
            iell = pre_factor*0.5*np.cos(np.pi*alpha_m/2)*gamma(alpha_m-2) \
                * ((1+R)**(2-alpha_m))

        return iell

    def partial2(self, z1, bin_z_j, bin_lbd_j, approx=True):
        """Romberg integration of a function using scipy.integrate.romberg
        Faster and more reliable than quad used in partial
        Approximation: Put the integral_mass outside looping in m 
        TODO: Check the romberg convergence!
        """
        romb_k = 6

        if (z1<=np.average(self.z_bins)):    
            vec_left = np.linspace(max(self.z_lower_limit, z1-6*self.z_bin_range),z1, 2**(romb_k-1)+1)
            vec_right = np.linspace(z1, z1+(z1-vec_left[0]), 2**(romb_k-1)+1)
            vec_final = np.append(vec_left, vec_right[1:])
        else: 
            vec_right = np.linspace(z1, min(self.z_upper_limit, z1+6*self.z_bin_range), 2**(romb_k-1)+1)
            vec_left = np.linspace(z1-(vec_right[-1]-z1),z1, 2**(romb_k-1)+1)
            vec_final = np.append(vec_left, vec_right[1:])

        romb_range = (vec_final[-1]-vec_final[0])/(2**romb_k)
        kernel = np.zeros(2**romb_k+1)

        if approx:
            for m in range(2**romb_k+1):
                try:
                    kernel[m] = self.dV(vec_final[m],bin_z_j) \
                                * ccl.growth_factor(self.cosmo, 1/(1+vec_final[m])) \
                                * self.double_bessel_integral(z1,vec_final[m])
                except Exception as ex:
                    print(f'{ex=}')
                                
            factor_approx = self.integral_mass(z1, bin_lbd_j)

        else:
            for m in range(2**romb_k+1):
                kernel[m] = self.dV(vec_final[m],bin_z_j)\
                            * ccl.growth_factor(self.cosmo, 1/(1+vec_final[m]))\
                            * self.double_bessel_integral(z1,vec_final[m])\
                            * self.integral_mass(vec_final[m],bin_lbd_j)
                factor_approx = 1 

        return (romb(kernel, dx=romb_range)) * factor_approx


    def double_bessel_integral(self, z1, z2):
        """Calculates the double bessel integral from I-ell algorithm, as function of z1 and z2
        """

        # definition of t, forcing it to be <= 1
        r1 = ccl.comoving_radial_distance(self.cosmo, 1/(1+z1))
        r2 = r1
        R = 1
        if z1 != z2:
            r2 = ccl.comoving_radial_distance(self.cosmo, 1/(1+z2))
            R = min(r1, r2)/max(r1, r2)

        I_ell_vec = [self.I_ell(m, R ) for m in range(self.N//2+1)]

        back_FFT_vec = np.fft.irfft(self.Phi_vec*I_ell_vec)*self.N  # FFT back
        two_fast_vec = (1/np.pi)*(self.ko**3)*((self.r_vec/self.ro) ** (-self.bias_fft))*back_FFT_vec/self.G

        imin = self.imin
        imax = self.imax
       
        # we will use this to interpolate the exact r(z1)
        f = interp1d(self.r_vec[imin:imax], two_fast_vec[imin:imax], kind='cubic') 
        try:
            return f(max(r1, r2))
        except Exception as err:
            print(err,f"""\nValue you tried to interpolate: {max(r1,r2)} Mpc, 
                Input r {r1}, {r2}
            Valid range range: [{self.r_vec[self.imin]}, {self.r_vec[self.imax]}] Mpc""")

        #CHECK THE INDEX NUMBERS
        # TODO test interpolkind
   
    def eval_sigma_vec(self):
        sigma_vec = np.zeros((self.n_z, self.n_z))

        for i in range (self.n_z):
            for j in range (i, self.n_z):

                sigma_vec[i,j] = self.double_bessel_integral(self.z_true_vec[i],self.z_true_vec[j])
                sigma_vec[j,i] = sigma_vec[i,j]

        return sigma_vec
    
    def get_min_radial_idx(self):
        return np.argwhere(self.r_vec < 0.95*self.radial_lower_limit)[-1][0]
    
    def get_max_radial_idx(self):
        return np.argwhere(self.r_vec  > 1.05*self.radial_upper_limit)[0][0]



class CovarianceClusterCounts(CovarianceClusters):
    # Figure out
    # cov_type = 'gauss'
    # _reshape_order = 'F'

    def __init__(self, config):
        super().__init__(config)

        # Find out correct way to pull from sacc
        self.ssc_conf = self.config.get('SSC', {})

        romberg_num = 2**6+1
        self.Z1_true_vec = np.zeros((self.num_z_bins, romberg_num))
        self.G1_true_vec = np.zeros((self.num_z_bins, romberg_num))
        self.dV_true_vec = np.zeros((self.num_z_bins, romberg_num))
        self.M1_true_vec = np.zeros((self.num_richness_bins, self.num_z_bins, romberg_num))
    
    def get_covariance_block(self, tracer_comb1=None, tracer_comb2=None,
                             integration_method=None,
                             include_b_modes=True):
        
        romberg_num = 2**6+1
        # Computes the geometric true vectors
        self.eval_true_vec(romberg_num)
        # Pre computes the true vectors M1 for Cov_N_N
        self.eval_M1_true_vec(romberg_num)

        final_array = np.zeros((self.num_richness_bins, self.num_richness_bins, self.num_z_bins, self.num_z_bins))

        # li, lj: richness_i, richness_j
        # zi, zj: redshift_i, redshift_j
        matrixElement = namedtuple('matrixElement', ['li', 'lj', 'zi', 'zj'])
        elemList = []

        for richness_i in range(self.num_richness_bins):   
            for richness_j in range(richness_i, self.num_richness_bins):
                for z_i in range (self.num_z_bins):
                    for z_j in range(z_i, self.num_z_bins):
                        elemList.append(matrixElement(richness_i, richness_j, z_i, z_j))

        for e in elemList:
        
            cov = self._covariance_cluster_NxN(e.zi, e.zj, e.li, e.lj, romberg_num)
            shot_noise = 0
            if (e.li == e.lj and e.zi == e.zj):
                shot_noise = self.shot_noise(e.zi, e.li)
                        
            cov_term = shot_noise + cov
            final_array[e.li,e.lj,e.zi,e.zj]= cov_term
            final_array[e.li,e.lj,e.zj,e.zi]= final_array[e.li,e.lj,e.zi,e.zj]
            final_array[e.lj,e.li,e.zi,e.zj]= final_array[e.li,e.lj,e.zi,e.zj]      

        return final_array

    def _covariance_cluster_NxN(self, z_i, z_j, richness_i, richness_j, romberg_num):
        """ Cluster counts covariance
        Args:
            bin_z_i (float or ?array): tomographic bins in z_i or z_j
            bin_lbd_i (float or ?array): bins of richness (usually log spaced)
        Returns:
            float: Covariance at given bins
        """
        dz = (self.Z1_true_vec[z_i, -1]-self.Z1_true_vec[z_i, 0])/(romberg_num-1)
        
        partial_vec = [
            self.partial2(self.Z1_true_vec[z_i, m], z_j, richness_j) for m in range(romberg_num)
        ]
        
        romb_vec = partial_vec*self.dV_true_vec[z_i]*self.M1_true_vec[richness_i, z_i]*self.G1_true_vec[z_i]

        return (self.survey_area**2)*romb(romb_vec, dx=dz)


    def eval_true_vec(self, romb_num):
        """ Computes the -geometric- true vectors Z1, G1, dV for Cov_N_N. 
        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 
        Returns:
            (array) Z1_true_vec
            (array) G1_true_vec
            (array) dV_true_vec
        """

        self.Z1_true_vec = np.zeros((self.num_z_bins, romb_num))
        self.G1_true_vec = np.zeros((self.num_z_bins, romb_num))
        self.dV_true_vec = np.zeros((self.num_z_bins, romb_num))

        for i in range(self.num_z_bins):

            z_low_limit = max(self.z_lower_limit, self.z_bins[i]-4*self.z_bin_range)
            z_upper_limit = min(self.z_upper_limit, self.z_bins[i+1]+6*self.z_bin_range)
            
            self.Z1_true_vec[i] = np.linspace(z_low_limit, z_upper_limit, romb_num)
            self.G1_true_vec[i] = ccl.growth_factor(self.cosmo, 1/(1+self.Z1_true_vec[i]))
            self.dV_true_vec[i] = [self.dV(self.Z1_true_vec[i, m], i) for m in range(romb_num)]


    def eval_M1_true_vec(self, romb_num):
        """ Pre computes the true vectors M1 for Cov_N_N. 
        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 
        """

        print('evaluating M1_true_vec (this may take some time)...')

        self.M1_true_vec = np.zeros((self.num_richness_bins, self.num_z_bins, romb_num))

        for lbd in range(self.num_richness_bins):
            for z in range(self.num_z_bins):
                for m in range(romb_num):

                    self.M1_true_vec[lbd, z, m] = self.integral_mass(self.Z1_true_vec[z, m], lbd)


    def get_covariance_block(self, tracer_comb1=None, tracer_comb2=None,
                             integration_method=None,
                             include_b_modes=True):
        """
        Compute a single SSC covariance matrix for a given pair of C_ell. If
        outdir is set, it will save the covariance to a file called
        `ssc_tr1_tr2_tr3_tr4.npz`. This file will be read and its output
        returned if found.

        Blocks of the B-modes are assumed 0 so far.

        Parameters:
        -----------
            tracer_comb 1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb 2 (list): List of the pair of tracer names of C_ell^2
            ccl_tracers (dict): Dictionary with necessary ccl_tracers with keys
            the tracer names
            integration_method (string): integration method to be used
            for the Limber integrals. Possibilities: 'qag_quad' (GSL's `qag`
            method backed up by `quad` when it fails) and 'spline' (the
            integrand is splined and then integrated analytically). If given,
            it will take priority over the specified in the configuration file
            through config['SSC']['integration_method']. Elsewise, it will use
            'qag_quad'.
            include_b_modes (bool): If True, return the full SSC with zeros in
            for B-modes (if any). If False, return the non-zero block. This
            option cannot be modified through the configuration file to avoid
            breaking the compatibility with the NaMaster covariance.

        Returns:
        --------
            cov (array):  Super sample covariance matrix for a pair of C_ell.

        """
        fname = 'ssc_{}_{}_{}_{}.npz'.format(*tracer_comb1, *tracer_comb2)
        fname = os.path.join(self.io.outdir, fname)
        if os.path.isfile(fname):
            cf = np.load(fname)
            return cf['cov' if include_b_modes else 'cov_nob']

        if integration_method is None:
            integration_method = self.ssc_conf.get('integration_method',
                                                   'qag_quad')

        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2

        cosmo = self.cosmo
        mass_def = ccl.halos.MassDef200m()
        hmf = ccl.halos.MassFuncTinker08(cosmo,
                                         mass_def=mass_def)
        hbf = ccl.halos.HaloBiasTinker10(cosmo,
                                         mass_def=mass_def)
        nfw = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(mass_def),
                                       fourier_analytic=True)
        hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)

        # Get range of redshifts. z_min = 0 for compatibility with the limber
        # integrals
        sacc_file = self.io.get_sacc_file()
        z_max = []
        for i in range(4):
            tr_sacc = sacc_file.tracers[tr[i + 1]]
            z, nz = tr_sacc.z, tr_sacc.nz
            # z_min.append(z[np.where(nz > 0)[0][0]])
            # z_max.append(z[np.where(np.cumsum(nz)/np.sum(nz) > 0.999)[0][0]])
            z_max.append(z.max())

        z_max = np.min(z_max)

        # Array of a.
        # Use the a's in the pk spline
        na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        a, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)
        a = a[1/a < z_max + 1]

        bias1 = self.bias_lens.get(tr[1], 1)
        bias2 = self.bias_lens.get(tr[2], 1)
        bias3 = self.bias_lens.get(tr[3], 1)
        bias4 = self.bias_lens.get(tr[4], 1)

        ccl_tracers, _ = self.get_tracer_info()

        isnc1 = isinstance(ccl_tracers[tr[1]], ccl.NumberCountsTracer)
        isnc2 = isinstance(ccl_tracers[tr[2]], ccl.NumberCountsTracer)
        isnc3 = isinstance(ccl_tracers[tr[3]], ccl.NumberCountsTracer)
        isnc4 = isinstance(ccl_tracers[tr[4]], ccl.NumberCountsTracer)

        tk3D = ccl.halos.halomod_Tk3D_SSC_linear_bias(cosmo=cosmo, hmc=hmc,
                                                      prof=nfw,
                                                      bias1=bias1,
                                                      bias2=bias2,
                                                      bias3=bias3,
                                                      bias4=bias4,
                                                      is_number_counts1=isnc1,
                                                      is_number_counts2=isnc2,
                                                      is_number_counts3=isnc3,
                                                      is_number_counts4=isnc4,
                                                      )

        masks = self.get_masks_dict(tr, {})
        # TODO: Optimize this, avoid computing the mask_wl for all blocks.
        # Note that this is correct for same footprint cross-correlations. In
        # case of multisurvey analyses this approximation might break.
        m12 = masks[1] * masks[2]
        m34 = masks[3] * masks[4]
        area = hp.nside2pixarea(hp.npix2nside(m12.size))

        alm = hp.map2alm(m12)
        blm = hp.map2alm(m34)

        mask_wl = hp.alm2cl(alm, blm)
        mask_wl *= (2 * np.arange(mask_wl.size) + 1)
        mask_wl /= np.sum(m12) * np.sum(m34) * area**2

        sigma2_B = ccl.sigma2_B_from_mask(cosmo, a=a, mask_wl=mask_wl)

        ell = self.get_ell_eff()
        cov_ssc = ccl.covariances.angular_cl_cov_SSC(cosmo,
                                                     cltracer1=ccl_tracers[tr[1]],
                                                     cltracer2=ccl_tracers[tr[2]],
                                                     cltracer3=ccl_tracers[tr[3]],
                                                     cltracer4=ccl_tracers[tr[4]],
                                                     ell=ell,
                                                     tkka=tk3D,
                                                     sigma2_B=(a, sigma2_B),
                                                     integration_method=integration_method)

        nbpw = ell.size
        ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
        ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
        cov_full = np.zeros((nbpw, ncell1, nbpw, ncell2))
        cov_full[:, 0, :, 0] = cov_ssc
        cov_full = cov_full.reshape((nbpw * ncell1, nbpw * ncell2),
                                    order=self._reshape_order)

        np.savez_compressed(fname, cov=cov_full, cov_nob=cov_ssc)

        if not include_b_modes:
            return cov_ssc

        return cov_full


class MassRichnessRelation(object):

    @staticmethod
    def MurataCostanzi(ln_true_mass, richness_bin, richness_bin_next, h0):
        """
        Define lognormal mass-richness relation 
        (leveraging paper from Murata et. alli - ArxIv 1707.01907 and Costanzi et al ArxIv 1810.09456v1)
       
        Args:
            ln_true_mass: ln(true mass)
            richness_bin: ith richness bin
            richness_bin_next: i+1th richness bin
            h0: 
        Returns:
            The probability that the true mass ln(ln_true_mass) is observed within 
            the bins richness_bin and richness_bin_next
        """

        alpha = 3.207           # Murata
        beta = 0.75             # Costanzi
        sigma_zero = 2.68       # Costanzi
        q = 0.54                # Costanzi
        m_pivot = 3.e+14/h0     # in solar masses , Murata and Costanzi use it

        sigma_lambda = sigma_zero+q*(ln_true_mass - np.log(m_pivot))
        average = alpha+beta*(ln_true_mass-np.log(m_pivot))

        def integrand(richness):
            return (1./richness) * np.exp(-(np.log(richness)-average)**2. / (2.*sigma_lambda**2.))\
                / (np.sqrt(2.*np.pi) * sigma_lambda)
        
        return (quad(integrand, richness_bin, richness_bin_next)[0])