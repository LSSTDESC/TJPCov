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
    # get c from CCL
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
        if self.min_richness == 0:
            self.min_richness = 1.
        self.max_richness = max(max_richness)
        self.richness_bins = np.round(np.logspace(np.log10(self.min_richness),
                                             np.log10(self.max_richness), 
                                             self.num_richness_bins+1), 2)

        # Define arrays for bins for Photometric z and z grid
        self.z_max = max(max_redshifts)
        self.z_min = min(min_redshifts)
        if self.z_min == 0:
            self.z_min = 0.01
        self.z_bins = np.round(np.linspace(self.z_min, self.z_max, self.num_z_bins+1), 2)
        self.z_bin_range = (self.z_max - self.z_min)/self.num_z_bins

        #   minimum log mass in solar masses; 
        self.min_mass = np.log(min_mass)
        #   maximum log mass in solar masses; above this HMF < 10^-10
        self.max_mass = np.log(1e16)

        # Cosmology
        self.survey_area = survey_area
        self.ovdelta = ovdelta
        # TODO Read from SACC
        self.h0 = 0.6736

        # FFT parameters:
        # TODO test 
        self.ro = 1/self.kmax
        self.rmax = 1/self.ko
        self.G = np.log(self.kmax/self.ko)
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

        self.z_true_vec = np.linspace(self.z_bins[0], self.z_bins[self.num_z_bins], self.num_z_bins)
        # Eq 7.6
        self.sigma_vec = self.eval_sigma_vec()

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

    # TODO vectorize 
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


    #TODO Most important integral 
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

        I_ell_vec = [self.I_ell(m, R) for m in range(self.N//2+1)]

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
        sigma_vec = np.zeros((self.num_z_bins, self.num_z_bins))

        for i in range (self.num_z_bins):
            for j in range (i, self.num_z_bins):

                sigma_vec[i,j] = self.double_bessel_integral(self.z_true_vec[i],self.z_true_vec[j])
                sigma_vec[j,i] = sigma_vec[i,j]

        return sigma_vec
    
    def get_min_radial_idx(self):
        return np.argwhere(self.r_vec < 0.95*self.radial_lower_limit)[-1][0]
    
    def get_max_radial_idx(self):
        return np.argwhere(self.r_vec  > 1.05*self.radial_upper_limit)[0][0]

    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        raise NotImplementedError("Not implemented")


class CovarianceClusterCounts(CovarianceClusters):
    # Figure out
    # cov_type = 'fourier'
    _reshape_order = 'F'

    def __init__(self, config):
        super().__init__(config)

        # Find out correct way to pull from sacc
        self.ssc_conf = self.config.get('SSC', {})

        self.romberg_num = 2**6+1
        self.Z1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.G1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.dV_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.M1_true_vec = np.zeros((self.num_richness_bins, self.num_z_bins, self.romberg_num))

        #TODO  this should be moved to evaluate 1 entry at a time, so we can use parallelization
        # Computes the geometric true vectors
        self.eval_true_vec()
        # Pre computes the true vectors M1 for Cov_N_N
        self.eval_M1_true_vec()
    
    def get_covariance_block_for_sacc(self, tracer_comb1, tracer_comb2, **kwargs):
        self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2, kwargs)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2, kwargs)

    def get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2, **kwargs):

        # Compute a single covariance entry 'clusters_redshift_richness' e.g.
        # tracer_comb1 = ('clusters_0_0',) 
        # tracer_comb2 = ('clusters_0_0',)
        tracer_split1 = tracer_comb1[0].split('_')
        tracer_split2 = tracer_comb2[0].split('_')
        
        z_i = tracer_split1[1]
        richness_i = tracer_split1[2]

        z_j = tracer_split2[1]
        richness_j = tracer_split2[2]


        dz = (self.Z1_true_vec[z_i, -1]-self.Z1_true_vec[z_i, 0])/(self.romberg_num-1)
        
        partial_vec = [
            self.partial2(self.Z1_true_vec[z_i, m], z_j, richness_j) for m in range(self.romberg_num)
        ]
        
        romb_vec = partial_vec*self.dV_true_vec[z_i]*self.M1_true_vec[richness_i, z_i]*self.G1_true_vec[z_i]

        cov = (self.survey_area**2)*romb(romb_vec, dx=dz)

        shot_noise = 0
        if (richness_i == richness_j and z_i == z_j):
            shot_noise = self.shot_noise(z_i, richness_i)
                    
        cov_total = shot_noise + cov
        
        # store metadata in some header/log file
        # covariance_io
        return cov_total

    def _covariance_cluster_NxN(self, z_i, z_j, richness_i, richness_j):
        """ Cluster counts covariance
        Args:
            bin_z_i (float or ?array): tomographic bins in z_i or z_j
            bin_lbd_i (float or ?array): bins of richness (usually log spaced)
        Returns:
            float: Covariance at given bins
        """



    def eval_true_vec(self):
        """ Computes the -geometric- true vectors Z1, G1, dV for Cov_N_N. 
        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 
        Returns:
            (array) Z1_true_vec
            (array) G1_true_vec
            (array) dV_true_vec
        """

        for i in range(self.num_z_bins):

            z_low_limit = max(self.z_lower_limit, self.z_bins[i]-4*self.z_bin_range)
            z_upper_limit = min(self.z_upper_limit, self.z_bins[i+1]+6*self.z_bin_range)
            
            self.Z1_true_vec[i] = np.linspace(z_low_limit, z_upper_limit, self.romberg_num)
            self.G1_true_vec[i] = ccl.growth_factor(self.cosmo, 1/(1+self.Z1_true_vec[i]))
            self.dV_true_vec[i] = [self.dV(self.Z1_true_vec[i, m], i) for m in range(self.romberg_num)]


    def eval_M1_true_vec(self, romb_num):
        """ Pre computes the true vectors M1 for Cov_N_N. 
        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 
        """

        print('evaluating M1_true_vec (this may take some time)...')

        for lbd in range(self.num_richness_bins):
            for z in range(self.num_z_bins):
                for m in range(romb_num):
                    self.M1_true_vec[lbd, z, m] = self.integral_mass(self.Z1_true_vec[z, m], lbd)

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