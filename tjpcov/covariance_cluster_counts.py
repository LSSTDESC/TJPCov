from .covariance_builder import CovarianceBuilder
from .clusters_helpers import MassRichnessRelation, FFTHelper
import numpy as np
import pyccl as ccl
from scipy.integrate import quad, romb
from scipy.integrate import simpson as simps
from sacc import standard_types

from scipy.special import spherical_jn, eval_legendre

from scipy.integrate import simpson as simps
        
class CovarianceClusterCounts(CovarianceBuilder):
    """Class to calculate covariance of cluster counts."""

    space_type = "Fourier"
    _tracer_types = (
        standard_types.cluster_counts,
        standard_types.cluster_counts,
    )

    def __init__(self, config, min_halo_mass=1e13):
        """Class to calculate covariance of cluster counts.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
            min_halo_mass (float, optional): Minimum halo mass.
        """
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if "cluster_counts" not in sacc_file.get_data_types():
            raise ValueError(
                "Cluster count covariance was requested but cluster count data"
                + " points were not included in the sacc file."
            )

        self.load_from_sacc(sacc_file, min_halo_mass)

        cosmo = self.get_cosmology()
        self.load_from_cosmology(cosmo)

        # Quick key to skip P(Richness|M)
        self.has_mproxy = self.config.get("has_mproxy", True)
        self.covariance_block_data_type = standard_types.cluster_counts

    def load_from_cosmology(self, cosmo):
        """Load parameters from a CCL cosmology object.

        Derived attributes from the cosmology are set here.

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
        """
        self.cosmo = cosmo
        self.c  = ccl.physical_constants.CLIGHT / 1000
        self.h0 = float(self.config["parameters"].get("h"))
        
        mass_def = ccl.halos.MassDef200m                                 #### better to def in config file?
        self.mass_func = ccl.halos.MassFuncTinker08(mass_def=mass_def)   #### better to def in config file?  
        self.hbias = ccl.halos.HaloBiasTinker10(mass_def=mass_def)       #### better to def in config file?
        self.fullsky = False                                             #### better to def in config file?

        # photo-z scatter
        self.sigma_0 = float(self.config["photo-z"].get("sigma_0"))  
        
        # mass-observable relation parameters
        self.mor_alpha = float(self.config["mor_parameters"].get("alpha"))   
        self.mor_beta = float(self.config["mor_parameters"].get("beta"))     
        self.mor_sigma_zero = float(self.config["mor_parameters"].get("sigma_zero"))   
        self.mor_q = float(self.config["mor_parameters"].get("q"))   
        self.mor_m_pivot = float(self.config["mor_parameters"].get("m_pivot")) / 0.7   # Msun (convert units using a fiducial value for h, if the self.h0 is used this would add an extra dependence on h)
        
        
    def load_from_sacc(self, sacc_file, min_halo_mass):
        """Set class attributes based on data from the SACC file.

        Cluster covariance has special parameters set in the SACC file. This
        informs the code that the data to calculate the cluster covariance is
        there.  We set extract those values from the sacc file here, and set
        the attributes here.

        Args:
            sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object, already
            loaded.
        """

        z_tracer_type = "bin_z"
        survey_tracer_type = "survey"
        richness_tracer_type = "bin_richness"

        survey_tracer = [
            x
            for x in sacc_file.tracers.values()
            if x.tracer_type == survey_tracer_type
        ]
        if len(survey_tracer) == 0:
            self.survey_tracer_nm = ""
            self.survey_area = 4 * np.pi
            print(
                "Survey tracer not provided in sacc file.\n"
                + "We will use the default value.",
                flush=True,
            )
        else:
            self.survey_area = survey_tracer[0].sky_area * (np.pi / 180) ** 2

        # Setup redshift bins
        z_bins = [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == z_tracer_type
        ]
        self.num_z_bins = len(z_bins)
        self.z_min = z_bins[0].lower
        self.z_max = z_bins[-1].upper
        self.z_bins = np.round(
            np.linspace(self.z_min, self.z_max, self.num_z_bins + 1), 2
        )
        self.z_bin_spacing = (self.z_max - self.z_min) / self.num_z_bins
        self.z_lower_limit = max(0.02, self.z_bins[0] - 4 * self.z_bin_spacing)
        self.z_upper_limit = self.z_bins[-1] + 0.4 * self.z_bins[-1]      # Set upper limit to be 40% higher than max redshift

        # Setup richness bins
        richness_bins = [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == richness_tracer_type
        ]
        self.num_richness_bins = len(richness_bins)
        self.min_richness = 10 ** richness_bins[0].lower
        self.max_richness = 10 ** richness_bins[-1].upper
        self.richness_bins = np.round(
            np.logspace(
                np.log10(self.min_richness),
                np.log10(self.max_richness),
                self.num_richness_bins + 1,
            ),
            2,
        )

        self.min_mass = np.log(min_halo_mass)
        self.max_mass = np.log(1e16)

    def _quad_integrate(self, argument, from_lim, to_lim):
        """Numerically integrate argument between bounds using scipy quad.

        Args:
            argument (callable): Function to integrate between bounds
            from_lim (float): lower limit
            to_lim (float): upper limit

        Returns:
            float: Value of the integral
        """

        integral_value = quad(argument, from_lim, to_lim)
        return integral_value[0]


    def observed_photo_z(self, z_true, z_i, sigma_0):
        """Implementation of the photometric redshift uncertainty distribution.

        We don't assume that redshift can be measured exactly, so we include
        a measurement of the uncertainty around photometric redshifts. Assume,
        given a true redshift z, the measured redshift will be gaussian. The
        uncertainty will increase with redshift bin.

        See section 2.3 of N. Ferreira

        Args:
            z_true (float): True redshift
            z_i (float): Photometric redshift bin index            
        Returns:
            float: Probability weighted photo-z
        """

        sigma_z = sigma_0 * (1 + z_true)

        def integrand(z_phot):
            prefactor = 1 / (np.sqrt(2.0 * np.pi) * sigma_z)
            dist = np.exp(-(1 / 2) * ((z_phot - z_true) / sigma_z) ** 2.0)
            return prefactor * dist

        # Using the formula for a truncated normal distribution
        numerator = self._quad_integrate(
            integrand, self.z_bins[z_i], self.z_bins[z_i + 1]
        )
        denominator = 1.0 - self._quad_integrate(integrand, -np.inf, 0.0)

        return numerator / denominator
    

    def comoving_volume_element(self, z_true, z_i, sigma_0):  
        """Calculates the volume element for this bin.

        Given a true redshift, and a redshift bin, this will give the
        volume element for this bin including photo-z uncertainties.

        Args:
            z_true (float): True redshift
            z_i (float): Photometric redshift bin

        Returns:
            float: Photo-z-weighted comoving volume element per steridian
            for redshift bin i in units of Mpc^3
        """
        dV = (
            self.c
            * (ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z_true)) ** 2)
            / (100 * self.h0 * ccl.h_over_h0(self.cosmo, 1 / (1 + z_true)))
            * (self.observed_photo_z(z_true, z_i, sigma_0))
        )
        return dV
    

    def mass_richness(self, ln_true_mass, richness_i):   
        """Log-normal mass-richness relation without observational scatter.

        The probability that we observe richness given the true mass M, is
        given by the convolution of a Poisson distribution (relating observed
        richness to true richness) with a Gaussian distribution (relating true
        richness to M). Such convolution can be translated into a parametrized
        log-normal mass-richness distribution, done so here.

        Args:
            ln_true_mass (float): True mass
            richness_bin (int): Richness bin i
        Returns:
            float: The probability that the true mass ln(ln_true_mass)
            is observed within the richness bin i and richness bin i+1
        """

        richness_bin = self.richness_bins[richness_i]
        richness_bin_next = self.richness_bins[richness_i + 1]

        #### mass-obs relation params to be added as input params
        std_deviation, average = MassRichnessRelation.MurataCostanzi(
            ln_true_mass, self.h0, self.mor_alpha, self.mor_beta, self.mor_sigma_zero, self.mor_q, self.mor_m_pivot
        )                                 

        def integrand(richness):
            prefactor = 1.0 / (
                richness * (np.sqrt(2.0 * np.pi) * std_deviation)
            )
            distribution = np.exp(
                -(1 / 2) * ((np.log(richness) - average) / std_deviation) ** 2
            )
            return prefactor * distribution

        return self._quad_integrate(integrand, richness_bin, richness_bin_next)

    
    def mass_richness_integral(self, z, richness_i, remove_bias=False):
        """Integrates the HMF weighted by mass-richness relation.

        The halo mass function weighted by the probability that we measure
        observed richness lambda given true mass M.

        Args:
            z (float): Redshift
            lbd_i (int): Richness bin
            remove_bias (bool, optional): If TRUE, will remove halo_bias from
            the mass integral. Used for calculating the shot noise.
        Returns:
            float: The mass-richness weighed derivative of number density per
            fluctuation in background
        """

        def integrand(ln_m):
            argument = 1 / np.log(10.0)

            scale_factor = 1 / (1 + z)

            mass_func = self.mass_func(self.cosmo, np.exp(ln_m), scale_factor)

            argument *= mass_func

            if not remove_bias:
                halo_bias = self.hbias(
                    self.cosmo,
                    np.exp(ln_m),
                    scale_factor,
                )
                argument *= halo_bias

            if self.has_mproxy:
                argument *= self.mass_richness(ln_m, richness_i)

            return argument

        if self.has_mproxy:
            m_integ_lower, m_integ_upper = self.min_mass, self.max_mass
        else:
            m_integ_lower = np.log(10) * self.richness_bins[richness_i]
            m_integ_upper = np.log(10) * self.richness_bins[richness_i + 1]

        return self._quad_integrate(integrand, m_integ_lower, m_integ_upper)


    # spherical harmonics coefficients                                         
    def Kl_func(self, L, theta):
        """Harmonic expansion coefficients.

        Coefficients for the redshift-slice window function 
        See Costanzi+19 (arXiv:1810.09456v1) and Fumagalli+21 (arXiv:2102.08914v1).
        For L=0 full-sky approximation.

        Args:
            L (int): number of multipoles for the expansion (suggested for partial-sky: L=20)
            theta (float): angular aperture of the lightcone
        Returns:
            array: L coefficients
        """
        
        Kl    = np.array([np.sqrt(np.pi/(2.*l+1.))*
                          (eval_legendre(l-1,np.cos(theta))- eval_legendre(l+1,np.cos(theta)))
                          /(2.*np.pi*(1-np.cos(theta))) for l in range(L+1)])
        Kl[0] = 1/(2.*np.sqrt(np.pi))
        return Kl


    # window function
    def window_redshift_bin(self,k_arr,z_arr,iz,L,sigma_0):
        """Redshift-slice window function

        Window function describing the geometry of the lightcone redshift slice.

        Args:
            k_arr (array): wavenumbers in 1/Mpc
            z_arr (array): true redshift
            iz (int): photometric redshift bin
            L (int): number of multipoles for the expansion (suggested for partial-sky: L=20)
        Returns:
            array: growth factor times window function of the redshift slice, averaged over the redshift bin
        """        
        
        # harmonic expansion coefficeints
        theta_sky = np.arccos(1-self.survey_area/(2*np.pi))
        KL        = self.Kl_func(L, theta_sky)    

        # redshift-dependent quantities
        rz     = ccl.comoving_radial_distance(self.cosmo, 1/(1+z_arr))   # Mpc

        dVdzob = np.array([self.comoving_volume_element(z,iz,sigma_0) for z in z_arr])
        Vz     = simps(dVdzob, x=z_arr)
        D      = ccl.growth_factor(self.cosmo, 1/(1+z_arr))

        # integral over redshift
        jl_kz  = np.array([spherical_jn(l,k_arr[:,None]*rz) for l in range(L+1)]).T
        rint   = simps((dVdzob * D)[:,None,None] * jl_kz, x=z_arr, axis=0) / Vz

        return 4*np.pi * rint * KL



    def super_sample_covariance(self):
        """super-sample covariance

        super sample covariance term of the number counts covariance

        Args:
            k_arr (array): wavenumbers in 1/Mpc
            z_arr (array): true redshift
            iz (int): photometric redshift bin
            L (int): number of multipoles for the expansion (suggested for partial-sky: L=20)
            Omega_sky (float): survey area in deg^2
        Returns:
            array: growth factor times window function of the redshift slice, averaged over the redshift bin
        """  

        L = 20
        if self.fullsky==True:
            L = 0

        m_arr = np.logspace(self.min_mass*np.log10(np.e), self.max_mass*np.log10(np.e), 250)      # Msun
        z_arr = np.linspace(self.z_lower_limit,self.z_upper_limit, 300)
        k_arr = np.geomspace(1e-4,2e1,700)  # 1/Mpc

        #### number counts*bias and window function   
        Nb_lob_zob = np.zeros((self.num_richness_bins,self.num_z_bins))  
        Wi_l = np.zeros((self.num_z_bins,len(k_arr),L+1))
        
        for iz in range(self.num_z_bins):
            
            # true redshift for integration
            z_tr = np.linspace(max(self.z_bins[iz]-0.3, 0.02),min(self.z_bins[iz+1]+0.3,0.91),200)

            # observed volume element
            dVdzob = np.array([self.comoving_volume_element(z,iz,self.sigma_0) for z in z_tr])

            # number counts and bias in observed redshift and richness bins
            for il in range(self.num_richness_bins):

                Nb_lob_z = np.array([self.mass_richness_integral(z,il,remove_bias=False) for z in z_tr])
                
                Nb_lob_zob[il,iz] = simps(dVdzob * Nb_lob_z, x=z_tr, axis=0)

            # window function of the i-th redshift bin
            Wi_l[iz] = self.window_redshift_bin(k_arr,z_tr,iz,L,self.sigma_0)

        # sum over ell of W_i * W_j
        WiWj_sum = np.sum(Wi_l[:,None,:,:] * Wi_l[None,:,:,:], axis=-1)  

        # sample covariance
        pk0         = ccl.linear_matter_power(self.cosmo,k_arr, 1.) # Mpc^3
        sigma2_zizj = 1/(2*np.pi)**3 * simps(k_arr**2 * pk0 * WiWj_sum, x=k_arr, axis=-1) 

        # SSC, dim=[richness,richness,redshift,redshift]
        SSC = self.survey_area**2 * (Nb_lob_zob.reshape(1,self.num_richness_bins,1,self.num_z_bins) * 
                                     Nb_lob_zob.reshape(self.num_richness_bins,1,self.num_z_bins,1) * 
                                     sigma2_zizj.reshape(1,1,self.num_z_bins,self.num_z_bins))

        return SSC







