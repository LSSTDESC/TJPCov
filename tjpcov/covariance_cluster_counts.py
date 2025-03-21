from .covariance_builder import CovarianceBuilder
import numpy as np
import pyccl as ccl
from scipy.integrate import quad
from scipy.integrate import simpson as simps
from sacc import standard_types
from scipy.special import spherical_jn, eval_legendre
from firecrown.models.cluster.mass_proxy import MurataBinned
from .clusters_helpers import _load_from_sacc, mass_func_map, halo_bias_map


class CovarianceClusterCounts(CovarianceBuilder):
    """Class to calculate covariance of cluster counts."""

    space_type = "Fourier"
    _tracer_types = (
        standard_types.cluster_counts,
        standard_types.cluster_counts,
    )

    def __init__(self, config):
        """Class to calculate covariance of cluster counts.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if "cluster_counts" not in sacc_file.get_data_types():
            raise ValueError(
                "Cluster count covariance was requested but cluster count data"
                + " points were not included in the sacc file."
            )

        cosmo = self.get_cosmology()
        self.load_from_cosmology(cosmo)
        self.load_cluster_parameters()
        self.load_from_sacc(sacc_file)
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
        self.c = ccl.physical_constants.CLIGHT / 1000
        self.h0 = float(self.config["parameters"].get("h"))

    def load_cluster_parameters(self):
        """Load cluster parameters from the configuration file."""
        mass_func_name = self.config["mor_parameters"].get("mass_func")
        halo_bias_name = self.config["mor_parameters"].get("halo_bias")
        self.mass_def = self.config["mor_parameters"].get("mass_def")
        self.min_halo_mass = float(
            self.config["mor_parameters"].get("min_halo_mass")
        )
        self.max_halo_mass = float(
            self.config["mor_parameters"].get("max_halo_mass")
        )

        if mass_func_name not in mass_func_map:
            raise ValueError(f"Invalid mass function: {mass_func_name}")
        if halo_bias_name not in halo_bias_map:
            raise ValueError(f"Invalid halo bias: {halo_bias_name}")

        # Create the mass definition, mass function, and halo bias objects
        self.mass_func = mass_func_map[mass_func_name](mass_def=self.mass_def)
        self.hbias = halo_bias_map[halo_bias_name](mass_def=self.mass_def)

        self.fullsky = False

        # photo-z scatter
        self.sigma_0 = float(self.config["photo-z"].get("sigma_0"))

        # mass-observable relation parameters
        self.mor_m_pivot = float(self.config["mor_parameters"].get("m_pivot"))
        self.mor_mu_p0 = float(self.config["mor_parameters"].get("mu_p0"))
        self.mor_mu_p1 = float(self.config["mor_parameters"].get("mu_p1"))
        self.mor_mu_p2 = float(self.config["mor_parameters"].get("mu_p2"))
        self.mor_sigma_p0 = float(
            self.config["mor_parameters"].get("sigma_p0")
        )
        self.mor_sigma_p1 = float(
            self.config["mor_parameters"].get("sigma_p1")
        )
        self.mor_sigma_p2 = float(
            self.config["mor_parameters"].get("sigma_p2")
        )
        # Msun (convert units using a fiducial value for h,
        # if the self.h0 is used this would add an extra dependence on h)
        self.mor_z_pivot = float(self.config["mor_parameters"].get("z_pivot"))

    def load_from_sacc(self, sacc_file):
        """Set class attributes based on data from the SACC file.

        Cluster covariance has special parameters set in the SACC file. This
        informs the code that the data to calculate the cluster covariance is
        there.  We set extract those values from the sacc file here, and set
        the attributes here.

        Args:
            sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object, already
            loaded.
        """
        attributes = _load_from_sacc(
            sacc_file, self.min_halo_mass, self.max_halo_mass
        )
        for key, value in attributes.items():
            setattr(self, key, value)

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

    def mass_richness(self, ln_true_mass, z, richness_i):
        """Log-normal mass-richness relation without observational scatter.

        The probability that we observe richness given the true mass M, is
        given by the convolution of a Poisson distribution (relating observed
        richness to true richness) with a Gaussian distribution (relating true
        richness to M). Such convolution can be translated into a parametrized
        log-normal mass-richness distribution, done so here.

        Args:
            ln_true_mass (float): True mass
            z (float): Redshift
            richness_bin (int): Richness bin i
        Returns:
            float: The probability that the true mass ln(ln_true_mass)
            is observed within the richness bin i and richness bin i+1
        """
        richness_lower = np.log10(self.richness_bins[richness_i])
        richness_upper = np.log10(self.richness_bins[richness_i + 1])
        rich_bin = (richness_lower, richness_upper)
        mass_richness_prob = MurataBinned(
            np.log10(self.mor_m_pivot), self.mor_z_pivot
        )
        # mass-obs relation params to be added as input params
        mass_richness_prob.mu_p0 = self.mor_mu_p0
        mass_richness_prob.mu_p1 = self.mor_mu_p1
        mass_richness_prob.mu_p2 = self.mor_mu_p2
        mass_richness_prob.sigma_p0 = self.mor_sigma_p0
        mass_richness_prob.sigma_p1 = self.mor_sigma_p1
        mass_richness_prob.sigma_p2 = self.mor_sigma_p2
        ln_true_mass = np.atleast_1d(ln_true_mass).astype(np.float64)
        z = np.atleast_1d(z).astype(np.float64)
        result = mass_richness_prob.distribution(
            ln_true_mass / np.log(10), z, rich_bin
        )

        return result

    def mass_richness_integral(self, z, richness_i, remove_bias=False):
        """Integrates the HMF weighted by mass-richness relation.

        The halo mass function weighted by the probability that we measure
        observed richness lambda given true mass M.

        Args:
            z (float): Redshift
            richness_i (int): Richness bin
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
                argument *= self.mass_richness(
                    np.array([ln_m]), np.array([z]), np.array([richness_i])
                )

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
        See Costanzi+19 (arXiv:1810.09456v1)
        and Fumagalli+21 (arXiv:2102.08914v1).
        For L=0 full-sky approximation.

        Args:
            L (int): number of multipoles for the expansion
                     (suggested for partial-sky: L=20)
            theta (float): angular aperture of the lightcone
        Returns:
            array: L coefficients
        """

        Kl = np.array(
            [
                np.sqrt(np.pi / (2.0 * ell + 1.0))
                * (
                    eval_legendre(ell - 1, np.cos(theta))
                    - eval_legendre(ell + 1, np.cos(theta))
                )
                / (2.0 * np.pi * (1 - np.cos(theta)))
                for ell in range(L + 1)
            ]
        )
        Kl[0] = 1 / (2.0 * np.sqrt(np.pi))
        return Kl

    # window function
    def window_redshift_bin(self, k_arr, z_arr, iz, L, sigma_0):
        """Redshift-slice window function

        Window function of the lightcone redshift slice.

        Args:
            k_arr (array): wavenumbers in 1/Mpc
            z_arr (array): true redshift
            iz (int): photometric redshift bin
            L (int): number of multipoles for the expansion
                     (suggested for partial-sky: L=20)
        Returns:
            array: growth factor times window function of the redshift slice
        """

        # harmonic expansion coefficeints
        arccos_arg = 1 - self.survey_area / (2 * np.pi)
        if arccos_arg < -1:  # may happen due to numerical inaccuracy
            arccos_arg = -1
        theta_sky = np.arccos(arccos_arg)

        KL = self.Kl_func(L, theta_sky)

        # redshift-dependent quantities
        rz = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z_arr))  # Mpc

        dVdzob = np.array(
            [self.comoving_volume_element(z, iz, sigma_0) for z in z_arr]
        )
        Vz = simps(dVdzob, x=z_arr)
        D = ccl.growth_factor(self.cosmo, 1 / (1 + z_arr))

        # integral over redshift
        jl_kz = np.array(
            [spherical_jn(ell, k_arr[:, None] * rz) for ell in range(L + 1)]
        ).T
        rint = simps((dVdzob * D)[:, None, None] * jl_kz, x=z_arr, axis=0) / Vz

        return 4 * np.pi * rint * KL

    def super_sample_covariance(self):
        """super-sample covariance

        super sample covariance term of the number counts covariance
        """

        # number of multipoles for the expansion
        # (suggested for partial-sky: L=20)
        L = 20
        if self.fullsky is True:
            L = 0

        k_arr = np.geomspace(1e-4, 2e1, 700)  # 1/Mpc

        # number counts*bias and window function
        Nb_lob_zob = np.zeros((self.num_richness_bins, self.num_z_bins))
        Wi_l = np.zeros((self.num_z_bins, len(k_arr), L + 1))

        for iz in range(self.num_z_bins):

            # true redshift for integration
            z_tr = np.linspace(
                max(self.z_bins[iz] - 0.3, 0.02),
                min(self.z_bins[iz + 1] + 0.3, 0.91),
                200,
            )

            # observed volume element
            dVdzob = np.array(
                [
                    self.comoving_volume_element(z, iz, self.sigma_0)
                    for z in z_tr
                ]
            )

            # number counts and bias in observed redshift and richness bins
            for il in range(self.num_richness_bins):

                Nb_lob_z = np.array(
                    [
                        self.mass_richness_integral(z, il, remove_bias=False)
                        for z in z_tr
                    ]
                )

                Nb_lob_zob[il, iz] = simps(dVdzob * Nb_lob_z, x=z_tr, axis=0)

            # window function of the i-th redshift bin
            Wi_l[iz] = self.window_redshift_bin(
                k_arr, z_tr, iz, L, self.sigma_0
            )

        # sum over ell of W_i * W_j
        WiWj_sum = np.sum(Wi_l[:, None, :, :] * Wi_l[None, :, :, :], axis=-1)

        # sample covariance
        pk0 = ccl.linear_matter_power(self.cosmo, k_arr, 1.0)  # Mpc^3
        sigma2_zizj = (
            1
            / (2 * np.pi) ** 3
            * simps(k_arr**2 * pk0 * WiWj_sum, x=k_arr, axis=-1)
        )

        # SSC, dim=[richness,richness,redshift,redshift]
        SSC = self.survey_area**2 * (
            Nb_lob_zob.reshape(1, self.num_richness_bins, 1, self.num_z_bins)
            * Nb_lob_zob.reshape(self.num_richness_bins, 1, self.num_z_bins, 1)
            * sigma2_zizj.reshape(1, 1, self.num_z_bins, self.num_z_bins)
        )

        return SSC
