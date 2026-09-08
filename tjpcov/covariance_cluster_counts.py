from .covariance_builder import CovarianceBuilder
import numpy as np
import pyccl as ccl
from scipy.integrate import quad
from sacc import standard_types
from crow.cluster_modules.mass_proxy import MurataBinned
from .cluster_covariance_base import ClusterCovarianceBase
from .clusters_helpers import mass_func_map, halo_bias_map


class CovarianceClusterCounts(ClusterCovarianceBase, CovarianceBuilder):
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

    def load_cluster_parameters(self):
        """Load cluster parameters from the configuration file."""
        self._load_cluster_parameters()
        halo_bias_name = self.config["mor_parameters"].get("halo_bias")
        if halo_bias_name not in halo_bias_map:
            raise ValueError(f"Invalid halo bias: {halo_bias_name}")

        # Create the halo bias objects
        self.hbias = halo_bias_map[halo_bias_name](mass_def=self.mass_def)

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
        self.mor_z_pivot = float(self.config["mor_parameters"].get("z_pivot"))

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
        mass_richness_prob = MurataBinned(self.mor_m_pivot, self.mor_z_pivot)
        # mass-obs relation params to be added as input params
        mass_richness_prob.parameters["mu0"] = self.mor_mu_p0
        mass_richness_prob.parameters["mu1"] = self.mor_mu_p1
        mass_richness_prob.parameters["mu2"] = self.mor_mu_p2
        mass_richness_prob.parameters["sigma0"] = self.mor_sigma_p0
        mass_richness_prob.parameters["sigma1"] = self.mor_sigma_p1
        mass_richness_prob.parameters["sigma2"] = self.mor_sigma_p2
        ln_true_mass = np.atleast_1d(ln_true_mass).astype(np.float64)
        z = np.atleast_1d(z).astype(np.float64)
        result = mass_richness_prob.distribution(
            ln_true_mass / np.log(10), z, rich_bin
        )

        return result[0]

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
            m_integ_lower, m_integ_upper = (
                self.min_halo_ln_mass,
                self.max_halo_ln_mass,
            )
        else:
            m_integ_lower = np.log(10) * self.richness_bins[richness_i]
            m_integ_upper = np.log(10) * self.richness_bins[richness_i + 1]
        return self._quad_integrate(integrand, m_integ_lower, m_integ_upper)
