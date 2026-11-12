from .covariance_builder import CovarianceBuilder
import numpy as np
import pyccl as ccl
from scipy.integrate import quad, simpson
from sacc import standard_types
from crow.cluster_modules.completeness_models import (
    Completeness,
    CompletenessAguena16,
)
from crow.cluster_modules.mass_proxy import MurataBinned
from crow.cluster_modules.purity_models import (
    Purity,
    PurityAguena16,
    PurityAguena16LnProxy,
)
from .cluster_covariance_base import ClusterCovarianceBase
from .clusters_helpers import halo_bias_map

# Map completeness and purity model names to crow models
completeness_map = {"Aguena16": CompletenessAguena16}
purity_map = {
    "Aguena16": PurityAguena16,
    "Aguena16LnProxy": PurityAguena16LnProxy,
}


class CovarianceClusterCounts(ClusterCovarianceBase, CovarianceBuilder):
    """Class to calculate covariance of cluster counts.

    The cluster selection can include the completeness, c(M, z), and purity,
    p(richness, z), models of crow, given in the ``completeness`` and
    ``purity`` sections of the configuration, e.g.
    ``{"model": "Aguena16", "a_n": 1.1}`` (parameters not given take the
    crow defaults) or directly as crow objects. As in crow, the selection is
    multiplied by the completeness and divided by the purity of the observed
    richness, so that the counts include the false detections. They are
    assumed to be clustered as the true clusters.
    """

    space_type = "Fourier"
    _tracer_types = (
        standard_types.cluster_counts,
        standard_types.cluster_counts,
    )
    # Number of richness nodes per bin for the integral with the purity
    n_richness_purity = 32

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

        self.mass_richness_prob = MurataBinned(
            self.mor_m_pivot, self.mor_z_pivot
        )
        for p in ["mu", "sigma"]:
            for i in range(3):
                self.mass_richness_prob.parameters[f"{p}{i}"] = getattr(
                    self, f"mor_{p}_p{i}"
                )

        # Selection completeness and purity (None if not given)
        self.completeness = self._load_selection_model(
            "completeness", completeness_map, Completeness
        )
        self.purity = self._load_selection_model("purity", purity_map, Purity)

    def _load_selection_model(self, section, model_map, model_class):
        """Return the crow completeness or purity model of the configuration.

        Args:
            section (str): Configuration section, 'completeness' or 'purity'
            model_map (dict): Map of model names to crow models
            model_class (type): Base class of the crow models

        Returns:
            crow model or None if the section is not in the configuration
        """
        conf = self.config.get(section)
        if conf is None or isinstance(conf, model_class):
            return conf

        conf = dict(conf)
        name = conf.pop("model", None)
        if name not in model_map:
            raise ValueError(
                f"Invalid {section} model: {name}. "
                f"Must be one of {list(model_map)}"
            )
        model = model_map[name]()
        for key, value in conf.items():
            model.parameters[key] = float(value)
        return model

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
        return self.mass_richness_vec(ln_true_mass, z, richness_i)[0]

    def mass_richness_vec(self, ln_true_mass, z, richness_i):
        """Vectorized version of :meth:`mass_richness`.

        With a purity model, the probability density of the richness is
        divided by the purity of the observed richness before integrating
        over the bin, as in crow. It is then the number of detections, true
        or false, per halo.

        Args:
            ln_true_mass (array): True masses
            z (float or array): Redshift(s)
            richness_bin (int): Richness bin i
        Returns:
            array: The probability that each mass is observed within the
            richness bin i
        """
        rich_bin = (
            np.log10(self.richness_bins[richness_i]),
            np.log10(self.richness_bins[richness_i + 1]),
        )
        log_m = np.atleast_1d(ln_true_mass).astype(np.float64) / np.log(10)
        z = np.atleast_1d(z).astype(np.float64)
        if self.purity is None:
            return self.mass_richness_prob.distribution(log_m, z, rich_bin)

        log_rich = np.linspace(*rich_bin, self.n_richness_purity)[:, None]
        integrand = (
            self.mass_richness_prob.gaussian_kernel(log_m, z, log_rich)
            * np.log(10.0)
            / self.purity.distribution(log_rich, z)
        )
        return simpson(integrand, x=log_rich[:, 0], axis=0)

    def completeness_vec(self, ln_true_mass, z):
        """Completeness of the cluster selection.

        Args:
            ln_true_mass (array): True masses
            z (float or array): Redshift(s)
        Returns:
            array: Completeness for each mass, 1 without completeness model
        """
        ln_true_mass = np.atleast_1d(ln_true_mass).astype(np.float64)
        z = np.atleast_1d(z).astype(np.float64)
        if self.completeness is None:
            return np.ones(np.broadcast_shapes(ln_true_mass.shape, z.shape))
        return self.completeness.distribution(ln_true_mass / np.log(10), z)

    def mass_richness_integral(
        self, z, richness_i, remove_bias=False, log_mass_power=0
    ):
        """Integrates the HMF weighted by mass-richness relation.

        The halo mass function weighted by the probability that we measure
        observed richness lambda given true mass M.

        Args:
            z (float): Redshift
            richness_i (int): Richness bin
            remove_bias (bool, optional): If TRUE, will remove halo_bias from
            the mass integral. Used for calculating the shot noise.
            log_mass_power (int, optional): Power of log10(M) in the
            integrand. Used for the mean log10 mass.
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

            if log_mass_power:
                argument *= (ln_m / np.log(10.0)) ** log_mass_power

            argument *= self.completeness_vec(ln_m, z)[0]

            if self.has_mproxy:
                argument *= self.mass_richness(
                    np.array([ln_m]), np.array([z]), richness_i
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
