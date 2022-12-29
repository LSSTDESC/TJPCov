from .covariance_builder import CovarianceBuilder
from .clusters_helpers import MassRichnessRelation, FFTHelper
import numpy as np
import pyccl as ccl
from scipy.integrate import quad, romb


class CovarianceClusters(CovarianceBuilder):
    """The base class for calculating covariance that includes galaxy cluster
    number counts.
    """

    def __init__(self, config, survey_area=4 * np.pi):
        """Constructor for the base class, used to pass through config options
        for covariance calculation.

        Args:
            config (`dict` or `str`): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
            survey_area: The area of the survey on the sky.  This will be
            pulled from the sacc file eventually. Defaults to 4*np.pi.
        """
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if "clusters" not in str(sacc_file.tracers.keys()):
            print(
                "Clusters are not within the SACC file tracers."
                + "Not performing cluster covariances."
            )
            return

        self.overdensity_delta = 200
        self.h0 = float(self.config["parameters"].get("h"))
        self.load_from_sacc(sacc_file)

        # NOTE Survey area should be pulled from SACC file but is currently not
        self.survey_area = survey_area

        cosmo = self.get_cosmology()
        self.load_from_cosmology(cosmo)
        self.fft_helper = FFTHelper(
            cosmo, self.z_lower_limit, self.z_lower_limit
        )

    def load_from_cosmology(self, cosmo):
        """Values used by the covariance calculation that come from a CCL
        cosmology object.  Derived attributes from the cosmology are set here.

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
        """
        self.cosmo = cosmo
        mass_def = ccl.halos.MassDef200m()
        self.c = ccl.physical_constants.CLIGHT / 1000
        self.mass_func = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)

    def load_from_sacc(self, sacc_file):
        """Cluster covariance has special parameters set in the SACC file. This
        informs the code that the data to calculate the cluster covariance is
        there.  We set extract those values from the sacc file here, and set
        the attributes here.

        Args:
            sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object, already
            loaded.
        """
        # Read from SACC file relevant quantities
        self.num_z_bins = sacc_file.metadata["nbins_cluster_redshift"]
        self.num_richness_bins = sacc_file.metadata["nbins_cluster_richness"]
        min_mass = sacc_file.metadata["min_mass"]
        # survey_area = sacc_file.metadata['survey_area']

        min_redshifts = [
            sacc_file.tracers[x].metadata["z_min"]
            for x in sacc_file.tracers
            if x.__contains__("clusters")
        ]
        max_redshifts = [
            sacc_file.tracers[x].metadata["z_max"]
            for x in sacc_file.tracers
            if x.__contains__("clusters")
        ]
        min_richness = [
            sacc_file.tracers[x].metadata["Mproxy_min"]
            for x in sacc_file.tracers
            if x.__contains__("clusters")
        ]
        max_richness = [
            sacc_file.tracers[x].metadata["Mproxy_max"]
            for x in sacc_file.tracers
            if x.__contains__("clusters")
        ]

        # Setup Richness Bins
        self.min_richness = min(min_richness)
        if self.min_richness == 0:
            self.min_richness = 1.0
        self.max_richness = max(max_richness)
        self.richness_bins = np.round(
            np.logspace(
                np.log10(self.min_richness),
                np.log10(self.max_richness),
                self.num_richness_bins + 1,
            ),
            2,
        )

        # Define arrays for bins for Photometric z and z grid
        self.z_max = max(max_redshifts)
        self.z_min = min(min_redshifts)
        if self.z_min == 0:
            self.z_min = 0.01

        self.z_bins = np.round(
            np.linspace(self.z_min, self.z_max, self.num_z_bins + 1), 2
        )
        self.z_bin_spacing = (self.z_max - self.z_min) / self.num_z_bins
        self.z_lower_limit = max(0.02, self.z_bins[0] - 4 * self.z_bin_spacing)
        self.z_upper_limit = self.z_bins[-1] + 6 * self.z_bin_spacing

        self.min_mass = np.log(min_mass)
        self.max_mass = np.log(1e16)

    def _quad_integrate(self, argument, from_lim, to_lim):
        """Helper function to numerically integral arguments between bounds
        using scipy quad function.

        Args:
            argument: Function to integrate between bounds
            from_lim: lower limit
            to_lim: upper limit

        Returns:
            Value of the integral
        """

        integral_value = quad(argument, from_lim, to_lim)
        return integral_value[0]

    def _romb_integrate(self, kernel, spacing):
        """Helper function to numerically integral arguments between bounds
        using scipy romberg integration

        Args:
            kernel: Vector of equally spaced samples of a function
            spacing: Sample spacing

        Returns:
            Value of the integral
        """
        return romb(kernel, dx=spacing)

    def observed_photo_z(self, z_true, z_i, sigma_0=0.05):
        """We don't assume that redshift can be measured exactly, so we include
        a measurement of the uncertainty around photometric redshifts. Assume,
        given a true redshift z, the measured redshift will be gaussian. The
        uncertainty will increase with redshift bin.

        See section 2.3 of N. Ferreira

        Args:
            z_true (float): True redshift
            z_i (float): Photometric redshift bin index
            sigma_0 (float): Spread in the uncertainty of the photo-z
                distribution, defaults to 0.05 (DES Y1)
        Returns:

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

    def comoving_volume_element(self, z_true, z_i):
        """Given a true redshift, and a redshift bin, this will give the
        volume element for this bin including photo-z uncertainties.

        Args:
            z_true: True redshift
            z_i: Photometric redshift bin

        Returns:
            Photo-z-weighted comoving volume element per steridian for redshift
            bin i in units of Mpc^3
        """

        dV = (
            self.c
            * (ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z_true)) ** 2)
            / (100 * self.h0 * ccl.h_over_h0(self.cosmo, 1 / (1 + z_true)))
            * (self.observed_photo_z(z_true, z_i))
        )
        return dV

    def mass_richness(self, ln_true_mass, lbd_i):
        """The probability that we observe richness given the true mass M, is
        given by the convolution of a Poisson distribution (relating observed
        richness to true richness) with a Gaussian distribution (relating true
        richness to M). Such convolution can be translated into a parametrized
        log-normal mass-richness distribution, done so here.

        Args:
            ln_true_mass: True mass
            richness_bin: Richness bin i
            richness_bin_next: Richness bin i+1
        Returns:
            The probability that the true mass ln(ln_true_mass)
            is observed within the richness bin i and richness bin i+1
        """

        richness_bin = self.richness_bins[lbd_i]
        richness_bin_next = self.richness_bins[lbd_i + 1]

        std_deviation, average = MassRichnessRelation.MurataCostanzi(
            ln_true_mass, self.h0
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
        """The derivative of the number density of halos with variations in the
        background density (Eqn 3.31 N. Ferreira)

        Args:
            z (float): Redshift
            lbd_i (int): Richness bin
            remove_bias: If TRUE, will remove halo_bias from the mass integral.
            Used for calculating the shot noise.
        Returns:
            The mass-richness weighed derivative of number density per
            fluctuation in background
        """

        def integrand(ln_m):

            argument = 1 / np.log(10.0)

            scale_factor = 1 / (1 + z)

            mass_func = self.mass_func.get_mass_function(
                self.cosmo, np.exp(ln_m), scale_factor
            )

            argument *= mass_func

            if not remove_bias:
                halo_bias = ccl.halo_bias(
                    self.cosmo,
                    np.exp(ln_m),
                    scale_factor,
                    overdensity=self.overdensity_delta,
                )
                argument *= halo_bias

            mass_richness = self.mass_richness(ln_m, richness_i)
            argument *= mass_richness

            return argument

        return self._quad_integrate(integrand, self.min_mass, self.max_mass)

    def shot_noise(self, z_i, lbd_i):
        """The covariance of number counts is a sum of a super sample
        covariance (SSC) term plus a gaussian diagonal term.  The diagonal
        term is also referred to as "shot noise" which we compute here.

        Args:
            z_i (int): redshift bin i
            lbd_i (int): richness bin i

        """
        # Eqn B.7 or 1601.05779.pdf eqn 1
        def integrand(z):
            return (
                self.c
                * (ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z)) ** 2)
                / (100 * self.h0 * ccl.h_over_h0(self.cosmo, 1 / (1 + z)))
                * self.mass_richness_integral(z, lbd_i, remove_bias=True)
                * self.observed_photo_z(z, z_i)
            )

        result = self._quad_integrate(
            integrand, self.z_lower_limit, self.z_upper_limit
        )
        return self.survey_area * result

    def partial_SSC(self, z, bin_z_j, bin_lbd_j, approx=True):
        """Calculate part of the super sample covariance, or the non-diagonal
        correlation between two point functions whose observed modes are larger
        than the survey size

        Args:
            z1 (float): redshift
            bin_z_j (int): redshift bin i
            bin_lbd_j (int): richness bin j
            approx (bool, optional): Will only calculate the mass richness
            integral once and multiply at end. Defaults to True.

        """
        # Nelson tested and found convergence at 5 iterations
        romb_k = 5
        num_samples = 2 ** (romb_k - 1) + 1

        # Build an equally sampled redshift array based on input and bounds
        if z <= np.average(self.z_bins):
            min_z = max(self.z_lower_limit, z - 6 * self.z_bin_spacing)
            vec_left = np.linspace(min_z, z, num_samples)
            vec_right = np.linspace(z, z + (z - vec_left[0]), num_samples)
        else:
            max_z = min(self.z_upper_limit, z + 6 * self.z_bin_spacing)
            vec_right = np.linspace(z, max_z, num_samples)
            vec_left = np.linspace(z - (vec_right[-1] - z), z, num_samples)

        z_values = np.append(vec_left, vec_right[1:])
        romb_range = (z_values[-1] - z_values[0]) / (2**romb_k)
        fn_values = np.zeros(2**romb_k + 1)

        for i in range(2**romb_k + 1):
            fn_values[i] = (
                self.comoving_volume_element(z_values[i], bin_z_j)
                * ccl.growth_factor(self.cosmo, 1 / (1 + z_values[i]))
                * self.double_bessel_integral(z, z_values[i])
            )

            if approx:
                continue

            fn_values[i] *= self.mass_richness_integral(z_values[i], bin_lbd_j)

        integral_val = self._romb_integrate(fn_values, romb_range)

        factor_approx = 1
        if approx:
            factor_approx = self.mass_richness_integral(z, bin_lbd_j)

        return integral_val * factor_approx

    def double_bessel_integral(self, z1, z2):
        """Calculates the double bessel integral using 2-FAST algorithm,
        as function of z1 and z2. See section 7.1, 7.2 of N. Ferreira
        dissertation.

        Args:
            z1 (float): redshift lower bound
            z2 (float): redshift upper bound
        """
        return self.fft_helper.two_fast_algorithm(z1, z2)
