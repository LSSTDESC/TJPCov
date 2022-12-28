from .covariance_builder import CovarianceBuilder
import numpy as np
import pyccl as ccl

# Replace with CCL functions
from scipy.integrate import quad, romb
from scipy.special import gamma
from scipy.interpolate import interp1d


class CovarianceClusters(CovarianceBuilder):
    """The base class for calculating covariance that includes galaxy cluster
    number counts.
    """

    def __init__(self, config, survey_area=4 * np.pi):
        """Constructor for the base class, used to pass through config options
        for covariance calculation.

        Args:
            config: Path to the config file to be used
            survey_area: The area of the survey on the sky.  This will be pulled
            from the sacc file eventually. Defaults to 4*np.pi.
        """
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if "clusters" not in str(sacc_file.tracers.keys()):
            print(
                "Clusters are not within the SACC file tracers."
                + "Not performing cluster covariances."
            )
            return

        self.load_from_config()
        self.set_fft_params()
        self.load_from_sacc(sacc_file)
        # Cosmology
        self.survey_area = survey_area
        self.load_from_cosmology(self.get_cosmology())

    def load_from_config(self):
        """Some cosmology values and numerical integration methods are hard
        coded into the config file.  We extract those here, cast them to their
        proper types, and set them as attributes."""
        self.bias_fft = float(self.config["fft_params"].get("bias_fft"))
        self.ko = float(self.config["fft_params"].get("ko"))
        self.kmax = int(self.config["fft_params"].get("kmax"))
        self.N = int(self.config["fft_params"].get("N"))
        self.overdensity_delta = float(
            self.config["clusters_params"].get("overdensity_delta")
        )
        self.h0 = float(self.config["parameters"].get("h"))

    def set_fft_params(self):
        """The numerical implementation of the FFT needs some values
        set by some simple calculations.  Those are performed here."""
        self.ro = 1 / self.kmax
        self.rmax = 1 / self.ko
        self.G = np.log(self.kmax / self.ko)
        self.L = 2 * np.pi * self.N / self.G
        self.k_vec = np.logspace(
            np.log(self.ko), np.log(self.kmax), self.N, base=np.exp(1)
        )
        self.r_vec = np.logspace(
            np.log(self.ro), np.log(self.rmax), self.N, base=np.exp(1)
        )

    def load_from_cosmology(self, cosmo):
        """Values used by the covariance calculation that come from a CCL
        cosmology object.  Derived attributes from the cosmology are set here.

        Args:
            cosmo: CCL Cosmology Object
        """
        self.cosmo = cosmo
        mass_def = ccl.halos.MassDef200m()
        self.c = ccl.physical_constants.CLIGHT / 1000
        self.mass_func = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
        # TODO: optimize these ranges like Nelson did interpolation limits
        # for double_bessel_integral
        # zmin & zmax drawn from Z_true_vec
        self.radial_lower_limit = ccl.comoving_radial_distance(
            self.cosmo, 1 / (1 + self.z_lower_limit)
        )
        self.radial_upper_limit = ccl.comoving_radial_distance(
            self.cosmo, 1 / (1 + self.z_upper_limit)
        )
        self.imin = np.argwhere(self.r_vec < 0.95 * self.radial_lower_limit)[
            -1
        ][0]
        self.imax = np.argwhere(self.r_vec > 1.05 * self.radial_upper_limit)[
            0
        ][0]

        self.pk_vec = ccl.linear_matter_power(cosmo, self.k_vec, 1)
        self.fk_vec = (self.k_vec / self.ko) ** (
            3.0 - self.bias_fft
        ) * self.pk_vec
        self.Phi_vec = np.conjugate(np.fft.rfft(self.fk_vec)) / self.L

    def load_from_sacc(self, sacc_file):
        """Cluster covariance has special parameters set in the SACC file. This
        informs the code that the data to calculate the cluster covariance is
        there.  We set extract those values from the sacc file here, and set
        the attributes here.

        Args:
            sacc_file: SACC file object, already loaded.
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
        self.z_bin_range = (self.z_max - self.z_min) / self.num_z_bins
        self.z_lower_limit = max(0.02, self.z_bins[0] - 4 * self.z_bin_range)
        self.z_upper_limit = self.z_bins[-1] + 6 * self.z_bin_range

        self.min_mass = np.log(min_mass)
        self.max_mass = np.log(1e16)

    def integrate(self, argument, from_lim, to_lim):
        """Helper function to numerically integral arguments between bounds

        Args:
            argument: Function to integrate between bounds
            from_lim: lower limit
            to_lim: upper limit

        Returns:
            Value of the integral
        """
        integral_value = quad(argument, from_lim, to_lim)
        return integral_value[0]

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
        numerator = self.integrate(
            integrand, self.z_bins[z_i], self.z_bins[z_i + 1]
        )
        denominator = 1.0 - self.integrate(integrand, -np.inf, 0.0)[0]

        return numerator / denominator

    def dV(self, z_true, z_i):
        """Given a true redshift, and a redshift bin, this will give the
        volume element for this bin including photo-z uncertainties.

        Args:
            z_true (float): True redshift
            z_i (float): Photometric redshift bin

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

        return self.integrate(integrand, richness_bin, richness_bin_next)

    def mass_richness_integral(self, z, richness_i, remove_bias=False):
        """The derivative of the number density of halos with variations in the
        background density (Eqn 3.31)

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

            argument *= scale_factor
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

        return self.integrate(integrand, self.min_mass, self.max_mass)

    def Limber(self, z):
        """Calculating Limber approximation for double Bessel
        integral for l equal zero

        Args:
            z (float): redshift
        """

        return ccl.linear_matter_power(
            self.cosmo,
            0.5 / ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z)),
            1,
        ) / (4 * np.pi)

    def cov_Limber(self, z_i, z_j, lbd_i, lbd_j):
        """Calculating the covariance of diagonal terms using Limber
        (the delta transforms the double redshift integral into a
        single redshift integral)
        CAUTION: hard-wired ovdelta and survey_area!

        Args:
            z_i (int): redshift bin i
            z_j (int): redshift bin j
            lbd_i (int): richness bin i
            lbd_j (int): richness bin j

        """

        def integrand(z_true):
            return (
                self.dV(self.cosmo, z_true, z_i)
                * (ccl.growth_factor(self.cosmo, 1 / (1 + z_true)) ** 2)
                * self.observed_photo_z(z_true, z_j)
                * self.mass_richness_integral(
                    self.cosmo,
                    z_true,
                    lbd_i,
                    self.min_mass,
                    self.max_mass,
                    self.ovdelta,
                )
                * self.mass_richness_integral(
                    self.cosmo,
                    z_true,
                    lbd_j,
                    self.min_mass,
                    self.max_mass,
                    self.ovdelta,
                )
                * self.Limber(self.cosmo, z_true)
            )

        return (self.survey_area**2) * self.integrate(
            integrand, self.z_lower_limit, self.z_upper_limit
        )

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

        result = self.integrate(
            integrand, self.z_lower_limit, self.z_upper_limit
        )
        return self.survey_area * result

    def I_ell(self, m, R):
        """Calculating the function M_0_0
        the formula below only valid for R <=1, l = 0,
        formula B2 ASZ and 31 from 2-fast paper
        """

        t_m = 2 * np.pi * m / self.G
        alpha_m = self.bias_fft - 1.0j * t_m
        pre_factor = (self.ko * self.ro) ** (-alpha_m)

        if R < 1:
            iell = (
                pre_factor
                * 0.5
                * np.cos(np.pi * alpha_m / 2)
                * gamma(alpha_m - 2)
                * (1 / R)
                * ((1 + R) ** (2 - alpha_m) - (1 - R) ** (2 - alpha_m))
            )

        elif R == 1:
            iell = (
                pre_factor
                * 0.5
                * np.cos(np.pi * alpha_m / 2)
                * gamma(alpha_m - 2)
                * ((1 + R) ** (2 - alpha_m))
            )

        return iell

    def partial2(self, z1, bin_z_j, bin_lbd_j, approx=True):
        """The variation of cluster counts with regards to the background
        density

        Eqn 3.31

        Approximation: Put the integral_mass outside looping in m

        Args:
            z1 (float): redshift
            bin_z_j (int): redshift bin i
            bin_lbd_j (int): richness bin j
            approx (bool, optional): Defaults to True.

        """
        romb_k = 5

        if z1 <= np.average(self.z_bins):
            vec_left = np.linspace(
                max(self.z_lower_limit, z1 - 6 * self.z_bin_range),
                z1,
                2 ** (romb_k - 1) + 1,
            )
            vec_right = np.linspace(
                z1, z1 + (z1 - vec_left[0]), 2 ** (romb_k - 1) + 1
            )
            vec_final = np.append(vec_left, vec_right[1:])
        else:
            vec_right = np.linspace(
                z1,
                min(self.z_upper_limit, z1 + 6 * self.z_bin_range),
                2 ** (romb_k - 1) + 1,
            )
            vec_left = np.linspace(
                z1 - (vec_right[-1] - z1), z1, 2 ** (romb_k - 1) + 1
            )
            vec_final = np.append(vec_left, vec_right[1:])

        romb_range = (vec_final[-1] - vec_final[0]) / (2**romb_k)
        kernel = np.zeros(2**romb_k + 1)

        if approx:
            for m in range(2**romb_k + 1):
                try:
                    kernel[m] = (
                        self.dV(vec_final[m], bin_z_j)
                        * ccl.growth_factor(self.cosmo, 1 / (1 + vec_final[m]))
                        * self.double_bessel_integral(z1, vec_final[m])
                    )
                except Exception as ex:
                    print(ex)

            factor_approx = self.mass_richness_integral(z1, bin_lbd_j)

        else:
            for m in range(2**romb_k + 1):
                kernel[m] = (
                    self.dV(vec_final[m], bin_z_j)
                    * ccl.growth_factor(self.cosmo, 1 / (1 + vec_final[m]))
                    * self.double_bessel_integral(z1, vec_final[m])
                    * self.mass_richness_integral(vec_final[m], bin_lbd_j)
                )
                factor_approx = 1

        return (romb(kernel, dx=romb_range)) * factor_approx

    def double_bessel_integral(self, z1, z2):
        """Calculates the double bessel integral from I-ell algorithm,
        as function of z1 and z2

        Args:
            z1 (float): redshift lower bound
            z2 (float): redshift upper bound
        """

        # definition of t, forcing it to be <= 1
        r1 = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z1))
        r2 = r1
        R = 1
        if z1 != z2:
            r2 = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z2))
            R = min(r1, r2) / max(r1, r2)

        I_ell_vec = [self.I_ell(m, R) for m in range(self.N // 2 + 1)]

        back_FFT_vec = (
            np.fft.irfft(self.Phi_vec * I_ell_vec) * self.N
        )  # FFT back
        two_fast_vec = (
            (1 / np.pi)
            * (self.ko**3)
            * ((self.r_vec / self.ro) ** (-self.bias_fft))
            * back_FFT_vec
            / self.G
        )

        imin = self.imin
        imax = self.imax

        # we will use this to interpolate the exact r(z1)
        f = interp1d(
            self.r_vec[imin:imax], two_fast_vec[imin:imax], kind="cubic"
        )
        try:
            return f(max(r1, r2))
        except Exception as err:
            print(
                err,
                f"""\n
                Value you tried to interpolate: {max(r1,r2)} Mpc,
                Input r {r1}, {r2}
                Valid range range:
                [{self.r_vec[self.imin]}, {self.r_vec[self.imax]}]
                Mpc""",
            )


class MassRichnessRelation(object):
    """Helper class to hold different mass richness relations"""

    @staticmethod
    def MurataCostanzi(ln_true_mass, h0):
        """Uses constants from Murata et al - ArxIv 1707.01907 and Costanzi
        et al ArxIv 1810.09456v1 to return the parameterized average and spread
        of the log-normal mass-richness relation

        Args:
            ln_true_mass: True mass
            h0: Hubble's constant
        Returns:

        """

        alpha = 3.207  # Murata
        beta = 0.75  # Costanzi
        sigma_zero = 2.68  # Costanzi
        q = 0.54  # Costanzi
        m_pivot = 3.0e14 / h0  # in solar masses , Murata and Costanzi use it

        sigma_lambda = sigma_zero + q * (ln_true_mass - np.log(m_pivot))
        average = alpha + beta * (ln_true_mass - np.log(m_pivot))

        return sigma_lambda, average
