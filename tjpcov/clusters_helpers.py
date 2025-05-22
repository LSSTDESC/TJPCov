"""Helper functions and classes for cluster covariance calculations."""

import numpy as np
import pyccl as ccl

# Map mass function name to actual function
mass_func_map = {
    "Tinker08": ccl.halos.MassFuncTinker08,
    "Tinker10": ccl.halos.MassFuncTinker10,
    "Bocquet16": ccl.halos.MassFuncBocquet16,
    "Bocquet20": ccl.halos.MassFuncBocquet20,
    "Despali16": ccl.halos.MassFuncDespali16,
    "Jenkins01": ccl.halos.MassFuncJenkins01,
    "Nishimichi19": ccl.halos.MassFuncNishimichi19,
    "Press74": ccl.halos.MassFuncPress74,
    "Sheth99": ccl.halos.MassFuncSheth99,
    "Watson13": ccl.halos.MassFuncWatson13,
    "Angulo12": ccl.halos.MassFuncAngulo12,
}

# Map halo bias name to actual function
halo_bias_map = {
    "Tinker10": ccl.halos.HaloBiasTinker10,
    "Bhattacharya11": ccl.halos.HaloBiasBhattacharya11,
    "Sheth01": ccl.halos.HaloBiasSheth01,
    "Sheth99": ccl.halos.HaloBiasSheth99,
}


def _load_from_sacc(sacc_file, min_halo_mass, max_halo_mass):
    """Extract and compute attributes from a SACC file.

    Args:
        sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object, already loaded.
        min_halo_mass (float): Minimum halo mass.
        max_halo_mass (float): Maximum halo mass.

    Returns:
        dict: A dictionary containing all computed attributes.
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
        survey_area = 4 * np.pi
        print(
            "Survey tracer not provided in sacc file.\n"
            + "We will use the default value.",
            flush=True,
        )
    else:
        survey_area = survey_tracer[0].sky_area * (np.pi / 180) ** 2

    # Setup redshift bins
    z_bins = sorted(
        [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == z_tracer_type
        ],
        key=lambda z: z.lower,
    )
    num_z_bins = len(z_bins)
    z_min = np.min([zbin.lower for zbin in z_bins])
    z_max = np.max([zbin.upper for zbin in z_bins])
    z_bins = np.array([round(z_bins[0].lower, 2)] + [
        round(zbin.upper, 2) for zbin in z_bins
    ])
    z_bin_spacing = (z_max - z_min) / num_z_bins
    z_lower_limit = max(0.02, z_bins[0] - 4 * z_bin_spacing)
    z_upper_limit = (
        z_bins[-1] + 0.4 * z_bins[-1]
    )  # Set upper limit to be 40% higher than max redshift

    # Setup richness bins
    richness_bins = sorted(
        [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == richness_tracer_type
        ],
        key=lambda rich: rich.lower,
    )
    num_richness_bins = len(richness_bins)
    min_richness = 10 ** np.min([rbin.lower for rbin in richness_bins])
    max_richness = 10 ** np.max([rbin.upper for rbin in richness_bins])
    richness_bins = np.array(
        [10 ** richness_bins[0].lower]
        + [10**rbin.upper for rbin in richness_bins]
    )
    richness_bins = np.round(richness_bins, 2)
    # Compute mass-related attributes
    min_mass = np.log(min_halo_mass)
    max_mass = np.log(max_halo_mass)

    # Return all computed attributes as a dictionary
    return {
        "survey_area": survey_area,
        "num_z_bins": num_z_bins,
        "z_min": z_min,
        "z_max": z_max,
        "z_bins": z_bins,
        "z_bin_spacing": z_bin_spacing,
        "z_lower_limit": z_lower_limit,
        "z_upper_limit": z_upper_limit,
        "num_richness_bins": num_richness_bins,
        "min_richness": min_richness,
        "max_richness": max_richness,
        "richness_bins": richness_bins,
        "min_mass": min_mass,
        "max_mass": max_mass,
    }


def extract_indices_rich_z(tracer_comb):
    """Extract richness and redshift indices from a tracer combination."""
    if len(tracer_comb) == 1:
        # Handle input type 2: ('clusters_0_1',)
        parts = tracer_comb[0].split("_")
        richness = int(parts[-2])  # Second-to-last part is richness
        z = int(parts[-1])  # Last part is redshift
    else:
        # Handle input type 1: ('survey', 'bin_richness_1', 'bin_z_0')
        richness = None
        z = None
        for part in tracer_comb:
            if part.startswith("bin_richness_") or part.startswith(
                "bin_rich_"
            ):  # Handle both prefixes
                richness = int(part.split("_")[-1])
            elif part.startswith("bin_z_"):
                z = int(part.split("_")[-1])
        if richness is None or z is None:
            raise ValueError(
                "Could not extract richness or z from tracer combination: "
                f"{tracer_comb}"
            )
    return richness, z


class FFTHelper(object):
    """Fft helper class.

    Cluster covariance needs to use fast fourier transforms in combination
    with numerical approximations to evaluate rapidly oscillating integrals
    that appear in the calculation of the covariance.  These are stored in this
    helper class.
    """

    bias_fft = 1.4165
    k_min = 1e-4
    k_max = 3
    N = 1024

    def __init__(self, cosmo, z_min, z_max):
        """Constructor for the FFTHelper class.

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
            z_min (float): Lower bound on redshift integral
            z_max (float): Upper bound on redshift integral
        """
        self.cosmo = cosmo
        self._set_fft_params(z_min, z_max)

    def _set_fft_params(self, z_min, z_max):
        """Function to set fft parameters.

        The numerical implementation of the FFT needs some values
        set by some simple calculations.  Those are performed here.

        See Eqn 7.16 N. Ferreira disseration.

        Args:
            z_min (float): Lower bound on redshift integral
            z_max (float): Upper bound on redshift integral
        """
        import pyccl as ccl

        self.r_min = 1 / self.k_max
        self.r_max = 1 / self.k_min
        self.G = np.log(self.k_max / self.k_min)
        self.L = 2 * np.pi * self.N / self.G

        self.k_grid = np.logspace(
            np.log(self.k_min), np.log(self.k_max), self.N, base=np.exp(1)
        )

        self.r_grid = np.logspace(
            np.log(self.r_min), np.log(self.r_max), self.N, base=np.exp(1)
        )

        radial_min = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z_min))
        radial_max = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z_max))

        self.idx_min = np.argwhere(self.r_grid < 0.95 * radial_min)[-1][0]
        self.idx_max = np.argwhere(self.r_grid > 1.05 * radial_max)[0][0]

        self.pk_grid = ccl.linear_matter_power(self.cosmo, self.k_grid, 1)

        # Eqn 7.3 N. Ferreira
        self.fk_grid = (self.k_grid / self.k_min) ** (
            3.0 - self.bias_fft
        ) * self.pk_grid

    def two_fast_algorithm(self, z1, z2):
        """2-FAST algorithm implementation used to evaluate the double bessel
        integral.  See https://arxiv.org/pdf/1709.02401v3.pdf for more details

        See Eqn 7.4 of N. Ferreira

        Args:
            z1 (float): Lower redshift bound
            z2 (float): Upper redshift bound

        Returns:
            float: Numerical approximation of double bessel function
        """

        import pyccl as ccl
        from scipy.interpolate import interp1d

        r1 = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z1))
        r2 = ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z2))
        ratio = min(r1, r2) / max(r1, r2)

        I_ell_vec = [
            self._I_ell_algorithm(i, ratio) for i in range(self.N // 2 + 1)
        ]
        # Eqn 7.4 N. Ferreira
        fk_inv_fourier_xform = np.conjugate(np.fft.rfft(self.fk_grid)) / self.L

        back_FFT_grid = np.fft.irfft(fk_inv_fourier_xform * I_ell_vec) * self.N

        two_fast_grid = (
            (1 / np.pi)
            * (self.k_min**3)
            * ((self.r_grid / self.r_min) ** (-self.bias_fft))
            * back_FFT_grid
            / self.G
        )

        idx_min = self.idx_min
        idx_max = self.idx_max

        interpolation = interp1d(
            self.r_grid[idx_min:idx_max],
            two_fast_grid[idx_min:idx_max],
            kind="cubic",
        )
        try:
            return float(interpolation(max(r1, r2)))
        except Exception as err:
            print(
                err,
                f"""\n
                    Value you tried to interpolate: {max(r1, r2)} Mpc,
                    Input r {r1}, {r2}
                    Valid range range:
                    [{self.r_grid[self.idx_min]}, {self.r_grid[self.idx_max]}]
                    Mpc""",
            )

    def _I_ell_algorithm(self, i, ratio):
        """Calculating the function M_0_0 the formula below only valid for
        R <=1, l = 0, formula B2 ASZ and 31 from 2-fast paper
        https://arxiv.org/pdf/1709.02401v3.pdf

        Args:
            i (int): iteration
            ratio (float): Ratio between comoving coordinates

        Returns:
            float: Fourier transform of spherical bessel function
        """
        from scipy.special import gamma

        t_m = 2 * np.pi * i / self.G
        alpha_m = self.bias_fft - 1.0j * t_m
        pre_factor = (self.k_min * self.r_min) ** (-alpha_m)

        return_val = (
            pre_factor * 0.5 * np.cos(np.pi * alpha_m / 2) * gamma(alpha_m - 2)
        )

        if ratio < 1:
            return_val *= (1 / ratio) * (
                (1 + ratio) ** (2 - alpha_m) - (1 - ratio) ** (2 - alpha_m)
            )

        elif ratio == 1:
            return_val *= (1 + ratio) ** (2 - alpha_m)

        return return_val
