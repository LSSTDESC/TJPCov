import numpy as np


class MassRichnessRelation(object):
    """Helper class to hold different mass richness relations"""

    @staticmethod
    def MurataCostanzi(ln_true_mass, h0):
        """Uses constants from Murata et al - ArxIv 1707.01907 and Costanzi
        et al ArxIv 1810.09456v1 to derive the log-normal mass-richness
        relation

        Args:
            ln_true_mass (float): True mass
            h0 (float): Hubble's constant
        Returns:
            `tuple` of float: The parameterized average and spread of the
            log-normal mass-richness relation
        """

        alpha = 3.207  # Murata
        beta = 0.75  # Costanzi
        sigma_zero = 2.68  # Costanzi
        q = 0.54  # Costanzi
        m_pivot = 3.0e14 / h0  # in solar masses , Murata and Costanzi use it

        sigma_lambda = sigma_zero + q * (ln_true_mass - np.log(m_pivot))
        average = alpha + beta * (ln_true_mass - np.log(m_pivot))

        return sigma_lambda, average


class FFTHelper(object):
    """Cluster covariance needs to use fast fourier transforms in combination
    with numerical approximations to evaluate rapidly oscillating integrals
    that appear in the calculation of the covariance.  These are stored in this
    helper class.
    """

    bias_fft = 1.4165
    k_min = 1e-4
    k_max = 3
    N = 1024

    def __init__(self, cosmo, z_min, z_max):
        """Constructor for the FFTHelper class

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
            z_min (float): Lower bound on redshift integral
            z_max (float): Upper bound on redshift integral
        """
        self.cosmo = cosmo
        self._set_fft_params(z_min, z_max)

    def _set_fft_params(self, z_min, z_max):
        """The numerical implementation of the FFT needs some values
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
            return interpolation(max(r1, r2))
        except Exception as err:
            print(
                err,
                f"""\n
                    Value you tried to interpolate: {max(r1,r2)} Mpc,
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
