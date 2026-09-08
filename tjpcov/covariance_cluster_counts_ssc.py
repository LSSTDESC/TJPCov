from .covariance_cluster_counts import CovarianceClusterCounts
import numpy as np
import pyccl as ccl
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn, eval_legendre

class ClusterCountsSSC(CovarianceClusterCounts):
    """Implementation of the SSC cluster covariance term.

    Calculates the sample variance contribution to the autocorrelation of
    cluster counts (NxN) following N. Ferreira 2019.
    """

    cov_type = "SSC"

    def __init__(self, config):
        """Class to calculate the SSC covariance of cluster counts

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.ssc_total = None  # initialise the full ssc covariance

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_richness_1', 'bin_z_0')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_richness_0', 'bin_z_0')
        Returns:
            float: Covariance for a single block
        """
        return self._get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def _get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
            ('survey', 'bin_richness_1', 'bin_z_0') or ('clusters_0_1',)
            tracer_comb2 (`tuple` of str): e.g.
            ('survey', 'bin_richness_0', 'bin_z_0') or ('clusters_0_0',)

        Returns:
            array_like: Covariance for a single block
        """

        # Extract richness and redshift indices for both tracer combinations
        richness_i, z_i = self.extract_indices_rich_z(tracer_comb1)
        richness_j, z_j = self.extract_indices_rich_z(tracer_comb2)

        # Compute the full SSC covariance only once
        if self.ssc_total is None:
            self.ssc_total = self.super_sample_covariance()

        # Read the single entries of the total SSC covariance
        # ssc_total dim = [richness, richness, redshift, redshift]
        cov_full = np.array(self.ssc_total[richness_i, richness_j, z_i, z_j])

        return cov_full

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)
        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_cluster_counts(tracer_comb1, tracer_comb2)


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
