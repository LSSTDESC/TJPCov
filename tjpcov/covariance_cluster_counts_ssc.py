from .covariance_cluster_counts import CovarianceClusterCounts
from . import tools
import healpy as hp
import numpy as np
import pyccl as ccl
from sacc import standard_types
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn, eval_legendre


class ClusterCountsSSC(CovarianceClusterCounts):
    """Implementation of the SSC cluster covariance term.

    Calculates the sample variance contribution to the autocorrelation of
    cluster counts (NxN) following N. Ferreira 2019.

    The long mode is expanded in multipoles up to ``L`` (config
    ``ClusterCountsSSC: L``, default 20 for a spherical cap and 100 for a
    mask). If the configuration has a mask for the survey
    tracer of the clusters (``mask_file``, and ``mask_names`` for hdf5
    files), the multipoles are weighted by the mask power spectrum,
    (2l+1) C_l / Omega^2, as for the C_ell SSC with masks. Otherwise, the
    footprint is a spherical cap of the survey area.

    If the sacc file contains the mean log10 mass of the clusters, <log M>,
    it also computes the SSC of <log M> with the counts and with itself. The
    long mode changes the counts in a bin by Nb delta_b, with Nb the number
    of clusters times their bias, and <log M> = L / N, with L the sum of
    log10 masses, by (Nb_L - <log M> Nb) / N delta_b to linear order, with
    Nb_L the bias weighted L. Hence, the covariance between two bins is
    w_1 w_2 sigma2_ij, with w = Nb for the counts and
    w = (Nb_L - <log M> Nb) / N for <log M>.
    """

    cov_type = "SSC"
    # Wavenumbers in 1/Mpc and number of multipoles for the expansion of the
    # redshift bin windows for a spherical cap (suggested for partial-sky:
    # L=20; converged to a few percent) and for a mask
    k_arr = np.geomspace(1e-4, 2e1, 700)
    L = 20
    L_mask = 100

    def __init__(self, config):
        """Class to calculate the SSC covariance of cluster counts

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.ssc_total = None  # initialise the full ssc covariance

        # Footprint of the clusters for the long mode. Masks have more power
        # at high multipoles than a spherical cap.
        self.mask = self.get_cluster_mask()
        self.L = self.config.get("ClusterCountsSSC", {}).get(
            "L", self.L if self.mask is None else self.L_mask
        )

        # With the mean log mass, blocks contain both cluster data types
        sacc_file = self.io.get_sacc_file()
        self.has_mean_log_mass = (
            standard_types.cluster_mean_log_mass in sacc_file.get_data_types()
        )
        if self.has_mean_log_mass:
            self.covariance_block_data_type = None

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'clusters_redshift_richness'

        If the sacc file contains the mean log mass, the block contains the
        counts and the mean log mass of the two bins, in the sacc order.

        Args:
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_richness_1', 'bin_z_0')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_richness_0', 'bin_z_0')
        Returns:
            array_like: Covariance for a single block
        """
        if not self.has_mean_log_mass:
            return self._get_covariance_cluster_counts(
                tracer_comb1, tracer_comb2
            )

        if self.ssc_total is None:
            self.ssc_total = self.super_sample_covariance()

        # Response of each data point to the long mode
        s = self.io.get_sacc_file()
        resp = {
            standard_types.cluster_counts: self.Nb,
            standard_types.cluster_mean_log_mass: self.Nb_mean_log_mass,
        }
        w, z = [], []
        for tracer_comb in (tracer_comb1, tracer_comb2):
            richness_i, z_i = self.extract_indices_rich_z(tracer_comb)
            dtypes = [
                s.data[i].data_type for i in s.indices(tracers=tracer_comb)
            ]
            w.append(np.array([resp[dt][richness_i, z_i] for dt in dtypes]))
            z.append(z_i)

        return np.outer(w[0], w[1]) * self.sigma2_zizj[z[0], z[1]]

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

    def get_cluster_mask(self):
        """Return the footprint mask of the clusters.

        Returns:
            array: HEALPix mask of the survey tracer of the clusters, or None
            if the configuration has no mask for it.
        """
        sacc_file = self.io.get_sacc_file()
        survey = [
            name
            for name, tr in sacc_file.tracers.items()
            if tr.tracer_type == "survey"
        ]
        mask_file = self.mask_files
        if isinstance(mask_file, dict):
            mask_file = mask_file.get(survey[0]) if survey else None
        if mask_file is None:
            return None
        if isinstance(mask_file, np.ndarray):
            return mask_file
        name = (self.mask_names or {}).get(survey[0]) if survey else None
        return tools.read_map(mask_file, name, self.nside)

    def get_mask_wl(self, mask1, mask2):
        """Return the multipole weights of the long mode for two masks.

        Same normalization as the C_ell SSC with masks:
        (2l+1) C_l^{12} / (Omega_1 Omega_2), with C_l^{12} the cross power
        spectrum of the masks and Omega their areas. For a spherical cap, it
        is ``Kl_func**2``.

        Args:
            mask1 (array): HEALPix mask
            mask2 (array): HEALPix mask

        Returns:
            array: Weights for multipoles 0 to L
        """
        nside = hp.npix2nside(mask1.size)
        if self.L > 3 * nside - 1:
            raise ValueError(
                f"L = {self.L} is larger than 3 nside - 1 of the masks."
            )
        area = hp.nside2pixarea(nside)
        mask_wl = hp.alm2cl(
            hp.map2alm(mask1, lmax=self.L), hp.map2alm(mask2, lmax=self.L)
        )
        mask_wl *= 2 * np.arange(self.L + 1) + 1
        return mask_wl / (np.sum(mask1) * np.sum(mask2) * area**2)

    def get_ell_weights(self):
        """Return the multipole weights of the long mode of the clusters.

        Returns:
            array: Weights for multipoles 0 to L, from the cluster mask (see
            :meth:`get_mask_wl`) or for a spherical cap of the survey area.
        """
        if self.mask is not None:
            return self.get_mask_wl(self.mask, self.mask)
        # arccos argument may be < -1 due to numerical inaccuracy
        theta_sky = np.arccos(max(1 - self.survey_area / (2 * np.pi), -1))
        return self.Kl_func(self.L, theta_sky) ** 2

    # window function
    def window_redshift_bin(self, k_arr, z_arr, iz, L, sigma_0):
        """Redshift-slice window function

        Window function of the lightcone redshift slice for each multipole,
        without the footprint (see :meth:`get_ell_weights`).

        Args:
            k_arr (array): wavenumbers in 1/Mpc
            z_arr (array): true redshift
            iz (int): photometric redshift bin
            L (int): number of multipoles for the expansion
                     (suggested for partial-sky: L=20)
        Returns:
            array: growth factor times window function of the redshift slice
        """

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

        return 4 * np.pi * rint

    def get_z_true(self, iz):
        """True redshifts for the integrals over a photometric redshift bin.

        Args:
            iz (int): photometric redshift bin
        Returns:
            array: true redshifts
        """
        return np.linspace(
            max(self.z_bins[iz] - 0.3, 0.02),
            min(self.z_bins[iz + 1] + 0.3, 0.91),
            200,
        )

    def super_sample_covariance(self):
        """super-sample covariance

        super sample covariance term of the number counts covariance. It
        also sets the responses to the long mode of the counts, ``Nb``, and
        of the mean log mass, ``Nb_mean_log_mass``, and the covariance of the
        long mode between redshift bins, ``sigma2_zizj``.
        """
        L, k_arr = self.L, self.k_arr

        # number counts*bias and window function
        Nb_lob_zob = np.zeros((self.num_richness_bins, self.num_z_bins))
        Nb_mlm = np.zeros((self.num_richness_bins, self.num_z_bins))
        Wi_l = np.zeros((self.num_z_bins, len(k_arr), L + 1))

        for iz in range(self.num_z_bins):

            # true redshift for integration
            z_tr = self.get_z_true(iz)

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

                if self.has_mean_log_mass:
                    # N, L and the bias weighted L per steradian
                    n, l_m, nb_l = [
                        simps(
                            dVdzob
                            * np.array(
                                [
                                    self.mass_richness_integral(
                                        z, il, remove_bias=rb, log_mass_power=p
                                    )
                                    for z in z_tr
                                ]
                            ),
                            x=z_tr,
                        )
                        for rb, p in [(True, 0), (True, 1), (False, 1)]
                    ]
                    Nb_mlm[il, iz] = (nb_l - l_m / n * Nb_lob_zob[il, iz]) / n

            # window function of the i-th redshift bin
            Wi_l[iz] = self.window_redshift_bin(
                k_arr, z_tr, iz, L, self.sigma_0
            )

        # sum over ell of W_i * W_j, weighted by the footprint
        WiWj_sum = np.einsum(
            "ikl,jkl,l->ijk", Wi_l, Wi_l, self.get_ell_weights()
        )

        # sample covariance
        pk0 = ccl.linear_matter_power(self.cosmo, k_arr, 1.0)  # Mpc^3
        sigma2_zizj = (
            1
            / (2 * np.pi) ** 3
            * simps(k_arr**2 * pk0 * WiWj_sum, x=k_arr, axis=-1)
        )

        # Responses to the long mode, used for the blocks with <log M>
        self.Nb = self.survey_area * Nb_lob_zob
        self.Nb_mean_log_mass = Nb_mlm
        self.sigma2_zizj = sigma2_zizj

        # SSC, dim=[richness,richness,redshift,redshift]
        SSC = self.survey_area**2 * (
            Nb_lob_zob.reshape(1, self.num_richness_bins, 1, self.num_z_bins)
            * Nb_lob_zob.reshape(self.num_richness_bins, 1, self.num_z_bins, 1)
            * sigma2_zizj.reshape(1, 1, self.num_z_bins, self.num_z_bins)
        )

        return SSC
