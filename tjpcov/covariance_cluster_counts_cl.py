import numpy as np
import pyccl as ccl
from sacc import BandpowerWindow, standard_types
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from .clusters_helpers import get_halo_model
from .covariance_builder import CovarianceFourier
from .covariance_cluster_counts import CovarianceClusterCounts
from .covariance_cluster_counts_ssc import ClusterCountsSSC


class CovarianceClusterCountsCl(CovarianceClusterCounts, CovarianceFourier):
    """Base class for the C_ell x cluster covariance.

    It computes the covariance between the angular power spectra in the sacc
    file and the cluster counts, N, and mean log10 mass, <log M>, in each
    richness and redshift bin with the squeezed halo model bispectrum of CCL
    (see :mod:`pyccl.halos.pk_3pt`).

    The clusters are described by the selection function
    S(M, z) = P(richness bin|M, z) P(z bin|z), as for the cluster counts
    covariance. The covariance with <log M> = L / N, with L the sum of log10
    masses in the bin, follows from linear error propagation:
    Cov(<log M>, C) = Cov(L - <log M> N, C) / N. This is the covariance with
    the counts for the selection S (log10 M - <log M>) / N.

    Lensing tracers are described by an NFW profile and galaxies by the HOD
    in config['HOD'], which sets their bias (i.e. the CCL tracers of galaxies
    have unit bias).

    If the C_ell in the sacc file have bandpower windows, the covariance is
    that of the bandpowers, C_b = sum_ell W_b(ell) C_ell, which includes the
    mode coupling of the mask. Otherwise, it is evaluated at the effective
    ell of the bandpowers.
    """

    _tracer_types = ("cl", standard_types.cluster_counts)

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)
        # Blocks contain both, C_ell and cluster data types
        self.covariance_block_data_type = None

        # Number of redshift nodes per bin for the bispectrum splines
        self.nz = self.config.get("ClusterCl", {}).get("nz", 48)
        # Number of multipoles to interpolate the covariance to all
        # multipoles of the bandpower windows
        self.nell = self.config.get("ClusterCl", {}).get("nell", 100)
        self.fsky = self.survey_area / (4 * np.pi)
        # Same halo model as the C_ell x C_ell covariances
        self.hmc, cM, self.nfw = get_halo_model(self.config)
        hod = self.config.get("HOD")
        self.hod = None
        if hod is not None:
            self.hod = ccl.halos.HaloProfileHOD(
                mass_def=self.mass_def, concentration=cM, **hod
            )
        # Caches of the cluster profiles, bispectra and tracers
        self.bin_profiles = {}
        self.bin_means = {}
        self.bispectra = {}
        self.cl_tracers = {}

    def is_galaxy(self, tracer):
        """Return True if the tracer is a galaxy number counts tracer."""
        tr = self.io.get_sacc_file().tracers[tracer]
        return (tr.quantity == "galaxy_density") or ("lens" in tracer)

    def get_ccl_tracer(self, tracer):
        """Return the CCL tracer as in :meth:`get_tracer_info`, but galaxies
        have unit bias (set by the HOD).

        Args:
            tracer (str): Tracer name

        Returns:
            :obj:`pyccl.Tracer`: CCL tracer
        """
        if tracer not in self.cl_tracers:
            cosmo = self.get_cosmology()
            tr = self.io.get_sacc_file().tracers[tracer]
            if tr.quantity == "cmb_convergence":
                ccl_tr = ccl.CMBLensingTracer(cosmo, z_source=1100)
            elif self.is_galaxy(tracer):
                ccl_tr = ccl.NumberCountsTracer(
                    cosmo,
                    has_rsd=False,
                    dndz=(tr.z, tr.nz),
                    bias=(tr.z, np.ones_like(tr.z)),
                )
            else:
                ia = None if self.IA is None else (tr.z, self.IA + 0 * tr.z)
                ccl_tr = ccl.WeakLensingTracer(
                    cosmo, dndz=(tr.z, tr.nz), ia_bias=ia
                )
            self.cl_tracers[tracer] = ccl_tr
        return self.cl_tracers[tracer]

    def get_tracer_profiles(self, tracer_comb):
        """Return the halo profiles of a pair of tracers.

        Args:
            tracer_comb (list): List of the pair of tracer names of C_ell

        Returns:
            tuple:
                - key (tuple): (galaxy 1, galaxy 2, same galaxy sample)
                - profs (list): Halo profiles of the two tracers
                - prof_2pt (:obj:`pyccl.halos.Profile2pt`): Their 2-pt moment
        """
        gal = tuple(self.is_galaxy(tr) for tr in tracer_comb)
        if any(gal) and self.hod is None:
            raise ValueError("Galaxy tracers need the HOD configuration.")
        same = gal[0] and (tracer_comb[0] == tracer_comb[1])
        profs = [self.hod if g else self.nfw for g in gal]
        if same:
            prof_2pt = ccl.halos.Profile2ptHOD()
        else:
            prof_2pt = ccl.halos.Profile2pt()
        return gal + (same,), profs, prof_2pt

    def get_bin_profiles(self, richness_i, z_i):
        """Return the cluster profiles of a richness and redshift bin.

        Args:
            richness_i (int): Richness bin
            z_i (int): Redshift bin

        Returns:
            tuple:
                - a (array): Scale factors where the selection is not 0
                - profs (list): :obj:`pyccl.halos.HaloProfileSelection` for
                  the counts and the mean log mass. The model mean number of
                  clusters and log10 mass are stored in ``self.bin_means``.
        """
        if (richness_i, z_i) in self.bin_profiles:
            return self.bin_profiles[(richness_i, z_i)]

        cosmo = self.get_cosmology()

        def sel(M, a):
            z = 1 / a - 1
            ln_m = np.log(M)
            in_range = (ln_m >= self.min_halo_ln_mass) & (
                ln_m <= self.max_halo_ln_mass
            )
            return (
                in_range
                * self.completeness_vec(ln_m, z)
                * self.mass_richness_vec(ln_m, z, richness_i)
                * self.observed_photo_z(z, z_i, self.sigma_0)
            )

        # Redshifts where the selection is not 0 (5 sigma photo-z)
        z_lo, z_hi = self.z_bins[z_i], self.z_bins[z_i + 1]
        dz = 5 * self.sigma_0 * (1 + z_hi)
        z = np.linspace(max(z_lo - dz, 1e-3), z_hi + dz, self.nz)
        a = 1 / (1 + z[::-1])

        # Mean number of clusters and log10 mass in the bin
        N, L = [
            self.integrate_volume(
                a,
                lambda aa: self.hmc.integrate_over_massfunc(
                    lambda M: sel(M, aa) * np.log10(M) ** p, cosmo, aa
                ),
            )
            for p in (0, 1)
        ]
        mean_log_m = L / N
        self.bin_means[(richness_i, z_i)] = (N, mean_log_m)

        profs = [
            ccl.halos.HaloProfileSelection(mass_def=self.mass_def, selection=s)
            for s in [
                sel,
                lambda M, a: sel(M, a) * (np.log10(M) - mean_log_m) / N,
            ]
        ]
        self.bin_profiles[(richness_i, z_i)] = (a, profs)
        return a, profs

    def integrate_volume(self, a, func):
        """Integrate a function of the scale factor over the survey volume.

        Args:
            a (array): Increasing scale factors
            func (callable): Function of the scale factor, per Mpc^3

        Returns:
            float: Integral over the survey volume
        """
        chi = ccl.comoving_radial_distance(self.get_cosmology(), a)
        f = np.array([func(aa) for aa in a])
        # chi decreases with a
        return -self.survey_area * np.trapezoid(chi**2 * f, x=chi)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, ell=None):
        """Compute the covariance of a C_ell with a cluster bin.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell
            tracer_comb2 (list): Cluster tracers, e.g.
                ('survey', 'bin_richness_1', 'bin_z_0')
            ell (array, optional): Multipoles. If None, the effective ell of
                the sacc file.

        Returns:
            array: Covariance with shape (ell.size, 2) with columns the
            counts and the mean log mass.
        """
        raise NotImplementedError("Not implemented")

    def get_covariance_bandpowers(self, tracer_comb1, tracer_comb2, bpw):
        """Compute the covariance of the bandpowers of a C_ell with a cluster
        bin.

        The covariance is linear in C_ell, so that of the bandpowers
        C_b = sum_ell W_b(ell) C_ell is sum_ell W_b(ell) Cov(C_ell, N). If the
        windows have more than ``nell`` multipoles, the covariance is
        computed at ``nell`` multipoles, log spaced in ell + 1/2, and
        interpolated with a cubic spline.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell
            tracer_comb2 (list): Cluster tracers, e.g.
                ('survey', 'bin_richness_1', 'bin_z_0')
            bpw (:obj:`sacc.BandpowerWindow`): Bandpower windows

        Returns:
            array: Covariance with shape (number of bandpowers, 2) with
            columns the counts and the mean log mass.
        """
        # Only the multipoles that contribute to the bandpowers
        used = np.any(bpw.weight != 0, axis=1)
        ell, weight = bpw.values[used], bpw.weight[used]

        if ell.size <= self.nell:
            cov = self.get_covariance_block(
                tracer_comb1, tracer_comb2, ell=ell
            )
        else:
            x = np.log(ell + 0.5)
            x_grid = np.linspace(x.min(), x.max(), self.nell)
            cov = self.get_covariance_block(
                tracer_comb1, tracer_comb2, ell=np.exp(x_grid) - 0.5
            )
            cov = CubicSpline(x_grid, cov)(x)

        return weight.T @ cov

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Return the covariance block as needed for the sacc file.

        B-modes are not correlated with the clusters and their covariance is
        set to 0. If the C_ell have bandpower windows, it is the covariance
        of the bandpowers (see :meth:`get_covariance_bandpowers`).

        Args:
            tracer_comb1 (list): Tracer names of the C_ell or cluster bin
            tracer_comb2 (list): Tracer names of the C_ell or cluster bin
            **kwargs: Not used

        Returns:
            array: Covariance block
        """
        cc, mlm = (
            standard_types.cluster_counts,
            standard_types.cluster_mean_log_mass,
        )
        if cc in self.get_tracer_comb_data_types(tracer_comb1):
            return self._get_covariance_block_for_sacc(
                tracer_comb2, tracer_comb1
            ).T

        s = self.get_sacc_with_concise_dtypes()
        ix1 = s.indices(tracers=tracer_comb1)
        ix2 = s.indices(tracers=tracer_comb2)
        cols = [
            np.searchsorted(ix2, s.indices(tracers=tracer_comb2, data_type=dt))
            for dt in (cc, mlm)
        ]

        cov = np.zeros((ix1.size, ix2.size))
        for dt in ["cl_00", "cl_0e", "cl_ee"]:
            ix = s.indices(tracers=tracer_comb1, data_type=dt)
            if ix.size == 0:
                continue
            if isinstance(s.data[ix[0]].get_tag("window"), BandpowerWindow):
                cov_block = self.get_covariance_bandpowers(
                    tracer_comb1, tracer_comb2, s.get_bandpower_windows(ix)
                )
            else:
                ell, _ = s.get_ell_cl(dt, *tracer_comb1)
                cov_block = self.get_covariance_block(
                    tracer_comb1, tracer_comb2, ell=ell
                )
            rows = np.searchsorted(ix1, ix)
            for i, col in enumerate(cols):
                cov[np.ix_(rows, col)] = cov_block[:, i, None]

        return cov


class ClusterClcNGHaloModel(CovarianceClusterCountsCl):
    """Connected (1- and 2-halo) C_ell x cluster covariance.

    Cov(N, C_ell) = Int dchi / chi^2 q_1 q_2 B_N((ell + 1/2) / chi, a), with
    B_N the squeezed bispectrum for unnormalized counts of
    :func:`pyccl.halos.halomod_bispectrum_squeezed`. For galaxy clustering
    autocorrelations it includes the one-galaxy (shot noise) term, the
    covariance of the counts with the Poisson noise of galaxies in the
    clusters' halos:
    Int dM n(M) S(M) <N_g|M> / n_g^2.
    """

    cov_type = "cNG"

    def get_bin_bispectra(self, richness_i, z_i, tracer_comb):
        """Return the counts and <log M> bispectra of a cluster bin.

        Args:
            richness_i (int): Richness bin
            z_i (int): Redshift bin
            tracer_comb (list): List of the pair of tracer names of C_ell

        Returns:
            list: :obj:`pyccl.Pk2D` for the counts and the mean log mass
        """
        key, (prof2, prof3), prof23_2pt = self.get_tracer_profiles(tracer_comb)
        if (richness_i, z_i, key) in self.bispectra:
            return self.bispectra[(richness_i, z_i, key)]

        cosmo, hmc = self.get_cosmology(), self.hmc
        a, profs = self.get_bin_profiles(richness_i, z_i)

        # Pad with zeros to cover the range of the Limber integrals. Several
        # nodes avoid ringing of the splines.
        lk = cosmo.get_pk_spline_lk()
        a_lo = np.linspace(cosmo.get_pk_spline_a()[0], a[0], 8)[:-1]
        a_hi = np.linspace(a[-1], 1, 8)[1:]
        a_arr = np.concatenate([a_lo, a, a_hi])

        bks = []
        for prof in profs:
            bk = np.zeros((a_arr.size, lk.size))
            bk_bin = ccl.halos.halomod_bispectrum_squeezed(
                cosmo,
                hmc,
                np.exp(lk),
                a,
                prof,
                prof2=prof2,
                prof3=prof3,
                prof23_2pt=prof23_2pt,
            )
            if key[2]:
                # One-galaxy term
                bk_bin += np.array(
                    [
                        hmc.integrate_over_massfunc(
                            lambda M: prof.selection(M, aa)
                            * prof2.fourier(cosmo, 1e-4, M, aa),
                            cosmo,
                            aa,
                        )
                        / prof2.get_normalization(cosmo, aa, hmc=hmc) ** 2
                        for aa in a
                    ]
                )[:, None]
            bk[a_lo.size : a_lo.size + a.size] = bk_bin
            bks.append(
                ccl.Pk2D(a_arr=a_arr, lk_arr=lk, pk_arr=bk, is_logp=False)
            )

        self.bispectra[(richness_i, z_i, key)] = bks
        return bks

    def get_covariance_block(self, tracer_comb1, tracer_comb2, ell=None):
        if ell is None:
            ell = self.get_ell_eff()

        richness_i, z_i = self.extract_indices_rich_z(tracer_comb2)
        tr1, tr2 = [self.get_ccl_tracer(tr) for tr in tracer_comb1]
        bks = self.get_bin_bispectra(richness_i, z_i, tracer_comb1)
        # CCL returns the covariance with the counts per steradian
        cov = [
            ccl.angular_cl_cov_counts(
                self.cosmo, tr1, tr2, ell=ell, b_of_k_a=bk, fsky=self.fsky
            )
            for bk in bks
        ]
        return self.survey_area * np.array(cov).T


class ClusterClSSCHaloModel(CovarianceClusterCountsCl, ClusterCountsSSC):
    """Super-sample C_ell x cluster covariance.

    The long mode is projected with the same redshift bin windows of the
    cluster counts SSC (:class:`ClusterCountsSSC`), not assumed to be
    diagonal in chi:

    Cov(N_i, C_ell) = Nb_i Int dchi / chi^2 q_1 q_2
    dP/ddelta_b((ell + 1/2) / chi, a) sigma2_i(chi),

    with Nb_i the number of clusters times their bias, dP/ddelta_b the
    response of the power spectrum to the long mode (from
    :func:`pyccl.halos.halomod_bispectrum_squeezed_SSC`) and sigma2_i(chi)
    the covariance between the long mode averaged over the redshift bin
    window and over a thin shell at chi.

    With a cluster mask (see :class:`ClusterCountsSSC`), the long mode of
    the C_ell is averaged over the product of the masks of their tracers, as
    for the C_ell SSC with masks, and the multipoles are weighted by its
    cross power spectrum with the cluster mask.
    """

    cov_type = "SSC"

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)
        self.responses = {}
        self.sigma2 = {}
        self.ell_weights_cl = {}
        # Number of shells for the projection of the long mode
        self.na = self.config.get("ClusterCl", {}).get("na", 800)

        # Scale factors of the P(k) spline up to the maximum redshift of the
        # C_ell tracers, including the next node so that z_max is in range.
        # At higher z, the HOD can have no galaxies and its response is nan.
        sacc_file = self.io.get_sacc_file()
        z_max = max(
            tr.z.max()
            for tr in sacc_file.tracers.values()
            if hasattr(tr, "nz")
        )
        a = self.get_cosmology().get_pk_spline_a()
        self.a_arr = a[max(np.sum(1 / a > z_max + 1) - 1, 0) :]

    def get_response(self, tracer_comb):
        """Return the response of the C_ell power spectrum to a long mode.

        Args:
            tracer_comb (list): List of the pair of tracer names of C_ell

        Returns:
            :obj:`pyccl.Pk2D`: dP/ddelta_b(k, a)
        """
        key, (prof2, prof3), prof23_2pt = self.get_tracer_profiles(tracer_comb)
        if key not in self.responses:
            cosmo, a = self.get_cosmology(), self.a_arr
            # The SSC bispectrum of a selection S(M) = 1 is
            # I_1_1(S) dP/ddelta_b
            unit = ccl.halos.HaloProfileSelection(
                mass_def=self.mass_def, selection=lambda M, a: np.ones_like(M)
            )
            bk = ccl.halos.halomod_Pk2D_bispectrum_squeezed_SSC(
                cosmo,
                self.hmc,
                unit,
                prof2=prof2,
                prof3=prof3,
                prof23_2pt=prof23_2pt,
                a_arr=a,
            )
            _, lk, bk = bk.get_spline_arrays()
            i11 = [self.hmc.I_1_1(cosmo, 1e-4, aa, unit) for aa in a]
            self.responses[key] = ccl.Pk2D(
                a_arr=a,
                lk_arr=lk,
                pk_arr=bk / np.array(i11)[:, None],
                is_logp=False,
            )
        return self.responses[key]

    def get_shell_window(self, a, ells=None):
        """Harmonic coefficients of the window of thin shells.

        Same normalization as :meth:`ClusterCountsSSC.window_redshift_bin`.

        Args:
            a (array): Scale factors of the shells
            ells (array, optional): Multipoles. If None, 0 to L.

        Returns:
            array: Growth factor times window, shape (a.size, k.size,
            ells.size)
        """
        if ells is None:
            ells = range(self.L + 1)
        cosmo = self.get_cosmology()
        chi = ccl.comoving_radial_distance(cosmo, a)
        jl = np.array(
            [spherical_jn(ell, np.outer(chi, self.k_arr)) for ell in ells]
        ).transpose(1, 2, 0)
        D = ccl.growth_factor(cosmo, a)
        return 4 * np.pi * D[:, None, None] * jl

    def get_ell_weights_cl(self, tracer_comb):
        """Return the multipole weights of the long mode of C_ell x clusters.

        Args:
            tracer_comb (list): List of the pair of tracer names of C_ell

        Returns:
            array: Weights for multipoles 0 to L, from the cross power
            spectrum of the cluster mask and the product of the masks of the
            C_ell tracers, or for a spherical cap of the survey area if there
            is no cluster mask.
        """
        if self.mask is None:
            return self.get_ell_weights()
        key = tuple(tracer_comb)
        if key not in self.ell_weights_cl:
            tr = dict(zip([1, 2, 3, 4], 2 * list(tracer_comb)))
            masks = self.get_masks_dict(tr)
            self.ell_weights_cl[key] = self.get_mask_wl(
                self.mask, masks[1] * masks[2]
            )
        return self.ell_weights_cl[key]

    def get_sigma2(self, z_i, tracer_comb):
        """Return the covariance of the long mode in a cluster bin and shells.

        Args:
            z_i (int): Redshift bin
            tracer_comb (list): List of the pair of tracer names of C_ell,
                whose masks set the footprint of the shells

        Returns:
            tuple:
                - a (array): Scale factors of the shells
                - sigma2 (array): Covariance for each shell
        """
        if z_i not in self.sigma2:
            cosmo, k = self.get_cosmology(), self.k_arr
            # sigma2 changes on the scale of the cluster bin and has long
            # tails, so the shells sample densely all redshifts
            a = np.union1d(
                np.linspace(self.a_arr[0], 1, self.na),
                1 / (1 + self.get_z_true(z_i)),
            )
            w_bin = self.window_redshift_bin(
                k, self.get_z_true(z_i), z_i, self.L, self.sigma_0
            )
            pk0 = ccl.linear_matter_power(cosmo, k, 1.0)
            # Contribution of each multipole, one at a time to save memory
            sigma2_ell = np.zeros((a.size, self.L + 1))
            for ell in range(self.L + 1):
                w_shell = self.get_shell_window(a, [ell])[..., 0]
                sigma2_ell[:, ell] = simpson(
                    k**2 * pk0 * w_bin[:, ell] * w_shell, x=k, axis=-1
                )
            self.sigma2[z_i] = (a, sigma2_ell / (2 * np.pi) ** 3)

        a, sigma2_ell = self.sigma2[z_i]
        return a, sigma2_ell @ self.get_ell_weights_cl(tracer_comb)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, ell=None):
        if ell is None:
            ell = self.get_ell_eff()

        cosmo = self.get_cosmology()
        richness_i, z_i = self.extract_indices_rich_z(tracer_comb2)
        tr1, tr2 = [self.get_ccl_tracer(tr) for tr in tracer_comb1]

        # Projected response weighted by the long mode covariance
        a, sigma2 = self.get_sigma2(z_i, tracer_comb1)
        resp = self.get_response(tracer_comb1)
        lk = cosmo.get_pk_spline_lk()
        resp = np.array([resp(np.exp(lk), aa) for aa in a])
        pk = ccl.Pk2D(
            a_arr=a, lk_arr=lk, pk_arr=resp * sigma2[:, None], is_logp=False
        )
        cl = ccl.angular_cl(cosmo, tr1, tr2, ell, p_of_k_a=pk)

        # Number of clusters times their bias
        a_bin, profs = self.get_bin_profiles(richness_i, z_i)
        nb = [
            self.integrate_volume(
                a_bin, lambda aa: self.hmc.I_1_1(cosmo, 1e-4, aa, prof)
            )
            for prof in profs
        ]
        return np.outer(cl, nb)
