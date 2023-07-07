#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_clusters import CovarianceClusters
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC
from tjpcov.clusters_helpers import FFTHelper
import pyccl.halos.hmfunc as hmf
import pytest
import itertools

INPUT_YML = "./tests/data/conf_covariance_clusters.yaml"


@pytest.fixture
def mock_cosmo():
    Omg_c = 0.26
    Omg_b = 0.04
    h0 = 0.67  # so H0 = 100h0 will be in km/s/Mpc
    A_s_value = 2.1e-9
    n_s_value = 0.96
    w_0 = -1.0
    w_a = 0.0

    cosmo = ccl.Cosmology(
        Omega_c=Omg_c,
        Omega_b=Omg_b,
        h=h0,
        A_s=A_s_value,
        n_s=n_s_value,
        w0=w_0,
        wa=w_a,
    )
    return cosmo


@pytest.fixture
def mock_sacc():
    # Using values from
    # https://github.com/nrussofer/Cosmological-Covariance-matrices
    # /blob/master/Full%20covariance%20N_N%20part%20vfinal.ipynb
    # As reference.
    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = 10, 100

    z_tracers = []
    z_edges = np.linspace(z_min, z_max, 19)
    for i, zbin in enumerate(zip(z_edges[:-1], z_edges[1:])):
        bin_z_label = f"bin_z_{i}"
        s.add_tracer("bin_z", bin_z_label, zbin[0], zbin[1])
        z_tracers.append(bin_z_label)

    richness_tracers = []
    richness_edges = np.linspace(richness_min, richness_max, 4)
    for i, richness_bin in enumerate(
        zip(richness_edges[:-1], richness_edges[1:])
    ):
        bin_richness_label = f"bin_richness_{i}"
        s.add_tracer(
            "bin_richness",
            bin_richness_label,
            richness_bin[0],
            richness_bin[1],
        )
        richness_tracers.append(bin_richness_label)

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_counts,
            (bin_z_label, bin_richness_label),
            int(counts),
        )

    return s


@pytest.fixture
def mock_covariance_gauss(mock_sacc, mock_cosmo):
    cc_cov = ClusterCountsGaussian(INPUT_YML)
    cc_cov.load_from_sacc(mock_sacc, min_halo_mass=1e13)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mock_cosmo)
    cc_cov.h0 = 0.67
    return cc_cov


@pytest.fixture
def mock_covariance_ssc(mock_sacc, mock_cosmo):
    cc_cov = ClusterCountsSSC(INPUT_YML)
    cc_cov.load_from_sacc(mock_sacc, min_halo_mass=1e13)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mock_cosmo)
    cc_cov.h0 = 0.67
    return cc_cov


# Tests start


def test_is_not_null():
    cc_cov = ClusterCountsSSC(INPUT_YML)
    assert cc_cov is not None
    cc_cov = None

    cc_cov = ClusterCountsGaussian(INPUT_YML)
    assert cc_cov is not None


def test_load_from_sacc(mock_covariance_gauss: CovarianceClusters):
    assert mock_covariance_gauss.min_mass == np.log(1e13)
    assert mock_covariance_gauss.num_richness_bins == 3
    assert mock_covariance_gauss.num_z_bins == 18
    assert mock_covariance_gauss.min_richness == 10
    assert mock_covariance_gauss.max_richness == 100
    assert mock_covariance_gauss.z_min == 0.3
    assert mock_covariance_gauss.z_max == 1.2


def test_load_from_cosmology(mock_covariance_gauss: CovarianceClusters):
    cosmo = ccl.CosmologyVanillaLCDM()
    mock_covariance_gauss.load_from_cosmology(cosmo)

    assert mock_covariance_gauss.cosmo == cosmo


@pytest.mark.parametrize(
    "z, ref_val",
    [
        (0.3, 1.463291259900985e-05),
        (0.35, 1.4251538328691035e-05),
    ],
)
def test_integral_mass_no_bias(
    mock_covariance_gauss: CovarianceClusters, z, ref_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0, remove_bias=True)
    assert test == pytest.approx(ref_val, rel=1e-4)


def test_double_bessel_integral(mock_covariance_gauss: CovarianceClusters):
    ref = 8.427201745032292e-05
    test = mock_covariance_gauss.double_bessel_integral(0.3, 0.3)
    assert test == pytest.approx(ref, rel=1e-4)


def test_shot_noise(mock_covariance_gauss: ClusterCountsGaussian):
    ref = 63973.635143644424
    test = mock_covariance_gauss.shot_noise(0, 0)
    assert test == pytest.approx(ref, 1e-4)


@pytest.mark.parametrize(
    "z, reference_val",
    [
        (0.5, 2.596895139062984e-05),
        (0.55, 2.5910691906342223e-05),
    ],
)
def test_integral_mass(
    mock_covariance_gauss: CovarianceClusters, z, reference_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0)
    assert test == pytest.approx(reference_val, rel=1e-4)


def test_mass_richness(mock_covariance_gauss: CovarianceClusters):
    reference_min = 0.0009528852621284171

    test_min = [
        mock_covariance_gauss.mass_richness(mock_covariance_gauss.min_mass, i)
        for i in range(mock_covariance_gauss.num_richness_bins)
    ]
    assert np.sum(test_min) == pytest.approx(reference_min)


@pytest.mark.parametrize(
    "z_i, reference_val",
    [
        (0, 6613.739621696188),
        (4, 55940746.72160228),
        (8, 3781771343.1278453),
        (14, 252063237.8394578),
        (17, 1113852.72571463),
    ],
)
def test_calc_dv(
    mock_covariance_gauss: CovarianceClusters, z_i, reference_val
):
    z_true = 0.8
    test = mock_covariance_gauss.comoving_volume_element(z_true, z_i) / 1e4
    assert test == pytest.approx(reference_val / 1e4)


def test_cov_nxn(
    mock_covariance_gauss: ClusterCountsGaussian,
    mock_covariance_ssc: ClusterCountsSSC,
):
    ref_sum = 130462.91921818888

    cov_00_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("clusters_0_0",), ("clusters_0_0",)
    )
    cov_00_ssc = mock_covariance_ssc.get_covariance_block_for_sacc(
        ("clusters_0_0",), ("clusters_0_0",)
    )
    assert cov_00_gauss + cov_00_ssc == pytest.approx(ref_sum, rel=1e-4)
