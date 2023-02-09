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

cosmo = ccl.Cosmology.read_yaml("./tests/data/cosmo_desy1.yaml")
input_yml = "./tests/data/conf_covariance_clusters.yaml"

# Test Fixtures


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
    s.metadata["nbins_cluster_redshift"] = 18
    s.metadata["nbins_cluster_richness"] = 3
    s.metadata["min_mass"] = 1e13

    # This isnt how tracers actually look, but sort of
    # hacks the class to work without building
    # an entire sacc file for this test.
    s.add_tracer(
        "misc",
        "clusters_0_0",
        metadata={
            "Mproxy_name": "richness",
            "Mproxy_min": 10,
            "Mproxy_max": 100,
            "z_name": "redshift",
            "z_min": 0.3,
            "z_max": 1.2,
        },
    )

    return s


@pytest.fixture
def mock_covariance_gauss(mock_sacc, mock_cosmo):

    cc_cov = ClusterCountsGaussian(input_yml)
    cc_cov.load_from_sacc(mock_sacc)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mock_cosmo)
    cc_cov.h0 = 0.67
    return cc_cov


@pytest.fixture
def mock_covariance_ssc(mock_sacc, mock_cosmo):

    cc_cov = ClusterCountsSSC(input_yml)
    cc_cov.load_from_sacc(mock_sacc)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mock_cosmo)
    cc_cov.h0 = 0.67
    return cc_cov


# Tests start


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
    assert test == pytest.approx(ref_val)


def test_double_bessel_integral(mock_covariance_gauss: CovarianceClusters):
    ref = 8.427201745032292e-05
    test = mock_covariance_gauss.double_bessel_integral(0.3, 0.3)
    assert test == pytest.approx(ref)


def test_shot_noise(mock_covariance_gauss: ClusterCountsGaussian):
    ref = 63973.635143644424
    test = mock_covariance_gauss.shot_noise(0, 0)
    assert test == pytest.approx(ref, 1e-5)


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
    assert test == pytest.approx(reference_val)


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
    assert cov_00_gauss + cov_00_ssc == pytest.approx(ref_sum, 1e-5)
