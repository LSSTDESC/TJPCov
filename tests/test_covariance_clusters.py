#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_cluster_counts import ClusterCounts
from tjpcov.clusters_helpers import FFTHelper
import pyccl.halos.hmfunc as hmf

cosmo = ccl.Cosmology.read_yaml("tests/data/cosmo_desy1.yaml")
input_yml = "./tests/data/conf_covariance_clusters.yaml"


def get_mock_cosmo():
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


def get_mock_sacc():
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


def get_mock_covariance():

    cc_cov = ClusterCounts(input_yml)
    cc_cov.load_from_sacc(get_mock_sacc())
    cc_cov.load_from_cosmology(get_mock_cosmo())
    cc_cov.fft_helper = FFTHelper(
        get_mock_cosmo(), cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(get_mock_cosmo())
    cc_cov.h0 = 0.67
    return cc_cov


def test_integral_mass_no_bias():
    ref1 = 1.463291259900985e-05
    ref2 = 1.4251538328691035e-05
    cc_cov = get_mock_covariance()

    test1 = cc_cov.mass_richness_integral(0.3, 0, remove_bias=True)
    test2 = cc_cov.mass_richness_integral(0.35, 0, remove_bias=True)

    np.testing.assert_almost_equal(ref1, test1)
    np.testing.assert_almost_equal(ref2, test2)


test_integral_mass_no_bias()


def test_double_bessel_integral():
    ref = 8.427201745032292e-05
    cc_cov = get_mock_covariance()
    test = cc_cov.double_bessel_integral(0.3, 0.3)
    np.testing.assert_almost_equal(ref, test)


test_double_bessel_integral()


def test_shot_noise():
    ref = 63973.635143644424
    cc_cov = get_mock_covariance()
    test = cc_cov.shot_noise(0, 0)
    np.testing.assert_almost_equal(test, ref)


test_shot_noise()


def test_integral_mass():
    cc_cov = get_mock_covariance()
    ref1 = 2.596895139062984e-05
    ref2 = 2.5910691906342223e-05

    test1 = cc_cov.mass_richness_integral(0.5, 0)
    test2 = cc_cov.mass_richness_integral(0.55, 0)

    np.testing.assert_almost_equal(ref1, test1)
    np.testing.assert_almost_equal(ref2, test2)


test_integral_mass()


def test_mass_richness():
    cc_cov = get_mock_covariance()
    reference_min = 0.0009528852621284171

    test_min = [
        cc_cov.mass_richness(cc_cov.min_mass, i)
        for i in range(cc_cov.num_richness_bins)
    ]
    np.testing.assert_almost_equal(np.sum(test_min), reference_min)


test_mass_richness()


def test_calc_dv():
    reference_values = [
        6613.739621696188,
        55940746.72160228,
        3781771343.1278453,
        252063237.8394578,
        1113852.72571463,
    ]

    cc_cov = get_mock_covariance()
    z_true = 0.8

    for i, z_i in enumerate([0, 4, 8, 14, 17]):
        np.testing.assert_almost_equal(
            cc_cov.comoving_volume_element(z_true, z_i) / 1e4,
            reference_values[i] / 1e4,
        )


test_calc_dv()


def test_cov_nxn():
    ref_sum = 130462.91921818888

    cc_cov = get_mock_covariance()

    cov_00 = cc_cov.get_covariance_cluster_counts(
        ("clusters_0_0",), ("clusters_0_0",)
    )

    np.testing.assert_almost_equal(ref_sum, cov_00)


test_cov_nxn()
