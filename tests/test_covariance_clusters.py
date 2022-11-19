#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_cluster_counts import ClusterCounts
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
    # Using values from https://github.com/nrussofer/Cosmological-Covariance-matrices/blob/master/Full%20covariance%20N_N%20part%20vfinal.ipynb
    # As reference.
    s = sacc.Sacc()
    s.metadata["nbins_cluster_redshift"] = 18
    s.metadata["nbins_cluster_richness"] = 3
    s.metadata["min_mass"] = 1e13

    # This isnt how tracers actually look, but sort of hacks the class to work without building
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
    cc_cov.mass_func = hmf.MassFuncTinker10(get_mock_cosmo())
    cc_cov.h0 = 0.67
    return cc_cov


def test_integral_mass_no_bias():
    ref1 = 1.463291259900985e-05
    ref2 = 1.4251538328691035e-05
    cc_cov = get_mock_covariance()

    test1 = cc_cov.integral_mass_no_bias(0.3, 0)
    test2 = cc_cov.integral_mass_no_bias(0.35, 0)

    np.testing.assert_almost_equal(ref1, test1)
    np.testing.assert_almost_equal(ref2, test2)


def test_double_bessel_integral():
    ref = 8.427201745032292e-05
    cc_cov = get_mock_covariance()
    test = cc_cov.double_bessel_integral(0.3, 0.3)
    np.testing.assert_almost_equal(ref, test)


def test_shot_noise():
    ref = 63973.635143644424
    cc_cov = get_mock_covariance()
    test = cc_cov.shot_noise(0, 0)
    np.testing.assert_almost_equal(test, ref)


def test_calc_G1():

    ref_sum_0 = 53.77019327633519
    ref_sum_1 = 52.42500031312957

    cc_cov = get_mock_covariance()

    Z1_true_0 = cc_cov.calc_Z1(0)
    test_G1_0 = cc_cov.calc_G1(Z1_true_0)

    Z1_true_1 = cc_cov.calc_Z1(1)
    test_G1_1 = cc_cov.calc_G1(Z1_true_1)

    np.testing.assert_almost_equal(np.sum(test_G1_0), ref_sum_0)
    np.testing.assert_almost_equal(np.sum(test_G1_1), ref_sum_1)


def test_calc_dV():

    ref_sum_0 = 41736789276.57224
    ref_sum_1 = 52065518985.98286

    cc_cov = get_mock_covariance()

    Z1_true = cc_cov.calc_Z1(0)
    test_dV_0 = cc_cov.calc_dV(Z1_true, 0)

    Z1_true = cc_cov.calc_Z1(1)
    test_dV_1 = cc_cov.calc_dV(Z1_true, 1)

    np.testing.assert_almost_equal(np.sum(test_dV_0), ref_sum_0)
    np.testing.assert_almost_equal(np.sum(test_dV_1) / 1e12, ref_sum_1 / 1e12)


def test_calc_M1():

    ref_0_0 = 0.0016602249035099581
    ref_1_1 = 0.0008823472776646072

    cc_cov = get_mock_covariance()

    Z1_true = cc_cov.calc_Z1(0)
    test_0_0 = cc_cov.calc_M1(Z1_true, 0)

    Z1_true = cc_cov.calc_Z1(1)
    test_1_1 = cc_cov.calc_M1(Z1_true, 1)

    np.testing.assert_almost_equal(np.sum(test_0_0), ref_0_0)
    np.testing.assert_almost_equal(np.sum(test_1_1), ref_1_1)


def test_calc_Z1():

    ref_sum_0 = 24.374999999999996
    ref_sum_1 = 27.624999999999996

    cc_cov = get_mock_covariance()

    test_Z1_0 = cc_cov.calc_Z1(0)
    test_Z1_1 = cc_cov.calc_Z1(1)

    np.testing.assert_almost_equal(np.sum(test_Z1_0), ref_sum_0)
    np.testing.assert_almost_equal(np.sum(test_Z1_1), ref_sum_1)


def test_integral_mass():
    cc_cov = get_mock_covariance()
    ref1 = 2.596895139062984e-05
    ref2 = 2.5910691906342223e-05

    test1 = cc_cov.integral_mass(0.5, 0)
    test2 = cc_cov.integral_mass(0.55, 0)

    np.testing.assert_almost_equal(ref1, test1)
    np.testing.assert_almost_equal(ref2, test2)


def test_mass_richness():
    cc_cov = get_mock_covariance()
    reference_min = 0.0009528852621284171

    test_min = [
        cc_cov.mass_richness(cc_cov.min_mass, i)
        for i in range(cc_cov.num_richness_bins)
    ]
    np.testing.assert_almost_equal(np.sum(test_min), reference_min)


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
            cc_cov.dV(z_true, z_i), reference_values[i]
        )


def test_cov_nxn():
    ref_sum = 130462.91921818888

    cc_cov = get_mock_covariance()

    cov_00 = cc_cov.get_covariance_cluster_counts(
        ("clusters_0_0",), ("clusters_0_0",)
    )

    np.testing.assert_almost_equal(ref_sum, cov_00)
