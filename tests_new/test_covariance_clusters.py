#!/usr/bin/python3
import os
import numpy as np
import pyccl as ccl
import sacc
from tjpcov_new.covariance_cluster_counts import CovarianceClusterCounts
import pyccl.halos.hmfunc as hmf

# INPUT
# CCL and sacc input:
os.makedirs("tests/tmp/", exist_ok=True)
cosmo_filename = "tests/data/cosmo_desy1.yaml"
cosmo = ccl.Cosmology.read_yaml(cosmo_filename)

root = "./examples/clusters/"
input_yml = os.path.join(root, "tjpcov_conf_minimal_clusters.yaml")


def get_mock_cosmo():
    Omega_c = 0.26
    Omega_b = 0.04
    h0 = 0.67
    A_s = 2.1e-9
    n_s = 0.96
    w0 = -1.0
    wa = 0.0

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, h=h0, A_s=A_s, n_s=n_s, w0=w0, wa=wa
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

    cc_cov = CovarianceClusterCounts(input_yml)
    cc_cov.load_from_sacc(get_mock_sacc())
    cc_cov.load_from_cosmology(get_mock_cosmo())
    cc_cov.mass_func = hmf.MassFuncTinker10(get_mock_cosmo())
    cc_cov.h0 = 0.67
    cc_cov.setup_vectors()
    return cc_cov


def test_integral_mass_no_bias():
    ref1 = 1.463291259900985e-05
    ref2 = 1.4251538328691035e-05
    cc_cov = get_mock_covariance()

    test1 = cc_cov.integral_mass_no_bias(0.3, 0)
    test2 = cc_cov.integral_mass_no_bias(0.35, 0)

    np.testing.assert_almost_equal(ref1, test1)
    np.testing.assert_almost_equal(ref2, test2)


def test_eval_M1_true_vec():

    ref_sum = 0.048185959642970705
    cc_cov = get_mock_covariance()

    cc_cov.eval_true_vec()
    cc_cov.eval_M1_true_vec()

    np.testing.assert_almost_equal(np.sum(cc_cov.M1_true_vec), ref_sum)


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


def test_eval_true_vec():
    ref_z1_sum = 936.0
    ref_g1_sum = 795.6517795142859
    ref_dv_sum = 2295974227982.374

    cc_cov = get_mock_covariance()
    cc_cov.eval_true_vec()

    z1_sum = np.sum(cc_cov.Z1_true_vec)
    g1_sum = np.sum(cc_cov.G1_true_vec)
    dv_sum = np.sum(cc_cov.dV_true_vec)

    np.testing.assert_almost_equal(z1_sum, ref_z1_sum)
    np.testing.assert_almost_equal(g1_sum, ref_g1_sum)
    np.testing.assert_almost_equal(dv_sum / 1e10, ref_dv_sum / 1e10)


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


def test_dv():

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
