#!/usr/bin/python3
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new import bin_cov
from tjpcov_new.covariance_gaussian_fsky import \
    CovarianceFourierGaussianFsky, CovarianceRealGaussianFsky
from tjpcov_new.covariance_io import CovarianceIO
import yaml
import healpy as hp
import sacc
import shutil

# Some useful functions
def depickling(name):
    """get the target data
    """
    try:
        with open(f"tests/val_{name}.pkl", 'rb') as ff:
            return pickle.load(ff)
    except:
        print(f"missing {name}")
        print(os.listdir("tests/"))


def check_numerr(a, b, f=np.array_equal):
    """ wrapper for test
    """
    print('ok' if f(a, b) else 'Check for numerical errors')

# INPUT
# CCL and sacc input:
os.makedirs('tests/tmp/', exist_ok=True)
cosmo_filename = "tests/data/cosmo_desy1.yaml"
cosmo = ccl.Cosmology.read_yaml(cosmo_filename)
with open("tests/tmp/cosmos_desy1_v2p1p0.pkl", 'wb') as ff:
    pickle.dump(cosmo, ff)

xi_fn = "examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
cl_fn = "examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits"

# Reference Output extracted from ipynb:
with open("tests/data/tjpcov_cl.pkl", "rb") as ff:
    ref_cov0cl = pickle.load(ff)
with open("tests/data/tjpcov_xi.pkl", "rb") as ff:
    ref_cov0xi = pickle.load(ff)

ref_covnobin = depickling("covnobin")  # ell datavectors bins #EDGES
ref_md_ell_bins = depickling('metadata_ell_bins')

# SETUP
input_yml = "tests_new/data/conf_tjpcov_minimal.yaml"
input_yml_real = "tests_new/data/conf_tjpcov_minimal_real.yaml"
cfsky = CovarianceFourierGaussianFsky(input_yml)
ccl_tracers, tracer_Noise = cfsky.get_tracer_info()

class CRGF_tester(CovarianceRealGaussianFsky):
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

# cfsky_real = CovarianceRealGaussianFsky(input_yml)
cfsky_real = CRGF_tester(input_yml_real)

def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')


def get_config():
    return CovarianceIO(input_yml).config


def test_smoke():
    cfsky = CovarianceFourierGaussianFsky(input_yml)
    # cfsky = CovarianceRealGaussianFsky(input_yml)

    # Check it raises an error if fsky is not given
    config = get_config()
    config['GaussianFsky'] = {}
    with pytest.raises(ValueError):
        cfsky = CovarianceFourierGaussianFsky(config)


def test_Fourier_get_binning_info():
    cfsky = CovarianceFourierGaussianFsky(input_yml)
    ell, ell_eff, ell_edges = cfsky.get_binning_info()

    assert np.all(ell_eff == cfsky.get_ell_eff())
    assert np.allclose((ell_edges[1:]+ell_edges[:-1])/2, ell_eff)

    with pytest.raises(NotImplementedError):
        cfsky.get_binning_info('log')


def test_Fourier_get_covariance_block():
    # Test made independent of pickled objects
    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')

    s = cfsky.io.sacc_file

    ell, ell_bins, ell_edges = cfsky.get_binning_info()
    ccltr = ccl_tracers['lens0']
    cl = ccl.angular_cl(cosmo, ccltr, ccltr, ell) + tracer_Noise['lens0']

    fsky = cfsky.fsky
    dl = np.gradient(ell)
    cov = np.diag(2 * cl**2 / ((2 * ell + 1) * fsky * dl))
    lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

    gcov_cl_1 = cfsky.get_covariance_block(tracer_comb1=tracer_comb1,
                                           tracer_comb2=tracer_comb2,
                                           include_b_modes=False)
    np.testing.assert_allclose(gcov_cl_1, cov)

    trs = ('src0', 'src0')
    gcov_cl_1 = cfsky.get_covariance_block(tracer_comb1=trs,
                                           tracer_comb2=trs,
                                           include_b_modes=False)
    gcov_cl_1b = cfsky.get_covariance_block(tracer_comb1=trs,
                                           tracer_comb2=trs,
                                           include_b_modes=True)

    nbpw = lb.size
    assert np.all(gcov_cl_1b[:nbpw][:, :nbpw] == gcov_cl_1)
    gcov_cl_1b = gcov_cl_1b.reshape((nbpw, 4, nbpw, 4), order='F')
    gcov_cl_1b[:, 0, :, 0] -= gcov_cl_1
    assert not np.any(gcov_cl_1b)


def test_Real_get_binning_info():
    # Check we recover the ell effective from the edges
    theta, theta_eff, theta_edges = \
        cfsky_real.get_binning_info(in_radians=False)

    assert np.all(theta_eff == cfsky_real.get_theta_eff())
    assert np.allclose((theta_edges[1:]+theta_edges[:-1])/2, theta_eff)

    # Check in_radians work
    theta2, theta_eff2, theta_edges2 = \
        cfsky_real.get_binning_info(in_radians=True)
    arcmin_rad = np.pi / 180 / 60
    assert np.all(theta * arcmin_rad == theta2)
    assert np.all(theta_eff * arcmin_rad == theta_eff2)
    assert np.all(theta_edges * arcmin_rad == theta_edges2)

    with pytest.raises(NotImplementedError):
        cfsky_real.get_binning_info('linear')


def test_Real_get_covariance_block():
    # Test made independent of pickled objects
    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')

    print(f"Checking covariance block. \
          Tracer combination {tracer_comb1} {tracer_comb2}")
    s = cfsky_real.io.get_sacc_file()

    ell = np.arange(2, cfsky_real.lmax + 1)
    ccltr = ccl_tracers['lens0']
    cl = ccl.angular_cl(cosmo, ccltr, ccltr, ell) + tracer_Noise['lens0']

    fsky = cfsky_real.fsky
    dl = np.gradient(ell)
    cov = np.diag(2 * cl**2 / (4 * np.pi * fsky))

    WT = cfsky_real.get_Wigner_transform()
    s1_s2 = cfsky_real.get_cov_WT_spin(tracer_comb=tracer_comb1)
    th, cov = WT.projected_covariance2(l_cl=ell, s1_s2=s1_s2,
                                       s1_s2_cross=s1_s2, cl_cov=cov)

    gcov_xi_1 = cfsky_real.get_covariance_block(tracer_comb1=tracer_comb1,
                                                tracer_comb2=tracer_comb2,
                                                binned=False)
    np.testing.assert_allclose(gcov_xi_1, cov)

    gcov_xi_1 = cfsky_real.get_covariance_block(tracer_comb1=tracer_comb1,
                                                tracer_comb2=tracer_comb2,
                                                binned=True)

    theta, _, theta_edges = cfsky_real.get_binning_info(in_radians=False)
    _, cov = bin_cov(r=theta, r_bins=theta_edges, cov=cov)
    assert gcov_xi_1.shape == (20, 20)
    assert np.max(np.abs((gcov_xi_1+1e-100) / (cov + 1e-100) - 1)) < 1e-5
