#!/usr/bin/python3
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new import bin_cov
from tjpcov_new.covariance_gaussian_fsky import CovarianceFourierGaussianFsky
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
cfsky = CovarianceFourierGaussianFsky(input_yml)
ccl_tracers, tracer_Noise = cfsky.get_tracer_info()


def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')


def get_config():
    return CovarianceIO(input_yml).config


def test_smoke():
    cfsky = CovarianceFourierGaussianFsky(input_yml)

    # Check it raises an error if fsky is not given
    config = get_config()
    config['GaussianFsky'] = {}
    with pytest.raises(ValueError):
        cfsky = CovarianceFourierGaussianFsky(config)


def test_get_binning_info():
    cfsky = CovarianceFourierGaussianFsky(input_yml)
    ell, ell_eff, ell_edges = cfsky.get_binning_info()

    assert np.all(ell_eff == cfsky.get_ell_eff())
    assert np.allclose((ell_edges[1:]+ell_edges[:-1])/2, ell_eff)

    with pytest.raises(NotImplementedError):
        cfsky.get_binning_info('log')


def test_get_covariance_block():
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
                                           tracer_comb2=tracer_comb2)
    np.testing.assert_allclose(gcov_cl_1,
                               cov)
