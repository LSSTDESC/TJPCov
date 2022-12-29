#!/usr/bin/python
from tjpcov.covariance_calculator import CovarianceCalculator
from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt
from tjpcov.covariance_fourier_ssc import FourierSSCHaloModel
import os
import pytest
import numpy as np
import sacc
import shutil


input_yml = "./tests/data/conf_covariance_calculator.yml"
outdir = "tests/tmp"


def clean_tmp():
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir)


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    clean_tmp()


def test_smoke():
    CovarianceCalculator(input_yml)


# TODO: Test with "clxN" when clusters are implemented
def test_get_covariance_classes():
    cc = CovarianceCalculator(input_yml)
    classes = cc.get_covariance_classes()

    assert isinstance(classes["gauss"], dict)
    assert isinstance(classes["SSC"], dict)
    assert isinstance(classes["gauss"][("cl", "cl")], FourierGaussianNmt)
    assert isinstance(classes["SSC"][("cl", "cl")], FourierSSCHaloModel)

    # Test it raises an error if two gauss contributions are requested
    config = cc.config.copy()
    config["tjpcov"]["cov_type"] = ["FourierGaussianFsky"] * 2
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()

    # Test that it raises an error if you request Fourier and Real space covs
    config = cc.config.copy()
    config["tjpcov"]["cov_type"] = [
        "FourierGaussianFsky",
        "RealGaussianFsky",
    ]
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()


def test_get_covariance():
    cc = CovarianceCalculator(input_yml)
    cov = cc.get_covariance() + 1e-100

    cov_gauss = FourierGaussianNmt(input_yml).get_covariance()
    cov_ssc = FourierSSCHaloModel(input_yml).get_covariance()
    cov2 = (cov_gauss + cov_ssc) + 1e-100

    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)


def test_get_covariance_terms():
    cc = CovarianceCalculator(input_yml)
    cov_terms = cc.get_covariance_terms()

    cov_gauss = FourierGaussianNmt(input_yml).get_covariance()
    cov_ssc = FourierSSCHaloModel(input_yml).get_covariance()

    assert np.all(cov_terms["gauss"] == cov_gauss)
    assert np.all(cov_terms["SSC"] == cov_ssc)


def test_create_sacc_cov():
    cc = CovarianceCalculator(input_yml)
    cov = cc.get_covariance() + 1e-100

    # Check returned file
    s = cc.create_sacc_cov()
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)
    # Check saved file
    s = sacc.Sacc.load_fits(outdir + "/cls_cov.fits")
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Test the different terms are saved
    assert os.path.isfile(outdir + "/cls_cov_gauss.fits")
    assert os.path.isfile(outdir + "/cls_cov_SSC.fits")
    os.remove(outdir + "/cls_cov.fits")
    os.remove(outdir + "/cls_cov_gauss.fits")
    os.remove(outdir + "/cls_cov_SSC.fits")

    # Custom name
    # Check returned file
    s = cc.create_sacc_cov("test.fits", save_terms=False)
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)
    # Check saved file
    s = sacc.Sacc.load_fits(outdir + "/test.fits")
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Test the different terms are not saved
    assert not os.path.isfile(outdir + "/test_gauss.fits")
    assert not os.path.isfile(outdir + "/test_SSC.fits")

    os.remove(outdir + "/test.fits")
