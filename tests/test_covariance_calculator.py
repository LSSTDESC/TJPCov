#!/usr/bin/python
from tjpcov.covariance_calculator import CovarianceCalculator
from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt
from tjpcov.covariance_fourier_ssc import FourierSSCHaloModel
import os
import pytest
import numpy as np
import sacc
import shutil


INPUT_YML = "./tests/data/conf_covariance_calculator.yml"
OUTDIR = "./tests/tmp/"


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)


def teardown_module():
    shutil.rmtree(OUTDIR)


@pytest.fixture
def mock_cov_calc():
    return CovarianceCalculator(INPUT_YML)


def test_smoke():
    CovarianceCalculator(INPUT_YML)


# TODO: Test with "clxN" when clusters are implemented
def test_get_covariance_classes(mock_cov_calc):
    classes = mock_cov_calc.get_covariance_classes()

    assert isinstance(classes["gauss"], dict)
    assert isinstance(classes["SSC"], dict)
    assert isinstance(classes["gauss"][("cl", "cl")], FourierGaussianNmt)
    assert isinstance(classes["SSC"][("cl", "cl")], FourierSSCHaloModel)

    # Test it raises an error if two gauss contributions are requested
    config = mock_cov_calc.config.copy()
    config["tjpcov"]["cov_type"] = ["FourierGaussianFsky"] * 2
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()

    # Test that it raises an error if you request Fourier and Real space covs
    config = mock_cov_calc.config.copy()
    config["tjpcov"]["cov_type"] = [
        "FourierGaussianFsky",
        "RealGaussianFsky",
    ]
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()


def test_get_covariance(mock_cov_calc):
    cov = mock_cov_calc.get_covariance() + 1e-100

    cov_gauss = FourierGaussianNmt(INPUT_YML).get_covariance()
    cov_ssc = FourierSSCHaloModel(INPUT_YML).get_covariance()
    cov2 = (cov_gauss + cov_ssc) + 1e-100

    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)


def test_get_covariance_terms(mock_cov_calc):
    cov_terms = mock_cov_calc.get_covariance_terms()

    cov_gauss = FourierGaussianNmt(INPUT_YML).get_covariance()
    cov_ssc = FourierSSCHaloModel(INPUT_YML).get_covariance()

    assert np.all(cov_terms["gauss"] == cov_gauss)
    assert np.all(cov_terms["SSC"] == cov_ssc)


def test_create_sacc_cov(mock_cov_calc):
    cov = mock_cov_calc.get_covariance() + 1e-100

    # Check returned file
    s = mock_cov_calc.create_sacc_cov()
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Check saved file
    s = sacc.Sacc.load_fits(OUTDIR + "cls_cov.fits")
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Test the different terms are saved
    assert os.path.isfile(OUTDIR + "cls_cov_gauss.fits")
    assert os.path.isfile(OUTDIR + "cls_cov_SSC.fits")

    # Custom name
    # Check returned file
    s = mock_cov_calc.create_sacc_cov("test.fits", save_terms=False)
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Check saved file
    s = sacc.Sacc.load_fits(OUTDIR + "test.fits")
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Test the different terms are not saved
    assert not os.path.isfile(OUTDIR + "test_gauss.fits")
    assert not os.path.isfile(OUTDIR + "test_SSC.fits")
