#!/usr/bin/python
from tjpcov_new.covariance_calculator import CovarianceCalculator
from tjpcov_new.covariance_gaussian_fsky import CovarianceFourierGaussianFsky
from tjpcov_new.covariance_fourier_ssc import FourierSSCHaloModel
import os
import pytest
import numpy as np
import sacc


input_yml = "./tests_new/data/config_covariance_calculator.yml"
outdir = 'tests/tmp'

def test_smoke():
    CovarianceCalculator(input_yml)


def test_get_covariance_classes():
    cc = CovarianceCalculator(input_yml)
    classes = cc.get_covariance_classes()

    assert isinstance(classes['gauss'], CovarianceFourierGaussianFsky)
    assert isinstance(classes['SSC'], FourierSSCHaloModel)

    # Test it raises an error if two gauss contributions are requested
    config = cc.config.copy()
    config['tjpcov']['cov_type'] = ['CovarianceFourierGaussianFsky'] * 2
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()

    # Test that it raises an error if you request Fourier and Real space covs
    config = cc.config.copy()
    config['tjpcov']['cov_type'] = ['CovarianceFourierGaussianFsky',
                                    'CovarianceRealGaussianFsky']
    with pytest.raises(ValueError):
        cc = CovarianceCalculator(config)
        cc.get_covariance_classes()


def test_get_covariance():
    cc = CovarianceCalculator(input_yml)
    cov = cc.get_covariance() + 1e-100

    cov_gauss = CovarianceFourierGaussianFsky(input_yml).get_covariance()
    cov_ssc = CovarianceFourierGaussianFsky(input_yml).get_covariance()
    cov2 = (cov_gauss + cov_ssc) + 1e-100

    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)


def test_create_sacc_cov():
    cc = CovarianceCalculator(input_yml)
    cov = cc.get_covariance() + 1e-100

    cc.create_sacc_cov()
    s = sacc.Sacc.load_fits(outdir + '/cls_cov.fits')
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)

    # Custom name
    cc.create_sacc_cov('prueba.fits')
    s = sacc.Sacc.load_fits(outdir + '/prueba.fits')
    cov2 = s.covariance.covmat + 1e-100
    assert np.max(np.abs(cov / cov2 - 1) < 1e-10)
