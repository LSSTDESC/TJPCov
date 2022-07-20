#!/usr/bin/python
import numpy as np
import os
import pytest
import tjpcov.main as cv
from tjpcov.parser import parse
import yaml
import sacc
import glob
import pyccl as ccl
import healpy as hp
import shutil
import pymaster as nmt


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml_nmt = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_yml_ssc = os.path.join(root, "tjpcov_conf_minimal_ssc.yaml")
input_yml_nmt_ssc = os.path.join(root, "tjpcov_conf_minimal_nmt_ssc.yaml")

def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')

    if os.path.isdir('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/'):
        shutil.rmtree('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')
    os.makedirs('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')

clean_tmp()


def get_config(input_file):
    with open(input_file) as f:
        config = yaml.safe_load(f)
    return config


def get_CovarianceCalculator(input_file):
    config = get_config(input_file)
    return cv.CovarianceCalculator(config)


def get_nmt_bin():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def test_get_final_cov():
    cc_nmt = get_CovarianceCalculator(input_yml_nmt)
    cc_ssc = get_CovarianceCalculator(input_yml_ssc)
    cc_nmt_ssc = get_CovarianceCalculator(input_yml_nmt_ssc)

    cache = {'bins': get_nmt_bin()}
    cov_nmt = cc_nmt.get_final_cov(gauss_kwargs={'cache': cache})
    cov_ssc = cc_ssc.get_final_cov()
    cov_nmt_ssc = cc_nmt_ssc.get_final_cov(gauss_kwargs={'cache': cache})

    assert np.max(np.abs((cov_nmt+cov_ssc+1e-100)/(cov_nmt_ssc+1e-100) -1) < 1e-5)

    # Check the ssc_kwargs is read correctly
    clean_tmp()
    cov_ssc2 = cc_ssc.get_final_cov(ssc_kwargs={'integration_method':
                                                'spline'})
    assert np.max(np.abs((cov_ssc + 1e-100)/(cov_ssc2 + 1e-100) - 1)) > 1e-5


def test_create_sacc_cov():
    cc_nmt_ssc = get_CovarianceCalculator(input_yml_nmt_ssc)
    cov_nmt_ssc = cc_nmt_ssc.get_final_cov()

    fname = './tests/tmp/tmp.sacc'
    cc_nmt_ssc.create_sacc_cov(fname)
    s = sacc.Sacc.load_fits(fname)
    cov = s.covariance.covmat

    assert np.all(np.abs((cov + 1e-100)/(cov_nmt_ssc + 1e-100) -1) < 1e-5)

clean_tmp()
