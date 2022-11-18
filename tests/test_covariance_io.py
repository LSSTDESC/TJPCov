#!/usr/bin/python
from tjpcov.covariance_io import CovarianceIO
import os
import pytest
import numpy as np
import sacc
import shutil
from datetime import datetime
from glob import glob


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = root + "tjpcov_tmp/"
input_yml = os.path.join(root, "conf_covariance_gaussian_fourier_nmt.yaml")
input_sacc = sacc.Sacc.load_fits(root + "cls_cov.fits")


def clean_tmp():
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir)


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    clean_tmp()


def get_diag_covariance():
    ndata = input_sacc.mean.size
    return np.diag(np.ones(ndata))


def test_smoke_input():
    # CovarianceIO should accept dictionary or file path
    CovarianceIO(input_yml)
    config = CovarianceIO._parse(input_yml)
    CovarianceIO(config)
    with pytest.raises(ValueError):
        CovarianceIO(["hello"])

    # Check outdir is created
    if os.path.isdir(outdir):
        os.system(f"rm -rf {outdir}")
    CovarianceIO(input_yml)
    assert os.path.isdir(outdir)


def test_create_sacc_cov():
    cio = CovarianceIO(input_yml)
    # Circunvent the NotImplementedError
    cov = get_diag_covariance()
    s = cio.create_sacc_cov(cov)
    s2 = sacc.Sacc.load_fits(outdir + "cls_cov.fits")

    assert np.all(s.mean == input_sacc.mean)
    assert np.all(s.covariance.covmat == get_diag_covariance())
    assert np.all(s.mean == s2.mean)
    assert np.all(s.covariance.covmat == s2.covariance.covmat)

    # Check that it also writes the file with a different name
    s2 = cio.create_sacc_cov(cov, "cls_cov2.fits")
    s2 = sacc.Sacc.load_fits(outdir + "cls_cov2.fits")

    # Check that it will not overwrite a file but create a new one with the utc
    # time stamped
    s = cio.create_sacc_cov(cov)
    date = datetime.utcnow()
    # Timestamp without the seconds since there can be a delay between this
    # timestamp and the one when creating the sacc file.
    timestamp = date.strftime("%Y%m%d%H%M")
    files = glob(outdir + f"cls_cov.fits_{timestamp}*")
    assert len(files) == 1

    # Check that it will overwrite if overwrite is True
    cio.create_sacc_cov(0 * cov, overwrite=True)
    s2 = sacc.Sacc.load_fits(outdir + "cls_cov.fits")
    assert np.all(s2.covariance.covmat == 0)


def test_get_outdir():
    cio = CovarianceIO(input_yml)
    assert os.path.samefile(cio.get_outdir(), outdir)


def test_get_sacc_file():
    cio = CovarianceIO(input_yml)
    s = cio.get_sacc_file()

    assert np.all(s.mean == input_sacc.mean)
