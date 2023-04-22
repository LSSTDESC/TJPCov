#!/usr/bin/python
from tjpcov.covariance_io import CovarianceIO
import os
import pytest
import numpy as np
import sacc
import shutil
from datetime import datetime
from glob import glob

ROOT_DIR = "./tests/benchmarks/32_DES_tjpcov_bm/"
OUT_DIR = ROOT_DIR + "tmp/test_covariance_io/"
INPUT_YML = os.path.join(ROOT_DIR, "conf_covariance_gaussian_fourier_nmt.yaml")


def teardown_module():
    """whole test run finishes."""
    shutil.rmtree(OUT_DIR)


def setup_module():
    os.makedirs(OUT_DIR, exist_ok=True)


@pytest.fixture
def mock_sacc():
    input_sacc = sacc.Sacc.load_fits(ROOT_DIR + "cls_cov.fits")
    return input_sacc


@pytest.fixture
def mock_covariance_io():
    return CovarianceIO(INPUT_YML)


def get_diag_covariance(sacc):
    ndata = sacc.mean.size
    return np.diag(np.ones(ndata))


def test_smoke_input():
    # CovarianceIO should accept dictionary or file path
    CovarianceIO(INPUT_YML)
    config = CovarianceIO._parse(INPUT_YML)
    CovarianceIO(config)
    with pytest.raises(ValueError):
        CovarianceIO(["hello"])

    # Check outdir is created
    if os.path.isdir(OUT_DIR):
        os.system(f"rm -rf {OUT_DIR}")
    CovarianceIO(INPUT_YML)
    assert os.path.isdir(OUT_DIR)


def test_create_sacc_cov(mock_covariance_io, mock_sacc):
    cov = get_diag_covariance(mock_sacc)
    s = mock_covariance_io.create_sacc_cov(cov)
    s2 = sacc.Sacc.load_fits(OUT_DIR + "cls_cov.fits")

    assert np.all(s.mean == mock_sacc.mean)
    assert np.all(s.covariance.covmat == get_diag_covariance(mock_sacc))
    assert np.all(s.mean == s2.mean)
    assert np.all(s.covariance.covmat == s2.covariance.covmat)

    # Check that it also writes the file with a different name
    s2 = mock_covariance_io.create_sacc_cov(cov, "cls_cov2.fits")
    s2 = sacc.Sacc.load_fits(OUT_DIR + "cls_cov2.fits")

    # Check that it will not overwrite a file but create a new one with the utc
    # time stamped
    s = mock_covariance_io.create_sacc_cov(cov)
    date = datetime.utcnow()
    # Timestamp without the seconds since there can be a delay between this
    # timestamp and the one when creating the sacc file.
    timestamp = date.strftime("%Y%m%d%H%M")
    files = glob(OUT_DIR + f"cls_cov.fits_{timestamp}*")
    assert len(files) == 1

    # Check that it will overwrite if overwrite is True
    mock_covariance_io.create_sacc_cov(0 * cov, overwrite=True)
    s2 = sacc.Sacc.load_fits(OUT_DIR + "cls_cov.fits")
    assert np.all(s2.covariance.covmat == 0)


def test_get_outdir(mock_covariance_io):
    assert os.path.samefile(mock_covariance_io.get_outdir(), OUT_DIR)


def test_get_sacc_file(mock_covariance_io, mock_sacc):
    s = mock_covariance_io.get_sacc_file()

    assert np.all(s.mean == mock_sacc.mean)
