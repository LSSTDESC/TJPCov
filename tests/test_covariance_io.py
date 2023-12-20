#!/usr/bin/python
from tjpcov.covariance_io import CovarianceIO
import os
import pytest
import numpy as np
import sacc
import shutil
from datetime import datetime
from glob import glob

OUT_DIR = "./tests/tmp/"
INPUT_YML = "./tests/data/conf_covariance_gaussian_fourier_nmt.yaml"


def teardown_module():
    """Cleanup after this test module completes."""
    shutil.rmtree(OUT_DIR)


def setup_module():
    """Prep work for this test module to run"""
    os.makedirs(OUT_DIR, exist_ok=True)


@pytest.fixture(name="sacc_file")
def fixture_sacc_file():
    sacc_file = sacc.Sacc()

    sacc_file.add_tracer("misc", "a_bin_1")
    sacc_file.add_tracer("misc", "b_bin_1")
    sacc_file.add_tracer("misc", "a_bin_2")
    sacc_file.add_tracer("misc", "b_bin_2")

    sacc_file.add_data_point("counts", ("a_bin_1", "b_bin_1"), 1)
    sacc_file.add_data_point("counts", ("a_bin_2", "b_bin_2"), 1)

    sacc_file.add_covariance(np.diag(np.ones(2)))
    return sacc_file


@pytest.fixture(name="covariance_matrix")
def fixture_covariance_matrix():
    return np.diag(np.ones(2))


@pytest.fixture(name="covariance_io")
def fixture_covariance_io():
    return CovarianceIO(INPUT_YML)


def test_config_requires_tjpcov():
    with pytest.raises(
        ValueError, match="tjpcov section not found in configuration"
    ):
        CovarianceIO({"key": {"outdir": "./"}})


def test_config_requires_tjpcov_is_dict():
    with pytest.raises(
        ValueError, match="tjpcov section must be a dictionary"
    ):
        CovarianceIO({"tjpcov": 1})


def test_config_requires_tjpcov_has_outdir():
    with pytest.raises(
        ValueError, match="outdir not found in the tjpcov configuration"
    ):
        CovarianceIO({"tjpcov": {"key": "./"}})


def test_throws_when_invalid_string():
    with pytest.raises(FileNotFoundError):
        CovarianceIO("")


def test_throws_when_not_dict_or_string():
    with pytest.raises(TypeError):
        CovarianceIO(1)

    with pytest.raises(TypeError):
        CovarianceIO(["sherkaner underhill"])


def test_accepts_dict():
    cio = CovarianceIO({"tjpcov": {"outdir": "./"}})
    assert isinstance(cio, CovarianceIO)
    assert cio is not None


def test_accepts_yaml_filename():
    cio = CovarianceIO(INPUT_YML)
    assert isinstance(cio, CovarianceIO)
    assert cio is not None


def test_get_dict_from_yaml():
    cio = CovarianceIO.get_dict_from_yaml(INPUT_YML)
    assert cio is not None
    assert isinstance(cio, dict)


def test_get_outdir(covariance_io):
    assert os.path.samefile(covariance_io.outdir, OUT_DIR)


def test_create_output_directory():
    shutil.rmtree(OUT_DIR)
    assert not os.path.isdir(OUT_DIR)
    CovarianceIO(INPUT_YML)
    assert os.path.isdir(OUT_DIR)


def test_create_sacc_cov(covariance_io, sacc_file, covariance_matrix):
    # Overwrite the sacc file coming from the yaml file for testing
    covariance_io.sacc_file = sacc_file

    new_sacc_file = covariance_io.create_sacc_cov(covariance_matrix)

    assert np.all(new_sacc_file.mean == sacc_file.mean)
    assert np.all(
        new_sacc_file.covariance.covmat == sacc_file.covariance.covmat
    )
    assert os.path.isfile(OUT_DIR + "cls_cov.fits")


def test_create_sacc_cov_new_name(covariance_io, sacc_file, covariance_matrix):
    # Overwrite the sacc file coming from the yaml file for testing
    covariance_io.sacc_file = sacc_file

    new_sacc_file = covariance_io.create_sacc_cov(
        covariance_matrix, "new_cov_file.fits"
    )

    assert np.all(new_sacc_file.mean == sacc_file.mean)
    assert np.all(
        new_sacc_file.covariance.covmat == sacc_file.covariance.covmat
    )
    assert os.path.isfile(OUT_DIR + "new_cov_file.fits")


def test_create_sacc_cov_file_exists(
    covariance_io, sacc_file, covariance_matrix
):
    # Overwrite the sacc file coming from the yaml file for testing
    covariance_io.sacc_file = sacc_file

    _ = covariance_io.create_sacc_cov(
        covariance_matrix, output="cov_file_exists.fits"
    )
    assert os.path.isfile(OUT_DIR + "cov_file_exists.fits")

    _ = covariance_io.create_sacc_cov(
        covariance_matrix, output="cov_file_exists.fits"
    )
    assert os.path.isfile(OUT_DIR + "cov_file_exists.fits")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    assert len(glob(OUT_DIR + f"cov_file_exists.fits_{timestamp}*")) == 1


def test_create_sacc_cov_overwrite(
    covariance_io, sacc_file, covariance_matrix
):
    # Overwrite the sacc file coming from the yaml file for testing
    covariance_io.sacc_file = sacc_file

    _ = covariance_io.create_sacc_cov(
        covariance_matrix, output="cov_file_overwrite.fits"
    )
    assert os.path.isfile(OUT_DIR + "cov_file_overwrite.fits")

    _ = covariance_io.create_sacc_cov(
        covariance_matrix, output="cov_file_overwrite.fits", overwrite=True
    )
    assert os.path.isfile(OUT_DIR + "cov_file_overwrite.fits")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    assert len(glob(OUT_DIR + f"cov_file_overwrite.fits_{timestamp}*")) == 0


def test_get_sacc_file(covariance_io):
    sf = covariance_io.get_sacc_file()
    assert sf is not None
    assert isinstance(sf, sacc.Sacc)


def test_get_sacc_file_requires_sacc_file_key():
    cio = CovarianceIO({"tjpcov": {"outdir": "./"}})
    with pytest.raises(
        ValueError, match="sacc_file not found in the tjpcov configuration"
    ):
        cio.get_sacc_file()


def test_get_sacc_file_requires_sacc_file_string():
    cio = CovarianceIO({"tjpcov": {"outdir": "./", "sacc_file": sacc.Sacc()}})
    with pytest.raises(
        ValueError, match="sacc_file entry in the config file must be a string"
    ):
        cio.get_sacc_file()
