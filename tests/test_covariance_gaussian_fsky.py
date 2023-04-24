#!/usr/bin/python3
import os
import pickle

import numpy as np
import pyccl as ccl
import pytest
import shutil

from tjpcov.wigner_transform import bin_cov
from tjpcov.covariance_gaussian_fsky import (
    FourierGaussianFsky,
    RealGaussianFsky,
)
from tjpcov.covariance_io import CovarianceIO

# INPUT
OUTDIR = "tests/tmp/"
COSMO_FILENAME = "tests/data/cosmo_desy1.yaml"
INPUT_YML = "tests/data/conf_covariance_gaussian_fsky_fourier.yaml"
INPUT_YML_REAL = "tests/data/conf_covariance_gaussian_fsky_real.yaml"


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)
    # CCL and sacc input:
    cosmo = ccl.Cosmology.read_yaml(COSMO_FILENAME)
    with open(OUTDIR + "cosmos_desy1_v2p1p0.pkl", "wb") as ff:
        pickle.dump(cosmo, ff)


def teardown_module():
    shutil.rmtree(OUTDIR)


@pytest.fixture
def mock_cosmo():
    return ccl.Cosmology.read_yaml(COSMO_FILENAME)


@pytest.fixture
def cov_fg_fsky():
    return FourierGaussianFsky(INPUT_YML)


@pytest.fixture
def cov_rg_fsky():
    return RealGaussianFsky(INPUT_YML_REAL)


def get_config():
    return CovarianceIO(INPUT_YML).config


def test_smoke():
    FourierGaussianFsky(INPUT_YML)
    RealGaussianFsky(INPUT_YML_REAL)

    # Check it raises an error if fsky is not given
    config = get_config()
    config["GaussianFsky"] = {}
    with pytest.raises(ValueError):
        FourierGaussianFsky(config)


def test_Fourier_get_binning_info(cov_fg_fsky):
    ell, ell_eff, ell_edges = cov_fg_fsky.get_binning_info()

    assert np.all(ell_eff == cov_fg_fsky.get_ell_eff())
    assert np.allclose((ell_edges[1:] + ell_edges[:-1]) / 2, ell_eff)

    with pytest.raises(NotImplementedError):
        cov_fg_fsky.get_binning_info("log")


def test_Fourier_get_covariance_block(cov_fg_fsky, mock_cosmo):
    # Test made independent of pickled objects
    tracer_comb1 = ("lens0", "lens0")
    tracer_comb2 = ("lens0", "lens0")

    ell, _, ell_edges = cov_fg_fsky.get_binning_info()
    ccl_tracers, tracer_noise = cov_fg_fsky.get_tracer_info()

    ccltr = ccl_tracers["lens0"]
    cl = ccl.angular_cl(mock_cosmo, ccltr, ccltr, ell) + tracer_noise["lens0"]

    fsky = cov_fg_fsky.fsky
    dl = np.gradient(ell)
    cov = np.diag(2 * cl**2 / ((2 * ell + 1) * fsky * dl))
    lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

    cov2 = cov_fg_fsky.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=False,
    )
    np.testing.assert_allclose(cov2, cov)

    # Check B-modes
    trs = ("src0", "src0")
    cov2 = cov_fg_fsky.get_covariance_block(
        tracer_comb1=trs, tracer_comb2=trs, include_b_modes=False
    )
    cov2b = cov_fg_fsky.get_covariance_block(
        tracer_comb1=trs, tracer_comb2=trs, include_b_modes=True
    )

    nbpw = lb.size
    cov2b = cov2b.reshape((nbpw, 4, nbpw, 4))
    assert np.all(cov2b[:, 0, :, 0] == cov2)
    cov2b[:, 0, :, 0] -= cov2
    assert not np.any(cov2b)

    # Check for_real
    # 1. Check that it request lmax
    with pytest.raises(ValueError):
        cov_fg_fsky.get_covariance_block(
            tracer_comb1=trs, tracer_comb2=trs, for_real=True
        )
    # 2. Check block
    cov2 = cov_fg_fsky.get_covariance_block(
        tracer_comb1=trs, tracer_comb2=trs, for_real=True, lmax=30
    )
    ell = np.arange(30 + 1)
    ccltr = ccl_tracers["src0"]
    cl = ccl.angular_cl(mock_cosmo, ccltr, ccltr, ell) + tracer_noise["src0"]
    cov = np.diag(2 * cl**2)
    assert cov2.shape == (ell.size, ell.size)
    np.testing.assert_allclose(cov2, cov)


@pytest.mark.parametrize(
    "tracer_comb1",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
@pytest.mark.parametrize(
    "tracer_comb2",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
def test_Real_get_fourier_block(
    cov_rg_fsky, cov_fg_fsky, tracer_comb1, tracer_comb2
):
    cov = cov_rg_fsky._get_fourier_block(tracer_comb1, tracer_comb2)
    cov2 = cov_fg_fsky.get_covariance_block(
        tracer_comb1, tracer_comb2, for_real=True, lmax=cov_rg_fsky.lmax
    )

    norm = np.pi * 4 * cov_rg_fsky.fsky
    assert np.all(cov == cov2 / norm)


def test_smoke_get_covariance(cov_fg_fsky, cov_rg_fsky):
    # Check that we can get the full covariance
    cov_fg_fsky.get_covariance()
    cov_rg_fsky.get_covariance()
