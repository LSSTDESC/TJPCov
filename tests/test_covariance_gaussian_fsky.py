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
# CCL and sacc input:
outdir = "tests/tmp/"
os.makedirs(outdir, exist_ok=True)
cosmo_filename = "tests/data/cosmo_desy1.yaml"
cosmo = ccl.Cosmology.read_yaml(cosmo_filename)
with open(outdir + "cosmos_desy1_v2p1p0.pkl", "wb") as ff:
    pickle.dump(cosmo, ff)

# SETUP
input_yml = "tests/data/conf_covariance_gaussian_fsky_fourier.yaml"
input_yml_real = "tests/data/conf_covariance_gaussian_fsky_real.yaml"
cfsky = FourierGaussianFsky(input_yml)
cfsky_real = RealGaussianFsky(input_yml_real)
ccl_tracers, tracer_Noise = cfsky.get_tracer_info()


def clean_tmp():
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir)


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    clean_tmp()


def get_config():
    return CovarianceIO(input_yml).config


def test_smoke():
    FourierGaussianFsky(input_yml)
    RealGaussianFsky(input_yml_real)

    # Check it raises an error if fsky is not given
    config = get_config()
    config["GaussianFsky"] = {}
    with pytest.raises(ValueError):
        FourierGaussianFsky(config)


def test_Fourier_get_binning_info():
    cfsky = FourierGaussianFsky(input_yml)
    ell, ell_eff, ell_edges = cfsky.get_binning_info()

    assert np.all(ell_eff == cfsky.get_ell_eff())
    assert np.allclose((ell_edges[1:] + ell_edges[:-1]) / 2, ell_eff)

    with pytest.raises(NotImplementedError):
        cfsky.get_binning_info("log")


def test_Fourier_get_covariance_block():
    # Test made independent of pickled objects
    tracer_comb1 = ("lens0", "lens0")
    tracer_comb2 = ("lens0", "lens0")

    ell, ell_bins, ell_edges = cfsky.get_binning_info()
    ccltr = ccl_tracers["lens0"]
    cl = ccl.angular_cl(cosmo, ccltr, ccltr, ell) + tracer_Noise["lens0"]

    fsky = cfsky.fsky
    dl = np.gradient(ell)
    cov = np.diag(2 * cl**2 / ((2 * ell + 1) * fsky * dl))
    lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

    cov2 = cfsky.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=False,
    )
    np.testing.assert_allclose(cov2, cov)

    # Check B-modes
    trs = ("src0", "src0")
    cov2 = cfsky.get_covariance_block(
        tracer_comb1=trs, tracer_comb2=trs, include_b_modes=False
    )
    cov2b = cfsky.get_covariance_block(
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
        cfsky.get_covariance_block(
            tracer_comb1=trs, tracer_comb2=trs, for_real=True
        )
    # 2. Check block
    cov2 = cfsky.get_covariance_block(
        tracer_comb1=trs, tracer_comb2=trs, for_real=True, lmax=30
    )
    ell = np.arange(30 + 1)
    ccltr = ccl_tracers["src0"]
    cl = ccl.angular_cl(cosmo, ccltr, ccltr, ell) + tracer_Noise["src0"]
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
def test_Real_get_fourier_block(tracer_comb1, tracer_comb2):
    cov = cfsky_real._get_fourier_block(tracer_comb1, tracer_comb2)
    cov2 = cfsky.get_covariance_block(
        tracer_comb1, tracer_comb2, for_real=True, lmax=cfsky_real.lmax
    )

    norm = np.pi * 4 * cfsky_real.fsky
    assert np.all(cov == cov2 / norm)


def test_smoke_get_covariance():
    # Check that we can get the full covariance
    cfsky.get_covariance()
    # Real test commented out because we don't have a method to build the full
    # covariance atm
    cfsky_real.get_covariance()
