#!/usr/bin/python3
import os
import shutil

import healpy as hp
import numpy as np
import pyccl as ccl
import pytest
import sacc
import yaml

from tjpcov.covariance_fourier_cNG import FouriercNGHaloModel

ROOT = "./tests/benchmarks/32_DES_tjpcov_bm/"
INPUT_YML_cNG = "./tests/data/conf_covariance_cNG.yaml"
OUTDIR = "./tests/tmp/"
NSIDE = 32


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)


def teardown_module():
    shutil.rmtree(OUTDIR)


@pytest.fixture(autouse=True)
def teardown_test():
    clean_outdir()


def clean_outdir():
    os.system(f"rm -f {OUTDIR}*")


@pytest.fixture
def sacc_file():
    return sacc.Sacc.load_fits(ROOT + "cls_cov.fits")


@pytest.fixture
def cov_fcNG():
    return FouriercNGHaloModel(INPUT_YML_cNG)


def get_config():
    with open(INPUT_YML_cNG) as f:
        config = yaml.safe_load(f)
    return config


def get_halo_model(cosmo):
    md = ccl.halos.MassDef200m
    mf = ccl.halos.MassFuncTinker08(mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(mass_def=md)
    hmc = ccl.halos.HMCalculator(mass_function=mf, halo_bias=hb, mass_def=md)

    return hmc


def get_NFW_profile():
    md = ccl.halos.MassDef200m
    cm = ccl.halos.ConcentrationDuffy08(mass_def=md)
    pNFW = ccl.halos.HaloProfileNFW(mass_def=md, concentration=cm)

    return pNFW


def get_fsky(tr1, tr2, tr3, tr4):
    config = get_config()
    mf = config["tjpcov"]["mask_file"]

    # TODO: do we need the hp area?
    # area = hp.nside2pixarea(32)
    m1 = hp.read_map(mf[tr1])
    m2 = hp.read_map(mf[tr2])
    m3 = hp.read_map(mf[tr3])
    m4 = hp.read_map(mf[tr4])

    return np.mean(m1 * m2 * m3 * m4)


def test_smoke():
    FouriercNGHaloModel(INPUT_YML_cNG)


@pytest.mark.parametrize(
    "tracer_comb1,tracer_comb2",
    [
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESgc__0")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
    ],
)
def test_get_covariance_block(cov_fcNG, tracer_comb1, tracer_comb2):
    # TJPCov covariance
    cosmo = cov_fcNG.get_cosmology()
    s = cov_fcNG.io.get_sacc_file()
    ell, _ = s.get_ell_cl("cl_00", "DESgc__0", "DESgc__0")

    cov_cNG = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=False,
    )

    # Check saved file
    covf = np.load(
        OUTDIR + "cng_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)
    )
    assert (
        np.max(np.abs((covf["cov_nob"] + 1e-100) / (cov_cNG + 1e-100) - 1))
        < 1e-10
    )

    # CCL covariance
    na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
    a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)

    # TODO: Need to make 1h TK3D object with HOD
    # & weight non-HOD TK3D object with gbias factors
    # & combine together before call to 
    # angular_cl_cov_cNG for proper comparison
    bias1 = bias2 = bias3 = bias4 = 1
    if "gc" in tracer_comb1[0]:
        bias1 = cov_fcNG.bias_lens[tracer_comb1[0]]

    if "gc" in tracer_comb1[1]:
        bias2 = cov_fcNG.bias_lens[tracer_comb1[1]]

    if "gc" in tracer_comb2[0]:
        bias3 = cov_fcNG.bias_lens[tracer_comb2[0]]

    if "gc" in tracer_comb2[0]:
        bias4 = cov_fcNG.bias_lens[tracer_comb2[1]]
    
    hmc = get_halo_model(cosmo)
    nfw_profile = get_NFW_profile()
    tkk_cNG = ccl.halos.halomod_Tk3D_cNG(
        cosmo,
        hmc,
        prof=nfw_profile,
    )

    ccl_tracers, _ = cov_fcNG.get_tracer_info()
    tr1 = ccl_tracers[tracer_comb1[0]]
    tr2 = ccl_tracers[tracer_comb1[1]]
    tr3 = ccl_tracers[tracer_comb2[0]]
    tr4 = ccl_tracers[tracer_comb2[1]]

    fsky = get_fsky(tr1, tr2, tr3, tr4)

    cov_ccl = ccl.covariances.angular_cl_cov_cNG(
        cosmo,
        tracer1=tr1,
        tracer2=tr2,
        tracer3=tr3,
        tracer4=tr4,
        ell=ell,
        t_of_kk_a=tkk_cNG,
        fsky=fsky,
    )

    assert np.max(np.fabs(np.diag(cov_cNG / cov_ccl - 1))) < 1e-5
    assert np.max(np.fabs(cov_cNG / cov_ccl - 1)) < 1e-3

    # Check you get zeroed B-modes
    cov_cNG_zb = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=True,
    )
    # Check saved
    assert (
        np.max(np.abs((covf["cov"] + 1e-100) / (cov_cNG_zb + 1e-100) - 1))
        < 1e-10
    )

    ix1 = s.indices(tracers=tracer_comb1)
    ix2 = s.indices(tracers=tracer_comb2)
    ncell1 = int(ix1.size / ell.size)
    ncell2 = int(ix2.size / ell.size)

    # The covariance will have all correlations, including when EB == BE
    if (ncell1 == 3) and (tracer_comb1[0] == tracer_comb1[1]):
        ncell1 += 1
    if (ncell2 == 3) and (tracer_comb2[0] == tracer_comb2[1]):
        ncell2 += 1

    assert cov_cNG_zb.shape == (ell.size * ncell1, ell.size * ncell2)
    # Check the blocks
    cov_cNG_zb = cov_cNG_zb.reshape((ell.size, ncell1, ell.size, ncell2))
    # Check the reshape has the correct ordering
    assert cov_cNG_zb[:, 0, :, 0].flatten() == pytest.approx(
        cov_cNG.flatten(), rel=1e-10
    )
    assert np.all(cov_cNG_zb[:, 1::, :, 1::] == 0)

    # Check get_cNG_cov reads file
    covf = np.load(
        OUTDIR + "cng_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)
    )
    cov_cNG = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=False,
    )
    assert np.all(covf["cov_nob"] == cov_cNG)

    cov_cNG_zb = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=True,
    )

    assert np.all(covf["cov"] == cov_cNG_zb)
