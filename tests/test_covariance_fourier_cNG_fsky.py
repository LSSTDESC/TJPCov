#!/usr/bin/python3
import os
import shutil

import numpy as np
import pyccl as ccl
import pytest
import sacc
import yaml

from tjpcov.covariance_fourier_cNG_fsky import FouriercNGHaloModelFsky

ROOT = "./tests/benchmarks/32_DES_tjpcov_bm/"
INPUT_YML_cNG = "./tests/data/conf_covariance_cNG_fsky.yaml"
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
    return FouriercNGHaloModelFsky(INPUT_YML_cNG)


def get_config():
    with open(INPUT_YML_cNG) as f:
        config = yaml.safe_load(f)
    return config


def get_hod_model():
    obj = FouriercNGHaloModelFsky(INPUT_YML_cNG)
    mass_def = ccl.halos.MassDef200m
    cM = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    hod = ccl.halos.HaloProfileHOD(
        mass_def=mass_def,
        concentration=cM,
        log10Mmin_0=obj.HOD_dict["log10Mmin_0"],
        log10Mmin_p=obj.HOD_dict["log10Mmin_p"],
        siglnM_0=obj.HOD_dict["siglnM_0"],
        siglnM_p=obj.HOD_dict["siglnM_p"],
        log10M0_0=obj.HOD_dict["log10M0_0"],
        log10M0_p=obj.HOD_dict["log10M0_p"],
        log10M1_0=obj.HOD_dict["log10M1_0"],
        log10M1_p=obj.HOD_dict["log10M1_p"],
        alpha_0=obj.HOD_dict["alpha_0"],
        alpha_p=obj.HOD_dict["alpha_p"],
        fc_0=obj.HOD_dict["fc_0"],
        fc_p=obj.HOD_dict["fc_p"],
        bg_0=obj.HOD_dict["bg_0"],
        bg_p=obj.HOD_dict["bg_p"],
        bmax_0=obj.HOD_dict["bmax_0"],
        bmax_p=obj.HOD_dict["bmax_p"],
        a_pivot=obj.HOD_dict["a_pivot"],
        ns_independent=obj.HOD_dict["ns_independent"],
        is_number_counts=obj.HOD_dict["is_number_counts"],
    )

    return hod


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
    fsky = config["GaussianFsky"].get("fsky", None)
    return fsky


def test_smoke():
    FouriercNGHaloModelFsky(INPUT_YML_cNG)


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
    integration_method = "spline"
    cov_cNG = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=False,
        integration_method=integration_method,
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
    tr = {}
    tr[1], tr[2] = tracer_comb1
    tr[3], tr[4] = tracer_comb2
    z_max = []
    for i in range(4):
        tr_sacc = s.tracers[tr[i + 1]]
        z = tr_sacc.z
        z_max.append(z.max())
    # Divide by zero errors happen when default a_arr used for 1h term
    z_max = np.min(z_max)

    # Array of a.
    # Use the a's in the pk spline
    na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
    a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)
    # Cut the array for efficiency
    sel = 1 / a_arr < z_max + 1
    # Include the next node so that z_max is in the range
    sel[np.sum(~sel) - 1] = True
    a_arr = a_arr[sel]
    bias1 = bias2 = bias3 = bias4 = 1
    if "gc" in tracer_comb1[0]:
        bias1 = cov_fcNG.bias_lens[tracer_comb1[0]]

    if "gc" in tracer_comb1[1]:
        bias2 = cov_fcNG.bias_lens[tracer_comb1[1]]

    if "gc" in tracer_comb2[0]:
        bias3 = cov_fcNG.bias_lens[tracer_comb2[0]]

    if "gc" in tracer_comb2[0]:
        bias4 = cov_fcNG.bias_lens[tracer_comb2[1]]

    biases = bias1 * bias2 * bias3 * bias4

    hmc = get_halo_model(cosmo)
    nfw_profile = get_NFW_profile()
    hod = get_hod_model()
    prof_2pt = ccl.halos.profiles_2pt.Profile2ptHOD()

    tkk_cNG = ccl.halos.halomod_Tk3D_cNG(
        cosmo,
        hmc,
        prof=nfw_profile,
        separable_growth=True,
        a_arr=a_arr,
    )
    tkk_1h_nfw = ccl.halos.halomod_Tk3D_1h(
        cosmo,
        hmc,
        prof=nfw_profile,
        a_arr=a_arr,
    )
    tkk_1h_hod = ccl.halos.halomod_Tk3D_1h(
        cosmo,
        hmc,
        prof=hod,
        prof12_2pt=prof_2pt,
        prof34_2pt=prof_2pt,
        a_arr=a_arr,
    )

    ccl_tracers, _ = cov_fcNG.get_tracer_info()
    tr1 = ccl_tracers[tracer_comb1[0]]
    tr2 = ccl_tracers[tracer_comb1[1]]
    tr3 = ccl_tracers[tracer_comb2[0]]
    tr4 = ccl_tracers[tracer_comb2[1]]

    fsky = get_fsky(*tracer_comb1, *tracer_comb2)

    cov_ccl = ccl.covariances.angular_cl_cov_cNG(
        cosmo,
        tracer1=tr1,
        tracer2=tr2,
        tracer3=tr3,
        tracer4=tr4,
        ell=ell,
        t_of_kk_a=tkk_cNG,
        fsky=fsky,
        integration_method=integration_method,
    )

    cov_ccl_1h_nfw = ccl.covariances.angular_cl_cov_cNG(
        cosmo,
        tracer1=tr1,
        tracer2=tr2,
        tracer3=tr3,
        tracer4=tr4,
        ell=ell,
        t_of_kk_a=tkk_1h_nfw,
        fsky=fsky,
        integration_method=integration_method,
    )

    cov_ccl_1h_hod = ccl.covariances.angular_cl_cov_cNG(
        cosmo,
        tracer1=tr1,
        tracer2=tr2,
        tracer3=tr3,
        tracer4=tr4,
        ell=ell,
        t_of_kk_a=tkk_1h_hod,
        fsky=fsky,
        integration_method=integration_method,
    )
    # An unfortunately messy way to to calculate the 234h terms
    # with an NFW Profile and only the 1h term with an HOD
    # using current CCL infrastructure.
    cov_ccl = biases * (cov_ccl - cov_ccl_1h_nfw) + cov_ccl_1h_hod

    assert np.max(np.fabs(np.diag(cov_cNG / cov_ccl - 1))) < 1e-5
    assert np.max(np.fabs(cov_cNG / cov_ccl - 1)) < 1e-3

    # Check you get zeroed B-modes
    cov_cNG_zb = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=True,
        integration_method=integration_method,
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
        integration_method=integration_method,
    )
    assert np.all(covf["cov_nob"] == cov_cNG)

    cov_cNG_zb = cov_fcNG.get_covariance_block(
        tracer_comb1=tracer_comb1,
        tracer_comb2=tracer_comb2,
        include_b_modes=True,
        integration_method=integration_method,
    )

    assert np.all(covf["cov"] == cov_cNG_zb)
