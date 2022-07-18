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


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml_ssc = os.path.join(root, "tjpcov_conf_minimal_ssc.yaml")

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.97,
                      sigma8=0.8, m_nu=0.0)

def get_config():
    with open(input_yml_ssc) as f:
        config = yaml.safe_load(f)
    return config


def get_halomodel_calculator_and_NFW_profile():
    md = ccl.halos.MassDef200m()
    mf = ccl.halos.MassFuncTinker10(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mdef=md)

    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    pNFW = ccl.halos.HaloProfileNFW(cm)

    return hmc, pNFW


def get_cl_footprint(tr1, tr2, tr3, tr4):
    config = get_config()
    mf = config['tjpcov']['mask_file']
    mf12 = hp.read_map(mf[tr1]) * hp.read_map(mf[tr2])
    mf34 = hp.read_map(mf[tr3]) * hp.read_map(mf[tr4])

    alm = hp.map2alm(mf12)
    blm = hp.map2alm(mf34)

    cl = hp.alm2cl(alm * blm)
    cl *= (2 * np.arange(cl.size) + 1)

    return cl


def get_CovarianceCalculator():
    config = get_config()
    return cv.CovarianceCalculator(config)


def test_get_SSC_cov():
    # TJPCov covariance
    cc  = get_CovarianceCalculator()
    s = cc.cl_data
    ell, _ = s.get_ell_cl('cl_ee', 'DESgc__0', 'DESgc__0')

    tracer_comb1 = tracer_comb2 = ('DESgc__0', 'DESgc__0')
    ccl_tracers, _ = cc.get_tracer_info(s)
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                             tracer_comb2=tracer_comb2,
                             ccl_tracers=ccl_tracers,
                             integration_method='qag_quad',
                             include_b_modes=False)

    # CCL covariance
    hmc, prof = get_halomodel_calculator_and_NFW_profile()
    tkk_ssc = ccl.halos.halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof, bias1=2,
                                                     bias2=3, bias3=4, bias4=5,
                                                     is_number_counts1=True,
                                                     is_number_counts2=True,
                                                     is_number_counts3=True,
                                                     is_number_counts4=True)

    a_arr = 1./(1+np.linspace(0, 3, 15)[::-1])
    cl_mask = get_cl_footprint(*tracer_comb1, *tracer_comb2)
    sigma2_B = ccl.sigma2_B_from_mask(cosmo, a=a_arr, mask_wl=cl_mask)
    tr = ccl_tracers['DESgc__0']
    cov_ccl = ccl.angular_cl_cov_SSC(cosmo, cltracer1=tr, cltracer2=tr,
                                     ell=ell, tkka=tkk_ssc,
                                     sigma2_B=(a_arr, sigma2_B),
                                     cltracer3=tr, cltracer4=tr)

    print(cov_ccl)
    print(cov_ccl.shape)
    assert np.all(np.fabs(cov_ssc/cov_ccl - 1) < 1e-3)

    # Check you get zeroed B-modes
    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=True)

    assert cov_ssc_zb.shape == (4 * ell.size, 4 * ell.size)
    cov_ssc_zb = cov_ssc_zb.reshape((ell.size, 4, ell.size, 4))
    assert np.all(cov_ssc_zb[:, 0, :, 0] == cov_ssc)
    cov_ssc_zb[:, 0, :, 0] -= cov_ssc
    assert np.all(cov_ssc_zb == np.zeros_like(cov_ssc_zb))


def test_get_all_cov_SSC():
    # Smoke
    cc  = get_CovarianceCalculator()
    cov_ssc = cc.get_all_cov_SSC() + 1e-100

    # Check the covariance has the right terms for 00, 0e and ee
    s = cc.cl_data.copy()
    s.covariance.covmat = cov_ssc
    s.remove_selection(data_type='cl_0b')
    s.remove_selection(data_type='cl_eb')
    s.remove_selection(data_type='cl_be')
    s.remove_selection(data_type='cl_bb')

    trs = s.get_tracer_combinations()
    ccl_tracers, _ = cc.get_tracer_info(s)
    for tracer_comb1 in trs:
        i1 = s.indices(tracers=tracer_comb1)
        for tracer_comb2 in trs:
            i2 = s.indices(tracers=tracer_comb2)
            cov_ssc2 = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                      tracer_comb2=tracer_comb2,
                                      ccl_tracers=ccl_tracers,
                                      integration_method='qag_quad',
                                      include_b_modes=False) + 1e-100

            cov_i1i2 = s.covariance.covmat[i1][:, i2]
            assert np.max(np.abs(cov_i1i2 / cov_ssc2 - 1)) < 1e-5

    # Check the covariance is 0 for all the b-modes
    s = cc.cl_data.copy()
    s.covariance.covmat = cov_ssc
    s.remove_selection(data_type='cl_00')
    s.remove_selection(data_type='cl_0e')
    s.remove_selection(data_type='cl_ee')

    assert np.all(s.covariance.covmat == 1e-100)
