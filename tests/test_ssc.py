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


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml_ssc = os.path.join(root, "tjpcov_conf_minimal_ssc.yaml")


def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')

    if os.path.isdir('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/'):
        shutil.rmtree('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')
    os.makedirs('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')

clean_tmp()

def get_config():
    with open(input_yml_ssc) as f:
        config = yaml.safe_load(f)
    return config


def get_halomodel_calculator_and_NFW_profile(cosmo):
    md = ccl.halos.MassDef200m()
    mf = ccl.halos.MassFuncTinker08(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mdef=md)

    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    pNFW = ccl.halos.HaloProfileNFW(cm)

    return hmc, pNFW


def get_cl_footprint(tr1, tr2, tr3, tr4):
    config = get_config()
    mf = config['tjpcov']['mask_file']

    area = hp.nside2pixarea(32)
    m1 = hp.read_map(mf[tr1])
    m2 = hp.read_map(mf[tr2])
    m3 = hp.read_map(mf[tr3])
    m4 = hp.read_map(mf[tr4])

    m12 = m1 * m2
    m34 = m3 * m4

    alm = hp.map2alm(m12)
    blm = hp.map2alm(m34)

    cl = hp.alm2cl(alm, blm)
    cl *= (2 * np.arange(cl.size) + 1)
    cl /= np.sum(m12) * np.sum(m34) * area**2

    return cl


def get_CovarianceCalculator():
    config = get_config()
    return cv.CovarianceCalculator(config)


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_get_SSC_cov(tracer_comb1, tracer_comb2):
    # TJPCov covariance
    cc  = get_CovarianceCalculator()
    cosmo = cc.cosmo
    s = cc.cl_data
    ell, _ = s.get_ell_cl('cl_00', 'DESgc__0', 'DESgc__0')

    ccl_tracers, _ = cc.get_tracer_info(s)
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                             tracer_comb2=tracer_comb2,
                             ccl_tracers=ccl_tracers,
                             integration_method='qag_quad',
                             include_b_modes=False)

    # Check saved file
    covf = np.load(cc.outdir + '/ssc_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                             *tracer_comb2))
    assert np.max(np.abs((covf['cov_nob'] + 1e-100) / (cov_ssc + 1e-100) - 1)) < 1e-10
    clean_tmp()

    # CCL covariance
    na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
    a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)

    bias1 = 1
    is_nc1 = False
    if 'gc' in tracer_comb1[0]:
        bias1 = cc.bias_lens[tracer_comb1[0]]
        is_nc1 = True

    bias2 = 1
    is_nc2 = False
    if 'gc' in tracer_comb1[1]:
        bias2 = cc.bias_lens[tracer_comb1[1]]
        is_nc2 = True

    bias3 = 1
    is_nc3 = False
    if 'gc' in tracer_comb2[0]:
        bias3 = cc.bias_lens[tracer_comb2[0]]
        is_nc3 = True

    bias4 = 1
    is_nc4 = False
    if 'gc' in tracer_comb2[0]:
        bias4 = cc.bias_lens[tracer_comb2[1]]
        is_nc4 = True


    hmc, prof = get_halomodel_calculator_and_NFW_profile(cosmo)
    tkk_ssc = ccl.halos.halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof,
                                                     bias1=bias1,
                                                     bias2=bias2, bias3=bias3,
                                                     bias4=bias4,
                                                     is_number_counts1=is_nc1,
                                                     is_number_counts2=is_nc2,
                                                     is_number_counts3=is_nc3,
                                                     is_number_counts4=is_nc4)

    cl_mask = get_cl_footprint(*tracer_comb1, *tracer_comb2)
    sigma2_B = ccl.sigma2_B_from_mask(cosmo, a=a_arr, mask_wl=cl_mask)

    tr1 = ccl_tracers[tracer_comb1[0]]
    tr2 = ccl_tracers[tracer_comb1[1]]
    tr3 = ccl_tracers[tracer_comb2[0]]
    tr4 = ccl_tracers[tracer_comb2[1]]
    cov_ccl = ccl.angular_cl_cov_SSC(cosmo, cltracer1=tr1, cltracer2=tr2,
                                     ell=ell, tkka=tkk_ssc,
                                     sigma2_B=(a_arr, sigma2_B),
                                     cltracer3=tr3, cltracer4=tr4)

    assert np.max(np.fabs(np.diag(cov_ssc/cov_ccl - 1))) < 1e-5
    assert np.max(np.fabs(cov_ssc/cov_ccl - 1)) < 1e-3

    # Check you get zeroed B-modes
    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=True)
    # Check saved
    assert np.max(np.abs((covf['cov'] + 1e-100) / (cov_ssc_zb + 1e-100) - 1)) < 1e-10

    ix1 = s.indices(tracers=tracer_comb1)
    ix2 = s.indices(tracers=tracer_comb2)
    ncell1 = int(ix1.size / ell.size)
    ncell2 = int(ix2.size / ell.size)

    # The covariance will have all correlations, including when EB == BE
    if (ncell1 == 3) and (tracer_comb1[0] == tracer_comb1[1]):
        ncell1 += 1
    if (ncell2 == 3) and (tracer_comb2[0] == tracer_comb2[1]):
        ncell2 += 1

    assert cov_ssc_zb.shape == (ell.size * ncell1, ell.size * ncell2)
    cov_ssc_zb = cov_ssc_zb.reshape((ell.size, ncell1, ell.size, ncell2))
    assert np.all(cov_ssc_zb[:, 0, :, 0] == cov_ssc)
    cov_ssc_zb[:, 0, :, 0] -= cov_ssc
    assert np.all(cov_ssc_zb == np.zeros_like(cov_ssc_zb))


    # Check get_SSC_cov reads file
    covf = np.load(cc.outdir + '/ssc_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                             *tracer_comb2))
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=False)
    assert np.all(covf['cov_nob'] == cov_ssc)

    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=True)

    assert np.all(covf['cov'] == cov_ssc_zb)


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

clean_tmp()
