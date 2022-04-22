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

data_dir = os.path.join("tests/benchmarks/SSC/")

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.97,
                      sigma8=0.8, m_nu=0.0)

def get_dummy_sacc():
    ell = np.loadtxt(os.path.join(data_dir, "ssc_WL_ell.txt"))
    z, nofz = np.loadtxt(os.path.join(data_dir, "ssc_WL_nofz.txt"),
                         unpack=True)

    s = sacc.Sacc()
    s.add_tracer('NZ', 'source_0', quantity='galaxy_shear', spin=2,
                 z=z, nz=nofz)
    s.add_ell_cl('cl_ee', 'source_0', 'source_0', ell, np.ones_like(ell)*1e-8)

    return s

def get_CovarianceCalculator():
    config = {}
    config['tjpcov'] = {'do_xi': False,
                        'cl_file': get_dummy_sacc(),
                        'cosmo': cosmo, 'fsky': 0.05,
                        'binning_info': 'ignore',
              }
    return cv.CovarianceCalculator(config)

def test_get_SSC_cov():
    # Test follows CCL's SSC benchmark:
    # Compare against Benjamin Joachimi's code. An overview of the methodology
    # is given in appendix E.2 of 2007.01844.
    cc  = get_CovarianceCalculator()
    s = cc.cl_data
    ell, _ = s.get_ell_cl('cl_ee', 'source_0', 'source_0')

    tracer_comb1 = tracer_comb2 = ('source_0', 'source_0')
    ccl_tracers, _ = cc.get_tracer_info(s)
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                             tracer_comb2=tracer_comb2,
                             ccl_tracers=ccl_tracers, fsky=None,
                             integration_method='qag_quad',
                             include_b_modes=False)

    var_ssc_ccl = np.diag(cov_ssc)
    off_diag_1_ccl = np.diag(cov_ssc, k=1)

    cov_ssc_bj = np.loadtxt(os.path.join(data_dir, "ssc_WL_cov_matrix.txt"))

    # At large scales, CCL uses a different convention for the Limber
    # approximation. This factor accounts for this difference
    ccl_limber_shear_fac = np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+1/2)**2
    cov_ssc_bj_corrected = cov_ssc_bj * np.outer(ccl_limber_shear_fac**2,
                                                 ccl_limber_shear_fac**2)
    var_bj = np.diag(cov_ssc_bj_corrected)
    off_diag_1_bj = np.diag(cov_ssc_bj_corrected, k=1)

    assert np.all(np.fabs(var_ssc_ccl/var_bj - 1) < 3e-2)
    assert np.all(np.fabs(off_diag_1_ccl/off_diag_1_bj - 1) < 3e-2)
    assert np.all(np.fabs(cov_ssc/cov_ssc_bj_corrected - 1) < 3e-2)

    # Check it reads the fsky:
    cov_ssc2 = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                              tracer_comb2=tracer_comb2,
                              ccl_tracers=ccl_tracers, fsky=0.3,
                              integration_method='qag_quad',
                              include_b_modes=False)

    var_ssc_ccl2 = np.diag(cov_ssc2)
    assert not np.all(np.fabs(var_ssc_ccl2/var_bj - 1) < 3e-2)

    # Check you get zeroed B-modes
    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers, fsky=None,
                                integration_method='qag_quad',
                                include_b_modes=True)

    assert cov_ssc_zb.shape == (4 * ell.size, 4 * ell.size)
    cov_ssc_zb = cov_ssc_zb.reshape((ell.size, 4, ell.size, 4))
    assert np.all(cov_ssc_zb[:, 0, :, 0] == cov_ssc)
    cov_ssc_zb[:, 0, :, 0] -= cov_ssc
    assert np.all(cov_ssc_zb == np.zeros_like(cov_ssc_zb))

