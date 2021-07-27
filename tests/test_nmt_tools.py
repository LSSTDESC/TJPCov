#!/usr/bin/python
import healpy as hp
import numpy as np
import os
import pymaster as nmt
import pytest
import sacc
import yaml
import tjpcov.main as cv
from tjpcov import nmt_tools

root = "./tests/benchmarks/32_DES_tjpcov_bm/"
sacc_path = os.path.join(root, 'cls_cov.fits')
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
xcell_yml = os.path.join(root, "desy1_tjpcov_bm.yml")

sacc_file = sacc.Sacc.load_fits(sacc_path)

def get_sacc():
    return sacc_file


def get_dummy_sacc():
    s = sacc.Sacc()
    s.add_tracer('map', 'PLAcv', quantity='cmb_convergence', spin=0,
                 ell=None, beam=None)
    s.add_tracer('NZ', 'DESgc__0', quantity='galaxy_density', spin=0,
                 nz=None, z=None)
    s.add_tracer('NZ', 'DESwl__0', quantity='galaxy_shear', spin=2,
                 nz=None, z=None)
    s.add_tracer('misc', 'ForError', quantity='generic')

    return s


def get_tjpcov():
    return cv.CovarianceCalculator(input_yml)


def get_mask(dtype):
    if dtype == 'galaxy_clustering':
        fname = os.path.join(root, 'catalogs', 'mask_DESgc__0.fits.gz')
    elif dtype == 'galaxy_shear':
        fname = os.path.join(root, 'catalogs',
                             'DESwlMETACAL_mask_zbin0_ns32.fits.gz')

    return hp.read_map(fname)


def get_workspace(dtype):
    w = nmt.NmtWorkspace()
    if dtype == 'galaxy_clustering':
        fname = os.path.join(root,
                             'DESgc_DESgc/w__mask_DESgc__mask_DESgc.fits')
    elif dtype == 'galaxy_shear':
        fname = os.path.join(root,
                             'DESwl_DESwl/w__mask_DESwl0__mask_DESwl0.fits')
    w.read_from(fname)

    return w


def get_cl(dtype, fiducial=False):
    subfolder = ''
    if fiducial:
        subfolder = 'fiducial'

    if dtype == 'galaxy_clustering':
        fname = os.path.join(root, subfolder,
                             'DESgc_DESgc/cl_DESgc__0_DESgc__0.npz')
    elif dtype == 'galaxy_shear':
        fname = os.path.join(root, subfolder,
                             'DESwl_DESwl/cl_DESwl__0_DESwl__0.npz')

    return np.load(fname)


def test_get_tracer_dof():
    s = get_dummy_sacc()

    with pytest.raises(ValueError):
        nmt_tools.get_tracer_dof(s, 'ForError')

    assert nmt_tools.get_tracer_dof(s, 'PLAcv') == 1
    assert nmt_tools.get_tracer_dof(s, 'DESgc__0') == 1
    assert nmt_tools.get_tracer_dof(s, 'DESwl__0') == 2


def test_get_tracer_spin():
    s = get_dummy_sacc()
    assert nmt_tools.get_tracer_spin(s, 'PLAcv') == 0
    assert nmt_tools.get_tracer_spin(s, 'DESgc__0') == 0
    assert nmt_tools.get_tracer_spin(s, 'DESwl__0') == 2


def test_get_tracer_comb_spin():
    s = get_dummy_sacc()
    tracers = ['PLACv', 'DESgc__0', 'DESwl__0']

    for tr1 in tracers:
        s1 = nmt_tools.get_tracer_spin(s, tr1)
        for tr2 in tracers:
            s2 = nmt_tools.get_tracer_spin(s, tr2)
            assert (s1, s2) == nmt_tools.get_tracer_comb_spin(s, (tr1, tr2))


def test_get_tracer_comb_spin():
    s = get_dummy_sacc()
    tracers = ['PLACv', 'DESgc__0', 'DESwl__0']

    assert nmt_tools.get_tracer_comb_dof(s, ('PLAcv', 'PLAcv')) == 1
    assert nmt_tools.get_tracer_comb_dof(s, ('PLAcv', 'DESgc__0')) == 1
    assert nmt_tools.get_tracer_comb_dof(s, ('DESgc__0', 'DESgc__0')) == 1
    assert nmt_tools.get_tracer_comb_dof(s, ('PLAcv', 'DESwl__0')) == 2
    assert nmt_tools.get_tracer_comb_dof(s, ('DESgc__0', 'DESwl__0')) == 2
    assert nmt_tools.get_tracer_comb_dof(s, ('DESwl__0', 'DESwl__0')) == 4


def test_get_datatypes_from_dof():
    with pytest.raises(ValueError):
        nmt_tools.get_datatypes_from_dof(0)

    with pytest.raises(ValueError):
        nmt_tools.get_datatypes_from_dof(3)

    assert nmt_tools.get_datatypes_from_dof(1) == ['cl_00']
    assert nmt_tools.get_datatypes_from_dof(2) == ['cl_0e', 'cl_0b']
    assert nmt_tools.get_datatypes_from_dof(4) == ['cl_ee', 'cl_eb', 'cl_be',
                                                   'cl_bb']


def test_get_cl_for_cov():
    # We just need to test for one case as the function will complain if the
    # Cell inputted has the wrong shape
    m = get_mask('galaxy_clustering')
    w = get_workspace('galaxy_clustering')
    wSh = get_workspace('galaxy_shear')

    cl = get_cl('galaxy_clustering', fiducial=False)
    cl_fid = get_cl('galaxy_clustering', fiducial=True)
    cl_fid_Sh = get_cl('galaxy_shear', fiducial=True)

    cl_cp = (w.couple_cell(cl_fid['cl']) + cl['nl_cp']) / np.mean(m**2)
    cl_cp_code = nmt_tools.get_cl_for_cov(cl_fid['cl'], cl['nl_cp'], m, m, w)
    assert np.abs(cl_cp / cl_cp_code - 1).max() < 1e-10

    # Create a non overlapping mask
    m2 = np.ones_like(m)
    m2[m != 0] = 0
    assert not np.all(nmt_tools.get_cl_for_cov(cl, cl['nl_cp'], m, m2, w))

    with pytest.raises(ValueError):
        nmt_tools.get_cl_for_cov(cl_fid_Sh, cl['nl_cp'], m, m, w)

    with pytest.raises(ValueError):
        nmt_tools.get_cl_for_cov(cl_fid, cl['nl'], m, m, w)

    with pytest.raises(ValueError):
        nmt_tools.get_cl_for_cov(cl_fid, cl['nl_cp'], m, m, wSh)


