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
outdir = os.path.join(root, "tjpcov_tmp")

# Try to create the tmp folder that should not exist. If for some reason it
# has not been deleted before, remove it here
if os.path.isdir(outdir):
    os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")
os.makedirs(outdir)

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
    elif dtype == 'cross':
        fname = os.path.join(root,
                             'DESgc_DESwl/w__mask_DESgc__mask_DESwl0.fits')
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
    elif dtype == 'cross':
        fname = os.path.join(root, subfolder,
                             'DESgc_DESwl/cl_DESgc__0_DESwl__0.npz')

    return np.load(fname)


def get_nmt_bin():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])

def remove_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)

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


@pytest.mark.parametrize('kwards', [{}, {'l_toeplitz': 10, 'l_exact': 10,
                                     'dl_band': 10, 'n_iter': 0 }])
def test_get_workspace(kwards):
    kwards_w = kwards.copy()

    # Compute NmtBins
    bins = get_nmt_bin()

    # Compute workspace
    m1 = get_mask('galaxy_clustering')
    m2 = get_mask('galaxy_shear')

    f1 = nmt.NmtField(m1, None, spin=0)
    f2 = nmt.NmtField(m2, None, spin=2)

    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, bins, **kwards)

    # Compute workspace with nmt_tools
    mn1 = 'mask_DESgc0'
    mn2 = 'mask_DESwl0'
    w_code = nmt_tools.get_workspace(f1, f2, mn1, mn2, bins, outdir, **kwards)

    # Check the file is created
    fname = os.path.join(outdir, f'w__{mn1}__{mn2}.fits')
    assert os.path.isfile(fname)

    # Check that you will read the same workspace if input the other way round
    # and check the symmetric file is not created
    w_code2 = nmt_tools.get_workspace(f2, f1, mn2, mn1, bins, outdir, **kwards)
    fname = os.path.join(outdir, f'w__{mn2}__{mn1}.fits')
    assert not os.path.isfile(fname)

    # Check that with recompute the original file is removed and the symmetric
    # remains
    w_code2 = nmt_tools.get_workspace(f2, f1, mn2, mn1, bins, outdir,
                                      recompute=True, **kwards)
    fname = os.path.join(outdir, f'w__{mn1}__{mn2}.fits')
    assert not os.path.isfile(fname)
    fname = os.path.join(outdir, f'w__{mn2}__{mn1}.fits')
    assert os.path.isfile(fname)

    # Load cl to apply the workspace on
    cl = get_cl('cross', fiducial=True)['cl']

    rdev = (w.couple_cell(cl) + 1e-100) / (w_code.couple_cell(cl) + 1e-100) - 1
    assert np.max(np.abs(rdev)) < 1e-10

    rdev = (w.couple_cell(cl) + 1e-100) / (w_code2.couple_cell(cl) + 1e-100) \
        - 1
    assert np.max(np.abs(rdev)) < 1e-10

    fname = os.path.join(outdir, f'w__{mn1}__{mn2}.fits')
    remove_file(fname)
    fname = os.path.join(outdir, f'w__{mn2}__{mn1}.fits')
    remove_file(fname)


@pytest.mark.parametrize('kwards', [{}, {'l_toeplitz': 10, 'l_exact': 10,
                                     'dl_band': 10, 'n_iter': 0 }])
def test_get_covariance_workspace(kwards):
    m1 = m2 = get_mask('galaxy_clustering')
    m3 = m4 = get_mask('galaxy_shear')

    f1 = f2 = nmt.NmtField(m1, None, spin=0)
    f3 = f4 = nmt.NmtField(m2, None, spin=2)

    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwards)

    cl = get_cl('cross', fiducial=False)
    cl_fid = get_cl('cross', fiducial=True)
    w13 = get_workspace('cross')
    cl_cov = nmt_tools.get_cl_for_cov(cl_fid['cl'], cl['nl_cp'], m1, m3, w13)
    cl13 = cl14 = cl23 = cl24 = cl_cov

    w12 = get_workspace('galaxy_clustering')
    w34 = get_workspace('galaxy_shear')
    cov = nmt.gaussian_covariance(cw, 0, 0, 2, 2, cl13, cl14, cl23, cl24,
                                  w12, w34, coupled=False)

    mn1, mn2, mn3, mn4 = '0', '1', '2', '3'

    combinations = [(f1, f2, f3, f4), (f2, f1, f3, f4), (f1, f2, f4, f3),
                    (f2, f1, f4, f3), (f3, f4, f1, f2), (f4, f3, f1, f2),
                    (f3, f4, f2, f1), (f4, f3, f2, f1)]

    combinations_names = [(mn1, mn2, mn3, mn4), (mn2, mn1, mn3, mn4),
                          (mn1, mn2, mn4, mn3), (mn2, mn1, mn4, mn3),
                          (mn3, mn4, mn1, mn2), (mn4, mn3, mn1, mn2),
                          (mn3, mn4, mn2, mn1), (mn4, mn3, mn2, mn1)]

    # Check only the first is written/computed created & that cw is correct

    for fields, masks_names in zip(combinations, combinations_names):
        cw_code = nmt_tools.get_covariance_workspace(*fields, *masks_names,
                                                     outdir, **kwards)
        fname = os.path.join(outdir,
                             'cw__{}__{}__{}__{}.fits'.format(*masks_names))
        if masks_names == (mn1, mn2, mn3, mn4):
            assert os.path.isfile(fname)
        else:
            assert not os.path.isfile(fname)

        cov2 = nmt.gaussian_covariance(cw_code, 0, 0, 2, 2, cl13, cl14, cl23,
                                       cl24, w12, w34, coupled=False)

        assert np.max(np.abs((cov + 1e-100) / (cov2 + 1e-100) - 1)) < 1e-10

    # Check that with recompute it deletes the existing file and creates a new
    # one

    cw_code = nmt_tools.get_covariance_workspace(f3, f4, f2, f1, mn3, mn4,
                                                 mn2, mn1, outdir,
                                                 recompute=True, **kwards)

    fname = os.path.join(outdir, f'cw__{mn1}__{mn2}__{mn3}__{mn3}.fits')
    assert not os.path.isfile(fname)

    fname = os.path.join(outdir, f'cw__{mn3}__{mn4}__{mn2}__{mn1}.fits')
    assert os.path.isfile(fname)

    remove_file(fname)

