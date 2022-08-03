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
from scipy.interpolate import interp1d

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
tjpcov_class = cv.CovarianceCalculator(input_yml)
nside = 32

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


def get_nmt_bin(lmax=3*nside):
    bpw_edges = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                           78, 84, 90, 96])
    if lmax != 3*nside:
        # lmax + 1 because the last ell is not included
        bpw_edges = bpw_edges[bpw_edges < lmax+1]
        bpw_edges[-1] = lmax+1

    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def get_tracers_dict_for_cov_as_in_tjpcov():
    tr = {1: 'DESgc__0', 2: 'DESgc__0', 3: 'DESwl__0', 4: 'DESwl__1'}
    return tr


def get_spins_dict_for_cov_as_in_tjpcov():
    return {1: 0, 2: 0, 3: 2, 4: 2}


def get_mask_names_dict_for_cov_as_in_tjpcov():
    mask_DESgc = tjpcov_class.mask_names['DESgc__0']
    mask_DESwl0 = tjpcov_class.mask_names['DESwl__0']
    mask_DESwl1 = tjpcov_class.mask_names['DESwl__1']
    m = {1: mask_DESgc, 2: mask_DESgc, 3: mask_DESwl0, 4: mask_DESwl1}
    return m


def get_masks_dict_for_cov_as_in_tjpcov():
    mask_DESgc = hp.read_map(tjpcov_class.mask_fn['DESgc__0'])
    mask_DESwl0 = hp.read_map(tjpcov_class.mask_fn['DESwl__0'])
    mask_DESwl1 = hp.read_map(tjpcov_class.mask_fn['DESwl__1'])
    m = {1: mask_DESgc, 2: mask_DESgc, 3: mask_DESwl0, 4: mask_DESwl1}
    return m


def get_fields_dict_for_cov_as_in_tjpcov(**nmt_conf):
    mask_DESgc = hp.read_map(tjpcov_class.mask_fn['DESgc__0'])
    mask_DESwl0 = hp.read_map(tjpcov_class.mask_fn['DESwl__0'])
    mask_DESwl1 = hp.read_map(tjpcov_class.mask_fn['DESwl__1'])

    f1 = f2 = nmt.NmtField(mask_DESgc, None, spin=0, **nmt_conf)
    f3 = nmt.NmtField(mask_DESwl0, None, spin=2, **nmt_conf)
    f4 = nmt.NmtField(mask_DESwl1, None, spin=2, **nmt_conf)

    return {1: f1, 2: f2, 3: f3, 4: f4}


def get_workspaces_dict_for_cov_as_in_tjpcov(**kwards):
    bins = get_nmt_bin()
    f = get_fields_dict_for_cov_as_in_tjpcov()

    w12 = nmt.NmtWorkspace()
    w12.compute_coupling_matrix(f[1], f[2], bins, **kwards)

    w34 = nmt.NmtWorkspace()
    w34.compute_coupling_matrix(f[3], f[4], bins, **kwards)

    w13 = nmt.NmtWorkspace()
    w13.compute_coupling_matrix(f[1], f[3], bins, **kwards)
    w23 = w13

    w14 = nmt.NmtWorkspace()
    w14.compute_coupling_matrix(f[1], f[4], bins, **kwards)
    w24 = w14

    return {13: w13, 23: w23, 14: w14, 24: w24, 12: w12, 34: w34}


def get_cl_dict_for_cov_as_in_tjpcov(**kwards):
    subfolder = 'fiducial'
    fname = os.path.join(root, subfolder,
                         'DESgc_DESgc/cl_DESgc__0_DESgc__0.npz')
    cl12 = np.load(fname)['cl']

    fname = os.path.join(root, subfolder,
                         'DESwl_DESwl/cl_DESwl__0_DESwl__1.npz')
    cl34 = np.load(fname)['cl']

    fname = os.path.join(root, subfolder,
                         'DESgc_DESwl/cl_DESgc__0_DESwl__0.npz')
    cl13 = cl23 = np.load(fname)['cl']

    fname = os.path.join(root, subfolder,
                         'DESgc_DESwl/cl_DESgc__0_DESwl__1.npz')
    cl14 = cl24 = np.load(fname)['cl']

    return {13: cl13, 23: cl23, 14: cl14, 24: cl24, 12: cl12, 34: cl34}


def remove_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)

def test_get_tracer_nmaps():
    s = get_dummy_sacc()

    with pytest.raises(NotImplementedError):
        nmt_tools.get_tracer_nmaps(s, 'ForError')

    assert nmt_tools.get_tracer_nmaps(s, 'PLAcv') == 1
    assert nmt_tools.get_tracer_nmaps(s, 'DESgc__0') == 1
    assert nmt_tools.get_tracer_nmaps(s, 'DESwl__0') == 2


def test_get_tracer_spin():
    s = get_dummy_sacc()

    with pytest.raises(NotImplementedError):
        nmt_tools.get_tracer_nmaps(s, 'ForError')

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

    assert nmt_tools.get_tracer_comb_ncell(s, ('PLAcv', 'PLAcv')) == 1
    assert nmt_tools.get_tracer_comb_ncell(s, ('PLAcv', 'DESgc__0')) == 1
    assert nmt_tools.get_tracer_comb_ncell(s, ('DESgc__0', 'DESgc__0')) == 1
    assert nmt_tools.get_tracer_comb_ncell(s, ('PLAcv', 'DESwl__0')) == 2
    assert nmt_tools.get_tracer_comb_ncell(s, ('DESgc__0', 'DESwl__0')) == 2
    assert nmt_tools.get_tracer_comb_ncell(s, ('DESwl__0', 'DESwl__0')) == 4
    assert nmt_tools.get_tracer_comb_ncell(s, ('DESwl__0', 'DESwl__0'),
                                               independent=True) == 3


def test_get_datatypes_from_ncell():
    with pytest.raises(ValueError):
        nmt_tools.get_datatypes_from_ncell(0)

    with pytest.raises(ValueError):
        nmt_tools.get_datatypes_from_ncell(3)

    assert nmt_tools.get_datatypes_from_ncell(1) == ['cl_00']
    assert nmt_tools.get_datatypes_from_ncell(2) == ['cl_0e', 'cl_0b']
    assert nmt_tools.get_datatypes_from_ncell(4) == ['cl_ee', 'cl_eb', 'cl_be',
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
    cl_cp_code = nmt_tools.get_cl_for_cov(cl_fid['cl'], cl['nl_cp'], m, m, w,
                                          nl_is_cp=True)
    assert np.abs(cl_cp / cl_cp_code - 1).max() < 1e-10

    # Inputting uncoupled noise.
    nlfill = np.ones_like(cl_fid['ell']) * cl['nl'][0, 0]
    cl_cp_code = nmt_tools.get_cl_for_cov(cl_fid['cl'], nlfill, m, m, w,
                                          nl_is_cp=False)
    assert np.abs(cl_cp[0] / cl_cp_code[0] - 1).max() < 1e-2

    # Check that if I input the coupled but nl_is_cp is False, we don't recover
    # cl_cp
    cl_cp_code = nmt_tools.get_cl_for_cov(cl_fid['cl'], cl['nl_cp'], m, m, w,
                                          nl_is_cp=False)
    assert np.abs(cl_cp / cl_cp_code - 1).max() > 0.4

    # Check that if I input the uncoupled but nl_is_cp is True, assert fails
    cl_cp_code = nmt_tools.get_cl_for_cov(cl_fid['cl'], nlfill, m, m, w,
                                          nl_is_cp=True)
    assert np.abs(cl_cp / cl_cp_code - 1).max() > 0.5

    # Create a non overlapping mask
    m2 = np.ones_like(m)
    m2[m != 0] = 0
    assert not np.all(nmt_tools.get_cl_for_cov(cl, cl['nl_cp'], m, m2, w,
                                               nl_is_cp=True))

    with pytest.raises(ValueError):
        nmt_tools.get_cl_for_cov(cl_fid_Sh, cl['nl_cp'], m, m, w, nl_is_cp=True)

    with pytest.raises(ValueError):
        # Uncoupled binned noise
        nmt_tools.get_cl_for_cov(cl_fid, cl['nl'], m, m, w, nl_is_cp=True)

    with pytest.raises(ValueError):
        nmt_tools.get_cl_for_cov(cl_fid, cl['nl_cp'], m, m, wSh, nl_is_cp=True)


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
    s1 = 0
    s2 = 2
    mn1 = 'mask_DESgc0'
    mn2 = 'mask_DESwl0'
    w_code = nmt_tools.get_workspace(f1, f2, mn1, mn2, bins, outdir, **kwards)

    # Check the file is created
    fname = os.path.join(outdir, f'w{s1}{s2}__{mn1}__{mn2}.fits')
    assert os.path.isfile(fname)

    # Check that you will read the same workspace if input the other way round
    # and check the symmetric file is not created
    w_code2 = nmt_tools.get_workspace(f2, f1, mn2, mn1, bins, outdir, **kwards)
    fname = os.path.join(outdir, f'w{s2}{s1}__{mn2}__{mn1}.fits')
    assert not os.path.isfile(fname)

    # Check that with recompute the original file is removed and the symmetric
    # remains
    w_code2 = nmt_tools.get_workspace(f2, f1, mn2, mn1, bins, outdir,
                                      recompute=True, **kwards)
    fname = os.path.join(outdir, f'w{s1}{s2}__{mn1}__{mn2}.fits')
    assert not os.path.isfile(fname)
    fname = os.path.join(outdir, f'w{s2}{s1}__{mn2}__{mn1}.fits')
    assert os.path.isfile(fname)

    # Load cl to apply the workspace on
    cl = get_cl('cross', fiducial=True)['cl']

    rdev = (w.couple_cell(cl) + 1e-100) / (w_code.couple_cell(cl) + 1e-100) - 1
    assert np.max(np.abs(rdev)) < 1e-10

    rdev = (w.couple_cell(cl) + 1e-100) / (w_code2.couple_cell(cl) + 1e-100) \
        - 1
    assert np.max(np.abs(rdev)) < 1e-10

    fname = os.path.join(outdir, f'w{s1}{s2}__{mn1}__{mn2}.fits')
    remove_file(fname)
    fname = os.path.join(outdir, f'w{s2}{s1}__{mn2}__{mn1}.fits')
    remove_file(fname)
    # Check that outdir can be None
    w_code = nmt_tools.get_workspace(f1, f2, mn1, mn2, bins, None, **kwards)
    fname = os.path.join(outdir, f'w{s1}{s2}__{mn1}__{mn2}.fits')
    assert not os.path.isfile(fname)


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
    cl_cov = nmt_tools.get_cl_for_cov(cl_fid['cl'], cl['nl_cp'], m1, m3, w13,
                                      nl_is_cp=True)
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
        spins = [fi.fl.spin for fi in fields]
        cw_code = nmt_tools.get_covariance_workspace(*fields, *masks_names,
                                                     outdir, **kwards)
        fname = os.path.join(outdir,
                             'cw{}{}{}{}__{}__{}__{}__{}.fits'.format(*spins,
                                                                      *masks_names))
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

    fname = os.path.join(outdir, f'cw0022__{mn1}__{mn2}__{mn3}__{mn3}.fits')
    assert not os.path.isfile(fname)

    fname = os.path.join(outdir, f'cw2200__{mn3}__{mn4}__{mn2}__{mn1}.fits')
    assert os.path.isfile(fname)

    remove_file(fname)
    # Check that outdir can be None
    cw_code = nmt_tools.get_covariance_workspace(f3, f4, f2, f1, mn3, mn4,
                                                 mn2, mn1, None,
                                                 recompute=True, **kwards)
    assert not os.path.isfile(fname)


def test_get_mask_names_dict():
    tr = get_tracers_dict_for_cov_as_in_tjpcov()
    mn = nmt_tools.get_mask_names_dict(tjpcov_class.mask_names, tr)
    assert len(mn) == 4
    for i in range(4):
        assert mn[i + 1] == tjpcov_class.mask_names[tr[i + 1]]


@pytest.mark.parametrize('kwards', [{'mask_fn': tjpcov_class.mask_fn,
                                     'nside': None},
                                    {'mask_fn':
                                     './tests/benchmarks/32_DES_tjpcov_bm/catalogs/DES_mask_ns32.hdf5',
                                     'nside': 32}])

def test_get_masks_dict(kwards):
    tr = get_tracers_dict_for_cov_as_in_tjpcov()
    mn = get_mask_names_dict_for_cov_as_in_tjpcov()
    m = get_masks_dict_for_cov_as_in_tjpcov()

    mask_fn = kwards['mask_fn']
    nside = kwards['nside']

    m2 = nmt_tools.get_masks_dict(mask_fn, mn, tr, cache={}, nside=nside)

    # Check the masks have been read correctly
    for i in range(4):
        assert np.all(m[i + 1] == m2[i + 1])
        assert m[i + 1] is not m2[i + 1]

    # Check that DESgc__0 mask is not read twice. tr[1] == tr[2]
    assert m2[1] is m2[2]

    # Check that cache works and avoid reading the files
    cache = {f'm{i + 1}': m[i + 1] for i in range(4)}
    m2 = nmt_tools.get_masks_dict(mask_fn, mn, tr, cache=cache, nside=nside)

    for i in range(4):
        # Check they are the same object, i.e. have not been read
        assert m[i + 1] is m2[i + 1]


@pytest.mark.parametrize('nmt_conf', [{}, {'n_iter': 0}])
def test_get_fields_dict(nmt_conf):
    m = get_masks_dict_for_cov_as_in_tjpcov()
    s = get_spins_dict_for_cov_as_in_tjpcov()
    mn = get_mask_names_dict_for_cov_as_in_tjpcov()
    tr = get_tracers_dict_for_cov_as_in_tjpcov()

    nmt_conf = {}
    f = get_fields_dict_for_cov_as_in_tjpcov(**nmt_conf)
    f2 = nmt_tools.get_fields_dict(m, s, mn, tr, nmt_conf, cache={})

    # Check that the DESgc fields are exactly the same (not generated twice)
    assert f2[1] is f2[2]

    # Check that if the mask of DESwl has the same name as that of DESgc, they
    # do not get messed up
    mn2 = mn.copy()
    mn2[3] = tjpcov_class.mask_names['DESgc__0']
    f2 = nmt_tools.get_fields_dict(m, s, mn2, tr, nmt_conf, cache={})
    assert f2[1] is not f2[3]

    # Check fields are the same by computing the workspace and coupling a
    # fiducial Cell
    cl = {}
    cl[1] = cl[2] = get_cl('galaxy_clustering', fiducial=True)['cl']
    cl[3] = cl[4] = get_cl('galaxy_shear', fiducial=True)['cl']

    bins = get_nmt_bin()
    for i in range(1, 5):
        w = nmt_tools.get_workspace(f[i], f[i], str(i), str(i), bins, outdir)
        w2 = nmt_tools.get_workspace(f[i], f[i], str(i), str(i), bins, outdir)

        cl1 = w.couple_cell(cl[i]) + 1e-100
        cl2 = w2.couple_cell(cl[i]) + 1e-100
        assert np.max(np.abs(cl1 / cl2 - 1)) < 1e-10

    # Check that cache works
    cache = {'f1': f[1], 'f2': f[2], 'f3': f[3], 'f4': f[4]}
    f2 = nmt_tools.get_fields_dict(m, s, mn, tr, nmt_conf, cache=cache)
    for i in range(1, 5):
        assert f[i] is f2[i]

    os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")

@pytest.mark.parametrize('kwards', [{}, {'l_toeplitz': 10, 'l_exact': 10,
                                     'dl_band': 10, 'n_iter': 0 }])
def test_get_workspace_dict(kwards):
    bins = get_nmt_bin()
    m = get_masks_dict_for_cov_as_in_tjpcov()
    f = get_fields_dict_for_cov_as_in_tjpcov()
    mn = get_mask_names_dict_for_cov_as_in_tjpcov()

    w = get_workspaces_dict_for_cov_as_in_tjpcov(**kwards)
    w2 = nmt_tools.get_workspaces_dict(f, m, mn, bins, outdir, kwards, cache={})

    # Check workspaces by comparing the coupled cells
    cl = get_cl_dict_for_cov_as_in_tjpcov()

    for i in [13, 23, 14, 24, 12, 34]:
        cl1 = w[i].couple_cell(cl[i]) + 1e-100
        cl2 = w2[i].couple_cell(cl[i]) + 1e-100
        assert np.max(np.abs(cl1 / cl2 - 1)) < 1e-10

    # Check that things are not read/computed twice
    assert w2[13] is w2[23]
    assert w2[14] is w2[24]

    # Check that cache works
    cache = {'w13': w[13], 'w23': w[23], 'w14': w[14], 'w24': w[24],
             'w12': w[12], 'w34': w[34]}
    w2 = nmt_tools.get_workspaces_dict(f, m, mn, bins, outdir, kwards,
                                       cache=cache)
    for i in [13, 23, 14, 24, 12, 34]:
        assert w[i] is w2[i]

    # Check that for non overlapping fields, the workspace is not computed (and
    # is None)
    # Create a non overlapping mask:
    m[1] = np.zeros_like(m[2])
    m[1][:1000] = 1
    m[3] = np.zeros_like(m[4])
    m[3][1000:2000] = 1

    w2 = nmt_tools.get_workspaces_dict(f, m, mn, bins, outdir, kwards, cache={})
    # w12, w34 should not be None as they are needed in nmt.gaussian_covariance
    assert w2[12] is not None
    assert w2[34] is not None
    # w13, w14, w23 should be None and w24 should be None because mn1 = mn2
    assert w2[13] is None
    assert w2[14] is None
    assert w2[13] is None
    assert w2[24] is None

    # Check that 'workspaces' cache also works. In this case, one will pass
    # paths, not instances
    gc0gc0 = os.path.join(root, 'DESgc_DESgc/w__mask_DESgc__mask_DESgc.fits')
    gc0wl0 = os.path.join(root, 'DESgc_DESwl/w__mask_DESgc__mask_DESwl0.fits')
    gc0wl1 = os.path.join(root, 'DESgc_DESwl/w__mask_DESgc__mask_DESwl1.fits')
    wl0wl0 = os.path.join(root, 'DESwl_DESwl/w__mask_DESwl0__mask_DESwl0.fits')
    wl0wl1 = os.path.join(root, 'DESwl_DESwl/w__mask_DESwl0__mask_DESwl1.fits')
    wl1wl1 = os.path.join(root, 'DESwl_DESwl/w__mask_DESwl1__mask_DESwl1.fits')
    cache = {'workspaces':
             {'00': {('mask_DESgc0', 'mask_DESgc0'): gc0gc0},
              '02': {('mask_DESgc0', 'mask_DESwl0'): gc0wl0,
                    ('mask_DESgc0', 'mask_DESwl1'): gc0wl1},
              '22': {('mask_DESwl0', 'mask_DESwl0'): wl0wl0,
                    ('mask_DESwl0', 'mask_DESwl1'): wl0wl1,
                    ('mask_DESwl1', 'mask_DESwl1'): wl1wl1}}}
    # bins to None to force it fail if it does not uses the cache
    w2 = nmt_tools.get_workspaces_dict(f, m, mn, None, outdir, kwards,
                                       cache=cache)

    # Check that it will compute the workspaces if one is missing
    del cache['workspaces']['02'][('mask_DESgc0', 'mask_DESwl1')]
    w2 = nmt_tools.get_workspaces_dict(f, m, mn, bins, outdir, kwards,
                                       cache=cache)
    # Check that '20' is also understood
    del cache['workspaces']['02']
    cache['workspaces']['20'] = {('mask_DESgc0', 'mask_DESwl0'): gc0wl0,
                                 ('mask_DESgc0', 'mask_DESwl1'): gc0wl1}
    w2 = nmt_tools.get_workspaces_dict(f, m, mn, None, outdir, kwards,
                                       cache=cache)


    os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")

def test_get_sacc_with_concise_dtypes():
    s = sacc_file.copy()
    for dp in s.data:
        dt = dp.data_type

        if dt == 'cl_00':
            dp.data_type = sacc.standard_types.galaxy_density_cl
        elif dt == 'cl_0e':
            dp.data_type = sacc.standard_types.galaxy_shearDensity_cl_e
        elif dt == 'cl_0b':
            dp.data_type = sacc.standard_types.galaxy_shearDensity_cl_b
        elif dt == 'cl_ee':
            dp.data_type = sacc.standard_types.galaxy_shear_cl_ee
        elif dt == 'cl_eb':
            dp.data_type = sacc.standard_types.galaxy_shear_cl_eb
        elif dt == 'cl_be':
            dp.data_type = sacc.standard_types.galaxy_shear_cl_be
        elif dt == 'cl_bb':
            dp.data_type = sacc.standard_types.galaxy_shear_cl_bb
        else:
            raise ValueError('Something went wrong. Data type not recognized')

    s2 = nmt_tools.get_sacc_with_concise_dtypes(s)
    dtypes = sacc_file.get_data_types()
    dtypes2 = s2.get_data_types()
    assert dtypes == dtypes2

    for dp, dp2 in zip(sacc_file.data, s2.data):
        assert dp.data_type == dp2.data_type
        assert dp.value == dp2.value
        assert dp.tracers == dp2.tracers
        for k in dp.tags:
            if k == 'window':
                # Don't check window as it points to a different memory address
                continue
            assert dp.tags[k] == dp2.tags[k]

def test_get_nbpw():
    s = get_sacc()

    nbpw = nmt_tools.get_nbpw(s)
    bins = get_nmt_bin()

    assert nbpw == bins.get_n_bands()


def test_get_nell():
    s = get_sacc()
    nell = 3 * nside
    bins = get_nmt_bin()
    w = get_workspace('galaxy_clustering')
    cache = {'workspaces': {'00': {('mask_DESgc0', 'mask_DESgc0'): w}}}

    assert nell == nmt_tools.get_nell(s)

    # Now with a sacc file without bandpower windows
    s = get_dummy_sacc()
    clf = get_cl('cross')
    s.add_ell_cl('cl_0e', 'DESgc__0', 'DESwl__0', clf['ell'], clf['cl'][0])

    assert nell == nmt_tools.get_nell(s, bins=bins)
    assert nell == nmt_tools.get_nell(s, nside=nside)
    assert nell == nmt_tools.get_nell(s, cache=cache)

    # Force ValueError (as when window is wrong)
    class s():
        def __init__(self):
            pass
        def get_data_types(self):
            raise ValueError

    with pytest.raises(ValueError):
        assert nell == nmt_tools.get_nell(s())
    # But it works if you pass the nside
    assert nell == nmt_tools.get_nell(s(), nside=nside)

    # Test lmax != 3*nside
    lmax = 50
    bins = get_nmt_bin(50)
    nell = 51
    assert nell == nmt_tools.get_nell(s, bins=bins)
    # Test that if bins nor workspace is given, it tries to use the sacc file
    # and when fails (if "binnint/ell_max" is not present in the metadata),
    # it defaults to nell = 3*nside
    assert 3 *nside == nmt_tools.get_nell(s(), nside=nside)

    # Check metadata
    s = sacc.Sacc()
    s.metadata["binning/ell_max"] = lmax
    assert nell == nmt_tools.get_nell(s)


def test_get_list_of_tracers_for_wsp():
    s = get_sacc()
    mask_names = tjpcov_class.mask_names
    trs_wsp = nmt_tools.get_list_of_tracers_for_wsp(s, mask_names)

    trs_wsp2 = [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                (('DESgc__0', 'DESwl__0'), ('DESgc__0', 'DESwl__0')),
                (('DESgc__0', 'DESwl__1'), ('DESgc__0', 'DESwl__1')),
                (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                (('DESwl__0', 'DESwl__1'), ('DESwl__0', 'DESwl__1')),
                (('DESwl__1', 'DESwl__1'), ('DESwl__1', 'DESwl__1'))]

    assert sorted(trs_wsp) == sorted(trs_wsp2)


def test_get_list_of_tracers_for_cov_wsp():
    s = get_sacc()
    mask_names = tjpcov_class.mask_names
    trs_cwsp = nmt_tools.get_list_of_tracers_for_cov_wsp(s, mask_names)

    trs_cwsp2 = [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                 (('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESwl__0')),
                 (('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESwl__1')),
                 (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                 (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__1')),
                 (('DESgc__0', 'DESgc__0'), ('DESwl__1', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__0'), ('DESgc__0', 'DESwl__0')),
                 (('DESgc__0', 'DESwl__0'), ('DESgc__0', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                 (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__1'), ('DESgc__0', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__1'), ('DESwl__0', 'DESwl__0')),
                 (('DESgc__0', 'DESwl__1'), ('DESwl__0', 'DESwl__1')),
                 (('DESgc__0', 'DESwl__1'), ('DESwl__1', 'DESwl__1')),
                 (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                 (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__1')),
                 (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                 (('DESwl__0', 'DESwl__1'), ('DESwl__0', 'DESwl__1')),
                 (('DESwl__0', 'DESwl__1'), ('DESwl__1', 'DESwl__1')),
                 (('DESwl__1', 'DESwl__1'), ('DESwl__1', 'DESwl__1')),
                 ]

    assert sorted(trs_cwsp) == sorted(trs_cwsp2)

    trs_cwsp = nmt_tools.get_list_of_tracers_for_cov_wsp(s, mask_names,
                                                         remove_trs_wsp=True)

    for trs in nmt_tools.get_list_of_tracers_for_wsp(s, mask_names):
        trs_cwsp2.remove(trs)

    assert trs_cwsp == trs_cwsp2


def test_get_list_of_tracers_for_cov():
    s = get_sacc()
    mask_names = tjpcov_class.mask_names
    trs_cov = nmt_tools.get_list_of_tracers_for_cov(s)

    # Test all tracers
    trs_cov2 = []
    tracers = s.get_tracer_combinations()
    for i, trs1 in enumerate(tracers):
        for trs2 in tracers[i:]:
            trs_cov2.append((trs1, trs2))

    assert trs_cov == trs_cov2

    # Test all tracers except those used for workspaces and cov workspaces
    trs_cov = nmt_tools.get_list_of_tracers_for_cov(s,
                                                    remove_trs_wsp_cwsp=True,
                                                    mask_names=mask_names)

    for trs in nmt_tools.get_list_of_tracers_for_cov_wsp(s, mask_names):
        trs_cov2.remove(trs)

    assert sorted(trs_cov) == sorted(trs_cov2)

def test_get_ell_eff():
    s = get_sacc()
    bins = get_nmt_bin()
    ells = bins.get_effective_ells()

    assert np.all(nmt_tools.get_ell_eff(s) == ells)



if os.path.isdir(outdir):
    os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")
