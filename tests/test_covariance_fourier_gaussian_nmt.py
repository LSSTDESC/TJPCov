#!/usr/bin/python3

import os

import healpy as hp
import numpy as np
import pymaster as nmt
import pytest
import sacc
import shutil
import yaml

from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt
from tjpcov.covariance_io import CovarianceIO


ROOT = "tests/benchmarks/32_DES_tjpcov_bm/"
OUTDIR = "tests/tmp/"
INPUT_YML = "tests/data/conf_covariance_gaussian_fourier_nmt.yaml"
NSIDE = 32


@pytest.fixture
def mock_sacc():
    return sacc.Sacc.load_fits(ROOT + "cls_cov.fits")


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
def cov_fg_nmt():
    return FourierGaussianNmt(INPUT_YML)


# Useful functions
def get_config(fname):
    return CovarianceIO._parse(fname)


def assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, threshold):
    cl1 = get_data_cl(*tracer_comb1, remove_be=True)
    cl2 = get_data_cl(*tracer_comb2, remove_be=True)

    clf1 = get_fiducial_cl(s, *tracer_comb1, remove_be=True)
    clf2 = get_fiducial_cl(s, *tracer_comb2, remove_be=True)

    ndim, nbpw = cl1.shape
    # This only runs if tracer_comb1 = tracer_comb2 (when the block covariance
    # is invertible)
    if (tracer_comb1[0] == tracer_comb1[1]) and (ndim == 3):
        cov = cov.reshape((nbpw, 4, nbpw, 4))
        cov = np.delete(np.delete(cov, 2, 1), 2, 3).reshape(3 * nbpw, -1)
        cov_bm = cov_bm.reshape((nbpw, 4, nbpw, 4))
        cov_bm = np.delete(np.delete(cov_bm, 2, 1), 2, 3).reshape(3 * nbpw, -1)

    delta1 = (clf1 - cl1).flatten()
    delta2 = (clf2 - cl2).flatten()
    chi2 = delta1.dot(np.linalg.inv(cov)).dot(delta2)
    chi2_bm = delta1.dot(np.linalg.inv(cov_bm)).dot(delta2)

    assert np.abs(chi2 / chi2_bm - 1) < threshold


def get_nmt_bin(lmax=3 * NSIDE):
    bpw_edges = np.array(
        [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    )
    if lmax != 3 * NSIDE:
        # lmax + 1 because the last ell is not included
        bpw_edges = bpw_edges[bpw_edges < lmax + 1]
        bpw_edges[-1] = lmax + 1

    return nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def get_pair_folder_name(tracer_comb):
    bn = []
    for tr in tracer_comb:
        bn.append(tr.split("__")[0])
    return "_".join(bn)


def get_cl(dtype, fiducial=False):
    subfolder = ""
    if fiducial:
        subfolder = "fiducial"

    if dtype == "galaxy_clustering":
        fname = os.path.join(
            ROOT, subfolder, "DESgc_DESgc/cl_DESgc__0_DESgc__0.npz"
        )
    elif dtype == "galaxy_shear":
        fname = os.path.join(
            ROOT, subfolder, "DESwl_DESwl/cl_DESwl__0_DESwl__0.npz"
        )
    elif dtype == "cross":
        fname = os.path.join(
            ROOT, subfolder, "DESgc_DESwl/cl_DESgc__0_DESwl__0.npz"
        )

    return np.load(fname)


def get_data_cl(tr1, tr2, remove_be=False):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(ROOT, bn, f"cl_{tr1}_{tr2}.npz")
    cl = np.load(fname)["cl"]

    # Remove redundant terms
    if remove_be and (tr1 == tr2) and (cl.shape[0] == 4):
        cl = np.delete(cl, 2, 0)
    return cl


def get_dummy_sacc():
    s = sacc.Sacc()
    s.add_tracer(
        "Map", "PLAcv", quantity="cmb_convergence", spin=0, ell=None, beam=None
    )
    s.add_tracer(
        "NZ", "DESgc__0", quantity="galaxy_density", spin=0, nz=None, z=None
    )
    s.add_tracer(
        "NZ", "DESwl__0", quantity="galaxy_shear", spin=2, nz=None, z=None
    )
    s.add_tracer("Misc", "ForError", quantity="generic")

    return s


def get_fiducial_cl(s, tr1, tr2, binned=True, remove_be=False):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(ROOT, "fiducial", bn, f"cl_{tr1}_{tr2}.npz")
    cl = np.load(fname)["cl"]
    if binned:
        s = s.copy()
        s.remove_selection(data_type="cl_0b")
        s.remove_selection(data_type="cl_eb")
        s.remove_selection(data_type="cl_be")
        s.remove_selection(data_type="cl_bb")
        ix = s.indices(tracers=(tr1, tr2))
        bpw = s.get_bandpower_windows(ix)

        cl0_bin = bpw.weight.T.dot(cl[0])

        cl_bin = np.zeros((cl.shape[0], cl0_bin.size))
        cl_bin[0] = cl0_bin
        cl = cl_bin
    else:
        cl

    # Remove redundant terms
    if remove_be and (tr1 == tr2) and (cl.shape[0] == 4):
        cl = np.delete(cl, 2, 0)
    return cl


def get_mask_from_dtype(dtype):
    if dtype == "galaxy_clustering":
        fname = os.path.join(ROOT, "catalogs", "mask_DESgc__0.fits.gz")
    elif dtype == "galaxy_shear":
        fname = os.path.join(
            ROOT, "catalogs", "DESwlMETACAL_mask_zbin0_ns32.fits.gz"
        )

    return hp.read_map(fname)


def get_tracer_noise(tr, cp=True):
    bn = get_pair_folder_name((tr, tr))
    fname = os.path.join(ROOT, bn, f"cl_{tr}_{tr}.npz")
    clfile = np.load(fname)
    if cp:
        return clfile["nl_cp"][0][-1]
    else:
        return clfile["nl"][0][0]


def get_benchmark_cov(tracer_comb1, tracer_comb2):
    (tr1, tr2), (tr3, tr4) = tracer_comb1, tracer_comb2
    fname = os.path.join(ROOT, "cov", f"cov_{tr1}_{tr2}_{tr3}_{tr4}.npz")
    return np.load(fname)["cov"]


def get_workspace_from_trs(tr1, tr2):
    config = get_xcell_yml()
    w = nmt.NmtWorkspace()
    bn = get_pair_folder_name((tr1, tr2))
    m1 = config["tracers"][tr1]["mask_name"]
    m2 = config["tracers"][tr2]["mask_name"]
    fname = os.path.join(ROOT, bn, f"w__{m1}__{m2}.fits")
    w.read_from(fname)
    return w


def get_workspace_from_dtype(dtype):
    w = nmt.NmtWorkspace()
    if dtype == "galaxy_clustering":
        fname = os.path.join(
            ROOT, "DESgc_DESgc/w__mask_DESgc__mask_DESgc.fits"
        )
    elif dtype == "galaxy_shear":
        fname = os.path.join(
            ROOT, "DESwl_DESwl/w__mask_DESwl0__mask_DESwl0.fits"
        )
    elif dtype == "cross":
        fname = os.path.join(
            ROOT, "DESgc_DESwl/w__mask_DESgc__mask_DESwl0.fits"
        )
    w.read_from(fname)

    return w


def get_covariance_workspace(tr1, tr2, tr3, tr4):
    config = get_xcell_yml()
    cw = nmt.NmtCovarianceWorkspace()
    m1 = config["tracers"][tr1]["mask_name"]
    m2 = config["tracers"][tr2]["mask_name"]
    m3 = config["tracers"][tr3]["mask_name"]
    m4 = config["tracers"][tr4]["mask_name"]
    fname = os.path.join(ROOT, "cov", f"cw__{m1}__{m2}__{m3}__{m4}.fits")
    cw.read_from(fname)
    return cw


def get_xcell_yml():
    fname = os.path.join(ROOT, "desy1_tjpcov_bm.yml")
    with open(fname) as f:
        config = yaml.safe_load(f)
    return config


def get_tracers_dict_for_cov():
    tr = {1: "DESgc__0", 2: "DESgc__0", 3: "DESwl__0", 4: "DESwl__1"}
    return tr


def get_fields_dict_for_cov(**nmt_conf):
    mask_fn = get_config(INPUT_YML)["tjpcov"]["mask_file"]
    mask_DESgc = hp.read_map(mask_fn["DESgc__0"])
    mask_DESwl0 = hp.read_map(mask_fn["DESwl__0"])
    mask_DESwl1 = hp.read_map(mask_fn["DESwl__1"])

    f1 = f2 = nmt.NmtField(mask_DESgc, None, spin=0, **nmt_conf)
    f3 = nmt.NmtField(mask_DESwl0, None, spin=2, **nmt_conf)
    f4 = nmt.NmtField(mask_DESwl1, None, spin=2, **nmt_conf)

    return {1: f1, 2: f2, 3: f3, 4: f4}


def get_workspaces_dict_for_cov(**kwargs):
    bins = get_nmt_bin()
    f = get_fields_dict_for_cov()

    w12 = nmt.NmtWorkspace()
    w12.compute_coupling_matrix(f[1], f[2], bins, **kwargs)

    w34 = nmt.NmtWorkspace()
    w34.compute_coupling_matrix(f[3], f[4], bins, **kwargs)

    w13 = nmt.NmtWorkspace()
    w13.compute_coupling_matrix(f[1], f[3], bins, **kwargs)
    w23 = w13

    w14 = nmt.NmtWorkspace()
    w14.compute_coupling_matrix(f[1], f[4], bins, **kwargs)
    w24 = w14

    return {13: w13, 23: w23, 14: w14, 24: w24, 12: w12, 34: w34}


def get_cl_dict_for_cov(**kwargs):
    subfolder = "fiducial"
    fname = os.path.join(
        ROOT, subfolder, "DESgc_DESgc/cl_DESgc__0_DESgc__0.npz"
    )
    cl12 = np.load(fname)["cl"]

    fname = os.path.join(
        ROOT, subfolder, "DESwl_DESwl/cl_DESwl__0_DESwl__1.npz"
    )
    cl34 = np.load(fname)["cl"]

    fname = os.path.join(
        ROOT, subfolder, "DESgc_DESwl/cl_DESgc__0_DESwl__0.npz"
    )
    cl13 = cl23 = np.load(fname)["cl"]

    fname = os.path.join(
        ROOT, subfolder, "DESgc_DESwl/cl_DESgc__0_DESwl__1.npz"
    )
    cl14 = cl24 = np.load(fname)["cl"]

    return {13: cl13, 23: cl23, 14: cl14, 24: cl24, 12: cl12, 34: cl34}


# Actual tests
def test_compute_all_blocks():
    # Test _compute_all_blocks function by modifying the
    # get_covariance_block method to output the block in the sacc file

    def _get_covariance_block_for_sacc(s, tracer_comb1, tracer_comb2):
        ix1 = s.indices(tracers=tracer_comb1)
        ix2 = s.indices(tracers=tracer_comb2)
        return s.covariance.covmat[ix1][:, ix2]

    class CNMTTester(FourierGaussianNmt):
        def _get_covariance_block_for_sacc(
            self, tracer_comb1, tracer_comb2, **kwargs
        ):
            s = self.io.sacc_file
            return _get_covariance_block_for_sacc(
                s, tracer_comb1, tracer_comb2
            )

    cnmt = CNMTTester(INPUT_YML)
    blocks, tracers_blocks = cnmt._compute_all_blocks()
    nblocks = len(cnmt.get_list_of_tracers_for_cov())
    assert nblocks == len(blocks)

    for bi, trs in zip(blocks, tracers_blocks):
        assert np.all(
            bi
            == _get_covariance_block_for_sacc(
                cnmt.io.sacc_file, trs[0], trs[1]
            )
        )


def test_get_cl_for_cov(cov_fg_nmt):
    # We just need to test for one case as the function will complain if the
    # Cell inputted has the wrong shape
    m = get_mask_from_dtype("galaxy_clustering")
    w = get_workspace_from_dtype("galaxy_clustering")
    wSh = get_workspace_from_dtype("galaxy_shear")

    cl = get_cl("galaxy_clustering", fiducial=False)
    cl_fid = get_cl("galaxy_clustering", fiducial=True)
    cl_fid_Sh = get_cl("galaxy_shear", fiducial=True)

    cl_cp = (w.couple_cell(cl_fid["cl"]) + cl["nl_cp"]) / np.mean(m**2)
    cl_cp_code = cov_fg_nmt.get_cl_for_cov(
        cl_fid["cl"], cl["nl_cp"], m, m, w, nl_is_cp=True
    )
    assert np.abs(cl_cp / cl_cp_code - 1).max() < 1e-10

    # Inputting uncoupled noise.
    nlfill = np.ones_like(cl_fid["ell"]) * cl["nl"][0, 0]
    cl_cp_code = cov_fg_nmt.get_cl_for_cov(
        cl_fid["cl"], nlfill, m, m, w, nl_is_cp=False
    )
    assert np.abs(cl_cp[0] / cl_cp_code[0] - 1).max() < 1e-2

    # Check that if I input the coupled but nl_is_cp is False, we don't recover
    # cl_cp
    cl_cp_code = cov_fg_nmt.get_cl_for_cov(
        cl_fid["cl"], cl["nl_cp"], m, m, w, nl_is_cp=False
    )
    assert np.abs(cl_cp / cl_cp_code - 1).max() > 0.4

    # Check that if I input the uncoupled but nl_is_cp is True, assert fails
    cl_cp_code = cov_fg_nmt.get_cl_for_cov(
        cl_fid["cl"], nlfill, m, m, w, nl_is_cp=True
    )
    assert np.abs(cl_cp / cl_cp_code - 1).max() > 0.5

    # Create a non overlapping mask
    m2 = np.ones_like(m)
    m2[m != 0] = 0
    assert not np.all(
        cov_fg_nmt.get_cl_for_cov(cl, cl["nl_cp"], m, m2, w, nl_is_cp=True)
    )

    with pytest.raises(ValueError):
        cov_fg_nmt.get_cl_for_cov(
            cl_fid_Sh, cl["nl_cp"], m, m, w, nl_is_cp=True
        )

    with pytest.raises(ValueError):
        # Uncoupled binned noise
        cov_fg_nmt.get_cl_for_cov(cl_fid, cl["nl"], m, m, w, nl_is_cp=True)

    with pytest.raises(ValueError):
        cov_fg_nmt.get_cl_for_cov(
            cl_fid, cl["nl_cp"], m, m, wSh, nl_is_cp=True
        )


@pytest.mark.parametrize(
    "tracer_comb1,tracer_comb2",
    [
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESgc__0")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
        (("DESwl__1", "DESwl__1"), ("DESwl__1", "DESwl__1")),
    ],
)
@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_get_covariance_block(tracer_comb1, tracer_comb2):
    # Load benchmark covariance
    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    # Pass the NmtBins through the config dictionary at initialization
    config = get_config(INPUT_YML)
    bins = get_nmt_bin()
    config["tjpcov"]["binning_info"] = bins
    cnmt = FourierGaussianNmt(config)
    cache = None

    # Check that it raises an Error when use_coupled_noise is True but not
    # coupled noise has been provided
    trs = tracer_comb1 + tracer_comb2
    auto = []
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        auto.append(trs[i - 1] == trs[j - 1])

    # Make sure any of the combinations require the computation of the noise.
    # Otherwise it will not fail
    if any(auto):
        with pytest.raises(ValueError):
            cov = cnmt.get_covariance_block(
                tracer_comb1, tracer_comb2, use_coupled_noise=True
            )

    # Load the coupled noise that we need for the benchmark covariance
    cnmt = FourierGaussianNmt(config)
    s = cnmt.io.get_sacc_file()
    tracer_noise = {}
    tracer_noise_cp = {}
    for tr in s.tracers.keys():
        nl_cp = get_tracer_noise(tr, cp=True)
        tracer_noise[tr] = get_tracer_noise(tr, cp=False)
        tracer_noise_cp[tr] = nl_cp
        cnmt.io.sacc_file.tracers[tr].metadata["n_ell_coupled"] = nl_cp

    # Cov with coupled noise (as in benchmark)
    cov = cnmt.get_covariance_block(tracer_comb1, tracer_comb2) + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-3
    assert cov.flatten() == pytest.approx(cov_bm.flatten(), rel=1e-3)

    # Test cov_tr1_tr2_tr3_tr4.npz cache
    fname = os.path.join(
        "./tests/tmp/",
        "cov_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2),
    )
    assert os.path.isfile(fname)
    assert np.all(np.load(fname)["cov"] + 1e-100 == cov)

    # Test you read it independently of what other arguments you pass
    cov2 = (
        cnmt.get_covariance_block(
            tracer_comb1, tracer_comb2, use_coupled_noise=False
        )
        + 1e-100
    )
    assert np.all(cov2 == cov)

    # Test error with 'bins' in cache different to that at initialization
    with pytest.raises(ValueError):
        cache2 = {"bins": nmt.NmtBin.from_nside_linear(32, bins.get_n_bands())}
        cov2 = cnmt.get_covariance_block(
            tracer_comb1, tracer_comb2, cache=cache2, clobber=True
        )

    # Test it runs with 'bins' in cache if they are the same
    cache2 = {"bins": bins}
    cov2 = (
        cnmt.get_covariance_block(
            tracer_comb1, tracer_comb2, clobber=True, cache=cache2
        )
        + 1e-100
    )

    # Assert relative difference to an absurd precision because the equality
    # test fails now for some reason.
    assert np.max(np.abs(cov / cov2) - 1) < 1e-10

    # Check it works if nl_cp is pass through cache
    cache = {}
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        ncell = cnmt.get_tracer_comb_ncell((trs[i - 1], trs[j - 1]))
        nl_arr = np.zeros((ncell, 96))

        if trs[i - 1] == trs[j - 1]:
            nl_arr[0] = nl_arr[-1] = tracer_noise_cp[trs[i - 1]]

        cache[f"SN{i}{j}"] = nl_arr

    cov2 = (
        cnmt.get_covariance_block(
            tracer_comb1,
            tracer_comb2,
            use_coupled_noise=True,
            cache=cache,
            clobber=True,
        )
        + 1e-100
    )
    assert np.max(np.abs(cov / cov2) - 1) < 1e-10

    # Cov with uncoupled noise cannot be used for benchmark as tracer_noise is
    # assumed to be flat but it is not when computed from the coupled due to
    # edge effects. However, we can test it runs, at least it through cache
    # and compare the chi2
    cache = {}
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        ncell = cnmt.get_tracer_comb_ncell((trs[i - 1], trs[j - 1]))
        nl_arr = np.zeros((ncell, 96))

        if trs[i - 1] == trs[j - 1]:
            nl_arr[0] = nl_arr[-1] = tracer_noise[trs[i - 1]]

        cache[f"SN{i}{j}"] = nl_arr

    cov2 = (
        cnmt.get_covariance_block(
            tracer_comb1,
            tracer_comb2,
            use_coupled_noise=False,
            clobber=True,
            cache=cache,
        )
        + 1e-100
    )
    if (tracer_comb1 == tracer_comb2) and ("DESgc__0" in tracer_comb1):
        # This test fails for weak lensing because there are orders of
        # magnitude between the coupled and decoupled noise that cannot be
        # reconciled by multiplying by <mask>.
        # TODO: Generalize this for weak lensing?

        # Only 1% accuracy since we are assuming a white decoupled noise, which
        # is not the case.
        assert_chi2(s, tracer_comb1, tracer_comb2, cov2, cov, 1e-2)

    # Check chi2, which is what we actually care about
    if tracer_comb1 == tracer_comb2:
        s = cnmt.io.get_sacc_file()
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-4)

    # Check that it runs if one of the masks does not overlap with the others
    if tracer_comb1 != tracer_comb2:
        cnmt.mask_files[
            tracer_comb1[0]
        ] = "./tests/benchmarks/32_DES_tjpcov_bm/catalogs/mask_nonoverlapping.fits.gz"  # noqa: E501
        cov = cnmt.get_covariance_block(
            tracer_comb1, tracer_comb2, clobber=True
        )


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
@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_get_covariance_block_cache(cov_fg_nmt, tracer_comb1, tracer_comb2):
    # In a separate function because the previous one is already too long
    # Add the coupled noise metadata information to the sacc file
    s = cov_fg_nmt.io.get_sacc_file()
    for tr in s.tracers.keys():
        nl_cp = get_tracer_noise(tr, cp=True)
        s.tracers[tr].metadata["n_ell_coupled"] = nl_cp

    (tr1, tr2), (tr3, tr4) = tracer_comb1, tracer_comb2

    cl13 = get_fiducial_cl(s, tr1, tr3, binned=False)
    cl24 = get_fiducial_cl(s, tr2, tr4, binned=False)
    cl14 = get_fiducial_cl(s, tr1, tr4, binned=False)
    cl23 = get_fiducial_cl(s, tr2, tr3, binned=False)

    cache = {
        # 'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
        # 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
        # 'w13': w13, 'w23': w23, 'w14': w14, 'w24': w24,
        # 'w12': w12, 'w34': w34,
        # 'cw': cw,
        "cl13": cl13,
        "cl24": cl24,
        "cl14": cl14,
        "cl23": cl23,
        # 'SN13': SN13, 'SN24': SN24, 'SN14': SN14, 'SN23': SN23,
        "bins": get_nmt_bin(),
    }

    cov = (
        cov_fg_nmt.get_covariance_block(
            tracer_comb1, tracer_comb2, cache=cache
        )
        + 1e-100
    )
    clean_outdir()
    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-5

    if tracer_comb1 == tracer_comb2:
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-5)

    w13 = get_workspace_from_trs(tr1, tr3)
    w23 = get_workspace_from_trs(tr2, tr3)
    w14 = get_workspace_from_trs(tr1, tr4)
    w24 = get_workspace_from_trs(tr2, tr4)
    w12 = get_workspace_from_trs(tr1, tr2)
    w34 = get_workspace_from_trs(tr3, tr4)
    cw = get_covariance_workspace(*tracer_comb1, *tracer_comb2)

    cache = {
        # 'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
        # 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
        "w13": w13,
        "w23": w23,
        "w14": w14,
        "w24": w24,
        "w12": w12,
        "w34": w34,
        "cw": cw,
        "cl13": cl13,
        "cl24": cl24,
        "cl14": cl14,
        "cl23": cl23,
        # 'SN13': SN13, 'SN24': SN24, 'SN14': SN14, 'SN23': SN23,
        "bins": get_nmt_bin(),
    }

    cov = (
        cov_fg_nmt.get_covariance_block(
            tracer_comb1, tracer_comb2, cache=cache, clobber=True
        )
        + 1e-100
    )
    clean_outdir()

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-6
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-6
    if tracer_comb1 == tracer_comb2:
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"l_toeplitz": 10, "l_exact": 10, "dl_band": 10, "n_iter": 0}],
)
def test_get_covariance_workspace(cov_fg_nmt, kwargs):
    m1 = get_mask_from_dtype("galaxy_clustering")
    m3 = get_mask_from_dtype("galaxy_shear")

    f1 = f2 = nmt.NmtField(m1, None, spin=0)
    f3 = f4 = nmt.NmtField(m3, None, spin=2)

    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwargs)

    cl = get_cl("cross", fiducial=False)
    cl_fid = get_cl("cross", fiducial=True)
    w13 = get_workspace_from_dtype("cross")
    cl_cov = cov_fg_nmt.get_cl_for_cov(
        cl_fid["cl"], cl["nl_cp"], m1, m3, w13, nl_is_cp=True
    )
    cl13 = cl14 = cl23 = cl24 = cl_cov

    w12 = get_workspace_from_dtype("galaxy_clustering")
    w34 = get_workspace_from_dtype("galaxy_shear")
    cov = nmt.gaussian_covariance(
        cw, 0, 0, 2, 2, cl13, cl14, cl23, cl24, w12, w34, coupled=False
    )

    mn1, mn2, mn3, mn4 = "0", "1", "2", "3"

    combinations = [
        (f1, f2, f3, f4),
        (f2, f1, f3, f4),
        (f1, f2, f4, f3),
        (f2, f1, f4, f3),
        (f3, f4, f1, f2),
        (f4, f3, f1, f2),
        (f3, f4, f2, f1),
        (f4, f3, f2, f1),
    ]

    combinations_names = [
        (mn1, mn2, mn3, mn4),
        (mn2, mn1, mn3, mn4),
        (mn1, mn2, mn4, mn3),
        (mn2, mn1, mn4, mn3),
        (mn3, mn4, mn1, mn2),
        (mn4, mn3, mn1, mn2),
        (mn3, mn4, mn2, mn1),
        (mn4, mn3, mn2, mn1),
    ]

    # Check only the first is written/computed created & that cw is correct

    for fields, masks_names in zip(combinations, combinations_names):
        spins = [fi.fl.spin for fi in fields]
        cw_code = cov_fg_nmt.get_covariance_workspace(
            *fields, *masks_names, **kwargs
        )
        fname = os.path.join(
            OUTDIR,
            "cw{}{}{}{}__{}__{}__{}__{}.fits".format(*spins, *masks_names),
        )
        if masks_names == (mn1, mn2, mn3, mn4):
            assert os.path.isfile(fname)
        else:
            assert not os.path.isfile(fname)

        cov2 = nmt.gaussian_covariance(
            cw_code,
            0,
            0,
            2,
            2,
            cl13,
            cl14,
            cl23,
            cl24,
            w12,
            w34,
            coupled=False,
        )

        assert np.max(np.abs((cov + 1e-100) / (cov2 + 1e-100) - 1)) < 1e-10

    # Check that with recompute it deletes the existing file and creates a new
    # one

    cw_code = cov_fg_nmt.get_covariance_workspace(
        f3, f4, f2, f1, mn3, mn4, mn2, mn1, recompute=True, **kwargs
    )

    fname = os.path.join(OUTDIR, f"cw0022__{mn1}__{mn2}__{mn3}__{mn3}.fits")
    assert not os.path.isfile(fname)

    fname = os.path.join(OUTDIR, f"cw2200__{mn3}__{mn4}__{mn2}__{mn1}.fits")
    assert os.path.isfile(fname)

    # Check that outdir can be None
    # At the moment outdir is always not None. Leaving this test in case we
    # revert the functionality in the future
    # cw_code = cnmt.get_covariance_workspace(f3, f4, f2, f1, mn3, mn4,
    #                                              mn2, mn1,
    #                                              recompute=True, **kwargs)
    # assert not os.path.isfile(fname)


@pytest.mark.parametrize("nmt_conf", [{}, {"n_iter": 0}])
def test_get_fields_dict(cov_fg_nmt, nmt_conf):
    tr = get_tracers_dict_for_cov()

    f = get_fields_dict_for_cov(**nmt_conf)
    f2 = cov_fg_nmt.get_fields_dict(tr, **nmt_conf)

    # Check that the DESgc fields are exactly the same (not generated twice)
    assert f2[1] is f2[2]

    # Check that if the mask of DESwl has the same name as that of DESgc, they
    # do not get messed up
    cov_fg_nmt.mask_names["DESwl__0"] = cov_fg_nmt.mask_names["DESgc__0"]
    f2 = cov_fg_nmt.get_fields_dict(tr, **nmt_conf)
    assert f2[1] is not f2[3]

    # Check fields are the same by computing the workspace and coupling a
    # fiducial Cell
    cl = {}
    cl[1] = cl[2] = get_cl("galaxy_clustering", fiducial=True)["cl"]
    cl[3] = cl[4] = get_cl("galaxy_shear", fiducial=True)["cl"]

    bins = get_nmt_bin()
    for i in range(1, 5):
        w = cov_fg_nmt.get_workspace(f[i], f[i], str(i), str(i), bins)
        w2 = cov_fg_nmt.get_workspace(f2[i], f2[i], str(i), str(i), bins)

        cl1 = w.couple_cell(cl[i]) + 1e-100
        cl2 = w2.couple_cell(cl[i]) + 1e-100
        assert np.max(np.abs(cl1 / cl2 - 1)) < 1e-10

    # Check that cache works
    cnmt = FourierGaussianNmt(INPUT_YML)
    cache = {"f1": f[1], "f2": f[2], "f3": f[3], "f4": f[4]}
    f2 = cnmt.get_fields_dict(tr, cache=cache, **nmt_conf)
    for i in range(1, 5):
        assert f[i] is f2[i]

    # Check that it does not read the masks again if provided
    cnmt = FourierGaussianNmt(INPUT_YML)
    m = cnmt.get_masks_dict(tr)
    cnmt.mask_files = None
    f2 = cnmt.get_fields_dict(tr, masks=m, **nmt_conf)

    for i in range(1, 5):
        w = cnmt.get_workspace(f[i], f[i], str(i), str(i), bins)
        w2 = cnmt.get_workspace(f2[i], f2[i], str(i), str(i), bins)

        cl1 = w.couple_cell(cl[i]) + 1e-100
        cl2 = w2.couple_cell(cl[i]) + 1e-100
        assert np.max(np.abs(cl1 / cl2 - 1)) < 1e-10


def test_get_list_of_tracers_for_wsp(cov_fg_nmt):
    trs_wsp = cov_fg_nmt.get_list_of_tracers_for_wsp()

    trs_wsp2 = [
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESgc__0")),
        (("DESgc__0", "DESwl__0"), ("DESgc__0", "DESwl__0")),
        (("DESgc__0", "DESwl__1"), ("DESgc__0", "DESwl__1")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__1"), ("DESwl__0", "DESwl__1")),
        (("DESwl__1", "DESwl__1"), ("DESwl__1", "DESwl__1")),
    ]

    assert sorted(trs_wsp) == sorted(trs_wsp2)


def test_get_list_of_tracers_for_cov_wsp(cov_fg_nmt):
    trs_cwsp = cov_fg_nmt.get_list_of_tracers_for_cov_wsp()

    trs_cwsp2 = [
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESgc__0")),
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESgc__0"), ("DESwl__1", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESgc__0", "DESwl__0")),
        (("DESgc__0", "DESwl__0"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESwl__1"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESwl__1", "DESwl__1")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__1")),
        (("DESwl__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
        (("DESwl__0", "DESwl__1"), ("DESwl__0", "DESwl__1")),
        (("DESwl__0", "DESwl__1"), ("DESwl__1", "DESwl__1")),
        (("DESwl__1", "DESwl__1"), ("DESwl__1", "DESwl__1")),
    ]

    assert sorted(trs_cwsp) == sorted(trs_cwsp2)

    trs_cwsp = cov_fg_nmt.get_list_of_tracers_for_cov_wsp(remove_trs_wsp=True)

    for trs in cov_fg_nmt.get_list_of_tracers_for_wsp():
        trs_cwsp2.remove(trs)

    assert trs_cwsp == trs_cwsp2


def test_get_list_of_tracers_for_cov_without_trs_wsp_cwsp(cov_fg_nmt):
    trs_cwsp = cov_fg_nmt.get_list_of_tracers_for_cov_without_trs_wsp_cwsp()

    trs_cwsp2 = [
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESgc__0")),
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESgc__0"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESgc__0"), ("DESwl__1", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESgc__0", "DESwl__0")),
        (("DESgc__0", "DESwl__0"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESwl__0"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESgc__0", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESwl__0", "DESwl__0")),
        (("DESgc__0", "DESwl__1"), ("DESwl__0", "DESwl__1")),
        (("DESgc__0", "DESwl__1"), ("DESwl__1", "DESwl__1")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__0")),
        (("DESwl__0", "DESwl__0"), ("DESwl__0", "DESwl__1")),
        (("DESwl__0", "DESwl__0"), ("DESwl__1", "DESwl__1")),
        (("DESwl__0", "DESwl__1"), ("DESwl__0", "DESwl__1")),
        (("DESwl__0", "DESwl__1"), ("DESwl__1", "DESwl__1")),
        (("DESwl__1", "DESwl__1"), ("DESwl__1", "DESwl__1")),
    ]

    trs_toremove = cov_fg_nmt.get_list_of_tracers_for_wsp()
    trs_toremove += cov_fg_nmt.get_list_of_tracers_for_cov_wsp(
        remove_trs_wsp=True
    )
    for trs in trs_toremove:
        trs_cwsp2.remove(trs)

    assert trs_cwsp == trs_cwsp2


def test_get_nell(cov_fg_nmt):
    nell = 3 * NSIDE
    bins = get_nmt_bin()
    w = get_workspace_from_dtype("galaxy_clustering")
    cache = {"workspaces": {"00": {("mask_DESgc0", "mask_DESgc0"): w}}}

    assert nell == cov_fg_nmt.get_nell()

    # Now with a sacc file without bandpower windows
    s = get_dummy_sacc()
    clf = get_cl("cross")
    s.add_ell_cl("cl_0e", "DESgc__0", "DESwl__0", clf["ell"], clf["cl"][0])
    cov_fg_nmt.io.sacc_file = s

    assert nell == cov_fg_nmt.get_nell(bins=bins)
    assert nell == cov_fg_nmt.get_nell(nside=NSIDE)
    assert nell == cov_fg_nmt.get_nell(cache=cache)

    # Force ValueError (as when window is wrong)
    class s:
        def __init__(self):
            self.metadata = {}
            pass

        def get_data_types(self):
            raise ValueError

    cov_fg_nmt.io.sacc_file = s()
    with pytest.raises(ValueError):
        assert nell == cov_fg_nmt.get_nell()
    # But it works if you pass the nside
    assert nell == cov_fg_nmt.get_nell(nside=NSIDE)

    # Test lmax != 3*nside
    lmax = 50
    bins = get_nmt_bin(50)
    nell = 51
    assert nell == cov_fg_nmt.get_nell(bins=bins)
    # Test that if bins nor workspace is given, it tries to use the sacc file
    # and when fails (if "binnint/ell_max" is not present in the metadata),
    # it defaults to nell = 3*nside
    assert 3 * NSIDE == cov_fg_nmt.get_nell(nside=NSIDE)

    # Check metadata
    cov_fg_nmt.io.sacc_file.metadata["binning/ell_max"] = lmax
    assert nell == cov_fg_nmt.get_nell()

    # Check that if ell_max > 3*nside-1 in metadata, (and nside given) we cut
    # nell = 3*nside
    # Check metadata
    cov_fg_nmt.io.sacc_file = s()
    cov_fg_nmt.io.sacc_file.metadata["binning/ell_max"] = 100
    nell = 3 * NSIDE
    assert nell == cov_fg_nmt.get_nell(nside=NSIDE)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"l_toeplitz": 10, "l_exact": 10, "dl_band": 10, "n_iter": 0}],
)
def test_get_workspace(cov_fg_nmt, kwargs):
    # Compute NmtBins
    bins = get_nmt_bin()

    # Compute workspace
    m1 = get_mask_from_dtype("galaxy_clustering")
    m2 = get_mask_from_dtype("galaxy_shear")

    f1 = nmt.NmtField(m1, None, spin=0)
    f2 = nmt.NmtField(m2, None, spin=2)

    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, bins, **kwargs)

    # Compute workspace with cov_fg_nmt
    s1 = 0
    s2 = 2
    mn1 = "mask_DESgc0"
    mn2 = "mask_DESwl0"
    w_code = cov_fg_nmt.get_workspace(f1, f2, mn1, mn2, bins, **kwargs)

    # Check the file is created
    fname = os.path.join(OUTDIR, f"w{s1}{s2}__{mn1}__{mn2}.fits")
    assert os.path.isfile(fname)

    # Check that you will read the same workspace if input the other way round
    # and check the symmetric file is not created
    w_code2 = cov_fg_nmt.get_workspace(f2, f1, mn2, mn1, bins, **kwargs)
    fname = os.path.join(OUTDIR, f"w{s2}{s1}__{mn2}__{mn1}.fits")
    assert not os.path.isfile(fname)

    # Check that with recompute the original file is removed and the symmetric
    # remains
    w_code2 = cov_fg_nmt.get_workspace(
        f2, f1, mn2, mn1, bins, recompute=True, **kwargs
    )
    fname = os.path.join(OUTDIR, f"w{s1}{s2}__{mn1}__{mn2}.fits")
    assert not os.path.isfile(fname)
    fname = os.path.join(OUTDIR, f"w{s2}{s1}__{mn2}__{mn1}.fits")
    assert os.path.isfile(fname)

    # Load cl to apply the workspace on
    cl = get_cl("cross", fiducial=True)["cl"]

    rdev = (w.couple_cell(cl) + 1e-100) / (w_code.couple_cell(cl) + 1e-100) - 1
    assert np.max(np.abs(rdev)) < 1e-10

    rdev = (w.couple_cell(cl) + 1e-100) / (
        w_code2.couple_cell(cl) + 1e-100
    ) - 1
    assert np.max(np.abs(rdev)) < 1e-10
    # Check that outdir can be None
    # At the moment outdir is always not None. Leaving this test in case we
    # revert the functionality in the future
    # w_code = cnmt.get_workspace(f1, f2, mn1, mn2, bins, None, **kwargs)
    # fname = os.path.join(outdir, f'w{s1}{s2}__{mn1}__{mn2}.fits')
    # assert not os.path.isfile(fname)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"l_toeplitz": 10, "l_exact": 10, "dl_band": 10, "n_iter": 0}],
)
def test_get_workspace_dict(cov_fg_nmt, kwargs):
    tracers = get_tracers_dict_for_cov()
    bins = get_nmt_bin()

    w = get_workspaces_dict_for_cov(**kwargs)
    w2 = cov_fg_nmt.get_workspaces_dict(tracers, bins, **kwargs)

    # Check workspaces by comparing the coupled cells
    cl = get_cl_dict_for_cov()

    for i in [13, 23, 14, 24, 12, 34]:
        cl1 = w[i].couple_cell(cl[i]) + 1e-100
        cl2 = w2[i].couple_cell(cl[i]) + 1e-100
        assert np.max(np.abs(cl1 / cl2 - 1)) < 1e-10

    # Check that things are not read/computed twice
    assert w2[13] is w2[23]
    assert w2[14] is w2[24]

    # Check that cache works
    cache = {
        "w13": w[13],
        "w23": w[23],
        "w14": w[14],
        "w24": w[24],
        "w12": w[12],
        "w34": w[34],
    }
    w2 = cov_fg_nmt.get_workspaces_dict(tracers, bins, cache=cache, **kwargs)
    for i in [13, 23, 14, 24, 12, 34]:
        assert w[i] is w2[i]

    # Check that for non overlapping fields, the workspace is not computed (and
    # is None)
    # Create a non overlapping mask:
    m = cov_fg_nmt.get_masks_dict(tracers)
    m[1] = np.zeros_like(m[2])
    m[1][:1000] = 1
    m[3] = np.zeros_like(m[4])
    m[3][1000:2000] = 1

    w2 = cov_fg_nmt.get_workspaces_dict(tracers, bins, masks=m, **kwargs)
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
    gc0gc0 = os.path.join(ROOT, "DESgc_DESgc/w__mask_DESgc__mask_DESgc.fits")
    gc0wl0 = os.path.join(ROOT, "DESgc_DESwl/w__mask_DESgc__mask_DESwl0.fits")
    gc0wl1 = os.path.join(ROOT, "DESgc_DESwl/w__mask_DESgc__mask_DESwl1.fits")
    wl0wl0 = os.path.join(ROOT, "DESwl_DESwl/w__mask_DESwl0__mask_DESwl0.fits")
    wl0wl1 = os.path.join(ROOT, "DESwl_DESwl/w__mask_DESwl0__mask_DESwl1.fits")
    wl1wl1 = os.path.join(ROOT, "DESwl_DESwl/w__mask_DESwl1__mask_DESwl1.fits")
    cache = {
        "workspaces": {
            "00": {("mask_DESgc0", "mask_DESgc0"): gc0gc0},
            "02": {
                ("mask_DESgc0", "mask_DESwl0"): gc0wl0,
                ("mask_DESgc0", "mask_DESwl1"): gc0wl1,
            },
            "22": {
                ("mask_DESwl0", "mask_DESwl0"): wl0wl0,
                ("mask_DESwl0", "mask_DESwl1"): wl0wl1,
                ("mask_DESwl1", "mask_DESwl1"): wl1wl1,
            },
        }
    }
    # bins to None to force it fail if it does not uses the cache
    w2 = cov_fg_nmt.get_workspaces_dict(tracers, None, cache=cache, **kwargs)

    # Check that it will compute the workspaces if one is missing
    del cache["workspaces"]["02"][("mask_DESgc0", "mask_DESwl1")]
    w2 = cov_fg_nmt.get_workspaces_dict(tracers, bins, cache=cache, **kwargs)
    # Check that '20' is also understood
    del cache["workspaces"]["02"]
    cache["workspaces"]["20"] = {
        ("mask_DESgc0", "mask_DESwl0"): gc0wl0,
        ("mask_DESgc0", "mask_DESwl1"): gc0wl1,
    }
    w2 = cov_fg_nmt.get_workspaces_dict(tracers, None, cache=cache, **kwargs)


@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_full_covariance_benchmark():
    config = get_config(INPUT_YML)
    bins = get_nmt_bin()
    config["tjpcov"]["binning_info"] = bins
    # Load the coupled noise that we need for the benchmark covariance
    cnmt = FourierGaussianNmt(config)
    s_nlcp = cnmt.io.get_sacc_file().copy()
    tracer_noise = {}
    tracer_noise_cp = {}
    for tr in s_nlcp.tracers.keys():
        nl_cp = get_tracer_noise(tr, cp=True)
        tracer_noise[tr] = get_tracer_noise(tr, cp=False)
        tracer_noise_cp[tr] = nl_cp
        s_nlcp.tracers[tr].metadata["n_ell_coupled"] = nl_cp

    cnmt.io.sacc_file = s_nlcp.copy()

    cov = cnmt.get_covariance() + 1e-100
    cov_bm = s_nlcp.covariance.covmat + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-3
    assert cov.flatten() == pytest.approx(cov_bm.flatten(), rel=1e-3)

    # Check chi2
    clf = np.array([])
    for trs in s_nlcp.get_tracer_combinations():
        cl_trs = get_fiducial_cl(s_nlcp, *trs, remove_be=True)
        clf = np.concatenate((clf, cl_trs.flatten()))
    cl = s_nlcp.mean

    delta = clf - cl
    chi2 = delta.dot(np.linalg.inv(cov)).dot(delta)
    chi2_bm = delta.dot(np.linalg.inv(cov_bm)).dot(delta)
    assert np.abs(chi2 / chi2_bm - 1) < 1e-4

    # Clean after the test
    clean_outdir()

    # Check that it also works if they don't use concise data_types
    s2 = s_nlcp.copy()
    for dp in s2.data:
        dt = dp.data_type

        if dt == "cl_00":
            dp.data_type = sacc.standard_types.galaxy_density_cl
        elif dt == "cl_0e":
            dp.data_type = sacc.standard_types.galaxy_shearDensity_cl_e
        elif dt == "cl_0b":
            dp.data_type = sacc.standard_types.galaxy_shearDensity_cl_b
        elif dt == "cl_ee":
            dp.data_type = sacc.standard_types.galaxy_shear_cl_ee
        elif dt == "cl_eb":
            dp.data_type = sacc.standard_types.galaxy_shear_cl_eb
        elif dt == "cl_be":
            dp.data_type = sacc.standard_types.galaxy_shear_cl_be
        elif dt == "cl_bb":
            dp.data_type = sacc.standard_types.galaxy_shear_cl_bb
        else:
            raise ValueError("Something went wrong. Data type not recognized")

    cnmt.cl_data = s2
    cov2 = cnmt.get_covariance() + 1e-100
    assert np.all(cov == cov2)

    # Clean after the test
    clean_outdir()

    # Check that it fails if tracer_noise is used instead of tracer_noise_cp
    cnmt = FourierGaussianNmt(config)
    cov2 = cnmt.get_covariance(use_coupled_noise=False) + 1e-100
    assert not np.all(cov == cov2)
    clean_outdir()

    # Check that binning can be passed through cache
    cnmt = FourierGaussianNmt(INPUT_YML)
    cnmt.io.sacc_file = s_nlcp.copy()
    cov2 = cnmt.get_covariance(cache={"bins": bins}) + 1e-100
    assert np.max(np.abs(cov / cov2 - 1)) < 1e-10


def test_txpipe_like_input():
    # We don't need to pass the bins because we have provided the workspaces
    # through the cache in the configuration file
    fname = "./tests/data/conf_covariance_gaussian_fourier_nmt_txpipe.yaml"
    cnmt = FourierGaussianNmt(fname)

    # Add the coupled noise metadata information to the sacc file
    s = cnmt.io.get_sacc_file()

    cov = cnmt.get_covariance() + 1e-100
    cov_bm = s.covariance.covmat + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-3
    assert cov.flatten() == pytest.approx(cov_bm.flatten(), rel=1e-3)

    # Check chi2
    clf = np.array([])
    for trs in s.get_tracer_combinations():
        cl_trs = get_fiducial_cl(s, *trs, remove_be=True)
        clf = np.concatenate((clf, cl_trs.flatten()))
    cl = s.mean

    delta = clf - cl
    chi2 = delta.dot(np.linalg.inv(cov)).dot(delta)
    chi2_bm = delta.dot(np.linalg.inv(cov_bm)).dot(delta)
    assert np.abs(chi2 / chi2_bm - 1) < 1e-4
