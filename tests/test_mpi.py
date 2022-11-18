#!/usr/bin/python
import os

import numpy as np
import pytest
import shutil
from mpi4py import MPI

from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt
from tjpcov.covariance_fourier_ssc import FourierSSCHaloModel
from tjpcov.covariance_calculator import CovarianceCalculator

root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = "./tests/tmp/"
input_yml_mpi = root + "conf_covariance_gaussian_fourier_nmt_txpipe_mpi.yaml"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
if rank == 0:
    # Create temporal folder
    os.makedirs(outdir, exist_ok=True)


def clean_tmp():
    comm.Barrier()
    if (rank == 0) and os.path.isdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir)


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    clean_tmp()


# The _split_tasks_by_rank and _compute_all_blocks methods have been tested
# serially in tests_covariance_builder.py. Here, we will just make sure that
# they also work in MPI and that we don't have problems when saving the block
# covariances. That's why we will use FourierGaussianNmt.


def get_pair_folder_name(tracer_comb):
    bn = []
    for tr in tracer_comb:
        bn.append(tr.split("__")[0])
    return "_".join(bn)


def get_fiducial_cl(s, tr1, tr2, binned=True, remove_be=False):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(root, "fiducial", bn, f"cl_{tr1}_{tr2}.npz")
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


def test_split_tasks_by_rank():
    cnmt = FourierGaussianNmt(input_yml_mpi)
    tasks = list(range(100))
    tasks_splitted = list(cnmt._split_tasks_by_rank(tasks))

    assert tasks[cnmt.rank :: cnmt.size] == tasks_splitted


def test_compute_all_blocks():
    cssc = FourierSSCHaloModel(input_yml_mpi)
    blocks, tracers_blocks = cssc._compute_all_blocks()
    nblocks = len(
        list(cssc._split_tasks_by_rank(cssc.get_list_of_tracers_for_cov()))
    )
    assert nblocks == len(blocks)

    for bi, trs in zip(blocks, tracers_blocks):
        cov = cssc._get_covariance_block_for_sacc(trs[0], trs[1])
        assert np.max(np.abs((bi + 1e-100) / (cov + 1e-100) - 1)) < 1e-5


def test_compute_all_blocks_nmt():
    # FourierGaussianNmt has its own _compute_all_blocks
    cnmt = FourierGaussianNmt(input_yml_mpi)
    blocks, tracers_blocks = cnmt._compute_all_blocks()
    nblocks = len(
        list(cnmt._split_tasks_by_rank(cnmt.get_list_of_tracers_for_cov()))
    )
    assert nblocks == len(blocks)

    for bi, trs in zip(blocks, tracers_blocks):
        cov = cnmt._get_covariance_block_for_sacc(trs[0], trs[1])
        assert np.max(np.abs((bi + 1e-100) / (cov + 1e-100) - 1)) < 1e-5


def test_get_covariance():
    # This checks that there is no problem during the gathering of blocks

    # The coupled noise metadata information is in the sacc file and the
    # workspaces in the config file
    cnmt = FourierGaussianNmt(input_yml_mpi)
    s = cnmt.io.get_sacc_file()

    cov = cnmt.get_covariance() + 1e-100
    cov_bm = s.covariance.covmat + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-3

    # Check chi2
    clf = np.array([])
    for trs in s.get_tracer_combinations():
        cl_trs = get_fiducial_cl(s, *trs, remove_be=True)
        clf = np.concatenate((clf, cl_trs.flatten()))
    cl = s.mean

    delta = clf - cl
    chi2 = delta.dot(np.linalg.inv(cov)).dot(delta)
    chi2_bm = delta.dot(np.linalg.inv(cov_bm)).dot(delta)
    assert np.abs(chi2 / chi2_bm - 1) < 1e-5


def test_CovarianceCalculator():
    cc = CovarianceCalculator("./tests/data/conf_covariance_calculator.yml")
    cc_mpi = CovarianceCalculator(
        "./tests/data/conf_covariance_calculator_mpi.yml"
    )

    # Test get_covariance_terms
    cov = None
    cov_mpi = cc_mpi.get_covariance_terms()
    if rank == 0:
        # Avoid computing serially the covariance terms multiple times.
        # Broadcast it later
        cov = cc.get_covariance_terms()
    cov = comm.bcast(cov, root=0)
    for k in cov.keys():
        assert (
            np.max(np.abs((cov[k] + 1e-100) / (cov_mpi[k] + 1e-100) - 1))
            < 1e-5
        )
    comm.Barrier()

    # Test get_covariance
    cov_mpi = cc_mpi.get_covariance()
    if rank == 0:
        # Avoid computing serially the covariance multiple times. Broadcast it
        # later
        cov = cc.get_covariance()
    cov = comm.bcast(cov, root=0)
    assert np.max(np.abs((cov + 1e-100) / (cov_mpi + 1e-100) - 1)) < 1e-5
    comm.Barrier()

    # Test create_sacc_cov
    fname = f"cls_cov{rank}.fits"
    cc_mpi.create_sacc_cov(output=fname, save_terms=True)
    keys = ["gauss", "SSC"]
    if rank == 0:
        assert os.path.isfile(outdir + "cls_cov0.fits")
        for k in keys:
            assert os.path.isfile(outdir + f"cls_cov0_{k}.fits")
    else:
        fname = f"cls_cov{rank}_{k}.fits"
        assert not os.path.isfile(outdir + fname)
        for k in keys:
            assert not os.path.isfile(outdir + fname)
    comm.Barrier()
