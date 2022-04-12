#!/usr/bin/python
import os
import numpy as np
import pymaster as nmt
from mpi4py import MPI
import tjpcov.main as cv
from tjpcov import nmt_tools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml_mpi = os.path.join(root, "tjpcov_conf_minimal_mpi.yaml")


def get_nmt_bin():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def get_pair_folder_name(tracer_comb):
    bn = []
    for tr in tracer_comb:
        bn.append(tr.split('__')[0])
    return '_'.join(bn)


def get_tracer_noise(tr, cp=True):
    bn = get_pair_folder_name((tr, tr))
    fname = os.path.join(root, bn, f"cl_{tr}_{tr}.npz")
    clfile = np.load(fname)
    if cp:
        return clfile['nl_cp'][0, -1]
    else:
        return clfile['nl'][0, 0]


def get_tracer_noise_cp_dict(cl_data):
    tracer_noise_cp = {}
    for tr in cl_data.tracers:
        tracer_noise_cp[tr] = get_tracer_noise(tr)

    return tracer_noise_cp


def get_tracers_cov(cl_data):
    cl_tracers = cl_data.get_tracer_combinations()

    # Make a list of all pair of tracer combinations
    tracers_cov = []
    for i, tracer_comb1 in enumerate(cl_tracers):
        for tracer_comb2 in cl_tracers[i:]:
            tracers_cov.append((tracer_comb1, tracer_comb2))

    return tracers_cov


def test_split_tasks_by_rank():
    tjpcov_class = cv.CovarianceCalculator(input_yml_mpi)

    l = list(range(100))

    # Bassicaly what split_tasks_by_rank does
    tasks = []
    for i, task in enumerate(l):
        if i % size == rank:
            tasks.append(task)

    print('rank in tjpcov', tjpcov_class.rank)
    assert list(tjpcov_class.split_tasks_by_rank(l)) == tasks


def test_compute_all_blocks_nmt():
    tjpcov_class = cv.CovarianceCalculator(input_yml_mpi)
    s = tjpcov_class.cl_data
    mask_names = tjpcov_class.mask_names
    bins = get_nmt_bin()

    tracer_noise_cp = get_tracer_noise_cp_dict(s)

    blocks, tracers_blocks = tjpcov_class.compute_all_blocks_nmt(None,
                                                                 tracer_noise_cp,
                                                                 cache={'bins':
                                                                        bins})

    trs1 = nmt_tools.get_list_of_tracers_for_wsp(s, mask_names)
    trs2 = nmt_tools.get_list_of_tracers_for_cov_wsp(s, mask_names,
                                                     remove_trs_wsp=True)
    trs3 = nmt_tools.get_list_of_tracers_for_cov(s, remove_trs_wsp=True,
                                                 remove_trs_cwsp=True,
                                                 mask_names=mask_names)
    trs_blocks = list(tjpcov_class.split_tasks_by_rank(trs1))
    trs_blocks += list(tjpcov_class.split_tasks_by_rank(trs2))
    trs_blocks += list(tjpcov_class.split_tasks_by_rank(trs3))
    assert(tracers_blocks == trs_blocks)


    ell, _ = s.get_ell_cl('cl_00', 'DESgc__0', 'DESgc__0')
    nbpw = ell.size

    for itrs, trs in enumerate(tracers_blocks):
        tracer_comb1, tracer_comb2 = trs
        print(trs)

        cov = blocks[itrs] + 1e-100

        fname = os.path.join(root,
                             'cov/cov_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                              *tracer_comb2))
        cov_bm = np.load(fname)['cov'] + 1e-100

        assert np.max(np.abs(cov / cov_bm - 1)) < 1e-3

    comm.Barrier()
    if rank == 0:
        os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")

def test_get_all_cov_nmt_mpi():
    tjpcov_class = cv.CovarianceCalculator(input_yml_mpi)
    s = tjpcov_class.cl_data
    bins = get_nmt_bin()

    # tracer_noise_cp = {}
    # for tr in s.tracers:
    #     tracer_noise_cp[tr] = get_tracer_noise(tr)
    tracer_noise_cp = get_tracer_noise_cp_dict(s)

    cov = tjpcov_class.get_all_cov_nmt(tracer_noise_coupled=tracer_noise_cp,
                                       cache={'bins': bins})

    comm.Barrier()
    if rank == 0:
        cov += 1e-100
        cov_bm = s.covariance.covmat + 1e-100
        assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
        assert np.max(np.abs(cov / cov_bm - 1)) < 1e-3
        os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")

# Clean up after the tests
comm.Barrier()
if rank == 0:
    os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")
