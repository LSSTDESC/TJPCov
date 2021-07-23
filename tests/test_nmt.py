#!/usr/bin/python
import numpy as np
import os
import pymaster as nmt
import pytest
import tjpcov.main as cv


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")

def get_nmt_bin():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def get_pair_folder_name(tracer_comb):
    bn = []
    for tr in tracer_comb:
        bn.append(tr.split('__')[0])
    return '_'.join(bn)

def get_data_cl(tr1, tr2):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(root, bn, f"cl_{tr1}_{tr2}.npz")
    return np.load(fname)['cl']

def get_fiducial_cl(s, tr1, tr2):
    s = s.copy()
    s.remove_selection(data_type='cl_0b')
    s.remove_selection(data_type='cl_eb')
    s.remove_selection(data_type='cl_bb')
    ix = s.indices(tracers=(tr1, tr2))
    bpw = s.get_bandpower_windows(ix)

    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(root, 'fiducial', bn, f"cl_{tr1}_{tr2}.npz")
    cl = np.load(fname)['cl']
    cl0_bin = bpw.weight.T.dot(cl[0])

    cl_bin = np.zeros((cl.shape[0], cl0_bin.size))
    cl_bin[0] = cl0_bin
    return cl_bin

def get_tracer_noise(tr):
    bn = get_pair_folder_name((tr, tr))
    fname = os.path.join(root, bn, f"cl_{tr}_{tr}.npz")
    return np.load(fname)['nl_cp'][0, -1]

def get_benchmark_cov(tracer_comb1, tracer_comb2):
    (tr1, tr2), (tr3, tr4) = tracer_comb1, tracer_comb2
    fname = os.path.join(root, 'cov', f'cov_{tr1}_{tr2}_{tr3}_{tr4}.npz')
    return np.load(fname)['cov']


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_nmt_gaussian_cov(tracer_comb1, tracer_comb2):
    print(tracer_comb1)
    tjpcov_class = cv.CovarianceCalculator(input_yml)

    ccl_tracers, tracer_noise = tjpcov_class.get_tracer_info(tjpcov_class.cl_data)

    for tr in tracer_comb1 + tracer_comb2:
        tracer_noise[tr] = get_tracer_noise(tr)

    cache = {'bins': get_nmt_bin()}

    cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        ccl_tracers, tracer_noise,
                                        cache=cache)['final'] + 1e-100

    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 5e-3
    assert np.max(np.abs(cov / cov_bm - 1)) < 5e-1

    if tracer_comb1 == tracer_comb2:
        s = tjpcov_class.cl_data

        cl1 = get_data_cl(*tracer_comb1)
        cl2 = get_data_cl(*tracer_comb2)

        clf1 = get_fiducial_cl(s, *tracer_comb1)
        clf2 = get_fiducial_cl(s, *tracer_comb2)

        delta1 = (clf1 - cl1).flatten()
        delta2 = (clf2 - cl2).flatten()
        chi2 = delta1.dot(np.linalg.inv(cov)).dot(delta2)
        chi2_bm = delta1.dot(np.linalg.inv(cov_bm)).dot(delta2)

        assert np.abs(chi2 / chi2_bm - 1) < 1e-2


# def test_nmt_gaussian_cov_cache():
#     for tr in tracer_comb1 + tracer_comb2:
#         tracer_noise[tr] = get_tracer_noise(tr)
#
#     cache = {
#             'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
#              'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
#              'w13': w13, 'w23': w23, 'w14': w14, 'w24': w24,
#              'w12': w12, 'w34': w34,
#              'bins': bins,
#              'cw': cw,
#              'cl13': cl, 'cl24': cl, 'cl14': cl, 'cl23':cl,
#              'SN13': SN, 'SN24': SN, 'SN14': SN, 'SN23': SN
#     }
#
#     cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
#                                         ccl_tracers, tracer_noise, cache) + \
#         1e-100
#
#     cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100
#
#     assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-10
#     assert np.max(np.abs(cov / cov_bm - 1)) < 1e-10
