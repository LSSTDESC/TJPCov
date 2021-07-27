#!/usr/bin/python
import numpy as np
import os
import pymaster as nmt
import pytest
import tjpcov.main as cv
import yaml


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
xcell_yml = os.path.join(root, "desy1_tjpcov_bm.yml")


def get_xcell_yml():
    with open(xcell_yml) as f:
        config = yaml.safe_load(f)
    return config


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


def get_fiducial_cl(s, tr1, tr2, binned=True):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(root, 'fiducial', bn, f"cl_{tr1}_{tr2}.npz")
    cl = np.load(fname)['cl']
    if binned:
        s = s.copy()
        s.remove_selection(data_type='cl_0b')
        s.remove_selection(data_type='cl_eb')
        s.remove_selection(data_type='cl_be')
        s.remove_selection(data_type='cl_bb')
        ix = s.indices(tracers=(tr1, tr2))
        bpw = s.get_bandpower_windows(ix)

        cl0_bin = bpw.weight.T.dot(cl[0])

        cl_bin = np.zeros((cl.shape[0], cl0_bin.size))
        cl_bin[0] = cl0_bin
        return cl_bin
    else:
        return cl


def get_tracer_noise(tr):
    bn = get_pair_folder_name((tr, tr))
    fname = os.path.join(root, bn, f"cl_{tr}_{tr}.npz")
    return np.load(fname)['nl_cp'][0, -1]


def get_benchmark_cov(tracer_comb1, tracer_comb2):
    (tr1, tr2), (tr3, tr4) = tracer_comb1, tracer_comb2
    fname = os.path.join(root, 'cov', f'cov_{tr1}_{tr2}_{tr3}_{tr4}.npz')
    return np.load(fname)['cov']


def get_workspace(tr1, tr2):
    config = get_xcell_yml()
    w = nmt.NmtWorkspace()
    bn = get_pair_folder_name((tr1, tr2))
    m1 = config['tracers'][tr1]['mask_name']
    m2 = config['tracers'][tr2]['mask_name']
    fname = os.path.join(root, bn, f"w__{m1}__{m2}.fits")
    w.read_from(fname)
    return w


def get_covariance_workspace(tr1, tr2, tr3, tr4):
    config = get_xcell_yml()
    cw = nmt.NmtCovarianceWorkspace()
    m1 = config['tracers'][tr1]['mask_name']
    m2 = config['tracers'][tr2]['mask_name']
    m3 = config['tracers'][tr3]['mask_name']
    m4 = config['tracers'][tr4]['mask_name']
    fname = os.path.join(root, 'cov', f"cw__{m1}__{m2}__{m3}__{m4}.fits")
    cw.read_from(fname)
    return cw


def assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, threshold):
    cl1 = get_data_cl(*tracer_comb1)
    cl2 = get_data_cl(*tracer_comb2)

    clf1 = get_fiducial_cl(s, *tracer_comb1)
    clf2 = get_fiducial_cl(s, *tracer_comb2)

    delta1 = (clf1 - cl1).flatten()
    delta2 = (clf2 - cl2).flatten()
    chi2 = delta1.dot(np.linalg.inv(cov)).dot(delta2)
    chi2_bm = delta1.dot(np.linalg.inv(cov_bm)).dot(delta2)

    assert np.abs(chi2 / chi2_bm - 1) < 1e-5



@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_nmt_gaussian_cov(tracer_comb1, tracer_comb2):
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
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-2)


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_nmt_gaussian_cov_cache(tracer_comb1, tracer_comb2):
    tjpcov_class = cv.CovarianceCalculator(input_yml)

    ccl_tracers, tracer_noise = tjpcov_class.get_tracer_info(tjpcov_class.cl_data)

    for tr in tracer_comb1 + tracer_comb2:
        tracer_noise[tr] = get_tracer_noise(tr)

    (tr1, tr2), (tr3, tr4) = tracer_comb1, tracer_comb2

    s = None  # Not needed if binned=False
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
             'cl13': cl13, 'cl24': cl24, 'cl14': cl14, 'cl23':cl23,
             # 'SN13': SN13, 'SN24': SN24, 'SN14': SN14, 'SN23': SN23,
             'bins': get_nmt_bin()
    }

    cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        ccl_tracers, tracer_noise,
                                        cache=cache)['final'] + 1e-100

    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-5

    if tracer_comb1 == tracer_comb2:
        s = tjpcov_class.cl_data
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-5)

    w13 = get_workspace(tr1, tr3)
    w23 = get_workspace(tr2, tr3)
    w14 = get_workspace(tr1, tr4)
    w24 = get_workspace(tr2, tr4)
    w12 = get_workspace(tr1, tr2)
    w34 = get_workspace(tr3, tr4)
    cw = get_covariance_workspace(*tracer_comb1, *tracer_comb2)

    cache = {
             # 'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
             # 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
             'w13': w13, 'w23': w23, 'w14': w14, 'w24': w24,
             'w12': w12, 'w34': w34,
             'cw': cw,
             'cl13': cl13, 'cl24': cl24, 'cl14': cl14, 'cl23':cl23,
             # 'SN13': SN13, 'SN24': SN24, 'SN14': SN14, 'SN23': SN23,
             'bins': get_nmt_bin()
    }

    cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        ccl_tracers, tracer_noise,
                                        cache=cache)['final'] + 1e-100

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-5
    if tracer_comb1 == tracer_comb2:
        s = tjpcov_class.cl_data
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-5)

def test_get_all_cov_nmt():
    tjpcov_class = cv.CovarianceCalculator(input_yml)
    s = tjpcov_class.cl_data
    bins = get_nmt_bin()

    tracer_noise = {}
    for tr in s.tracers:
        tracer_noise[tr] = get_tracer_noise(tr)

    cov = tjpcov_class.get_all_cov_nmt(tracer_noise=tracer_noise,
                                       cache={'bins': bins}) + 1e-100

    cov_bm = s.covariance.covmat + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-3
    assert np.max(np.abs(cov / cov_bm - 1)) < 5e-3


# Clean up after the tests
os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")
