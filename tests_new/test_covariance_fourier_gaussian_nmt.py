#!/usr/bin/python3

import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new.covariance_fourier_gaussian_nmt import \
    CovarianceFourierGaussianNmt
from tjpcov_new.covariance_io import CovarianceIO
import yaml


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = root + 'tjpcov_tmp/'
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_yml_no_nmtc = os.path.join(root, "tjpcov_conf_minimal_no_nmtconf.yaml")
input_yml_txpipe = os.path.join(root, "tjpcov_conf_minimal_txpipe.yaml")
xcell_yml = os.path.join(root, "desy1_tjpcov_bm.yml")

input_sacc = sacc.Sacc.load_fits(root + 'cls_cov.fits')

# Create temporal folder
os.makedirs(outdir, exist_ok=True)


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


def get_nmt_bin():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def get_pair_folder_name(tracer_comb):
    bn = []
    for tr in tracer_comb:
        bn.append(tr.split('__')[0])
    return '_'.join(bn)


def get_data_cl(tr1, tr2, remove_be=False):
    bn = get_pair_folder_name((tr1, tr2))
    fname = os.path.join(root, bn, f"cl_{tr1}_{tr2}.npz")
    cl = np.load(fname)['cl']

    # Remove redundant terms
    if remove_be and (tr1 == tr2) and (cl.shape[0] == 4):
        cl = np.delete(cl, 2, 0)
    return cl


def get_fiducial_cl(s, tr1, tr2, binned=True, remove_be=False):
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
        cl = cl_bin
    else:
        cl

    # Remove redundant terms
    if remove_be and (tr1 == tr2) and (cl.shape[0] == 4):
        cl = np.delete(cl, 2, 0)
    return cl


def get_tracer_noise(tr, cp=True):
    bn = get_pair_folder_name((tr, tr))
    fname = os.path.join(root, bn, f"cl_{tr}_{tr}.npz")
    clfile = np.load(fname)
    if cp:
        return clfile['nl_cp'][0][-1]
    else:
        return clfile['nl'][0][0]


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


def get_xcell_yml():
    with open(xcell_yml) as f:
        config = yaml.safe_load(f)
    return config


def test_compute_all_blocks():
    pass


def test_get_cl_for_cov():
    pass


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          (('DESwl__1', 'DESwl__1'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_get_covariance_block(tracer_comb1, tracer_comb2):
    # TODO: Not sure why this is needed here, but otherwise it failed to save
    # the workspaces
    os.makedirs(outdir, exist_ok=True)

    # Load benchmark covariance
    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    # Pass the NmtBins through the config dictionary at initialization
    config = get_config(input_yml)
    bins = get_nmt_bin()
    config['tjpcov']['binning_info'] = bins
    cnmt = CovarianceFourierGaussianNmt(config)
    cache = None

    # Check that it raises an Error when use_coupled_noise is True but not
    # coupled noise has been provided
    trs = tracer_comb1 + tracer_comb2
    auto = []
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        auto.append(trs[i-1] == trs[j-1])

    # Make sure any of the combinations require the computation of the noise.
    # Otherwise it will not fail
    if any(auto):
        with pytest.raises(ValueError):
            cov = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                            use_coupled_noise=True)


    # Load the coupled noise that we need for the benchmark covariance
    cnmt = CovarianceFourierGaussianNmt(config)
    s = cnmt.io.get_sacc_file()
    tracer_noise = {}
    tracer_noise_cp = {}
    for tr in s.tracers.keys():
        nl_cp = get_tracer_noise(tr, cp=True)
        tracer_noise[tr] = get_tracer_noise(tr, cp=False)
        tracer_noise_cp[tr] = nl_cp
        cnmt.io.sacc_file.tracers[tr].metadata['n_ell_coupled'] = nl_cp


    # Cov with coupled noise (as in benchmark)
    cov = cnmt.get_covariance_block(tracer_comb1, tracer_comb2) + 1e-100
    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-5

    # Test cov_tr1_tr2_tr3_tr4.npz cache
    fname = os.path.join('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/',
                         'cov_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                      *tracer_comb2))
    assert os.path.isfile(fname)
    assert np.all(np.load(fname)['cov'] + 1e-100 == cov)

    # Test you read it independently of what other arguments you pass
    cov2 = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                     use_coupled_noise=False) + 1e-100
    assert np.all(cov2 == cov)
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Test error with 'bins' in cache different to that at initialization
    with pytest.raises(ValueError):
        cache2 = {'bins': nmt.NmtBin.from_nside_linear(32, bins.get_n_bands())}
        cov2 = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                         cache=cache2)

    # Test it runs with 'bins' in cache if they are the same
    cache2 = {'bins': bins}
    cov2 = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                     cache=cache2) + 1e-100
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Assert relative difference to an absurd precision because the equality
    # test fails now for some reason.
    assert np.max(np.abs(cov / cov2) - 1) < 1e-10

    # Check it works if nl_cp is pass through cache
    cache = {}
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        ncell = cnmt.get_tracer_comb_ncell((trs[i-1], trs[j-1]))
        nl_arr = np.zeros((ncell, 96))

        if trs[i-1] == trs[j-1]:
            nl_arr[0] = nl_arr[-1] = tracer_noise_cp[trs[i-1]]

        cache[f'SN{i}{j}'] = nl_arr

    cov2 = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                     use_coupled_noise=True,
                                     cache=cache) + 1e-100
    assert np.max(np.abs(cov / cov2) - 1) < 1e-10
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Cov with uncoupled noise cannot be used for benchmark as tracer_noise is
    # assumed to be flat but it is not when computed from the coupled due to
    # edge effects. However, we can test it runs, at least it through cache
    # and compare the chi2
    cache = {}
    for i, j in [(1, 3), (2, 4), (1, 4), (2, 3)]:
        ncell = cnmt.get_tracer_comb_ncell((trs[i-1], trs[j-1]))
        nl_arr = np.zeros((ncell, 96))

        if trs[i-1] == trs[j-1]:
            nl_arr[0] = nl_arr[-1] = tracer_noise[trs[i-1]]

        cache[f'SN{i}{j}'] = nl_arr

    cov2 = cnmt.get_covariance_block(tracer_comb1, tracer_comb2,
                                     use_coupled_noise=False,
                                     cache=cache) + 1e-100
    if (tracer_comb1 == tracer_comb2) and ('DESgc__0' in tracer_comb1):
        # This test fails for weak lensing because there are orders of
        # magnitude between the coupled and decoupled noise that cannot be
        # reconciled by multiplying by <mask>.
        # TODO: Generalize this for weak lensing?

        # Only 1% accuracy since we are assuming a white decoupled noise, which
        # is not the case.
        assert_chi2(s, tracer_comb1, tracer_comb2, cov2, cov, 1e-2)

    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Check chi2, which is what we actually care about
    if tracer_comb1 == tracer_comb2:
        s = cnmt.io.get_sacc_file()
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-5)

    # Check that it runs if one of the masks does not overlap with the others
    if tracer_comb1 != tracer_comb2:
        os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")
        cnmt.mask_files[tracer_comb1[0]] = \
        './tests/benchmarks/32_DES_tjpcov_bm/catalogs/mask_nonoverlapping.fits.gz'
        cov = cnmt.get_covariance_block(tracer_comb1, tracer_comb2)
        os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")


def test_get_covariance_workspace():
    pass


def test_get_fields_dict():
    pass


def test_get_list_of_tracers_for_wsp():
    pass


def test_get_list_of_tracers_for_cov_wsp():
    pass


def test_get_list_of_tracers_for_cov_without_trs_wsp_cwsp():
    pass


def test_get_nell():
    pass


def test_get_workspace():
    pass


def test_get_workspaces_dict():
    pass


# Clean up after the tests
os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")
