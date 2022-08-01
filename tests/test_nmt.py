#!/usr/bin/python
import numpy as np
import os
import pymaster as nmt
import pytest
import tjpcov.main as cv
from tjpcov.parser import parse
import yaml
import sacc
import glob


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_yml_no_nmtc = os.path.join(root, "tjpcov_conf_minimal_no_nmtconf.yaml")
input_yml_txpipe = os.path.join(root, "tjpcov_conf_minimal_txpipe.yaml")
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
        return clfile['nl_cp'][0, -1]
    else:
        return clfile['nl'][0, 0]


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


def test_nmt_conf_missing():
    """
    Check that input file might not have nmt_conf and it still works
    """
    tjpcov_class = cv.CovarianceCalculator(input_yml_no_nmtc)

    ccl_tracers, tracer_noise = tjpcov_class.get_tracer_info(tjpcov_class.cl_data)

    tracer_comb1 = tracer_comb2 = ('DESgc__0', 'DESgc__0')

    cache = {'bins': get_nmt_bin()}

    cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        ccl_tracers, tracer_noise,
                                        cache=cache)['final'] + 1e-100
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")


def test_tracer_info():
    # Check it returns the correct coupled noise
    tjpcov_class = cv.CovarianceCalculator(input_yml)
    s = tjpcov_class.cl_data

    # Check you can pass the coupled noise with the sacc metadata
    for tr in s.tracers:
        nl_cp = get_tracer_noise(tr)
        s.tracers[tr].metadata['n_ell_coupled'] = nl_cp

    _, _, noise = tjpcov_class.get_tracer_info(s, return_noise_coupled=True)

    for tr in s.tracers:
        assert np.all(s.tracers[tr].metadata['n_ell_coupled'] == noise[tr])

    # Check that it will default to None if one of them is missing
    del s.tracers[tr].metadata['n_ell_coupled']
    _, _, noise = tjpcov_class.get_tracer_info(s, return_noise_coupled=True)

    assert noise is None




@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_nmt_gaussian_cov(tracer_comb1, tracer_comb2):
    # tjpcov_class = cv.CovarianceCalculator(input_yml)
    # cache = {'bins': get_nmt_bin()}

    config, _= parse(input_yml)
    bins = get_nmt_bin()
    config['tjpcov']['binning_info'] = bins
    tjpcov_class = cv.CovarianceCalculator(config)
    cache = None

    ccl_tracers, tracer_noise = tjpcov_class.get_tracer_info(tjpcov_class.cl_data)

    for tr in tracer_comb1 + tracer_comb2:
        tracer_noise[tr] = get_tracer_noise(tr)

    # Test error with uncoupled and coupled noise provided
    with pytest.raises(ValueError):
        cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                            ccl_tracers,
                                            tracer_Noise=tracer_noise,
                                            tracer_Noise_coupled=tracer_noise,
                                            cache=cache)['final']
        os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Cov with coupled noise (as in benchmark)
    cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        ccl_tracers,
                                        tracer_Noise_coupled=tracer_noise,
                                        cache=cache)['final'] + 1e-100

    cov_bm = get_benchmark_cov(tracer_comb1, tracer_comb2) + 1e-100

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-5
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-5

    # Test cov_tr1_tr2_tr3_tr4.npz cache
    fname = os.path.join('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/',
                         'cov_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                      *tracer_comb2))
    assert os.path.isfile(fname)
    cf = np.load(fname)
    for k in ['cov', 'final', 'final_b']:
        assert np.all(cf[k] + 1e-100 == cov)

    # Test you read it independently of what other arguments you pass
    cov2 = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                        None,
                                        tracer_Noise_coupled=None,
                                        cache=None)['final'] + 1e-100
    assert np.all(cov2 == cov)
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Test error with 'bins' in cache different to that at initialization
    with pytest.raises(ValueError):
        cache2 = {'bins': nmt.NmtBin.from_nside_linear(32, bins.get_n_bands())}
        cov2 = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                             ccl_tracers,
                                             tracer_Noise=tracer_noise,
                                             tracer_Noise_coupled=tracer_noise,
                                             cache=cache2)['final']
        os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Test it runs with 'bins' in cache if they are the same
    cache2 = {'bins': bins}
    cov2 = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                         ccl_tracers,
                                         tracer_Noise_coupled=tracer_noise,
                                         cache=cache2)['final'] + 1e-100
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    # Assert relative difference to an absurd precision because the equality
    # test fails now for some reason.
    assert np.max(np.abs(cov / cov2) - 1) < 1e-10

    # Cov with uncoupled noise cannot be used for benchmark as tracer_noise is
    # assumed to be flat but it is not when computed from the coupled due to
    # edge effects

    if tracer_comb1 == tracer_comb2:
        s = tjpcov_class.cl_data
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-5)

    # Check that it runs if one of the masks does not overlap with the others
    if tracer_comb1 != tracer_comb2:
        os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")
        tjpcov_class.mask_fn[tracer_comb1[0]] = \
        './tests/benchmarks/32_DES_tjpcov_bm/catalogs/mask_nonoverlapping.fits.gz'
        cov = tjpcov_class.nmt_gaussian_cov(tracer_comb1, tracer_comb2,
                                            ccl_tracers,
                                            tracer_Noise_coupled=tracer_noise,
                                            cache=cache)
        os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
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
                                        ccl_tracers, tracer_Noise_coupled=tracer_noise,
                                        cache=cache)['final'] + 1e-100
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

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
                                        ccl_tracers, tracer_Noise_coupled=tracer_noise,
                                        cache=cache)['final'] + 1e-100
    os.system("rm ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/cov*npz")

    assert np.max(np.abs(np.diag(cov) / np.diag(cov_bm) - 1)) < 1e-6
    assert np.max(np.abs(cov / cov_bm - 1)) < 1e-6
    if tracer_comb1 == tracer_comb2:
        s = tjpcov_class.cl_data
        assert_chi2(s, tracer_comb1, tracer_comb2, cov, cov_bm, 1e-6)

def test_get_all_cov_nmt():
    tjpcov_class = cv.CovarianceCalculator(input_yml)
    s = tjpcov_class.cl_data
    bins = get_nmt_bin()

    tracer_noise = {}
    for tr in s.tracers:
        tracer_noise[tr] = get_tracer_noise(tr)

    # Test error with uncoupled and coupled noise provided
    with pytest.raises(ValueError):
        cov = tjpcov_class.get_all_cov_nmt(tracer_noise=tracer_noise,
                                           tracer_noise_coupled=tracer_noise,
                                           cache={'bins': bins})

    cov = tjpcov_class.get_all_cov_nmt(tracer_noise_coupled=tracer_noise,
                                       cache={'bins': bins}) + 1e-100

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

    # Check that it also works if they don't use concise data_types
    s2 = s.copy()
    for dp in s2.data:
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

    tjpcov_class.cl_data = s2
    cov2 = tjpcov_class.get_all_cov_nmt(tracer_noise_coupled=tracer_noise,
                                        cache={'bins': bins}) + 1e-100
    assert np.all(cov == cov2)

    # Check you can pass the coupled noise with the sacc metadata
    for tr in s.tracers:
        nl_cp = get_tracer_noise(tr)
        s.tracers[tr].metadata['n_ell_coupled'] = nl_cp

    tjpcov_class.cl_data = s
    cov2 = tjpcov_class.get_all_cov_nmt(cache={'bins': bins}) + 1e-100
    assert np.all(cov == cov2)

    # Check that it will use the passed one instead of the one in the sacc file
    # if given
    for tr in s.tracers:
        nl_cp = get_tracer_noise(tr)
        s.tracers[tr].metadata['n_ell_coupled'] = nl_cp + 2
    cov2 = tjpcov_class.get_all_cov_nmt(tracer_noise_coupled=tracer_noise,
                                        cache={'bins': bins}) + 1e-100
    assert np.all(cov == cov2)

    # Clean after the test
    os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")


def test_txpipe_like_input():
    tjpcov_class = cv.CovarianceCalculator(input_yml_txpipe)
    s = tjpcov_class.cl_data

    tracer_noise = {}
    for tr in s.tracers:
        tracer_noise[tr] = get_tracer_noise(tr)

    # We don't need to pass the bins because we have provided the workspaces
    cov = tjpcov_class.get_all_cov_nmt(tracer_noise_coupled=tracer_noise) \
        + 1e-100

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

    # Clean up after the test
    os.system("rm -f ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/*")


# Clean up after the tests
os.system("rm -rf ./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/")
