import healpy as hp
import os
import pytest
import numpy as np
import sacc
import shutil
import pickle
import pyccl as ccl
import pymaster as nmt

from tjpcov.covariance_builder import CovarianceBuilder
from scipy.linalg import block_diag

input_yml = "./tests/data/conf_covariance_builder_minimal.yaml"
input_sacc = sacc.Sacc.load_fits(
    "./tests/benchmarks/32_DES_tjpcov_bm/cls_cov.fits"
)
outdir = "tests/tmp/"

# Create temporal folder
os.makedirs(outdir, exist_ok=True)


def get_covariance_block(tracer_comb1, tracer_comb2, **kwargs):
    f1 = int(tracer_comb1[0].split("__")[1]) + 1
    f2 = int(tracer_comb1[1].split("__")[1]) + 1
    f3 = int(tracer_comb2[0].split("__")[1]) + 1
    f4 = int(tracer_comb2[1].split("__")[1]) + 1

    block = f1 * f2 * f3 * f4 * np.ones((10, 10))
    return block


class CovarianceBuilderTester(CovarianceBuilder):
    _tracer_types = ["cl", "cl"]

    # Based on https://stackoverflow.com/a/28299369
    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        super().get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        super()._get_covariance_block_for_sacc(
            tracer_comb1, tracer_comb2, **kwargs
        )


def get_nmt_bin(lmax=95):
    bpw_edges = np.array(
        [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    )
    if lmax != 95:
        # lmax + 1 because the upper edge is not included
        bpw_edges = bpw_edges[bpw_edges < lmax + 1]
        bpw_edges[-1] = lmax + 1

    return nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def clean_tmp():
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir)


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    clean_tmp()


def test_smoke():
    CovarianceBuilderTester(input_yml)


def test_nuisance_config():
    cb = CovarianceBuilderTester(input_yml)
    assert cb.bias_lens == {"DESgc__0": 1.48}
    assert cb.IA is None
    Ngal = 26 * 3600 / (np.pi / 180) ** 2
    assert cb.Ngal == {"DESgc__0": Ngal, "DESwl__0": Ngal, "DESwl__1": Ngal}
    assert cb.sigma_e == {"DESwl__0": 0.26, "DESwl__1": 0.26}


# Tested also in tests/test_mpi.py
def test_split_tasks_by_rank():
    cb = CovarianceBuilderTester(input_yml)
    tasks = list(range(100))
    tasks_splitted = list(cb._split_tasks_by_rank(tasks))

    assert tasks == tasks_splitted

    # Fake a mpi process with size = 2 and rank = 2
    cb.size = 2
    cb.rank = 1

    tasks_splitted = list(cb._split_tasks_by_rank(tasks))

    assert tasks[1::2] == tasks_splitted


def test_compute_all_blocks():
    class CovarianceBuilderTester(CovarianceBuilder):
        _tracer_types = ["cl", "cl"]

        # Based on https://stackoverflow.com/a/28299369
        def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
            super().get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

        def _get_covariance_block_for_sacc(
            self, tracer_comb1, tracer_comb2, **kwargs
        ):
            return get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

    cb = CovarianceBuilderTester(input_yml)
    blocks, tracers_blocks = cb._compute_all_blocks()
    nblocks = len(cb.get_list_of_tracers_for_cov())
    assert nblocks == len(blocks)

    for bi, trs in zip(blocks, tracers_blocks):
        assert np.all(bi == get_covariance_block(trs[0], trs[1]))


def test_get_cosmology():
    # Check that it reads the parameters from the yaml file
    cb = CovarianceBuilderTester(input_yml)
    config = cb.config.copy()
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check that it uses the cosmology if given
    cosmo = ccl.CosmologyVanillaLCDM()
    config["tjpcov"]["cosmo"] = cosmo
    cb = CovarianceBuilderTester(config)
    assert cb.get_cosmology() is cosmo

    # Check that it reads a cosmology from a yml file
    config["tjpcov"]["cosmo"] = "./tests/data/cosmo_desy1.yaml"
    cb = CovarianceBuilderTester(config)
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check it reads pickles too
    fname = os.path.join(outdir, "cosmos_desy1.pkl")
    with open(fname, "wb") as ff:
        pickle.dump(cosmo, ff)

    config["tjpcov"]["cosmo"] = fname
    cb = CovarianceBuilderTester(config)
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check that any other thing rises an error

    with pytest.raises(ValueError):
        config["tjpcov"]["cosmo"] = ["hello"]
        cb = CovarianceBuilderTester(config)
        cb.get_cosmology()


def test_get_covariance_block_not_implemented():
    with pytest.raises(NotImplementedError):
        cb = CovarianceBuilderTester(input_yml)
        cb.get_covariance_block([], [])


def test_get_covariance():
    def build_matrix_from_blocks(blocks, tracers_cov):
        tracers_cov_sorted = sorted(tracers_cov)
        ix = []
        for trs in tracers_cov_sorted:
            ix.append(tracers_cov.index(trs))
        blocks = list(np.array(blocks)[ix])
        return block_diag(*blocks)

    class CovarianceBuilderTester(CovarianceBuilder):
        _tracer_types = ["cl", "cl"]

        # Based on https://stackoverflow.com/a/28299369
        def _build_matrix_from_blocks(self, blocks, tracers_cov):
            return build_matrix_from_blocks(blocks, tracers_cov)

        def get_covariance_block(self, **kwargs):
            super().get_covariance_block(**kwargs)

        def _get_covariance_block_for_sacc(
            self, tracer_comb1, tracer_comb2, **kwargs
        ):
            return get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

    cb = CovarianceBuilderTester(input_yml)
    blocks, tracers_blocks = cb._compute_all_blocks()
    cov = cb.get_covariance()
    cov2 = build_matrix_from_blocks(blocks[::-1], tracers_blocks[::-1])

    assert np.all(cov2 == cov)

    # Check that a ValueError is rised if the covariance is full of 0's
    cb._tracer_types = ["cluster", "cluster"]
    cb.cov = None  # To remove the stored value
    with pytest.raises(ValueError):
        cb.get_covariance()


def test_get_covariance_block_for_sacc():
    class CovarianceBuilderTester(CovarianceBuilder):
        _tracer_types = ["cl", "cluster"]

        # Based on https://stackoverflow.com/a/28299369
        def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
            super().get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

        def _get_covariance_block_for_sacc(
            self, tracer_comb1, tracer_comb2, **kwargs
        ):
            return get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

    # Test it return 0's when the data types are not those of the class
    cb = CovarianceBuilderTester(input_yml)
    trs_cov = cb.get_list_of_tracers_for_cov()[0]
    cov = cb.get_covariance_block_for_sacc(*trs_cov)
    assert not np.any(cov)
    s = cb.io.get_sacc_file()
    ix1 = s.indices(tracers=trs_cov[0])
    ix2 = s.indices(tracers=trs_cov[1])
    cov2 = np.zeros((ix1.size, ix2.size))
    assert np.all(cov2 == cov)

    # Test it return _get_covariance_block_for_sacc  when the data types are
    # those of the class
    cb._tracer_types = ["cl", "cl"]
    cov = cb.get_covariance_block_for_sacc(*trs_cov)
    cov2 = get_covariance_block(*trs_cov)
    assert np.all(cov == cov2)

    # Check that if the order of the tracers are the opposite as in
    # _tracer_types, it computes the covariance
    class CovarianceBuilderTester(CovarianceBuilder):
        _tracer_types = ["cl", "cluster"]

        # Based on https://stackoverflow.com/a/28299369
        def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
            super().get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

        def _get_covariance_block_for_sacc(
            self, tracer_comb1, tracer_comb2, **kwargs
        ):
            return get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)

        def get_tracer_comb_data_types(self, tracer_comb):
            if tracer_comb == trs_cov[0]:
                return ["cluster"]
            elif tracer_comb == trs_cov[1]:
                return ["cl"]

    cb = CovarianceBuilderTester(input_yml)
    cov = cb.get_covariance_block_for_sacc(*trs_cov)
    cov2 = get_covariance_block(*trs_cov)
    assert np.all(cov == cov2)


def test_get_list_of_tracers_for_cov():
    cb = CovarianceBuilderTester(input_yml)
    trs_cov = cb.get_list_of_tracers_for_cov()

    # Test all tracers
    trs_cov2 = []
    tracers = cb.io.get_sacc_file().get_tracer_combinations()
    for i, trs1 in enumerate(tracers):
        for trs2 in tracers[i:]:
            trs_cov2.append((trs1, trs2))

    assert trs_cov == trs_cov2


def test_get_mask_names_dict():
    tracer_names = {1: "DESwl__0", 2: "DESgc__0", 3: "DESwl__1", 4: "DESwl__1"}
    cb = CovarianceBuilderTester(input_yml)
    mn = cb.get_mask_names_dict(tracer_names)

    assert isinstance(mn, dict)
    for i, mni in mn.items():
        tni = tracer_names[i]
        assert mni == cb.config["tjpcov"]["mask_names"][tni]


def test_get_masks_dict():
    tracer_names = {1: "DESwl__0", 2: "DESgc__0", 3: "DESwl__1", 4: "DESwl__1"}
    cb = CovarianceBuilderTester(input_yml)
    m = cb.get_masks_dict(tracer_names)

    assert isinstance(m, dict)
    for i, mni in m.items():
        tni = tracer_names[i]
        assert np.all(
            mni == hp.read_map(cb.config["tjpcov"]["mask_file"][tni])
        )

    mi = np.arange(100)
    cache = {f"m{i+1}": mi + i for i in range(4)}
    m = cb.get_masks_dict(tracer_names, cache)
    for i in range(4):
        assert np.all(m[i + 1] == mi + i)


def test_get_nbpw():
    cb = CovarianceBuilderTester(input_yml)
    assert cb.get_nbpw() == 16


def test_get_tracers_spin_dict():
    tracer_names = {1: "DESwl__0", 2: "DESgc__0", 3: "DESwl__1", 4: "DESwl__1"}
    cb = CovarianceBuilderTester(input_yml)
    s = cb.get_tracers_spin_dict(tracer_names)

    assert s == {1: 2, 2: 0, 3: 2, 4: 2}


def test_get_tracer_comb_spin():
    tracer_comb = ["DESwl__0", "DESgc__0"]
    cb = CovarianceBuilderTester(input_yml)
    assert cb.get_tracer_comb_spin(tracer_comb) == (2, 0)


def test_get_tracer_comb_data_types():
    cb = CovarianceBuilderTester(input_yml)

    tracer_comb = ["DESgc__0", "DESgc__0"]
    assert cb.get_tracer_comb_data_types(tracer_comb) == ["cl_00"]

    tracer_comb = ["DESgc__0", "DESwl__0"]
    assert cb.get_tracer_comb_data_types(tracer_comb) == ["cl_0e", "cl_0b"]

    tracer_comb = ["DESwl__0", "DESwl__0"]
    assert cb.get_tracer_comb_data_types(tracer_comb) == [
        "cl_ee",
        "cl_eb",
        "cl_bb",
    ]

    tracer_comb = ["DESwl__0", "DESwl__1"]
    assert cb.get_tracer_comb_data_types(tracer_comb) == [
        "cl_ee",
        "cl_eb",
        "cl_be",
        "cl_bb",
    ]


@pytest.mark.parametrize("tr", ["DESwl__0", "DESgc__0"])
def test_get_tracer_nmaps(tr):
    cb = CovarianceBuilderTester(input_yml)
    nmap = 2 if tr == "DESwl__0" else 1
    assert cb.get_tracer_nmaps(tr) == nmap
