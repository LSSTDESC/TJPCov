import os

import numpy as np
import pymaster as nmt
import pytest
import sacc
import shutil
import pyccl as ccl

from tjpcov.covariance_builder import CovarianceFourier

ROOT = "./tests/benchmarks/32_DES_tjpcov_bm/"
OUTDIR = "./tests/tmp/"
INPUT_YML = "./tests/data/conf_covariance_gaussian_fourier_nmt.yaml"


@pytest.fixture
def mock_sacc():
    return sacc.Sacc.load_fits(ROOT + "cls_cov.fits")


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)


def teardown_module():
    shutil.rmtree(OUTDIR)


class CovarianceFourierTester(CovarianceFourier):
    # Based on https://stackoverflow.com/a/28299369
    def get_covariance_block(self, **kwargs):
        super().get_covariance_block(**kwargs)


@pytest.fixture
def mock_cov_fourier():
    return CovarianceFourierTester(INPUT_YML)


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


def get_nmt_bin(lmax=95):
    bpw_edges = np.array(
        [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    )
    if lmax != 95:
        # lmax + 1 because the upper edge is not included
        bpw_edges = bpw_edges[bpw_edges < lmax + 1]
        bpw_edges[-1] = lmax + 1

    return nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def test_build_matrix_from_blocks():
    class CFT(CovarianceFourierTester):
        def get_covariance_block(self, trs1, trs2):
            fname = "cov_{}_{}_{}_{}.npz".format(*trs1, *trs2)
            return np.load(os.path.join(ROOT + "cov", fname))["cov"]

    cb = CFT(INPUT_YML)
    s = cb.io.get_sacc_file()
    cov = s.covariance.covmat + 1e-100
    trs_cov = cb.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        cov12 = cb._get_covariance_block_for_sacc(trs1, trs2)
        blocks.append(cov12)

    cov2 = cb._build_matrix_from_blocks(blocks, trs_cov) + 1e-100
    assert np.max(np.abs(cov / cov2 - 1)) < 1e-10


def test__get_covariance_block_for_sacc():
    # Test with matrices ordered as in C
    class CFT(CovarianceFourierTester):
        def get_covariance_block(self, trs1, trs2):
            fname = "cov_{}_{}_{}_{}.npz".format(*trs1, *trs2)
            return np.load(os.path.join(ROOT + "cov", fname))["cov"]

    cb = CFT(INPUT_YML)
    s = cb.io.get_sacc_file()
    cov = s.covariance.covmat + 1e-100

    # Now test all of them
    trs_cov = cb.get_list_of_tracers_for_cov()
    for trs1, trs2 in trs_cov:
        ix1 = s.indices(tracers=trs1)
        ix2 = s.indices(tracers=trs2)
        cov1 = cov[ix1][:, ix2]
        cov2 = cb._get_covariance_block_for_sacc(trs1, trs2) + 1e-100

        assert cov1.shape == cov2.shape
        assert np.max(np.abs(cov1 / cov2 - 1)) < 1e-10

        # Make sure we get what it's in the files looking at the first block
        # i.e 00, 0e or ee

        ncell1 = cb.get_tracer_comb_ncell(trs1)
        ncell2 = cb.get_tracer_comb_ncell(trs2)

        cov1 = cov[ix1[:16]][:, ix2[:16]]
        cov2 = cb.get_covariance_block(trs1, trs2) + 1e-100
        cov2 = cov2.reshape(16, ncell1, 16, ncell2)[:, 0, :, 0]

        assert np.max(np.abs(cov1 / cov2 - 1)) < 1e-10


def test_get_datatypes_from_ncell(mock_cov_fourier):
    with pytest.raises(ValueError):
        mock_cov_fourier.get_datatypes_from_ncell(0)

    with pytest.raises(ValueError):
        mock_cov_fourier.get_datatypes_from_ncell(3)

    assert mock_cov_fourier.get_datatypes_from_ncell(1) == ["cl_00"]
    assert mock_cov_fourier.get_datatypes_from_ncell(2) == ["cl_0e", "cl_0b"]
    assert mock_cov_fourier.get_datatypes_from_ncell(4) == [
        "cl_ee",
        "cl_eb",
        "cl_be",
        "cl_bb",
    ]


def test_get_ell_eff(mock_cov_fourier):
    # Using this because the data was generated with NaMaster. We could read
    # the sacc file but then we would be doing the same as in the code.
    bins = get_nmt_bin()
    ells = bins.get_effective_ells()

    assert np.all(mock_cov_fourier.get_ell_eff() == ells)


def test_get_sacc_with_concise_dtypes(mock_sacc, mock_cov_fourier):
    s = mock_sacc.copy()
    for dp in s.data:
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

    s2 = mock_cov_fourier.get_sacc_with_concise_dtypes()
    dtypes = mock_sacc.get_data_types()
    dtypes2 = s2.get_data_types()
    assert dtypes == dtypes2

    for dp, dp2 in zip(mock_sacc.data, s2.data):
        assert dp.data_type == dp2.data_type
        assert dp.value == dp2.value
        assert dp.tracers == dp2.tracers
        for k in dp.tags:
            if k == "window":
                # Don't check window as it points to a different memory address
                continue
            assert dp.tags[k] == dp2.tags[k]


def test_get_tracer_comb_ncell(mock_cov_fourier):
    # Use dummy file to test for cmb_convergence too
    mock_cov_fourier.io.sacc_file = get_dummy_sacc()

    assert mock_cov_fourier.get_tracer_comb_ncell(("PLAcv", "PLAcv")) == 1
    assert mock_cov_fourier.get_tracer_comb_ncell(("PLAcv", "DESgc__0")) == 1
    assert (
        mock_cov_fourier.get_tracer_comb_ncell(("DESgc__0", "DESgc__0")) == 1
    )
    assert mock_cov_fourier.get_tracer_comb_ncell(("PLAcv", "DESwl__0")) == 2
    assert (
        mock_cov_fourier.get_tracer_comb_ncell(("DESgc__0", "DESwl__0")) == 2
    )
    assert (
        mock_cov_fourier.get_tracer_comb_ncell(("DESwl__0", "DESwl__0")) == 4
    )
    assert (
        mock_cov_fourier.get_tracer_comb_ncell(
            ("DESwl__0", "DESwl__0"), independent=True
        )
        == 3
    )


def test_get_tracer_info(mock_cov_fourier):
    ccl_tracers1, tracer_noise1 = mock_cov_fourier.get_tracer_info()
    (
        ccl_tracers,
        tracer_noise,
        tracer_noise_coupled,
    ) = mock_cov_fourier.get_tracer_info(return_noise_coupled=True)

    # Check that when returnig the coupled noise, the previous output is the
    # same and the tracer_noise_coupled is a dictionary with all values None,
    # as no coupled noise info has been passed.
    assert ccl_tracers is ccl_tracers1
    assert tracer_noise is tracer_noise1
    assert tracer_noise.keys() == tracer_noise_coupled.keys()
    assert not any(tracer_noise_coupled.values())

    # Check noise from formula
    arc_min = 1 / 60 * np.pi / 180  # arc_min in radians
    Ngal = 26 / arc_min**2  # Number galaxy density
    sigma_e = 0.26

    for tr, nl in tracer_noise.items():
        if "gc" in tr:
            assert np.abs(nl / (1 / Ngal) - 1) < 1e-5
        else:
            assert np.abs(nl / (sigma_e**2 / Ngal) - 1) < 1e-5

    sacc = mock_cov_fourier.io.sacc_file
    cosmo = mock_cov_fourier.get_cosmology()
    for tr, ccltr in ccl_tracers.items():
        trsacc = sacc.tracers[tr]
        if "gc" in tr:
            z, nz = (trsacc.z, trsacc.nz)
            assert nz == pytest.approx(ccltr.get_dndz(z), rel=1e-4)
            chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + z))
            kz = ccltr.get_kernel(chi)[0]
            z_max_nz = z[np.argmax(nz)]
            z_max_kz = z[np.argmax(kz)]
            assert z_max_nz == pytest.approx(z_max_kz, rel=1e-2)
        elif "wl" in tr:
            z, nz = (trsacc.z, trsacc.nz)
            assert nz == pytest.approx(ccltr.get_dndz(z), rel=1e-4)
            chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + z))
            kz = ccltr.get_kernel(chi)[0]
            z_max_nz = z[np.argmax(nz)]
            z_max_kz = z[np.argmax(kz)]
            assert z_max_kz < z_max_nz
        elif "cv" in tr:
            assert not isinstance(ccltr, ccl.tracers.NzTracer)
            chi_max = ccltr.chi_max
            chi_max2 = ccl.comoving_radial_distance(cosmo, 1 / (1 + 1100))
            assert chi_max == pytest.approx(chi_max2, rel=1e-4)

    # Check tracer_noise_coupled. Modify the sacc file to add metadata
    # information for the tracer noise
    cb = CovarianceFourierTester(INPUT_YML)
    cb.io.get_sacc_file()  # To have the sacc file stored it in cb.io.sacc_file
    coupled_noise = {}
    for i, tr in enumerate(cb.io.sacc_file.tracers.keys()):
        coupled_noise[tr] = i
        cb.io.sacc_file.tracers[tr].metadata["n_ell_coupled"] = i

    ccl_tracers, tracer_noise, tracer_noise_coupled = cb.get_tracer_info(
        return_noise_coupled=True
    )

    for tr, nl in tracer_noise_coupled.items():
        assert coupled_noise[tr] == nl
