#!/usr/bin/python3
import numpy as np
import pytest
import sacc

from functools import partial

from tjpcov.wigner_transform import bin_cov, WignerTransform
from tjpcov.covariance_builder import (
    CovarianceProjectedReal,
    CovarianceReal,
)


class CovarianceRealTester(CovarianceReal):
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

    def get_covariance_block(self, tracer_comb1, tracer_comb2):
        super().get_covariance_block(tracer_comb1, tracer_comb2)

    def _get_covariance_block_for_sacc(self, tracer_comb1, tracer_comb2):
        super()._get_covariance_block_for_sacc(tracer_comb1, tracer_comb2)


class CovarianceProjectedRealTester(CovarianceProjectedReal):
    fourier = None

    def _get_fourier_block(self, tracer_comb1, tracer_comb2):
        super().get_covariance_block(tracer_comb1, tracer_comb2)


@pytest.fixture
def cov_real():
    input_yml_real = "tests/data/conf_covariance_gaussian_fsky_real.yaml"
    cr = CovarianceRealTester(input_yml_real)
    return cr


@pytest.fixture
def cov_prj_real():
    input_yml_real = "tests/data/conf_covariance_gaussian_fsky_real.yaml"
    cr = CovarianceProjectedRealTester(input_yml_real)
    return cr


@pytest.fixture
def sacc_file():
    xi_fn = (
        "examples/old_api/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
    )
    sacc_file = sacc.Sacc.load_fits(xi_fn)
    return sacc_file


def test_get_theta_eff(cov_real, sacc_file):
    theta_eff, _ = sacc_file.get_theta_xi(
        "galaxy_shear_xi_plus", "src0", "src0"
    )
    assert np.all(theta_eff == cov_real.get_theta_eff())


def test_get_binning_info(cov_prj_real):
    # Check we recover the ell effective from the edges
    theta, theta_eff, theta_edges = cov_prj_real.get_binning_info(
        in_radians=False
    )

    assert np.all(theta_eff == cov_prj_real.get_theta_eff())
    assert np.allclose((theta_edges[1:] + theta_edges[:-1]) / 2, theta_eff)

    # Check in_radians work
    (
        theta2,
        theta_eff2,
        theta_edges2,
    ) = cov_prj_real.get_binning_info(in_radians=True)
    arcmin_rad = np.pi / 180 / 60
    assert np.all(theta * arcmin_rad == theta2)
    assert np.all(theta_eff * arcmin_rad == theta_eff2)
    assert np.all(theta_edges * arcmin_rad == theta_edges2)

    with pytest.raises(NotImplementedError):
        cov_prj_real.get_binning_info("linear")


@pytest.mark.parametrize(
    "tr1,tr2",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
def test_get_cov_WT_spin(cov_prj_real, tr1, tr2):
    spin = cov_prj_real.get_cov_WT_spin((tr1, tr2))

    spin2 = []
    for tr in [tr1, tr2]:
        if "lens" in tr:
            spin2.append(0)
        elif "src" in tr:
            spin2.append(2)

    if spin2 == [2, 2]:
        spin2 = {"plus": (2, 2), "minus": (2, -2)}
    else:
        spin2 = tuple(spin2)

    assert spin == spin2


def test_get_Wigner_transform(cov_prj_real):
    wt = cov_prj_real.get_Wigner_transform()

    assert isinstance(wt, WignerTransform)
    assert np.all(wt.ell == np.arange(2, cov_prj_real.lmax + 1))
    assert np.all(wt.theta == cov_prj_real.get_binning_info()[0])
    assert wt.s1_s2s == [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)]


def test_build_matrix_from_blocks(cov_prj_real):
    s = cov_prj_real.io.get_sacc_file()
    ndata = s.mean.size
    cov = np.random.rand(ndata, ndata)
    # Make it symmetric
    cov += cov.T

    tracers = cov_prj_real.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in tracers:
        ix1 = s.indices(tracers=trs1)
        ix2 = s.indices(tracers=trs2)
        blocks.append(cov[ix1][:, ix2])

    cov2 = cov_prj_real._build_matrix_from_blocks(blocks, tracers)
    assert np.all(cov == cov2)


@pytest.mark.parametrize(
    "tracer_comb1",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
@pytest.mark.parametrize(
    "tracer_comb2",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
def test_get_covariance_block(cov_prj_real, tracer_comb1, tracer_comb2):
    lmax = cov_prj_real.lmax
    ell = np.arange(2, lmax + 1)
    fourier_block = np.random.rand(lmax + 1, lmax + 1)

    # Dynamically override the method on this instance.
    def override_fourier_block(self, tracer_comb1, tracer_comb2):
        return fourier_block

    cov_prj_real._get_fourier_block = partial(
        override_fourier_block, cov_prj_real
    )

    WT = cov_prj_real.get_Wigner_transform()
    s1_s2_1 = cov_prj_real.get_cov_WT_spin(tracer_comb=tracer_comb1)
    s1_s2_2 = cov_prj_real.get_cov_WT_spin(tracer_comb=tracer_comb2)
    if isinstance(s1_s2_1, dict):
        s1_s2_1 = s1_s2_1["plus"]
    if isinstance(s1_s2_2, dict):
        s1_s2_2 = s1_s2_2["plus"]
    th, cov = WT.projected_covariance(
        ell_cl=ell,
        s1_s2=s1_s2_1,
        s1_s2_cross=s1_s2_2,
        cl_cov=fourier_block[2:][:, 2:],
    )

    gcov_xi_1 = cov_prj_real.get_covariance_block(
        tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, binned=False
    )

    assert np.max(np.abs((gcov_xi_1 + 1e-100) / (cov + 1e-100) - 1)) < 1e-5

    gcov_xi_1 = cov_prj_real.get_covariance_block(
        tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, binned=True
    )

    theta, _, theta_edges = cov_prj_real.get_binning_info(in_radians=False)
    _, cov = bin_cov(r=theta, r_bins=theta_edges, cov=cov)
    assert gcov_xi_1.shape == (20, 20)
    assert np.max(np.abs((gcov_xi_1 + 1e-100) / (cov + 1e-100) - 1)) < 1e-5


@pytest.mark.parametrize(
    "tracer_comb1,tracer_comb2",
    [
        ("lens0", "lens0"),
        ("src0", "lens0"),
        ("lens0", "src0"),
        ("src0", "src0"),
    ],
)
def test__get_covariance_block_for_sacc(
    cov_prj_real, tracer_comb1, tracer_comb2
):
    lmax = cov_prj_real.lmax
    fourier_block = np.random.rand(lmax + 1, lmax + 1)

    # Dynamically override the method on this instance.
    def override_fourier_block(self, tracer_comb1, tracer_comb2):
        if tracer_comb1 == tracer_comb2:
            return fourier_block + fourier_block.T
        return fourier_block

    cov_prj_real._get_fourier_block = partial(
        override_fourier_block, cov_prj_real
    )

    nbpw = cov_prj_real.get_nbpw()
    cov = cov_prj_real._get_covariance_block_for_sacc(
        tracer_comb1, tracer_comb2
    )

    s = cov_prj_real.io.get_sacc_file()
    ix1 = s.indices(tracers=tracer_comb1)
    ix2 = s.indices(tracers=tracer_comb2)
    assert (ix1.size, ix2.size) == cov.shape

    dt1 = cov_prj_real.get_tracer_comb_data_types(tracer_comb1)
    dt2 = cov_prj_real.get_tracer_comb_data_types(tracer_comb2)

    cov = cov.reshape((nbpw, len(dt1), nbpw, len(dt2)))

    pm = ["plus", "minus"]

    for i, dt1i in enumerate(dt1):
        for j, dt2j in enumerate(dt2):
            assert np.all(
                cov[:, i, :, j]
                == cov_prj_real.get_covariance_block(
                    tracer_comb1,
                    tracer_comb2,
                    pm["minus" in dt1i],
                    pm["minus" in dt2j],
                )
            )
