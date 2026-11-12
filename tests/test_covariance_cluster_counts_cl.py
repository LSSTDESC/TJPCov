#!/usr/bin/python3
import copy

import jinja2
import numpy as np
import pyccl as ccl
import pytest
import sacc
import yaml
from scipy.integrate import simpson

from conftest import INPUT_YML, cap_mask
from tjpcov.covariance_calculator import CovarianceCalculator
from tjpcov.covariance_cluster_counts_cl import (
    ClusterClcNGHaloModel,
    ClusterClSSCHaloModel,
)
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC

pytestmark = pytest.mark.skipif(
    not hasattr(ccl, "angular_cl_cov_counts"),
    reason="pyccl without C_ell x cluster counts covariances",
)

SURVEY_NAME = "NC_mock_redshift_richness"
CL_COMBS = [("src0", "src0"), ("lens0", "src0"), ("lens0", "lens0")]
CLUSTER_COMB = (SURVEY_NAME, "bin_rich_1", "bin_z_2")
HOD_YML = "./tests/data/conf_covariance_cNG_fsky.yaml"


def _load_yaml(fname):
    with open(fname, "r") as fp:
        return yaml.load(
            jinja2.Template(fp.read()).render(), Loader=yaml.Loader
        )


@pytest.fixture(scope="module")
def config(save_cluster_sacc_data):
    """Config with a sacc file with cluster data, shear, galaxy-galaxy
    lensing and galaxy clustering C_ell."""
    config = _load_yaml(INPUT_YML)
    s = save_cluster_sacc_data.copy()
    s.covariance = None
    z = np.linspace(0, 2, 100)
    for name, zm, q in [
        ("src0", 0.8, "galaxy_shear"),
        ("lens0", 0.4, "galaxy_density"),
    ]:
        nz = np.exp(-0.5 * ((z - zm) / 0.1) ** 2)
        s.add_tracer("NZ", name, z, nz, quantity=q)
    for dt, comb in zip(
        [
            "galaxy_shear_cl_ee",
            "galaxy_shearDensity_cl_e",
            "galaxy_density_cl",
        ],
        CL_COMBS,
    ):
        s.add_ell_cl(dt, *comb, [100, 300, 500], np.ones(3))
    s.add_covariance(np.diag(np.ones(s.mean.size)))
    config["tjpcov"]["sacc_file"] = s
    config["HOD"] = _load_yaml(HOD_YML)["HOD"]
    config["ClusterCl"] = {"nz": 16, "na": 400}
    return config


@pytest.fixture(scope="module")
def cov_cng(config):
    return ClusterClcNGHaloModel(config)


@pytest.fixture(scope="module")
def cov_ssc(config):
    return ClusterClSSCHaloModel(config)


def test_bin_means(config, cov_ssc):
    # Model counts must agree with those of the cluster counts covariance
    cov_ssc.get_bin_profiles(1, 2)
    N, mean_log_m = cov_ssc.bin_means[(1, 2)]
    shot_noise = ClusterCountsGaussian(config).shot_noise(2, 1)
    assert N == pytest.approx(shot_noise, rel=1e-2)
    assert 13 < mean_log_m < 15


def test_bin_means_selection(config, cov_ssc):
    # Completeness and purity enter the selection as for the cluster counts
    config = copy.copy(config)
    config["completeness"] = {"model": "Aguena16"}
    config["purity"] = {"model": "Aguena16"}
    cov = ClusterClcNGHaloModel(config)
    cov.get_bin_profiles(1, 2)
    N, _ = cov.bin_means[(1, 2)]
    shot_noise = ClusterCountsGaussian(config).shot_noise(2, 1)
    assert N == pytest.approx(shot_noise, rel=1e-2)
    cov_ssc.get_bin_profiles(1, 2)
    assert N != pytest.approx(cov_ssc.bin_means[(1, 2)][0], rel=1e-2)


def test_counts_ssc_responses(config, cov_ssc):
    # Responses of the counts and <log M> to the long mode of the cluster
    # counts SSC agree with those of the C_ell x cluster SSC
    ssc = ClusterCountsSSC(config)
    ssc.super_sample_covariance()
    a, profs = cov_ssc.get_bin_profiles(1, 2)
    nb = [
        cov_ssc.integrate_volume(
            a, lambda aa: cov_ssc.hmc.I_1_1(cov_ssc.cosmo, 1e-4, aa, prof)
        )
        for prof in profs
    ]
    assert ssc.Nb[1, 2] == pytest.approx(nb[0], rel=1e-2)
    assert ssc.Nb_mean_log_mass[1, 2] == pytest.approx(nb[1], rel=1e-2)


def test_tracer_profiles(cov_ssc):
    for comb, key in zip(
        CL_COMBS, [(False, False, False), (True, False, False)]
    ):
        assert cov_ssc.get_tracer_profiles(comb)[0] == key
    key, profs, prof_2pt = cov_ssc.get_tracer_profiles(("lens0", "lens0"))
    assert key == (True, True, True)
    assert all(p is cov_ssc.hod for p in profs)
    assert isinstance(prof_2pt, ccl.halos.Profile2ptHOD)


@pytest.mark.parametrize("comb", [("src0", "src0"), ("lens0", "lens0")])
def test_ssc_response(cov_ssc, comb):
    # Response of P(k) to a long mode, also for number counts (HOD)
    cosmo, hmc = cov_ssc.cosmo, cov_ssc.hmc
    _, (prof, prof2), prof_2pt = cov_ssc.get_tracer_profiles(comb)
    a, k = 1 / 1.39, np.array([0.05, 0.2, 1.0])
    tk = ccl.halos.halomod_Tk3D_SSC(
        cosmo,
        hmc,
        prof=prof,
        prof2=prof2,
        prof12_2pt=prof_2pt,
        a_arr=np.linspace(0.3, 1, 32),
    )
    resp = cov_ssc.get_response(comb)
    assert resp(k, a) == pytest.approx(np.sqrt(np.diag(tk(k, a))), rel=1e-4)


def test_one_galaxy_term(cov_cng):
    # Galaxy clustering autocorrelations include the one-galaxy term
    cosmo, hmc = cov_cng.cosmo, cov_cng.hmc
    a_bin, (prof, _) = cov_cng.get_bin_profiles(1, 2)
    a, k = a_bin[8], np.array([0.05, 0.2, 1.0])
    _, profs, prof_2pt = cov_cng.get_tracer_profiles(("lens0", "lens0"))
    bk = ccl.halos.halomod_bispectrum_squeezed(
        cosmo, hmc, k, a, prof, prof2=profs[0], prof23_2pt=prof_2pt
    )
    ng = cov_cng.hod.get_normalization(cosmo, a, hmc=hmc)
    one_gal = hmc.integrate_over_massfunc(
        lambda M: prof.selection(M, a)
        * cov_cng.hod.fourier(cosmo, 1e-4, M, a),
        cosmo,
        a,
    )
    bk_cov = cov_cng.get_bin_bispectra(1, 2, ("lens0", "lens0"))[0](k, a)
    assert one_gal > 0
    assert bk_cov == pytest.approx(bk + one_gal / ng**2, rel=1e-4)


@pytest.mark.parametrize("comb", CL_COMBS)
def test_ssc_cauchy_schwarz(config, cov_ssc, comb):
    # |Corr_SSC(N, C_ell)| <= 1, with the C_ell SSC variance computed with
    # the same projection of the long mode. This checks the normalization.
    cosmo, ell = cov_ssc.cosmo, np.array([200.0])
    var_n = ClusterCountsSSC(config).super_sample_covariance()[1, 1, 2, 2]

    a = np.linspace(cov_ssc.a_arr[0], 1, 200)
    chi = ccl.comoving_radial_distance(cosmo, a)
    w = cov_ssc.get_shell_window(a)
    pk0 = ccl.linear_matter_power(cosmo, cov_ssc.k_arr, 1.0)
    ww = np.einsum("akl,bkl,l->abk", w, w, cov_ssc.get_ell_weights())
    sigma2 = simpson(cov_ssc.k_arr**2 * pk0 * ww, x=cov_ssc.k_arr, axis=-1)
    sigma2 /= (2 * np.pi) ** 3

    tr1, tr2 = [cov_ssc.get_ccl_tracer(tr) for tr in comb]
    resp = cov_ssc.get_response(comb)
    chi_use = np.maximum(chi, 1)
    f = np.array(
        [resp((ell[0] + 0.5) / c, aa) for c, aa in zip(chi_use, a)]
    ).squeeze()
    f *= tr1.get_kernel(chi)[0] * tr2.get_kernel(chi)[0] / chi_use**2
    f *= np.abs(np.gradient(chi))
    var_cl = f @ sigma2 @ f

    cov = cov_ssc.get_covariance_block(comb, CLUSTER_COMB, ell=ell)[0, 0]
    assert cov != 0
    assert np.abs(cov) / np.sqrt(var_n * var_cl) < 1


def test_ssc_mask(config, cov_ssc):
    # With masks of the spherical cap of the survey area, the covariance is
    # that of the spherical cap
    cap = cap_mask(cov_ssc.survey_area)
    names = [SURVEY_NAME, "src0", "lens0"]
    config = copy.copy(config)
    config["tjpcov"] = dict(
        config["tjpcov"],
        mask_file={name: cap for name in names},
        mask_names={name: name for name in names},
    )
    config["ClusterCountsSSC"] = {"L": cov_ssc.L}
    cov = ClusterClSSCHaloModel(config)
    for comb in CL_COMBS:
        assert cov.get_covariance_block(comb, CLUSTER_COMB) == pytest.approx(
            cov_ssc.get_covariance_block(comb, CLUSTER_COMB), rel=2e-2
        )

    # The weights are the cross power spectrum of the cluster mask and the
    # product of the masks of the C_ell tracers
    half = cap * (np.arange(cap.size) % 2)
    cov.mask_files["src0"] = half
    assert cov.get_ell_weights_cl(("lens0", "src0")) == pytest.approx(
        cov.get_mask_wl(cap, cap * half)
    )


@pytest.mark.parametrize("cov_name", ["cov_cng", "cov_ssc"])
@pytest.mark.parametrize("comb", CL_COMBS)
def test_get_covariance_block_for_sacc(cov_name, comb, request):
    cov = request.getfixturevalue(cov_name)
    s = cov.io.get_sacc_file()

    block = cov.get_covariance_block_for_sacc(comb, CLUSTER_COMB)
    # Columns are in the sacc order of the cluster data types
    dtypes = [s.data[i].data_type for i in s.indices(tracers=CLUSTER_COMB)]
    expected = cov.get_covariance_block(comb, CLUSTER_COMB)
    assert block.shape == (3, 2)
    assert block[:, dtypes.index("cluster_counts")] == pytest.approx(
        expected[:, 0]
    )
    assert block[:, dtypes.index("cluster_mean_log_mass")] == pytest.approx(
        expected[:, 1]
    )
    assert np.all(block != 0)
    assert np.all(
        cov.get_covariance_block_for_sacc(CLUSTER_COMB, comb) == block.T
    )

    # No contribution to C_ell x C_ell or clusters x clusters
    assert not np.any(cov.get_covariance_block_for_sacc(comb, comb))
    assert not np.any(
        cov.get_covariance_block_for_sacc(CLUSTER_COMB, CLUSTER_COMB)
    )


@pytest.mark.parametrize(
    "cov_class", [ClusterClcNGHaloModel, ClusterClSSCHaloModel]
)
def test_get_covariance_block_for_sacc_bpw(config, cov_class):
    # With bandpower windows, the covariance is that of the bandpowers
    ell = np.arange(600)
    weight = np.exp(
        -0.5 * ((ell[:, None] - np.array([100, 300, 500])) / 40) ** 2
    )
    weight /= weight.sum(axis=0)
    bpw = sacc.BandpowerWindow(ell, weight)

    s = config["tjpcov"]["sacc_file"].copy()
    s.covariance = None
    s.remove_selection(data_type="galaxy_shear_cl_ee")
    s.add_ell_cl(
        "galaxy_shear_cl_ee", *CL_COMBS[0], [100, 300, 500], np.ones(3), bpw
    )
    s.add_covariance(np.diag(np.ones(s.mean.size)))
    config = copy.copy(config)
    config["tjpcov"] = dict(config["tjpcov"], sacc_file=s)
    cov = cov_class(config)

    block = cov.get_covariance_block_for_sacc(CL_COMBS[0], CLUSTER_COMB)
    dtypes = [s.data[i].data_type for i in s.indices(tracers=CLUSTER_COMB)]
    block = block[
        :,
        [
            dtypes.index("cluster_counts"),
            dtypes.index("cluster_mean_log_mass"),
        ],
    ]
    expected = weight.T @ cov.get_covariance_block(
        CL_COMBS[0], CLUSTER_COMB, ell=ell
    )
    # Interpolated from cov.nell multipoles
    assert block == pytest.approx(expected, rel=1e-3)
    # Differs from the covariance at the effective ell
    eff = cov.get_covariance_block(CL_COMBS[0], CLUSTER_COMB)
    assert block != pytest.approx(eff, rel=1e-3)

    # Without interpolation, if there are fewer multipoles than nell
    cov.nell = ell.size
    assert cov.get_covariance_bandpowers(
        CL_COMBS[0], CLUSTER_COMB, bpw
    ) == pytest.approx(expected)


def test_galaxies_need_hod(config):
    config = copy.copy(config)
    config.pop("HOD")
    cov = ClusterClcNGHaloModel(config)
    cov.get_tracer_profiles(("src0", "src0"))
    with pytest.raises(ValueError):
        cov.get_tracer_profiles(("lens0", "src0"))


def test_covariance_calculator(config):
    config = copy.deepcopy(config)
    config["tjpcov"]["cov_type"] = [
        "ClusterCountsGaussian",
        "ClusterClcNGHaloModel",
        "ClusterClSSCHaloModel",
    ]
    cov_terms = CovarianceCalculator(config).get_covariance_terms()
    s = config["tjpcov"]["sacc_file"]
    icl = np.concatenate([s.indices(tracers=comb) for comb in CL_COMBS])
    icc = s.indices(data_type="cluster_counts")

    for term in ["cNG", "SSC"]:
        cov = cov_terms[term]
        assert np.all(cov == cov.T)
        assert np.all(cov[np.ix_(icl, icc)] != 0)
        assert not np.any(cov[np.ix_(icl, icl)])
        assert not np.any(cov[np.ix_(icc, icc)])
    assert not np.any(cov_terms["gauss"][np.ix_(icl, icc)])
