#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_cluster_counts import CovarianceClusterCounts
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC
from tjpcov.clusters_helpers import FFTHelper
import pyccl.halos.hmfunc as hmf
import pytest
import os
import shutil
import itertools
import jinja2
import yaml

INPUT_YML = "./tests/data/conf_covariance_clusters.yaml"
OUTDIR = "./tests/tmp/"
N_z_bins = 18
N_lambda_bins = 3

MASSDEF = ccl.halos.MassDef200m


def teardown_module():
    shutil.rmtree(OUTDIR)


def setup_module():
    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = np.log10(10), np.log10(100)

    richness_tracers, z_tracers = _setup_sacc_tracers(
        z_min, z_max, richness_min, richness_max, s
    )

    survey_area = 4 * np.pi * (180 / np.pi) ** 2
    survey_name = "mock_survey"
    s.add_tracer("survey", survey_name, survey_area)

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_counts,
            (survey_name, z_tracers, richness_tracers),
            int(counts),
        )

    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    s.save_fits(os.path.join(OUTDIR, "test_cl_sacc.fits"), overwrite=True)


def _setup_sacc_tracers(z_min, z_max, richness_min, richness_max, s_file):
    z_tracers = []
    z_edges = np.linspace(z_min, z_max, N_z_bins + 1)
    for i, zbin in enumerate(zip(z_edges[:-1], z_edges[1:])):
        bin_z_label = f"bin_z_{i}"
        s_file.add_tracer("bin_z", bin_z_label, zbin[0], zbin[1])
        z_tracers.append(bin_z_label)

    richness_tracers = []
    richness_edges = np.linspace(richness_min, richness_max, N_lambda_bins + 1)
    for i, richness_bin in enumerate(
        zip(richness_edges[:-1], richness_edges[1:])
    ):
        bin_richness_label = f"bin_rich_{i}"
        s_file.add_tracer(
            "bin_richness",
            bin_richness_label,
            richness_bin[0],
            richness_bin[1],
        )
        richness_tracers.append(bin_richness_label)

    return richness_tracers, z_tracers


@pytest.fixture
def mock_cosmo():
    Omg_c = 0.26
    Omg_b = 0.04
    h0 = 0.67  # so H0 = 100h0 will be in km/s/Mpc
    A_s_value = 2.1e-9
    n_s_value = 0.96
    w_0 = -1.0
    w_a = 0.0

    cosmo = ccl.Cosmology(
        Omega_c=Omg_c,
        Omega_b=Omg_b,
        h=h0,
        A_s=A_s_value,
        n_s=n_s_value,
        w0=w_0,
        wa=w_a,
    )
    return cosmo


@pytest.fixture
def mock_covariance_gauss(mock_cosmo):
    cc_cov = ClusterCountsGaussian(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mass_def=MASSDEF)
    cc_cov.h0 = 0.67
    return cc_cov


@pytest.fixture
def mock_covariance_ssc(mock_cosmo):
    cc_cov = ClusterCountsSSC(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    cc_cov.fft_helper = FFTHelper(
        mock_cosmo, cc_cov.z_lower_limit, cc_cov.z_upper_limit
    )
    cc_cov.mass_func = hmf.MassFuncTinker10(mass_def=MASSDEF)
    cc_cov.h0 = 0.67
    return cc_cov


# Tests start
def test_is_not_null():
    cc_cov = ClusterCountsSSC(INPUT_YML)
    assert cc_cov is not None
    cc_cov = None

    cc_cov = ClusterCountsGaussian(INPUT_YML)
    assert cc_cov is not None


def test_load_from_sacc(mock_covariance_gauss: CovarianceClusterCounts):
    assert mock_covariance_gauss.min_mass == np.log(1e13)
    assert mock_covariance_gauss.num_richness_bins == 3
    assert mock_covariance_gauss.num_z_bins == 18
    assert mock_covariance_gauss.min_richness == 10
    assert mock_covariance_gauss.max_richness == 100
    assert mock_covariance_gauss.z_min == 0.3
    assert mock_covariance_gauss.z_max == 1.2


def test_load_from_cosmology(mock_covariance_gauss: CovarianceClusterCounts):
    cosmo = ccl.CosmologyVanillaLCDM()
    mock_covariance_gauss.load_from_cosmology(cosmo)

    assert mock_covariance_gauss.cosmo == cosmo


@pytest.mark.parametrize(
    "z, ref_val",
    [
        (0.3, 1.463291259900985e-05),
        (0.35, 1.4251538328691035e-05),
    ],
)
def test_integral_mass_no_bias(
    mock_covariance_gauss: CovarianceClusterCounts, z, ref_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0, remove_bias=True)
    assert test == pytest.approx(ref_val, rel=1e-3)


def test_double_bessel_integral(
    mock_covariance_gauss: CovarianceClusterCounts,
):
    ref = 8.427201745032292e-05
    test = mock_covariance_gauss.double_bessel_integral(0.3, 0.3)
    assert test == pytest.approx(ref, rel=1e-3)


def test_shot_noise(mock_covariance_gauss: ClusterCountsGaussian):
    ref = 63973.635143644424
    test = mock_covariance_gauss.shot_noise(0, 0)
    assert test == pytest.approx(ref, rel=1e-3)


@pytest.mark.parametrize(
    "z, reference_val",
    [
        (0.5, 2.596895139062984e-05),
        (0.55, 2.5910691906342223e-05),
    ],
)
def test_integral_mass(
    mock_covariance_gauss: CovarianceClusterCounts, z, reference_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0)
    assert test == pytest.approx(reference_val, rel=1e-3)


@pytest.mark.parametrize(
    "z, reference_val",
    [
        (0.5, 3.8e-05),  # a proper value must be added here
    ],
)
def test_integral_mass_no_mproxy(
    mock_covariance_gauss: CovarianceClusterCounts, z, reference_val
):
    mock_covariance_gauss.richness_bins = np.linspace(13.5, 14, 4)
    mock_covariance_gauss.has_mproxy = False
    test = mock_covariance_gauss.mass_richness_integral(z, 0)
    assert test == pytest.approx(reference_val, rel=1e-1)


def test_mass_richness(mock_covariance_gauss: CovarianceClusterCounts):
    reference_min = 0.0009528852621284171

    test_min = [
        mock_covariance_gauss.mass_richness(mock_covariance_gauss.min_mass, i)
        for i in range(mock_covariance_gauss.num_richness_bins)
    ]
    assert np.sum(test_min) == pytest.approx(reference_min)


@pytest.mark.parametrize(
    "z_i, reference_val",
    [
        (0, 6613.739621696188),
        (4, 55940746.72160228),
        (8, 3781771343.1278453),
        (14, 252063237.8394578),
        (17, 1113852.72571463),
    ],
)
def test_calc_dv(
    mock_covariance_gauss: CovarianceClusterCounts, z_i, reference_val
):
    z_true = 0.8
    test = mock_covariance_gauss.comoving_volume_element(z_true, z_i) / 1e4
    assert test == pytest.approx(reference_val / 1e4)


def test_cov_gaussian_zero_offdiagonal(
    mock_covariance_gauss: ClusterCountsGaussian,
):
    cov_0111_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_0", "bin_rich_1"),
        ("mock_survey", "bin_z_1", "bin_rich_1"),
    )
    cov_1011_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_1", "bin_rich_0"),
        ("mock_survey", "bin_z_1", "bin_rich_1"),
    )
    cov_1001_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_1", "bin_rich_0"),
        ("mock_survey", "bin_z_0", "bin_rich_1"),
    )
    assert cov_0111_gauss == 0
    assert cov_1011_gauss == 0
    assert cov_1001_gauss == 0

    cov_10_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_1", "bin_rich_0"),
        ("mock_survey", "bin_z_1", "bin_rich_1"),
    )
    assert cov_10_gauss == 0


def test_cov_nxn(
    mock_covariance_gauss: ClusterCountsGaussian,
    mock_covariance_ssc: ClusterCountsSSC,
):
    ref_sum = 130462.91921818888
    # Need to include survey name from mock file here to ensure correct data
    # types are found
    cov_00_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_0", "bin_rich_0"),
        ("mock_survey", "bin_z_0", "bin_rich_0"),
    )
    cov_00_ssc = mock_covariance_ssc.get_covariance_block_for_sacc(
        ("mock_survey", "bin_z_0", "bin_rich_0"),
        ("mock_survey", "bin_z_0", "bin_rich_0"),
    )
    assert cov_00_gauss + cov_00_ssc == pytest.approx(ref_sum, rel=1e-3)


def test_cluster_count_tracer_missing_throws():
    # Create a mock sacc file without any cluster count data points

    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = np.log10(10), np.log10(100)

    survey_area = 4 * np.pi * (180 / np.pi) ** 2
    survey_name = "mock_survey"
    s.add_tracer("survey", survey_name, survey_area)

    richness_tracers, z_tracers = _setup_sacc_tracers(
        z_min, z_max, richness_min, richness_max, s
    )

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_mean_log_mass,
            (survey_name, z_tracers, richness_tracers),
            int(counts),
        )

    # Save the file
    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    bad_sacc_file = os.path.join(OUTDIR, "test_cl_fails_sacc.fits")
    s.save_fits(bad_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = bad_sacc_file

    with pytest.raises(
        ValueError, match="Cluster count covariance was requested"
    ):
        ClusterCountsGaussian(config)

    os.remove(bad_sacc_file)


def test_cluster_count_defaults_survey_area():
    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = np.log10(10), np.log10(100)

    richness_tracers, z_tracers = _setup_sacc_tracers(
        z_min, z_max, richness_min, richness_max, s
    )

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_counts,
            (z_tracers, richness_tracers),
            int(counts),
        )

    # Save the file
    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    new_sacc_file = os.path.join(OUTDIR, "test_cl_no_survey_sacc.fits")
    s.save_fits(new_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = new_sacc_file

    cc = ClusterCountsGaussian(config)
    assert cc.survey_area == 4 * np.pi

    os.remove(new_sacc_file)


def test_non_cluster_counts_covmat_zero():
    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = np.log10(10), np.log10(100)

    richness_tracers, z_tracers = _setup_sacc_tracers(
        z_min, z_max, richness_min, richness_max, s
    )

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_counts,
            (z_tracers, richness_tracers),
            int(counts),
        )
        s.add_data_point(
            sacc.standard_types.cluster_mean_log_mass,
            (z_tracers, richness_tracers),
            int(counts),
        )

    # Save the file
    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    new_sacc_file = os.path.join(OUTDIR, "test_cl_multiple_data_sacc.fits")
    s.save_fits(new_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = new_sacc_file

    cc = ClusterCountsGaussian(config)

    trs_cov = cc.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        blocks.append(np.array(1))

    cov = cc._build_matrix_from_blocks(blocks, trs_cov)

    # Only upper left should have been populated with values.
    assert np.all(cov[:54, :54] == 1)
    assert np.all(cov[55:, 55:] == 0)
    assert np.all(cov[55:, :54] == 0)
    assert np.all(cov[:54, 55:] == 0)
