#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_cluster_mass import ClusterMass
import pytest
import os
import shutil
import itertools
import jinja2
import yaml

INPUT_YML = "./tests/data/conf_covariance_clusters.yaml"
OUTDIR = "./tests/tmp/"


def teardown_module():
    shutil.rmtree(OUTDIR)


def setup_module():
    s = sacc.Sacc()
    z_min, z_max = 0.3, 1.2
    richness_min, richness_max = np.log10(10), np.log10(100)

    z_tracers = []
    z_edges = np.linspace(z_min, z_max, 19)
    for i, zbin in enumerate(zip(z_edges[:-1], z_edges[1:])):
        bin_z_label = f"bin_z_{i}"
        s.add_tracer("bin_z", bin_z_label, zbin[0], zbin[1])
        z_tracers.append(bin_z_label)

    richness_tracers = []
    richness_edges = np.linspace(richness_min, richness_max, 4)
    for i, richness_bin in enumerate(
        zip(richness_edges[:-1], richness_edges[1:])
    ):
        bin_richness_label = f"bin_rich_{i}"
        s.add_tracer(
            "bin_richness",
            bin_richness_label,
            richness_bin[0],
            richness_bin[1],
        )
        richness_tracers.append(bin_richness_label)

    survey_area = 4 * np.pi * (180 / np.pi) ** 2
    survey_name = "mock_survey"
    s.add_tracer("survey", survey_name, survey_area)

    tracer_combos = list(itertools.product(z_tracers, richness_tracers))
    cluster_counts = np.linspace(1, 1e4, len(tracer_combos))

    counts_and_edges = zip(cluster_counts.flatten(), tracer_combos)

    for counts, (z_tracers, richness_tracers) in counts_and_edges:
        s.add_data_point(
            sacc.standard_types.cluster_mean_log_mass,
            (survey_name, z_tracers, richness_tracers),
            int(counts),
        )

    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    s.save_fits(os.path.join(OUTDIR, "test_cl_sacc.fits"), overwrite=True)


def _setup_sacc_tracers(z_min, z_max, richness_min, richness_max, s_file):
    z_tracers = []
    z_edges = np.linspace(z_min, z_max, 19)
    for i, zbin in enumerate(zip(z_edges[:-1], z_edges[1:])):
        bin_z_label = f"bin_z_{i}"
        s_file.add_tracer("bin_z", bin_z_label, zbin[0], zbin[1])
        z_tracers.append(bin_z_label)

    richness_tracers = []
    richness_edges = np.linspace(richness_min, richness_max, 4)
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
def mock_covariance_mass(mock_cosmo):
    cc_cov = ClusterMass(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    return cc_cov


# Tests start


def test_is_not_null():
    cc_cov = ClusterMass(INPUT_YML)
    assert cc_cov is not None


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
            sacc.standard_types.cluster_counts,
            (survey_name, z_tracers, richness_tracers),
            int(counts),
        )

    # Save the file
    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    bad_sacc_file = os.path.join(OUTDIR, "test_cl_mass_fails_sacc.fits")
    s.save_fits(bad_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = bad_sacc_file

    with pytest.raises(
        ValueError, match="Cluster mass covariance was requested"
    ):
        ClusterMass(config)

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
            sacc.standard_types.cluster_mean_log_mass,
            (z_tracers, richness_tracers),
            int(counts),
        )

    # Save the file
    s.to_canonical_order()
    os.makedirs(OUTDIR, exist_ok=True)
    new_sacc_file = os.path.join(OUTDIR, "test_cl_mass_no_survey_sacc.fits")
    s.save_fits(new_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = new_sacc_file

    cc = ClusterMass(config)
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
    new_sacc_file = os.path.join(
        OUTDIR, "test_cl_mass_multiple_data_sacc.fits"
    )
    s.save_fits(new_sacc_file, overwrite=True)

    # Overwrite config file to point to new sacc file
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    config["tjpcov"]["sacc_file"] = new_sacc_file

    cc = ClusterMass(config)

    trs_cov = cc.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        blocks.append(np.array(1))

    cov = cc._build_matrix_from_blocks(blocks, trs_cov)

    # Only bottom right should have been populated with values.
    assert np.all(cov[:54, :54] == 0)
    assert np.all(cov[55:, 55:] == 1)
    assert np.all(cov[55:, :54] == 0)
    assert np.all(cov[:54, 55:] == 0)
