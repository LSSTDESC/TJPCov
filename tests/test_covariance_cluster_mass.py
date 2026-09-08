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
import copy

from conftest import OUTDIR, INPUT_YML

def teardown_module():
    shutil.rmtree(OUTDIR)


@pytest.fixture
def mock_covariance_mass(mock_cosmo):
    cc_cov = ClusterMass(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    return cc_cov


# Tests start


def test_is_not_null(save_cluster_sacc_data):
    cc_cov = ClusterMass(INPUT_YML)
    assert cc_cov is not None


def test_load_cluster_parameters(mock_covariance_mass: ClusterMass):
    """Test that the cluster parameters are loaded correctly."""
    # Test with valid parameters
    mock_covariance_mass.load_cluster_parameters()

    # Check that the parameters are loaded correctly
    assert mock_covariance_mass.mass_def == "200m"
    assert mock_covariance_mass.min_halo_ln_mass == np.log(1e13)
    assert mock_covariance_mass.max_halo_ln_mass == np.log(1e16)
    assert mock_covariance_mass.mass_func is not None

    # Test with invalid mass function
    invalid_mass_func = "InvalidMassFunc"
    config_copy = mock_covariance_mass.config.copy()
    config_copy["mor_parameters"]["mass_func"] = invalid_mass_func
    mock_covariance_mass.config = config_copy
    with pytest.raises(
        ValueError,
        match=f"Invalid mass function: {invalid_mass_func}",
    ):
        mock_covariance_mass.load_cluster_parameters()


def test_cluster_count_tracer_missing_throws(save_cluster_sacc_data):
    # Keep only cluster_counts, so cluster_mean_log_mass is "missing".
    # save_cluster_sacc_data is module-scoped and shared, so copy it before
    # mutating.
    s = copy.deepcopy(save_cluster_sacc_data)
    s.reorder(s.indices(data_type=sacc.standard_types.cluster_counts))
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


def test_cluster_mass_defaults_survey_area(save_cluster_sacc_data):
    """Test default survey area."""
    s = copy.deepcopy(save_cluster_sacc_data)
    del s.tracers["NC_mock_redshift_richness"]
    s.reorder(s.indices(data_type=sacc.standard_types.cluster_mean_log_mass))

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


def test_non_cluster_mass_covmat_zero(save_cluster_sacc_data):
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)

    cc = ClusterMass(config)

    trs_cov = cc.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        blocks.append(np.array(1))

    cov = cc._build_matrix_from_blocks(blocks, trs_cov)

    # Only bottom right should have been populated with values.
    assert np.all(cov[:18, :18] == 0)
    assert np.all(cov[19:, 19:] == 1)
    assert np.all(cov[19:, :18] == 0)
    assert np.all(cov[:18, 19:] == 0)
