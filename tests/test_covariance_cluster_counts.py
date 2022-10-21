#!/usr/bin/python3
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new import bin_cov
from tjpcov_new.covariance_clusters import CovarianceClusterCounts
from tjpcov_new.covariance_io import CovarianceIO
import yaml
import healpy as hp
import sacc
import shutil

# INPUT
# CCL and sacc input:
os.makedirs("tests/tmp/", exist_ok=True)
cosmo_filename = "tests/data/cosmo_desy1.yaml"
cosmo = ccl.Cosmology.read_yaml(cosmo_filename)

# SETUP
input_yml = "examples/clusters/tjpcov_conf_minimal_clusters.yaml"
cluster_count_cov = CovarianceClusterCounts(input_yml)


def get_config():
    return CovarianceIO(input_yml).config


def test_smoke():
    cc_cov = CovarianceClusterCounts(input_yml)

    def analytic_bessel_fn(xi, Beta):
        prefactor = 1 / (2 * np.pi * np.sqrt(np.pi * Beta) * (4 * xi**2))
        return prefactor * (1 - np.exp(-(xi**2) / Beta))
