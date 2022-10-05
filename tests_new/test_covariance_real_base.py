#!/usr/bin/python3
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new import bin_cov
from tjpcov_new.covariance_builder import CovarianceReal, CovarianceProjectedReal
from tjpcov_new.covariance_io import CovarianceIO
import yaml
import healpy as hp
import sacc
import shutil

input_yml_real = "tests_new/data/conf_tjpcov_minimal_real.yaml"
xi_fn = "examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
sacc_file = sacc.Sacc.load_fits(xi_fn)


class CovarianceRealTester(CovarianceReal):
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

    def get_covariance_block(self, tracer_comb1, tracer_comb2):
        super().get_covariance_block(tracer_comb1, tracer_comb2)


class CovarianceProjectedRealTester(CovarianceProjectedReal):
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

    def get_covariance_block(self, tracer_comb1, tracer_comb2):
        super().get_covariance_block(tracer_comb1, tracer_comb2)

def test_get_theta_eff():
    cr = CovarianceRealTester(input_yml_real)
    theta_eff, _ = sacc_file.get_theta_xi('galaxy_shear_xi_plus',
                                          'src0', 'src0')
    assert np.all(theta_eff == cr.get_theta_eff())


def test_get_binning_info():
    # Check we recover the ell effective from the edges
    cpr = CovarianceProjectedRealTester(input_yml_real)
    theta, theta_eff, theta_edges = \
        cpr.get_binning_info(in_radians=False)

    assert np.all(theta_eff == cpr.get_theta_eff())
    assert np.allclose((theta_edges[1:]+theta_edges[:-1])/2, theta_eff)

    # Check in_radians work
    theta2, theta_eff2, theta_edges2 = \
        cpr.get_binning_info(in_radians=True)
    arcmin_rad = np.pi / 180 / 60
    assert np.all(theta * arcmin_rad == theta2)
    assert np.all(theta_eff * arcmin_rad == theta_eff2)
    assert np.all(theta_edges * arcmin_rad == theta_edges2)

    with pytest.raises(NotImplementedError):
        cpr.get_binning_info('linear')


def test_get_cov_WT_spin():
    pass


def test_get_Wigner_transform():
    pass
