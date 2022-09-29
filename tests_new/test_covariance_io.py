#!/usr/bin/python
from tjpcov_new.covariance_io import CovarianceIO
import os
import pytest
import numpy as np
import sacc


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = root + 'tjpcov_tmp/'
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_sacc = sacc.Sacc.load_fits(root + 'cls_cov.fits')


def get_diag_covariance():
    ndata = input_sacc.mean.size
    return np.diag(np.ones(ndata))


def test_smoke_input():
    # CovarianceIO should accept dictionary or file path
    CovarianceIO(input_yml)
    config = CovarianceIO._parse(input_yml)
    CovarianceIO(config)
    with pytest.raises(ValueError):
        CovarianceIO(['hello'])

    # Check outdir is created
    if os.path.isdir(outdir):
        os.system(f"rm -rf {outdir}")
    CovarianceIO(input_yml)
    assert os.path.isdir(outdir)


def test_create_sacc_cov():
    cio = CovarianceIO(input_yml)
    # Circunvent the NotImplementedError
    cov = get_diag_covariance()
    s = cio.create_sacc_cov(cov)
    s2 = sacc.Sacc.load_fits(outdir + 'cls_cov.fits')

    assert np.all(s.mean == input_sacc.mean)
    assert np.all(s.covariance.covmat == get_diag_covariance())
    assert np.all(s.mean == s2.mean)
    assert np.all(s.covariance.covmat == s2.covariance.covmat)

    # Check that it also writes the file with a different name
    s2 = cio.create_sacc_cov(cov, 'cls_cov2.fits')
    s2 = sacc.Sacc.load_fits(outdir + 'cls_cov2.fits')


def test_get_outdir():
    cio = CovarianceIO(input_yml)
    assert os.path.samefile(cio.get_outdir(), outdir)


def test_get_sacc_file():
    cio = CovarianceIO(input_yml)
    s = cio.get_sacc_file()

    assert np.all(s.mean == input_sacc.mean)


def test_get_sacc_with_concise_dtypes():
    s = input_sacc.copy()
    for dp in s.data:
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

    cio = CovarianceIO(input_yml)
    s2 = cio.get_sacc_with_concise_dtypes()
    dtypes = input_sacc.get_data_types()
    dtypes2 = s2.get_data_types()
    assert dtypes == dtypes2

    for dp, dp2 in zip(input_sacc.data, s2.data):
        assert dp.data_type == dp2.data_type
        assert dp.value == dp2.value
        assert dp.tracers == dp2.tracers
        for k in dp.tags:
            if k == 'window':
                # Don't check window as it points to a different memory address
                continue
            assert dp.tags[k] == dp2.tags[k]
