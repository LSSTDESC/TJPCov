#!/usr/bin/python3

import healpy as hp
import numpy as np
import pytest

from tjpcov import tools

MASK_HDF5_FNAME = "./tests/data/mask.hdf5"
MASK_FNAME = "./tests/data/mask.fits.gz"
NSIDE = 32


@pytest.fixture
def mock_hp_mask():
    return hp.read_map(MASK_FNAME)


def test_read_map_from_hdf5(mock_hp_mask):
    mask = tools.read_map_from_hdf5(MASK_HDF5_FNAME, "mask", NSIDE)
    assert np.all(mock_hp_mask == mask)


def test_read_map(mock_hp_mask):
    mask = tools.read_map(MASK_FNAME)
    assert np.all(mock_hp_mask == mask)

    # hdf5 extension
    mask = tools.read_map(MASK_HDF5_FNAME, name="mask", nside=NSIDE)
    assert np.all(mock_hp_mask == mask)

    # Test errors for hdf5 call
    with pytest.raises(ValueError):
        mask = tools.read_map(MASK_HDF5_FNAME, name="mask")

    with pytest.raises(ValueError):
        mask = tools.read_map(MASK_HDF5_FNAME, nside=NSIDE)

    with pytest.raises(ValueError):
        mask = tools.read_map(MASK_HDF5_FNAME)
