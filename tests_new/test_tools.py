#!/usr/bin/python3

import healpy as hp
import numpy as np
import pytest

from tjpcov import tools

mask_hdf5_fname = "./tests/data/mask.hdf5"
mask_fname = "./tests/data/mask.fits.gz"
mask_hp = hp.read_map(mask_fname)
nside = 32


def test_read_map_from_hdf5():
    mask = tools.read_map_from_hdf5(mask_hdf5_fname, "mask", nside)
    assert np.all(mask_hp == mask)


def test_read_map():
    mask = tools.read_map(mask_fname)
    assert np.all(mask_hp == mask)

    # hdf5 extension
    mask = tools.read_map(mask_hdf5_fname, name="mask", nside=nside)
    assert np.all(mask_hp == mask)

    # Test errors for hdf5 call
    with pytest.raises(ValueError):
        mask = tools.read_map(mask_hdf5_fname, name="mask")

    with pytest.raises(ValueError):
        mask = tools.read_map(mask_hdf5_fname, nside=nside)

    with pytest.raises(ValueError):
        mask = tools.read_map(mask_hdf5_fname)
