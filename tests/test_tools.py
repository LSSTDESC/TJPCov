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


def test_global_lock():
    """Unit test for the global lock. This test verifies that the global lock
    works for multiple threads.  However, this does not verify that the lock
    works between *processes*.

    This is a bit trickier because we would need to have two python scripts
    run the same code block. Possible, but would require a bit more code.

    For now this accurately show that whichever thread enters the global lock
    first prevents other threads from acquiring that same lock, which
    effectively demonstrates the mutex capabilities.

    """
    import time
    import os
    from threading import Thread

    test_file = open("./test_file.txt", "w")

    # Fire off a process to delete the file, wait 2 seconds for GlobalLock to
    # activate
    def delete_file():
        # Wait 3 seconds to allow the parent caller to activate GlobalLock
        time.sleep(3)
        with tools.GlobalLock():
            os.remove("./test_file.txt")

    delete_process = Thread(target=delete_file)
    delete_process.start()

    with tools.GlobalLock():
        # Wait for 5 seconds to allow delete to try to delete file.
        time.sleep(5)
        try:
            test_file.write("File should exist")
        except Exception:
            pytest.fail("Could not write, GlobalLock failed.")

    # Wait for delete to finish.
    delete_process.join()
