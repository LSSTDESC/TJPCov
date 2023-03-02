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
