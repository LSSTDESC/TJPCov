#!/usr/bin/python3

import h5py
import healpy as hp
import numpy as np
import os


def read_map_from_hdf5(fname, name, nside):
    """Return the map stored in the hdf5 TXPipe-like file.

    Args:
        fname (str): Path to the hdf5 file
        name (str): Name of the map in th hdf5 file
        nside (int): Map's HEALPix nside.

    Returns:
        array: HEALPix map
    """
    with h5py.File(fname, "r") as f:
        pixel = f[f"maps/{name}/pixel"]
        value = f[f"maps/{name}/value"]

        m = np.zeros(hp.nside2npix(nside))
        m[pixel] = value

        return m


def read_map(fname, name=None, nside=None):
    """Return the map stored in the file given.

    Args:
        fname (str): Path to the map file. If hdf5 it will call
            read_map_from_hdf5 and name and nside arguments are needed.
            Elsewise, it will be assumed a HEALPix map and read it.
        name (str): Name of the map in the hdf5 file
        nside (int): Map's HEALPix nside.

    Returns:
        array: HEALPix map
    """
    if h5py.is_hdf5(fname):
        if (nside is None) or (name is None):
            raise ValueError(
                "If masks are in hdf5 format, you need to pass the nside and "
                "name of the maps"
            )
        m = read_map_from_hdf5(fname, name, nside)
    else:
        m = hp.read_map(fname)

    return m


if os.name == "nt":
    # Windows
    import msvcrt

    def portable_lock(fp):
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def portable_unlock(fp):
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)

else:
    # *nix
    import fcntl

    def portable_lock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def portable_unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class GlobalLock:
    """Global mutex implementation.

    Ensures only one process can access a block of code. Use:

    with Locker():
        # do something

    See https://stackoverflow.com/a/60214222/6419909 for details.

    """

    def __enter__(self):
        self.fp = open("./lockfile.lock", "wb")
        portable_lock(self.fp)

    def __exit__(self, _type, value, tb):
        portable_unlock(self.fp)
        self.fp.close()
