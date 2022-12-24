#!/usr/bin/python3

import h5py
import healpy as hp
import numpy as np


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
