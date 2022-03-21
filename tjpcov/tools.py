#!/usr/bin/python3

import h5py
import healpy as hp
import numpy as np

def read_map_from_hdf5(fname, name, nside):
    with h5py.File(fname, 'r') as f:
        pixel = f[f'maps/{name}/pixel']
        value = f[f'maps/{name}/value']

        # m = np.zeros(hp.nside2npix(nside))
        # Use hp.UNSEEN as in TXPipe
        m = np.repeat(hp.UNSEEN, hp.nside2npix(nside))
        m[pixel] = m[value]

        return m

def read_map(fname, name=None, nside=None):
    if h5py.is_hdf5(fname):
        if nside is None:
            raise ValueError('If masks are in hdf5 format, you need to pass ' +
                             'the nside of the maps')
        m = read_map_from_hdf5(fname, name, nside)
    else:
        m = hp.read_map(fname)

    return m
