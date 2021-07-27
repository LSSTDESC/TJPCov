#!/usr/bin/python
import healpy as hp
import os
import pymaster as nmt
import numpy as np




def get_tracer_dof(sacc_data, tracer):
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 1
    elif tr.quantity == 'galaxy_shear':
        return 2
    else:
        raise ValueError(f'tracer.quantity {tr.quantity} not implemented.')


def get_tracer_spin(sacc_data, tracer):
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 0
    elif tr.quantity == 'galaxy_shear':
        return 2


def get_tracer_comb_spin(sacc_data, tracer_comb):
    s1 = get_tracer_spin(sacc_data, tracer_comb[0])
    s2 = get_tracer_spin(sacc_data, tracer_comb[1])

    return s1, s2


def get_tracer_comb_dof(sacc_data, tracer_comb):
    dof1 = get_tracer_dof(sacc_data, tracer_comb[0])
    dof2 = get_tracer_dof(sacc_data, tracer_comb[1])

    return dof1 * dof2


def get_datatypes_from_dof(dof):
    # Copied from https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L202-L212
    if dof == 1:
        cl_types = ['cl_00']
    elif dof == 2:
        cl_types = ['cl_0e', 'cl_0b']
    elif dof == 4:
        cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
    else:
        raise ValueError('dof does not match 1, 2, or 4.')

    return cl_types


def get_cl_for_cov(clab, nlab_cp, ma, mb, w):
    """
    Computes the coupled Cell that goes into the covariance matrix

    Parameters:
    -----------
        clab (array): Fiducial Cell for the tracers a and b, used for in the
        covariance estimation
        nlab_cp (array): Coupled noise for the tracers a and b
        ma (array): Mask of the field a
        mb (array): Mask of the field b
        w (NmtWorkspace): NmtWorkspace of the fields a and b

    Returns:
    --------
        cl:  Coupled Cell with signal and noise

    """
    mean_mamb = np.mean(ma * mb)
    if not mean_mamb:
        cl_cp = np.zeros_like(nlab_cp)
    else:
        cl_cp = (w.couple_cell(clab) + nlab_cp) / mean_mamb

    return cl_cp


def get_workspace(f1, f2, m1, m2, bins, outdir, **kwards):
    """
    Return the workspace of the fields f1, f2

    Parameters:
    -----------
        f1 (NmtField):  Field 1
        f2 (NmtField):  Field 2
        m1 (string): Mask name assotiated to the field 1
        m2 (string): Mask name assotiated to the field 2
        bins (NmtBin):  NmtBin instance
        outdir (string): Path to the output folder where to store the
        workspace
        mask_names (dict): Dictionary with tracer names as key and maks names
        as values.
        **kwards:  Extra arguments to pass to
        `w.compute_coupling_matrix`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        w:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    fname = os.path.join(outdir, f'w__{m1}__{m2}.fits')
    isfile = os.path.isfile(fname)

    # The workspace of m1 x m2 and m2 x m1 is the same.
    fname2 = os.path.join(outdir, f'w__{m2}__{m1}.fits')
    isfile2 = os.path.isfile(fname2)

    w = nmt.NmtWorkspace()
    if 'recompute' in kwards:
        recompute = kwards.pop('recompute')
    else:
        recompute = False

    if recompute or ((not isfile) and (not isfile2)):
        w.compute_coupling_matrix(f1, f2, bins, **kwards)
        w.write_to(fname)
        if isfile2:
            # Remove the other to avoid later confusions
            os.remove(fname2)
    elif isfile:
        w.read_from(fname)
    else:
        w.read_from(fname2)

    return w


def get_covariance_workspace(f1, f2, f3, f4, m1, m2, m3, m4, outdir, **kwards):
    """
    Return the covariance workspace of the fields f1, f2, f3, f4

    Parameters:
    -----------
        f1 (NmtField):  Field 1
        f2 (NmtField):  Field 2
        f3 (NmtField):  Field 3
        f4 (NmtField):  Field 4
        m1 (string): Mask name assotiated to the field 1
        m2 (string): Mask name assotiated to the field 2
        m3 (string): Mask name assotiated to the field 3
        m4 (string): Mask name assotiated to the field 4
        **kwards:  Extra arguments to pass to
        `nmt.compute_coupling_coefficients`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        cw:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    # Any other symmetry?
    combinations = [(m1, m2, m3, m4), (m2, m1, m3, m4), (m1, m2, m4, m3),
                    (m2, m1, m4, m3), (m3, m4, m1, m2), (m4, m3, m1, m2),
                    (m3, m4, m2, m1), (m4, m3, m2, m1)]

    fnames = []
    isfiles = []
    for mn1, mn2, mn3, mn4 in combinations:
        f = os.path.join(outdir, f'cw__{mn1}__{mn2}__{mn3}__{mn4}.fits')
        if f not in fnames:
            fnames.append(f)
            isfiles.append(os.path.isfile(fnames[-1]))

    cw = nmt.NmtCovarianceWorkspace()
    if 'recompute' in kwards:
        recompute = kwards.pop('recompute')
    else:
        recompute = False
    if recompute or (not True in isfiles):
        cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwards)
        cw.write_to(fnames[0])
        for fn, isf in zip(fnames[1:], isfiles[1:]):
            # This will only be run if they don't exist or recompute = True
            if isf:
                # Remove old covariance workspace if you have recomputed it
                os.remove(fn)
    else:
        ix = isfiles.index(True)
        cw.read_from(fnames[ix])

    return cw


def get_mask_names_dict(mask_names, tracer_names):
    mn  = {}
    for i in [1, 2, 3, 4]:
        mn[i] = mask_names[tracer_names[i]]
    return mn


def get_masks_dict(mask_files, mask_names, tracer_names, cache):
    mask = {}
    mask_by_mask_name = {}
    for i in [1, 2, 3, 4]:
        # Mask
        key = f'm{i}'
        if key in cache:
            mask[i] = cache[key]
        else:
            k = mask_names[i]
            if k not in mask_by_mask_name:
                mask_by_mask_name[k] = hp.read_map(mask_files[tracer_names[i]])
            mask[i] = mask_by_mask_name[k]

    return mask


def get_fields_dict(masks, spins, mask_names, tracer_names, nmt_conf, cache):
    f = {}
    f_by_mask_name = {}
    for i in [1, 2, 3, 4]:
        key = f'f{i}'
        if key in cache:
            f[i] = cache[key]
        else:
            k = mask_names[i]
            if k not in f_by_mask_name:
                f_by_mask_name[k] = nmt.NmtField(masks[i], None, spin=spins[i])
            f[i] = f_by_mask_name[k]

    return f


def get_workspaces_dict(fields, mask_names, bins, outdir, nmt_conf, cache):
    w = {}
    w_by_mask_name = {}
    for i in [13, 23, 14, 24, 12, 34]:
        i1, i2 = [int(j) for j in str(i)]
        # Workspace
        key = f'w{i}'
        if key in cache:
            w[i] = cache[key]
        else:
            # In this case you have to check for m1 x m2 and m2 x m1
            k = (mask_names[i1], mask_names[i2])
            if k in w_by_mask_name:
                w[i] = w_by_mask_name[k]
            elif k[::-1] in w_by_mask_name:
                w[i] = w_by_mask_name[k[::-1]]
            else:
                w_by_mask_name[k] = get_workspace(fields[i1], fields[i2],
                                                  mask_names[i1],
                                                  mask_names[i2], bins,
                                                  outdir, **nmt_conf)
                w[i] = w_by_mask_name[k]

    return w
