#!/usr/bin/python
import healpy as hp
import os
import pymaster as nmt
import numpy as np




def get_tracer_dof(sacc_data, tracer):
    """
    Return the degrees of freedom of a given tracer

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer (str):  Tracer name

    Returns:
    --------
        dof (int):  Degrees of freedom

    """
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 1
    elif tr.quantity == 'galaxy_shear':
        return 2
    else:
        raise ValueError(f'tracer.quantity {tr.quantity} not implemented.')


def get_tracer_spin(sacc_data, tracer):
    """
    Return the spin of a given tracer

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer (str):  Tracer name

    Returns:
    --------
        spin (int):  Spin of the given tracer

    """
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 0
    elif tr.quantity == 'galaxy_shear':
        return 2


def get_tracer_comb_spin(sacc_data, tracer_comb):
    """
    Return the spins of a pair of tracers

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer_comb (tuple):  List or tuple of a pair of tracer names

    Returns:
    --------
        s1 (int):  Spin of the first tracer
        s2 (int):  Spin of the second tracer

    """
    s1 = get_tracer_spin(sacc_data, tracer_comb[0])
    s2 = get_tracer_spin(sacc_data, tracer_comb[1])

    return s1, s2


def get_tracer_comb_dof(sacc_data, tracer_comb):
    """
    Return the degrees of freedom of the Cell of a pair of tracers

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer_comb (tuple):  List or tuple of a pair of tracer names

    Returns:
    --------
        dof (int):  Degrees of freedom of the Cell of the pair of tracers
        given

    """
    dof1 = get_tracer_dof(sacc_data, tracer_comb[0])
    dof2 = get_tracer_dof(sacc_data, tracer_comb[1])

    return dof1 * dof2


def get_datatypes_from_dof(dof):
    """
    Return the possible datatypes (cl_00, cl_0e, cl_0b, etc.) given a number
    of degrees of freedom

    Parameters:
    -----------
        dof (int):  Degrees of freedom of the Cell of the pair of tracers
        given

    Returns:
    --------
        datatypes (list):  List of data types assotiated to the given degrees
        of freedom

    """
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
        m1 (str): Mask name assotiated to the field 1
        m2 (str): Mask name assotiated to the field 2
        bins (NmtBin):  NmtBin instance
        outdir (str): Path to the output folder where to store the
        workspace. If None, no file will be saved.
        mask_names (dict): Dictionary with tracer names as key and maks names
        as values.
        **kwards:  Extra arguments to pass to
        `w.compute_coupling_matrix`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        w:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    if outdir is not None:
        fname = os.path.join(outdir, f'w__{m1}__{m2}.fits')
        isfile = os.path.isfile(fname)

        # The workspace of m1 x m2 and m2 x m1 is the same.
        fname2 = os.path.join(outdir, f'w__{m2}__{m1}.fits')
        isfile2 = os.path.isfile(fname2)
    else:
        fname = isfile = fname2 = isfile2 = None

    w = nmt.NmtWorkspace()
    if 'recompute' in kwards:
        recompute = kwards.pop('recompute')
    else:
        recompute = False

    if recompute or ((not isfile) and (not isfile2)):
        w.compute_coupling_matrix(f1, f2, bins, **kwards)
        if fname:
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
        m1 (str): Mask name assotiated to the field 1
        m2 (str): Mask name assotiated to the field 2
        m3 (str): Mask name assotiated to the field 3
        m4 (str): Mask name assotiated to the field 4
        outdir (str): Path to the output folder where to store the
        workspace. If None, no file will be saved.
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

    if outdir is not None:
        fnames = []
        isfiles = []
        for mn1, mn2, mn3, mn4 in combinations:
            f = os.path.join(outdir, f'cw__{mn1}__{mn2}__{mn3}__{mn4}.fits')
            if f not in fnames:
                fnames.append(f)
                isfiles.append(os.path.isfile(fnames[-1]))
    else:
        fnames = [None]
        isfiles = [None]

    cw = nmt.NmtCovarianceWorkspace()
    if 'recompute' in kwards:
        recompute = kwards.pop('recompute')
    else:
        recompute = False
    if recompute or (not True in isfiles):
        cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwards)
        if fnames[0]:
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
    """
    Return a dictionary with the mask names assotiated to the fields to be
    correlated

    Parameters:
    -----------
        mask_names (dict):  Dictionary of the masks names assotiated to the
        fields to be correlated. It has to be given as {1: name1, 2: name2, 3:
        name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
        the first and second Cell you are computing the covariance for; i.e.
        <Cell^12 Cell^34>. In fact, the tjpcov.mask_names.
        tracer_names (dict):  Dictionary of the tracer names of the same form
        as mask_name.

    Returns:
    --------
        masks_names_dict (dict):  Dictionary with the mask names assotiated to the
        fields to be correlated.

    """
    mn  = {}
    for i in [1, 2, 3, 4]:
        mn[i] = mask_names[tracer_names[i]]
    return mn


def get_masks_dict(mask_files, mask_names, tracer_names, cache):
    """
    Return a dictionary with the masks assotiated to the fields to be
    correlated

    Parameters:
    -----------
        mask_files (dict): Dictionary of the masks, with the tracer names as
        keys and paths to the masks as values. In fact, the tjpcov.mask_fn.
        mask_names (dict):  Dictionary of the masks names assotiated to the
        fields to be correlated. It has to be given as {1: name1, 2: name2, 3:
        name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
        the first and second Cell you are computing the covariance for; i.e.
        <Cell^12 Cell^34>. In fact, the tjpcov.mask_names.
        tracer_names (dict):  Dictionary of the tracer names of the same form
        as mask_name.
        cache (dict): Dictionary with cached variables. It will use the cached
        masks if found. The keys must be 'm1', 'm2', 'm3' or 'm4' and the
        values the loaded maps.

    Returns:
    --------
        masks_dict (dict):  Dictionary with the masks assotiated to the fields
        to be correlated.

    """
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
    """
    Return a dictionary with the masks assotiated to the fields to be
    correlated

    Parameters:
    -----------
        masks (dict): Dictionary of the masks of the fields correlated with
        keys 1, 2, 3 or 4 and values the loaded masks.
        spins (dict): Dictionary of the spins of the fields correlated with
        keys 1, 2, 3 or 4 and values their spin.
        mask_names (dict):  Dictionary of the masks names assotiated to the
        fields to be correlated. It has to be given as {1: name1, 2: name2, 3:
        name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
        the first and second Cell you are computing the covariance for; i.e.
        <Cell^12 Cell^34>. In fact, the tjpcov.mask_names.
        tracer_names (dict):  Dictionary of the tracer names of the same form
        as mask_name.
        nmt_conf (dict): Dictionary with extra arguments to pass to NmtField.
        In fact, tjpcov.nmt_conf['f']
        cache (dict): Dictionary with cached variables. It will use the cached
        field if found. The keys must be 'f1', 'f2', 'f3' or 'f4' and the
        values the corresponding NmtFields.

    Returns:
    --------
        fields_dict (dict):  Dictionary with the masks assotiated to the fields
        to be correlated.

    """
    f = {}
    f_by_mask_name = {}
    for i in [1, 2, 3, 4]:
        key = f'f{i}'
        if key in cache:
            f[i] = cache[key]
        else:
            k = mask_names[i]
            if k not in f_by_mask_name:
                f_by_mask_name[k] = nmt.NmtField(masks[i], None,
                                                 spin=spins[i], **nmt_conf)
            f[i] = f_by_mask_name[k]

    return f


def get_workspaces_dict(fields, mask_names, bins, outdir, nmt_conf, cache):
    """
    Return a dictionary with the masks assotiated to the fields to be
    correlated

    Parameters:
    -----------
        field (dict): Dictionary of the NmtFields of the fields correlated
        with keys 1, 2, 3 or 4 and values the NmtFields.
        mask_names (dict):  Dictionary of the masks names assotiated to the
        fields to be correlated. It has to be given as {1: name1, 2: name2, 3:
        name3, 4: name4}, where 12 and 34 are the pair of tracers that go into
        the first and second Cell you are computing the covariance for; i.e.
        <Cell^12 Cell^34>. In fact, the tjpcov.mask_names.
        bins (NmtBin): NmtBin instance with the desired binning.
        outdir (str): Path to the directory where to save the computed
        workspaces.
        nmt_conf (dict): Dictionary with extra arguments to pass to NmtField.
        In fact, tjpcov.nmt_conf['w']
        cache (dict): Dictionary with cached variables. It will use the cached
        field if found. The keys must be 'w1', 'w2', 'w3' or 'w4' and the
        values the corresponding NmtFields.

    Returns:
    --------
        workspaces_dict (dict):  Dictionary with the workspaces assotiated to
        the different field combinations needed for the covariance. Its keys
        are 13, 23, 14, 24, 12, 34; with values the corresponding
        NmtWorkspaces.

    """
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