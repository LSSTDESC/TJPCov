#!/usr/bin/python
import os
import pymaster as nmt
import numpy as np
import sacc
from . import tools
import warnings


def get_tracer_nmaps(sacc_data, tracer):
    """
    Return the number of maps assotiated to the given tracer

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer (str):  Tracer name

    Returns:
    --------
        nmaps (int):  Number of maps assotiated to the tracer.

    """
    s = get_tracer_spin(sacc_data, tracer)
    if s == 0:
        return 1
    else:
        return 2


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
    if (tr.quantity in ['cmb_convergence', 'galaxy_density']) or \
       ('lens' in tracer):
        return 0
    elif (tr.quantity == 'galaxy_shear') or ('source' in tracer):
        return 2
    else:
        raise NotImplementedError(f'tracer.quantity {tr.quantity} not implemented.')


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


def get_tracer_comb_ncell(sacc_data, tracer_comb, independent=False):
    """
    Return the number of Cell for a pair of tracers (e.g. for shear-shear,
    ncell = 4: EE, EB, BE, BB)

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        tracer_comb (tuple):  List or tuple of a pair of tracer names
        independent (bool): If True, just return the number of independent
        Cell.

    Returns:
    --------
        ncell (int):  Number of Cell for the pair of tracers given

    """
    nmaps1 = get_tracer_nmaps(sacc_data, tracer_comb[0])
    nmaps2 = get_tracer_nmaps(sacc_data, tracer_comb[1])

    ncell = nmaps1 * nmaps2

    if independent and (tracer_comb[0] == tracer_comb[1]) and (ncell == 4):
        # Remove BE, because it will be the same as EB if tr1 == tr2
        ncell = 3

    return ncell


def get_datatypes_from_ncell(ncell):
    """
    Return the possible datatypes (cl_00, cl_0e, cl_0b, etc.) given a number
    of cells for a pair of tracers

    Parameters:
    -----------
        ncell (int):  Number of Cell for a pair of tracers
    Returns:
    --------
        datatypes (list):  List of data types assotiated to the given degrees
        of freedom

    """
    # Copied from https://github.com/xC-ell/xCell/blob/069c42389f56dfff3a209eef4d05175707c98744/xcell/cls/to_sacc.py#L202-L212
    if ncell == 1:
        cl_types = ['cl_00']
    elif ncell == 2:
        cl_types = ['cl_0e', 'cl_0b']
    elif ncell == 4:
        cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
    else:
        raise ValueError('ncell does not match 1, 2, or 4.')

    return cl_types


def get_cl_for_cov(clab, nlab, ma, mb, w, nl_is_cp):
    """
    Computes the coupled Cell that goes into the covariance matrix

    Parameters:
    -----------
        clab (array): Fiducial Cell for the tracers a and b, used for in the
        covariance estimation
        nlab (array): Coupled noise for the tracers a and b
        ma (array): Mask of the field a
        mb (array): Mask of the field b
        w (NmtWorkspace): NmtWorkspace of the fields a and b
        nl_is_cp (bool): True if nlab is coupled. False otherwise.
    Returns:
    --------
        cl:  Coupled Cell with signal and noise

    """
    mean_mamb = np.mean(ma * mb)
    if mean_mamb == 0:
        cl_cp = np.zeros_like(nlab)
    elif nl_is_cp:
        cl_cp = (w.couple_cell(clab) + nlab) / mean_mamb
    else:
        cl_cp = (w.couple_cell(clab) + np.mean(ma) * nlab) / mean_mamb

    return cl_cp


def get_workspace(f1, f2, m1, m2, bins, outdir, **kwargs):
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
        **kwargs:  Extra arguments to pass to
        `w.compute_coupling_matrix`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        w:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    if not isinstance(bins, nmt.NmtBin):
        raise ValueError('You must pass a NmtBin instance through the ' +
                         'cache or at initialization')

    s1, s2 = f1.fl.spin, f2.fl.spin

    if outdir is not None:
        fname = os.path.join(outdir, f'w{s1}{s2}__{m1}__{m2}.fits')
        isfile = os.path.isfile(fname)

        # The workspace of m1 x m2 and m2 x m1 is the same.
        fname2 = os.path.join(outdir, f'w{s2}{s1}__{m2}__{m1}.fits')
        isfile2 = os.path.isfile(fname2)
    else:
        fname = isfile = fname2 = isfile2 = None

    w = nmt.NmtWorkspace()
    if 'recompute' in kwargs:
        recompute = kwargs.pop('recompute')
    else:
        recompute = False

    if recompute or ((not isfile) and (not isfile2)):
        w.compute_coupling_matrix(f1, f2, bins, **kwargs)
        # Recheck that the file has not been written by other proccess
        if fname and not os.path.isfile(fname):
            w.write_to(fname)
        # Check if the other files exist. Recheck in case other process has
        # removed it in the mean time.
        if isfile2 and os.path.isfile(fname2):
            # Remove the other to avoid later confusions.
            os.remove(fname2)
    elif isfile:
        w.read_from(fname)
    else:
        w.read_from(fname2)

    return w


def get_covariance_workspace(f1, f2, f3, f4, m1, m2, m3, m4, outdir, **kwargs):
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
        **kwargs:  Extra arguments to pass to
        `nmt.compute_coupling_coefficients`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        cw:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    spins = {m1: f1.fl.spin, m2: f2.fl.spin, m3: f3.fl.spin, m4: f4.fl.spin}

    # Any other symmetry?
    combinations = [(m1, m2, m3, m4), (m2, m1, m3, m4), (m1, m2, m4, m3),
                    (m2, m1, m4, m3), (m3, m4, m1, m2), (m4, m3, m1, m2),
                    (m3, m4, m2, m1), (m4, m3, m2, m1)]

    if outdir is not None:
        fnames = []
        isfiles = []
        for mn1, mn2, mn3, mn4 in combinations:
            s1, s2, s3, s4 = [spins[mi] for mi in [mn1, mn2, mn3, mn4]]
            f = f'cw{s1}{s2}{s3}{s4}__{mn1}__{mn2}__{mn3}__{mn4}.fits'
            f = os.path.join(outdir, f)
            if f not in fnames:
                fnames.append(f)
                isfiles.append(os.path.isfile(fnames[-1]))
    else:
        fnames = [None]
        isfiles = [None]

    cw = nmt.NmtCovarianceWorkspace()
    if 'recompute' in kwargs:
        recompute = kwargs.pop('recompute')
    else:
        recompute = False
    if recompute or (not np.any(isfiles)):
        cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwargs)
        # Recheck that the file has not been written by other proccess
        if fnames[0] and not os.path.isfile(fnames[0]):
            cw.write_to(fnames[0])
        for fn, isf in zip(fnames[1:], isfiles[1:]):
            # This will only be run if they don't exist or recompute = True
            # Recheck the file exist in case other process has removed it in
            # the mean time.
            if isf and os.path.isfile(fn):
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


def get_masks_dict(mask_files, mask_names, tracer_names, cache, nside=None):
    """
    Return a dictionary with the masks assotiated to the fields to be
    correlated

    Parameters:
    -----------
        mask_files (dict or str): Dictionary of the masks, with the tracer
        names as keys and paths to the masks as values. In fact, the
        tjpcov.mask_fn. If str, it has to be a hdf5 file with keys the mask
        names in mask_names.
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
        nside (int): Healpy map nside. Needed to recreate the map from a TXPipe
        hdf5 map file.

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
                if type(mask_files) is str:
                    mf = mask_files
                else:
                    mf = mask_files[tracer_names[i]]
                if isinstance(mf, np.ndarray):
                    mask_by_mask_name[k] = mf
                else:
                    mask_by_mask_name[k] = tools.read_map(mf, k, nside)
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
            # We add the spin to make sure we distinguish fields of different
            # types even though they share the same mask
            k = mask_names[i] + str(spins[i])
            if k not in f_by_mask_name:
                f_by_mask_name[k] = nmt.NmtField(masks[i], None,
                                                 spin=spins[i], **nmt_conf)
            f[i] = f_by_mask_name[k]

    return f


def get_workspaces_dict(fields, masks, mask_names, bins, outdir, nmt_conf,
                        cache):
    """
    Return a dictionary with the masks assotiated to the fields to be
    correlated

    Parameters:
    -----------
        field (dict): Dictionary of the NmtFields of the fields correlated
        with keys 1, 2, 3 or 4 and values the NmtFields.
        masks (dict): Dictionary of the masks of the fields correlated with
        keys 1, 2, 3 or 4 and values the loaded masks.
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
        field if found. The keys must be 'w12', 'w34', 'w13', 'w23', 'w14' or
        'w24' and the values the corresponding NmtWorkspaces. Alternatively,
        you can pass a dictionary with keys as (mask_name1, mask_name2).

    Returns:
    --------
        workspaces_dict (dict):  Dictionary with the workspaces assotiated to
        the different field combinations needed for the covariance. Its keys
        are 13, 23, 14, 24, 12, 34; with values the corresponding
        NmtWorkspaces.

    """
    w = {}
    w_by_mask_name = {}
    # 12 and 34 must be first to avoid asigning them None if their maks do not
    # overlap
    for i in [12, 34, 13, 23, 14, 24]:
        i1, i2 = [int(j) for j in str(i)]
        # Workspace
        key = f'w{i}'
        if key in cache:
            w[i] = cache[key]
        else:
            s1, s2 = fields[i1].fl.spin, fields[i2].fl.spin
            # In this case you have to check for m1 x m2 and m2 x m1
            k = (mask_names[i1], mask_names[i2])
            sk = ''.join(sorted(f'{s1}{s2}'))

            # Look for the workspaces of appropriate spin combinations
            cache_wsp = cache.get('workspaces', None)
            if cache_wsp is not None:
                if sk in cache_wsp:
                    cache_wsp = cache_wsp[sk]
                elif sk[::-1] in cache_wsp:
                    cache_wsp = cache_wsp[sk[::-1]]

            if sk not in w_by_mask_name:
                w_by_mask_name_s = {}
                w_by_mask_name[sk] = w_by_mask_name_s
            else:
                w_by_mask_name_s =  w_by_mask_name[sk]

            if k in w_by_mask_name_s:
                w[i] = w_by_mask_name_s[k]
            elif k[::-1] in w_by_mask_name_s:
                w[i] = w_by_mask_name_s[k[::-1]]
            elif (i not in [12, 34]) and (np.mean(masks[i1] * masks[i2]) == 0):
                # w13, w23, w14, w24 are needed to couple the theoretical Cell
                # and are not needed if the masks do not overlap. However,
                # w12 and w34 are needed for nmt.gaussian_covariance, which
                # will complain if they are None
                w_by_mask_name_s[k] = None
                w[i] = w_by_mask_name_s[k]
            elif (cache_wsp is not None) and \
                 ((k in cache_wsp) or k[::-1] in cache_wsp):
                if k in cache_wsp:
                    fname = cache_wsp[k]
                else:
                    fname = cache_wsp[k[::-1]]
                wsp = nmt.NmtWorkspace()
                wsp.read_from(fname)
                w[i] = w_by_mask_name_s[k] = wsp
            else:
                w_by_mask_name_s[k] = get_workspace(fields[i1], fields[i2],
                                                    mask_names[i1],
                                                    mask_names[i2], bins,
                                                    outdir, **nmt_conf)
                w[i] = w_by_mask_name_s[k]

    return w


def get_sacc_with_concise_dtypes(sacc_data):
    """
    Return a copy of the sacc file with concise data types

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance

    Returns:
    --------
        sacc_data (Sacc): Data Sacc instance with concise data types
    """

    s = sacc_data.copy()
    dtypes = s.get_data_types()

    dt_long = []
    for dt in dtypes:
        if len(dt.split('_')) > 2:
            dt_long.append(dt)

    for dt in dt_long:
        pd = sacc.parse_data_type_name(dt)

        if pd.statistic != 'cl':
            raise ValueError(f'data_type {dt} not recognized. Is it a Cell?')

        if pd.subtype is None:
            dc = 'cl_00'
        elif pd.subtype == 'e':
            dc = 'cl_0e'
        elif pd.subtype == 'b':
            dc = 'cl_0b'
        elif pd.subtype == 'ee':
            dc = 'cl_ee'
        elif pd.subtype == 'bb':
            dc = 'cl_bb'
        elif pd.subtype == 'eb':
            dc = 'cl_eb'
        elif pd.subtype == 'be':
            dc = 'cl_be'
        else:
            raise ValueError(f'Data type subtype {pd.subtype} not recognized')


        # Change the data_type to its concise versio
        for dp in s.get_data_points(dt):
            dp.data_type = dc

    return s


def get_nbpw(sacc_data):
    """
    Return the number of bandpowers in which the data has been binned

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance

    Returns:
    --------
        nbpw (int): Number of bandpowers; i.e. ell_effective.size
    """
    dtype = sacc_data.get_data_types()[0]
    tracers = sacc_data.get_tracer_combinations(data_type=dtype)[0]
    ix = sacc_data.indices(data_type=dtype, tracers=tracers)
    nbpw = ix.size

    return nbpw


def get_nell(sacc_data, bins=None, nside=None, cache=None):
    """
    Return the number of ells for the fiducial Cells

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance. If the stored bandpowers are
        wrong. You will need to pass one of the other arguments.
        bins (NmtBin): NmtBin instance with the desired binning.
        nside (int): Healpy map nside.
        cache (dict): Dictionary with cached variables. It will use the cached
        workspaces to read the bandpower windows.

    Returns:
    --------
        nell (int): Number of ells for the fidicual Cells points; i.e. lmax or
        3*nside
    """
    # Extracting the workspace from the cache first to use it later in a easy
    # way.
    if (cache is not None) and ('workspaces' in cache):
        w = list(list(cache['workspaces'].values())[0].values())[0]
    else:
        w = None

    if isinstance(w, nmt.NmtWorkspace):
        # We don't want to read the workspace just to get the nell
        bpw = w.get_bandpower_windows()
        nell = bpw.shape[-1]
    elif bins is not None:
        nell = bins.lmax + 1
    else:
        try:
            dtype = sacc_data.get_data_types()[0]
            tracers = sacc_data.get_tracer_combinations(data_type=dtype)[0]
            ix = sacc_data.indices(data_type=dtype, tracers=tracers)
            bpw = sacc_data.get_bandpower_windows(ix)
            if bpw is None:
                raise ValueError
            nell = bpw.nell
        except ValueError as e:
            # If the window functions are wrong. Do magic
            warnings.warn('The window functions in the sacc file are wrong: ')
            warnings.warn(str(e))
            if "binning/ell_max" in sacc_data.metadata:
                warnings.warn('Trying to circunvent this error: we will use' +
                              'nell = lmax + 1 as given in the metadata')
                nell = sacc_data.metadata["binning/ell_max"] + 1
                if nside is not None and nell > 3*nside:
                    warnings.warn('lmax is larger than 3*nside. We will use ' +
                                  'nell = 3*nside')
                    nell = 3 * nside
            elif nside is not None:
                warnings.warn('Trying to circunvent this error: we will try' +
                              'with nell = 3*nside')
                nell = 3*nside
            else:
                raise ValueError('nside, NmtBin or NmtWorkspace instances ' +
                                 'must be passed')
    return nell


def get_list_of_tracers_for_wsp(sacc_data, mask_names):
    """
    Return the tracers needed to compute the independent workspaces.

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        mask_names (dict): Dictionary with the mask names for each tracer. Keys
        must be the tracer names and values the mask names.

    Returns:
    --------
        tracers (list of str): List of tracers needed to compute the
        independent workspaces.
    """

    tracers = sacc_data.get_tracer_combinations()

    fnames = []
    tracers_out = []
    for i, trs1 in enumerate(tracers):
        s1, s2 = get_tracer_comb_spin(sacc_data, trs1)
        mn1, mn2 = [mask_names[tri] for tri in trs1]

        for trs2 in tracers[i:]:
            s3, s4 = get_tracer_comb_spin(sacc_data, trs2)
            mn3, mn4 = [mask_names[tri] for tri in trs2]

            fname1 = f'w{s1}{s2}__{mn1}__{mn2}.fits'
            fname2 = f'w{s3}{s4}__{mn3}__{mn4}.fits'

            if (fname1 in fnames) or (fname2 in fnames):
                continue

            fnames.append(fname1)
            fnames.append(fname2)

            tracers_out.append((trs1, trs2))

    return tracers_out


def get_list_of_tracers_for_cov_wsp(sacc_data, mask_names, remove_trs_wsp=False):
    """
    Return the tracers needed to compute the independent covariance workspaces.

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        mask_names (dict): Dictionary with the mask names for each tracer. Keys
        must be the tracer names and values the mask names.
        remove_trs_wsp (bool): If True, remove the tracer combinations from
        used to generate the workspaces independently (i.e the output of
        `get_list_of_tracers_for_wsp`).

    Returns:
    --------
        tracers (list of str): List of tracers needed to compute the
        independent covariance workspaces.
    """

    tracers = sacc_data.get_tracer_combinations()

    fnames = []
    tracers_out = []
    for i, trs1 in enumerate(tracers):
        s1, s2 = get_tracer_comb_spin(sacc_data, trs1)
        mn1, mn2 = [mask_names[tri] for tri in trs1]
        for trs2 in tracers[i:]:
            s3, s4 = get_tracer_comb_spin(sacc_data, trs2)
            mn3, mn4 = [mask_names[tri] for tri in trs2]

            fname = f'cw{s1}{s2}{s3}{s4}__{mn1}__{mn2}__{mn3}__{mn4}.fits'
            if fname not in fnames:
                fnames.append(fname)
                tracers_out.append((trs1, trs2))

    if remove_trs_wsp:
        trs_wsp = get_list_of_tracers_for_wsp(sacc_data, mask_names)
        for trs in trs_wsp:
            tracers_out.remove(trs)

    return tracers_out


def get_list_of_tracers_for_cov(sacc_data, remove_trs_wsp_cwsp=False,
                                mask_names=None):
    """
    Return the covariance independent tracers combinations.

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance
        remove_trs_wsp_cwsp (bool): If True, remove the tracer combinations from
        used to generate the workspaces and covariance workspaces
        independently. If True, `must_names` must be provided.
        mask_names (dict): Dictionary with the mask names for each tracer. Keys
        must be the tracer names and values the mask names. Needed if
        `remove_trs_wsp_cwsp` is True.

    Returns:
    --------
        tracers (list of str): List of independent tracers combinations.
    """

    tracers = sacc_data.get_tracer_combinations()

    tracers_out = []
    for i, trs1 in enumerate(tracers):
        for trs2 in tracers[i:]:
            tracers_out.append((trs1, trs2))

    if remove_trs_wsp_cwsp:
        trs_cwsp = get_list_of_tracers_for_cov_wsp(sacc_data, mask_names)
        for trs in trs_cwsp:
            tracers_out.remove(trs)

    return tracers_out


def get_ell_eff(sacc_data):
    """
    Return the effective ell in the sacc file. It assume that all of them have
    the same effective ell (true with current TXPipe implementation).

    Parameters:
    -----------
        sacc_data (Sacc):  Data Sacc instance

    Returns:
    --------
        ell (array): Array with the effective ell in the sacc file.
    """
    dtype = sacc_data.get_data_types()[0]
    tracers = sacc_data.get_tracer_combinations(data_type=dtype)[0]
    ell, _ = sacc_data.get_ell_cl(dtype, *tracers)

    return ell
