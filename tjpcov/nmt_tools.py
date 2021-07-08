#!/usr/bin/python
import os
import pymaster as nmt
import numpy as np




def get_tracer_dof(sacc_data, tracer):
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 1
    elif tr.quantity == 'galaxy_shear':
        return 2


def get_tracer_spin(sacc_data, tracer):
    tr = sacc_data.get_tracer(tracer)
    if tr.quantity in ['cmb_convergence', 'galaxy_density']:
        return 0
    elif tr.quantity == 'galaxy_shear':
        return 1


def get_tracer_comb_spin(sacc_data, tracer_comb):
    s1 = get_tracer_spin(sacc_data, tracer_comb[0])
    s2 = get_tracer_spin(sacc_data, tracer_comb[1])

    return s1, s2


def get_tracer_comb_dof(sacc_data, tracer_comb):
    dof1 = get_tracer_dof(sacc_data, tracer_comb[0])
    dof2 = get_tracer_dof(sacc_data, tracer_comb[1])

    return dof1 * dof2


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


def get_covariance_workspace(f1, f2, f3, f4, **kwards):
    """
    Return the covariance workspace of the fields f1, f2, f3, f4

    Parameters:
    -----------
        f1 (NmtField):  Field 1
        f2 (NmtField):  Field 2
        f3 (NmtField):  Field 3
        f4 (NmtField):  Field 4
        **kwards:  Extra arguments to pass to
        `nmt.compute_coupling_coefficients`. In addition, if recompute=True is
        passed, the cw will be recomputed even if found in the disk.

    Returns:
    --------
        cw:  NmtCovarianceWorkspace of the fields f1, f2, f3, f4

    """
    fname = ''  # TODO: Agree on how to store tmp cw
    cw = nmt.NmtCovarianceWorkspace()
    recompute = kwards.get('recompute', False)
    if recompute or (not os.path.isfile(fname)):
        cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwards)
        if fname:
            cw.write_to(fname)
    else:
        cw.read_from(fname)

    return cw
