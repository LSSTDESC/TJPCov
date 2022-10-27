#!/usr/bin/python3
import numpy as np
import pytest

import tjpcov.wigner_transform as wigner_transform


def get_WT_kwargs():
    lmax = 96
    ell = np.arange(2, lmax + 1)
    theta = np.sort(np.pi / ell)[::2]  # Force them to have different sizes
    WT_kwargs = {
        "ell": ell,
        "theta": theta,
        "s1_s2": [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)],
    }
    return WT_kwargs


def get_matrix(ell):
    cl_cov = np.zeros((ell.size, ell.size))
    for i in range(ell.size):
        cl_cov[i, i:] = ell[i:]
        cl_cov[i:, i] = ell[i:]

    return cl_cov


def get_WT():
    WT_kwargs = get_WT_kwargs()
    return wigner_transform.WignerTransform(**WT_kwargs)


def bin_cov(r, cov, r_bins):
    """
    A slower function to test the bin_mat function above.
    """
    # This function was in wigner_transform.py before. I think I correctly
    # fixed bin_mat and moved it here as it is said to be slower. We will use
    # it to test bin_mat.
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])
    n_bins = len(bin_center)
    cov_int = np.zeros((n_bins, n_bins), dtype="float64")
    bin_idx = np.digitize(r, r_bins) - 1

    # this takes care of problems around bin edges
    r2 = np.sort(np.unique(np.append(r, r_bins)))
    dr = np.gradient(r2)
    r2_idx = [i for i in np.arange(len(r2)) if r2[i] in r]
    dr = dr[r2_idx]
    r_dr = r * dr
    cov_r_dr = cov * np.outer(r_dr, r_dr)

    for i in np.arange(min(bin_idx), n_bins):
        xi = bin_idx == i
        for j in np.arange(min(bin_idx), n_bins):
            xj = bin_idx == j
            norm_ij = np.sum(r_dr[xi]) * np.sum(r_dr[xj])
            if norm_ij == 0:
                continue
            cov_int[i][j] = np.sum(cov_r_dr[xi, :][:, xj]) / norm_ij
    return bin_center, cov_int


def test_smoke():
    WT_kwargs = get_WT_kwargs()
    wigner_transform.WignerTransform(**WT_kwargs)


def test_cl_grid():
    wt = get_WT()
    ell = np.arange(5, 96, 2)
    cl = ell
    cl2 = wt.cl_grid(ell, cl)
    sel = (wt.ell < ell[0]) + (wt.ell > ell[-1])
    assert np.all(cl2[sel] == 0)
    assert np.all(cl2[~sel] == wt.ell[~sel])


def test_cl_cov_grid():
    wt = get_WT()
    ell = np.arange(5, 96, 2)
    cl_cov = get_matrix(ell)

    cl_cov2 = wt.cl_cov_grid(ell, cl_cov)
    sel1 = wt.ell < ell[0]
    assert np.max(np.abs(cl_cov2[sel1][:, sel1] / ell[0] - 1) < 1e-10)
    sel2 = wt.ell > ell[-1]
    assert np.max(np.abs(cl_cov2[sel2][:, sel2] / ell[-1] - 1) < 1e-10)
    sel = ~(sel1 + sel2)
    cl_cov = get_matrix(wt.ell)
    assert np.max(
        np.abs(cl_cov2[sel][:, sel] / cl_cov[sel][:, sel] - 1) < 1e-10
    )


@pytest.mark.parametrize("s1_s2", [(0, 0), (0, 2), (2, 2), (2, -2)])
@pytest.mark.parametrize("s1_s2_cross", [(0, 0), (0, 2), (2, 2), (2, -2)])
def test_projected_covariance(s1_s2, s1_s2_cross):
    wt = get_WT()
    mat = get_matrix(wt.ell)
    with pytest.raises(NotImplementedError):
        wt.projected_covariance(wt.ell[10:], mat, s1_s2, s1_s2_cross)

    th, matb = wt.projected_covariance(wt.ell, mat, s1_s2, s1_s2_cross)
    wd_a = wigner_transform.wigner_d(*s1_s2, wt.theta, wt.ell)
    wd_b = wigner_transform.wigner_d(*s1_s2_cross, wt.theta, wt.ell)
    matb_2 = (
        (wd_a * np.sqrt(wt.norm) * wt.grad_ell)
        @ mat
        @ (wd_b * np.sqrt(wt.norm)).T
    )

    assert np.all(th == wt.theta)
    assert np.max(np.abs(matb / matb_2) - 1) < 1e-5


def test_taper():
    with pytest.raises(NotImplementedError):
        wt = get_WT()
        wt.taper(wt.ell)


def test_diagonal_err():
    wt = get_WT()
    assert np.all(wt.diagonal_err(get_matrix(wt.ell)) == np.sqrt(wt.ell))


# FIXME: I couldn't reproduce the values with an external code
# (https://github.com/ntessore/wigner) but I might be using it wrong. Felipe
# validated the implementation with Sukhdeep's code, so
# it should be all right. We need to add the validation here
# @pytest.mark.parametrize("s1,s2", [(0, 0), (0, 2), (2, 2), (2, -2)])
# def test_wigner_d(s1, s2):
#     kwargs = get_WT_kwargs()
#     theta = kwargs["theta"]
#     ell = kwargs["ell"]
#     wd = wigner_transform.wigner_d(s1, s2, np.atleast_1d(theta[20]),
#                                    np.atleast_1d(ell[20]))
#     # import wigner
#     # wd2 = wigner.wigner_dl(ell[20], ell[20], s1, s2, theta[20])
#     # assert np.max(np.abs(wd / wd2 - 1)) < 1e-5


@pytest.mark.parametrize("s1,s2", [(0, 0), (0, 2), (2, 2), (2, -2)])
def test_wigner_d_parallel(s1, s2):
    kwargs = get_WT_kwargs()
    theta = kwargs["theta"]
    ell = kwargs["ell"]
    wd = wigner_transform.wigner_d(s1, s2, theta, ell)
    wd2 = wigner_transform.wigner_d_parallel(s1, s2, theta, ell)
    assert np.all(wd == wd2)

    wd2 = wigner_transform.wigner_d_parallel(s1, s2, theta, ell, ncpu=4)
    assert np.all(wd == wd2)

    wd = wigner_transform.wigner_d(s1, s2, theta, ell, l_use_bessel=None)
    wd2 = wigner_transform.wigner_d_parallel(
        s1, s2, theta, ell, l_use_bessel=None
    )
    assert np.all(wd == wd2)


def test_bin_cov():
    kwargs = get_WT_kwargs()
    r = kwargs["theta"]
    mat = get_matrix(r)
    r_bins = np.logspace(np.log10(r.min()), np.log10(r.max()), 10)
    bin_center, bin_mat = wigner_transform.bin_cov(r, mat, r_bins)
    bin_center2, bin_mat2 = bin_cov(r, mat, r_bins)

    assert np.all(bin_center == bin_center2)
    assert np.all(bin_mat == bin_mat2)
