# Sukhdeep: This code is copied from Skylens. Skylens is not ready to be public yet, but TJPCov have our permission to use this code.
from scipy.special import jn, jn_zeros, jv
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.optimize import fsolve
from multiprocessing import cpu_count, Pool
from functools import partial
from scipy.special import binom, jn, loggamma
from scipy.special import eval_jacobi as jacobi
import numpy as np
import itertools

# FIXME:
# 1. change l->ell
# 2. Do we need to pass logger?
# 3. Need to add inverse transform functionality.


class wigner_transform:
    def __init__(
        self, theta=[], l=[], s1_s2=[(0, 0)], ncpu=None
    ):  # logger=None,
        """
        Class to compute curved sky Hankel transforms using the wigner-d matrices.

        Parameters
        ----------
        theta:
            Values of angular separation, theta, at which the Hankel transform is done. Should be in radians.
        l:
            ell values at which the Hankel transform is done. Should be integers
        s1_s2:
            List of spin pairs of the tracers. Each spin pair should be a tuple. e.g. for 3X2 analysis, pass [(0,0),(0,2),(2,2),(2,-2)].
            (0,0): (galaxy,galaxy)
            (0,2): (galaxy,shear). (2,0) is equivalent.
            (2,2): (shear,shear), xi+
            (2,-2): (shear,shear), xi-
        ncpu:
            Number of python processes to use when computing wigner-d matrices.
        """
        self.name = "Wigner"
        # self.logger=logger

        self.l = l
        self.grad_l = np.gradient(l)
        self.norm = (2 * l + 1.0) / (
            4.0 * np.pi
        )  # ignoring some factors of -1,
        # assuming sum and differences of s1,s2
        # are even for all correlations we need.

        # for inverse wigner transform
        self.grad_theta = np.gradient(theta)
        self.inv_norm = np.sin(theta) * 2 * np.pi
        self.inv_wig_norm = self.inv_norm * self.grad_theta

        self.wig_d = {}
        self.wig_3j = {}
        self.s1_s2s = s1_s2
        self.theta = {}
        self.theta = theta

        # compute the wigner-d matrices.
        for (s1, s2) in s1_s2:
            self.wig_d[(s1, s2)] = wigner_d_parallel(
                s1, s2, theta, self.l, ncpu=ncpu
            )

    #             self.wig_d[(s1,s2)]=wigner_d_recur(s1,s2,theta,self.l)
    # self.theta[(s1,s2)]=theta #FIXME: Ugly and unnecessary. Should be removed.

    def cl_grid(self, l_cl=[], cl=[], taper=False, **taper_kwargs):
        """
        Interpolate input C_ell in case the ell values of C_ell are different from the grid on which wigner-d matrices
        were computed during intialization.

        Parameters
        ----------
        cl:
            input C_ell
        l_cl:
            ell at which the input C_ell is computed.
        taper:
            if True apply the tapering to the input C_ell. Tapering can help in reducing ringing.
        """
        if taper:
            sself.taper_f = self.taper(l=l, **taper_kwargs)
            cl = cl * taper_f
        # if l==[]:#In this case pass a function that takes k with kwargs and outputs cl
        #     cl2=cl(l=self.l,**kwargs)
        # else:
        cl_int = interp1d(
            l_cl, cl, bounds_error=False, fill_value=0, kind="linear"
        )
        cl2 = cl_int(self.l)
        return cl2

    def cl_cov_grid(self, l_cl=[], cl_cov=[], taper=False, **taper_kwargs):
        """
        Same as cl_grid, but for the 2D covariance. Uses 2D interpolation.

        """
        if taper:  # FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(
                np.isclose(self.taper_f["l"], cl)
            ):
                self.taper_f = self.taper(l=l, **taper_kwargs)
                taper_f2 = np.outer(
                    self.taper_f["taper_f"], self.taper_f["taper_f"]
                )
                self.taper_f2 = {"l": l, "taper_f2": taper_f2}
            cl = cl * self.taper_f2["taper_f2"]
        if (
            l_cl_cl == []
        ):  # In this case pass a function that takes k with kwargs and outputs cl
            cl2 = cl_cov(l=self.l, **kwargs)
        else:
            cl_int = RectBivariateSpline(
                l_cl,
                l_cl,
                cl_cov,
            )  # bounds_error=False,fill_value=0,
            # kind='linear')
            # interp2d is slow. Make sure l_cl is on regular grid.
            cl2 = cl_int(self.l, self.l)
        return cl2

    def projected_correlation(
        self, l_cl=[], cl=[], s1_s2=(), taper=False, **taper_kwargs
    ):
        """
        Convert input C_ell to the correlation function.

        Parameters
        ----------
        cl:
            Input C_ell
        l_cl:
            ell values at which input C_ell is computer.
        s1_s2:
            Tuple of the spin factors of the tracers. Used to identify the correct wigner-d matrix to use.
        taper:
            If true, apply tapering to the input C_ell
        taper_kwargs:
            Arguments to be passed to the tapering function.
        """
        cl2 = self.cl_grid(l_cl=l_cl, cl=cl, taper=taper, **taper_kwargs)
        w = np.dot(self.wig_d[s1_s2] * self.grad_l * self.norm, cl2)
        return self.theta, w

    def inv_projected_correlation(
        self, theta_xi=[], xi=[], s1_s2=[], taper=False, **kwargs
    ):
        """
        Convert input xi to C_ell, the inverse hankel transform
        Parameters
        ----------
        xi:
            The input correlation function
        theta_xi:
            theta values at which xi is computed.
        s1_s2:
            Tuple of the spin factors of the tracers. Used to identify the correct wigner-d matrix to use.
        """
        wig_d = self.wig_d[s1_s2].T
        wig_theta = self.theta
        wig_norm = self.inv_wig_norm

        xi2 = self.cl_grid(
            l_cl=theta_xi, cl=xi, taper=taper, wig_l=wig_theta, **kwargs
        )
        cl = np.dot(wig_d * wig_norm, xi2)
        return self.l, cl

    def projected_covariance(
        self,
        l_cl=[],
        cl_cov=[],
        s1_s2=[],
        s1_s2_cross=None,
        taper=False,
        **kwargs
    ):
        """
        Convert C_ell covariance to correlation function.
        This function assumes that cl_cov is one dimensional array (i.e. only defined along the diagonal),
        useful for diagonal gaussian covariances.
        For the 2D covariances, use projected_covariance2 function.

        Parameters
        ----------
        cl_cov:
            Diagonal of the covariance matrix.
        l_cl:
            ell values at which input C_ell is computer.
        s1_s2:
            Tuple of the spin factors of the first set of tracers. Used to identify the correct wigner-d matrix to use.
        s1_s2_cross:
            Tuple of the spin factors of the second set of tracers, if different from s1_s2.
            Used to identify the correct wigner-d matrix to use.

        """
        if s1_s2_cross is None:
            s1_s2_cross = s1_s2
        # when cl_cov can be written as vector, eg. gaussian covariance
        cl2 = self.cl_grid(l_cl=l_cl, cl=cl_cov, taper=taper, **kwargs)
        cov = np.einsum(
            "rk,k,sk->rs",
            self.wig_d[s1_s2] * np.sqrt(self.norm),
            cl2 * self.grad_l,
            self.wig_d[s1_s2_cross] * np.sqrt(self.norm),
            optimize=True,
        )
        # FIXME: Check normalization
        return self.theta, cov

    def projected_covariance2(
        self,
        l_cl=[],
        cl_cov=[],
        s1_s2=[],
        s1_s2_cross=None,
        taper=False,
        **kwargs
    ):
        """
        Convert C_ell covariance to correlation function.
        This function assumes that cl_cov is 2D matrix.

        Parameters
        ----------
        cl_cov:
            C_ell covariance matrix.
        l_cl:
            ell values at which input C_ell is computed.
        s1_s2:
            Tuple of the spin factors of the first set of tracers. Used to identify the correct wigner-d matrix to use.
        s1_s2_cross:
            Tuple of the spin factors of the second set of tracers, if different from s1_s2.
            Used to identify the correct wigner-d matrix to use.
        """

        if s1_s2_cross is None:
            s1_s2_cross = s1_s2
        cl_cov2 = cl_cov  # self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)

        cov = np.einsum(
            "rk,kk,sk->rs",
            self.wig_d[s1_s2] * np.sqrt(self.norm) * self.grad_l,
            cl_cov2,
            self.wig_d[s1_s2_cross] * np.sqrt(self.norm),
            optimize=True,
        )
        #         cov=np.dot(self.wig_d[s1_s2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[s1_s2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        # FIXME: Check normalization
        return self.theta, cov

    def taper(
        self,
        l=[],
        large_k_lower=10,
        large_k_upper=100,
        low_k_lower=0,
        low_k_upper=1.0e-5,
    ):
        """
        Function to apply tapering to input C_ell. Tapering is useful to reduce the ringing.
        This function uses the cosine function to apply the tapering.
        See eq. 71 in https://arxiv.org/pdf/2105.04548.pdf for the function and meaning of input
        parameters.

        Parameters
        ----------
        l:
            ell values at which input C_ell is computed.
        """

        # FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(
            np.isclose(self.taper_f["k"], k)
        ):
            taper_f = np.zeros_like(k)
            x = k > large_k_lower
            taper_f[x] = np.cos(
                (k[x] - large_k_lower)
                / (large_k_upper - large_k_lower)
                * np.pi
                / 2.0
            )
            x = k < large_k_lower and k > low_k_upper
            taper_f[x] = 1
            x = k < low_k_upper
            taper_f[x] = np.cos(
                (k[x] - low_k_upper)
                / (low_k_upper - low_k_lower)
                * np.pi
                / 2.0
            )
            self.taper_f = {"taper_f": taper_f, "k": k}
        return self.taper_f

    def diagonal_err(self, cov=[]):
        """
        Returns the diagonal error from the covariance. Useful for errorbar plots.
        """
        return np.sqrt(np.diagonal(cov))


def wigner_d(s1, s2, theta, l, l_use_bessel=1.0e4):
    """
    Function to compute the wigner-d matrices
    Parameters
    ----------
    s1,s2:
        Spin factors for the wigner-d matrix.
    theta:
        Angular separation for which to compute the wigner-d matrix. The matrix depends on cos(theta).
    l:
        The spherical harmonics mode ell for which to compute the matrix.
    l_use_bessel:
        Due to numerical issues, we need to switch from wigner-d matrix to bessel functions at high ell (see the note below).
        This defines the scale at which the switch happens.
    """
    l0 = np.copy(l)
    if l_use_bessel is not None:
        # FIXME: This is not great. Due to a issues with the scipy hypergeometric function,
        # jacobi can output nan for large ell, l>1.e4
        # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the
        # bessel function. Fingers and toes crossed!!!
        # mpmath is slower and also has convergence issues at large ell.
        # https://github.com/scipy/scipy/issues/4446
        l = np.atleast_1d(l)
        x = l < l_use_bessel
        l = np.atleast_1d(l[x])
    k = np.amin([l - s1, l - s2, l + s1, l + s2], axis=0)
    a = np.absolute(s1 - s2)
    lamb = 0  # lambda
    if s2 > s1:
        lamb = s2 - s1
    b = 2 * l - 2 * k - a
    d_mat = (-1) ** lamb
    d_mat *= np.sqrt(
        binom(2 * l - k, k + a)
    )  # this gives array of shape l with elements choose(2l[i]-k[i], k[i]+a)
    d_mat /= np.sqrt(binom(k + b, b))
    d_mat = np.atleast_1d(d_mat)
    x = k < 0
    d_mat[x] = 0

    d_mat = d_mat.reshape(1, len(d_mat))
    theta = theta.reshape(len(theta), 1)
    d_mat = d_mat * ((np.sin(theta / 2.0) ** a) * (np.cos(theta / 2.0) ** b))
    d_mat *= jacobi(l, a, b, np.cos(theta))

    if l_use_bessel is not None:
        l = np.atleast_1d(l0)
        x = l >= l_use_bessel
        l = np.atleast_1d(l[x])
        #         d_mat[:,x]=jn(s1-s2,l[x]*theta)
        d_mat = np.append(d_mat, jn(s1 - s2, l * theta), axis=1)
    return d_mat


def wigner_d_parallel(s1, s2, theta, l, ncpu=None, l_use_bessel=1.0e4):
    """
    compute the wigner-d matrix in parallel using multiprocessing Pool.
    This function calls the wigner-d function defined above.

    Parameters
    ----------
    ncpu:
        number of processes to use for computing the matrix.
    """
    if ncpu is None:
        ncpu = cpu_count()
    p = Pool(ncpu)
    d_mat = np.array(
        p.map(partial(wigner_d, s1, s2, theta, l_use_bessel=l_use_bessel), l)
    )
    p.close()
    return d_mat[:, :, 0].T


def bin_mat(r=[], mat=[], r_bins=[]):  # works for cov and skewness
    """
    Function to apply the binning operator. This function works on both one dimensional vectors
    and two dimensional covariance matrices.

    Parameters
    ----------
    r:
        theta or ell values at which the un-binned vector is computed.
    mat:
        Unbinned vector of C_ell or xi or the unbinned covariance
    r_bins:
        theta or ell bins to which the values should be binned.
    """
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])
    n_bins = len(bin_center)
    ndim = len(mat.shape)
    mat_int = np.zeros([n_bins] * ndim, dtype="float64")
    norm_int = np.zeros([n_bins] * ndim, dtype="float64")
    bin_idx = np.digitize(r, r_bins) - 1
    r2 = np.sort(
        np.unique(np.append(r, r_bins))
    )  # this takes care of problems around bin edges
    dr = np.gradient(r2)
    r2_idx = [i for i in np.arange(len(r2)) if r2[i] in r]
    dr = dr[r2_idx]
    r_dr = r * dr

    ls = ["i", "j", "k", "l"]
    s1 = ls[0]
    s2 = ls[0]
    r_dr_m = r_dr
    for i in np.arange(ndim - 1):
        s1 = s2 + "," + ls[i + 1]
        s2 += ls[i + 1]
        r_dr_m = np.einsum(
            s1 + "->" + s2, r_dr_m, r_dr
        )  # works ok for 2-d case

    mat_r_dr = mat * r_dr_m
    for indxs in itertools.product(
        np.arange(min(bin_idx), n_bins), repeat=ndim
    ):
        x = {}  # np.zeros_like(mat_r_dr,dtype='bool')
        norm_ijk = 1
        mat_t = []
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            # x[nd]=bin_idx==indxs[nd]
            slc[nd] = bin_idx == indxs[nd]
            if nd == 0:
                mat_t = mat_r_dr[slc]
            else:
                mat_t = mat_t[slc]
            norm_ijk *= np.sum(r_dr[slc[nd]])
        if norm_ijk == 0:
            continue
        mat_int[indxs] = np.sum(mat_t) / norm_ijk
        norm_int[indxs] = norm_ijk
    return bin_center, mat_int


def bin_cov(r=[], cov=[], r_bins=[]):
    """
    A slower function to test the bin_mat function above.
    """
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
    # cov_int=np.nan_to_num(cov_int)
    return bin_center, cov_int
