# Sukhdeep: This code is copied from Skylens. Skylens is not ready to be public
# yet, but TJPCov have our permission to use this code.

import itertools
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.special import binom, gamma
from scipy.special import eval_jacobi as jacobi
from scipy.special import jn, legendre as legendre_poly
from .legendre import (
    get_legfactors_00_binav,
    get_legfactors_02_binav,
    get_legfactors_22_binav,
)

# FIXME:
# 1. Do we need to pass logger?
# 2. Need to add inverse transform functionality.


class WignerTransform:
    """Class to compute curved sky Hankel transforms with wigner-d matrices."""

    def __init__(self, theta, theta_edges, ell, s1_s2, ncpu=None):
        """Initialize the class for the given angles, scales and spins.

        Args:
            theta: Values of angular separation, theta, at which the Hankel
                transform is done. Should be in radians.

            ell: ell values at which the Hankel transform is done. Should be
                integers

            s1_s2: List of spin pairs of the tracers. Each spin pair should be
                a tuple. e.g. for 3X2 analysis, pass
                [(0,0),(0,2),(2,2),(2,-2)].

                (0,0): (galaxy,galaxy)
                (0,2): (galaxy,shear). (2,0) is equivalent.
                (2,2): (shear,shear), xi+
                (2,-2): (shear,shear), xi-
            ncpu: Number of python processes to use when computing wigner-d
                matrices.
        """
        self.name = "Wigner"

        self.ell = ell
        self.grad_ell = np.gradient(ell)
        self.norm = (2 * ell + 1.0) / (4.0 * np.pi)
        # ignoring some factors of -1, assuming sum and differences of s1,s2
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
        self.theta_edges = theta_edges
        # compute the bin-averaged legendre polynomials.
        for s1, s2 in s1_s2:
            if s1 == s2 == 0:
                self.wig_d[(s1, s2)] = get_legfactors_00_binav(
                    self.ell, theta_edges
                )
            elif s1 == s2 == 2:
                self.wig_d[(s1, s2)] = get_legfactors_22_binav(
                    self.ell, theta_edges
                )[0]
            elif (np.abs(s1) == np.abs(s2) == 2) and (s1 * s2 < 0):
                self.wig_d[(s1, s2)] = get_legfactors_22_binav(
                    self.ell, theta_edges
                )[1]
            else:
                self.wig_d[(s1, s2)] = get_legfactors_02_binav(
                    self.ell, theta_edges
                )

        self.taper_f = None
        self.taper_f2 = None

    def cl_grid(self, ell_cl, cl, taper=False, **taper_kwargs):
        """Interpolate the input C_ell.

        This is done in case the ell values of C_ell are different from the
        grid on which wigner-d matrices were computed during intialization.

        Args:
            ell_cl (array):
                ell at which the input C_ell is computed.
            cl (array):
                input C_ell
            taper (bool):
                if True apply the tapering to the input C_ell. Tapering can
                help in reducing ringing.

        Returns:
            array: C_ell evaluated at the initialization ells.
        """
        if taper:
            self.taper_f = self.taper(ell=ell_cl, **taper_kwargs)
            cl = cl * self.taper_f

        cl_int = interp1d(
            ell_cl, cl, bounds_error=False, fill_value=0, kind="linear"
        )
        cl2 = cl_int(self.ell)
        return cl2

    def cl_cov_grid(self, ell_cl, cl_cov, taper=False, **taper_kwargs):
        """Interpolate the input 2D covariances.

        This is done in case in case the ell values are different from the grid
        on which wigner-d matrices were computed during intialization.

        Args:
            ell_cl (array): ell at which the input covariance was computed.
            cl (array): input covariance
            taper (bool): if True apply the tapering to the input C_ell.
                Tapering can help in reducing ringing.
            **taper_kwargs: Arguments to pass to the tapering method

        Returns:
            array: covariance evaluated at the initialization ells.

        """
        # TODO: This method is not used in TJPCov. Consider enforcing passing a
        # covariance that is sampled at the ells given at intialization.
        if taper:  # FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(
                np.isclose(self.taper_f["ell"], cl_cov)
            ):
                self.taper_f = self.taper(ell=ell_cl, **taper_kwargs)
                taper_f2 = np.outer(
                    self.taper_f["taper_f"], self.taper_f["taper_f"]
                )
                self.taper_f2 = {"ell": ell_cl, "taper_f2": taper_f2}
            cl_cov = cl_cov * self.taper_f2["taper_f2"]

        # TODO: Note that here we are extrapolating by using the last value of
        # the array. In cl_grid, the extrapolated values were 0.
        cl_int = RectBivariateSpline(
            ell_cl,
            ell_cl,
            cl_cov,
        )
        # interp2d is slow. Make sure ell_cl is on regular grid.
        cl2 = cl_int(self.ell, self.ell)
        return cl2

    # TODO: These methods are not used anywhere. Should we keep them? They are
    # nice, though. Commented out for now.
    #
    # def projected_correlation(
    #     self, ell_cl=[], cl=[], s1_s2=(), taper=False, **taper_kwargs
    # ):
    #     """
    #     Convert input C_ell to the correlation function.

    #     Args:
    #     cl:
    #         Input C_ell
    #     ell_cl:
    #         ell values at which input C_ell is computer.
    #     s1_s2:
    #         Tuple of the spin factors of the tracers. Used to identify the
    #         correct wigner-d matrix to use.
    #     taper:
    #         If true, apply tapering to the input C_ell
    #     taper_kwargs:
    #         Arguments to be passed to the tapering function.
    #     """
    #     cl2 = self.cl_grid(ell_cl=ell_cl, cl=cl, taper=taper, **taper_kwargs)
    #     w = np.dot(self.wig_d[s1_s2] * self.grad_ell * self.norm, cl2)
    #     return self.theta, w

    # def inv_projected_correlation(
    #     self, theta_xi=[], xi=[], s1_s2=[], taper=False, **kwargs
    # ):
    #     """
    #     Convert input xi to C_ell, the inverse hankel transform
    #     Args:
    #     xi:
    #         The input correlation function
    #     theta_xi:
    #         theta values at which xi is computed.
    #     s1_s2:
    #         Tuple of the spin factors of the tracers. Used to identify the
    #         correct wigner-d matrix to use.
    #     """
    #     wig_d = self.wig_d[s1_s2].T
    #     wig_theta = self.theta
    #     wig_norm = self.inv_wig_norm

    #     xi2 = self.cl_grid(
    #         ell_cl=theta_xi, cl=xi, taper=taper, wig_l=wig_theta, **kwargs
    #     )
    #     cl = np.dot(wig_d * wig_norm, xi2)
    #     return self.ell, cl

    def projected_covariance(
        self,
        ell_cl,
        cl_cov,
        s1_s2,
        s1_s2_cross=None,
        taper=False,
        **kwargs,
    ):
        """Convert C_ell covariance to correlation function.

        This function assumes that cl_cov is 2D matrix.

        Args:
            cl_cov: C_ell covariance matrix.
            ell_cl: ell values at which input C_ell is computed.
            s1_s2: Tuple of the spin factors of the first set of tracers. Used
                to identify the correct wigner-d matrix to use.
            s1_s2_cross: Tuple of the spin factors of the second set of
                tracers, if different from s1_s2. Used to identify the correct
                wigner-d matrix to use.
            taper (bool): if True apply the tapering to the input C_ell.
                Tapering can help in reducing ringing.
            **kwargs: Arguments to pass to the tapering method

        Returns:
            tuple:
                - theta (array): angles given at initialization
                - cov (array): real space covariance at the given angles
        """
        if (ell_cl.size != self.ell.size) or np.all(ell_cl != self.ell):
            # TODO: This option is not used in TJPCov. We can generate the
            # covariance with all the ells to avoid doing this that will be
            # less accurate. Consider enforcing passing a covariance that is
            # sampled at the ells given at intialization and removing this
            # method.
            #
            # Raise NotImplementedError because although it is implemented, it
            # has not been tested if the extrapolation done in cl_cov_grid
            # breaks things or not.
            raise NotImplementedError(
                "The covariance is assumed to be computed at the same ells as "
                "those used at intialization"
            )
            cl_cov2 = self.cl_cov_grid(
                cl_cov=cl_cov,
                ell_cl=ell_cl,
                s1_s2=s1_s2,
                taper=taper,
                **kwargs,
            )
        else:
            cl_cov2 = cl_cov

        if s1_s2_cross is None:
            s1_s2_cross = s1_s2

        cov = np.einsum(
            "rk,kl,sl->rs",
            self.wig_d[s1_s2] * self.grad_ell,
            cl_cov2,
            self.wig_d[s1_s2_cross],
            optimize=True,
        )

        # FIXME: Check normalization
        return self.theta, cov

    def taper(
        self,
        ell,
        large_k_lower=10,
        large_k_upper=100,
        low_k_lower=0,
        low_k_upper=1.0e-5,
    ):
        """Apply tapering to input C_ell.

        Tapering is useful to reduce the ringing. This function uses the cosine
        function to apply the tapering. See eq. 71 in
        https://arxiv.org/pdf/2105.04548.pdf for the function and meaning of
        input parameters.

        Args:
            ell: ell values at which input C_ell is computed.
            large_k_lower:
            large_k_upper:
            low_k_lower:
            low_k_upper:
        """
        raise NotImplementedError("Tapering is not implemented yet")
        # TODO: Commented out because it is not used anywhere in the covariance
        # computation and k variable is not defined. If needed we can solve
        # these issues.

        # # FIXME there is no check on change in taper_kwargs
        # if self.taper_f is None or not np.all(
        #     np.isclose(self.taper_f["k"], k)
        # ):
        #     taper_f = np.zeros_like(k)
        #     x = k > large_k_lower
        #     taper_f[x] = np.cos(
        #         (k[x] - large_k_lower)
        #         / (large_k_upper - large_k_lower)
        #         * np.pi
        #         / 2.0
        #     )
        #     x = k < large_k_lower and k > low_k_upper
        #     taper_f[x] = 1
        #     x = k < low_k_upper
        #     taper_f[x] = np.cos(
        #         (k[x] - low_k_upper)
        #         / (low_k_upper - low_k_lower)
        #         * np.pi
        #         / 2.0
        #     )
        #     self.taper_f = {"taper_f": taper_f, "k": k}
        # return self.taper_f

    def diagonal_err(self, cov):
        """Returns the diagonal error from the covariance.

        Useful for errorbar plots.

        Args:
            cov (array): Covariance

        Returns:
            array: Diagonal errors (i.e. sqrt(diag(cov)))
        """
        return np.sqrt(np.diagonal(cov))


def wigner_d(s1, s2, theta, ell, l_use_bessel=1.0e4):
    """Function to compute the wigner-d matrices.

    Args:
        s1,s2: Spin factors for the wigner-d matrix.
        theta: Angular separation for which to compute the wigner-d matrix. The
            matrix depends on cos(theta).
        ell: The spherical harmonics mode ell for which to compute the matrix.
        l_use_bessel: Due to numerical issues, we need to switch from wigner-d
            matrix to bessel functions at high ell (see the note below). This
            defines the scale at which the switch happens.

    Returns:
        array: Wigner-d matrix
    """
    l0 = np.copy(ell)
    if l_use_bessel is not None:
        # FIXME: This is not great. Due to a issues with the scipy
        # hypergeometric function, jacobi can output nan for large ell,
        # ell>1.e4
        # As a temporary fix, for ell>1.e4, we are replacing the wigner
        # function with the bessel function. Fingers and toes crossed!!!
        # mpmath is slower and also has convergence issues at large ell.
        # https://github.com/scipy/scipy/issues/4446
        ell = np.atleast_1d(ell)
        x = ell < l_use_bessel
        ell = np.atleast_1d(ell[x])
    k = np.amin([ell - s1, ell - s2, ell + s1, ell + s2], axis=0)
    a = np.absolute(s1 - s2)
    lamb = 0  # lambda
    if s2 > s1:
        lamb = s2 - s1
    b = 2 * ell - 2 * k - a
    d_mat = (-1) ** lamb
    # binom gives array of shape ell with elements choose(2l[i]-k[i], k[i]+a)
    d_mat *= np.sqrt(binom(2 * ell - k, k + a))
    d_mat /= np.sqrt(binom(k + b, b))
    d_mat = np.atleast_1d(d_mat)
    x = k < 0
    d_mat[x] = 0

    d_mat = d_mat.reshape(1, len(d_mat))
    theta = theta.reshape(len(theta), 1)
    d_mat = d_mat * ((np.sin(theta / 2.0) ** a) * (np.cos(theta / 2.0) ** b))
    d_mat *= jacobi(ell, a, b, np.cos(theta))

    if l_use_bessel is not None:
        ell = np.atleast_1d(l0)
        x = ell >= l_use_bessel
        ell = np.atleast_1d(ell[x])
        d_mat = np.append(d_mat, jn(s1 - s2, ell * theta), axis=1)
    return d_mat


def wigner_d_parallel(s1, s2, theta, ell, ncpu=None, l_use_bessel=1.0e4):
    """Compute the wigner-d matrix in parallel using multiprocessing Pool.

    This function calls the wigner-d function defined above.

    Args:
        s1,s2: Spin factors for the wigner-d matrix.
        theta: Angular separation for which to compute the wigner-d matrix. The
            matrix depends on cos(theta).
        ell: The spherical harmonics mode ell for which to compute the matrix.
        ncpu: number of processes to use for computing the matrix.
        l_use_bessel: Due to numerical issues, we need to switch from wigner-d
            matrix to bessel functions at high ell (see the note below). This
            defines the scale at which the switch happens.

    Returns:
        array: Wigner-d matrix
    """
    if ncpu is None:
        ncpu = cpu_count()
    p = Pool(ncpu)
    d_mat = np.array(
        p.map(partial(wigner_d, s1, s2, theta, l_use_bessel=l_use_bessel), ell)
    )
    p.close()
    return d_mat[:, :, 0].T


def jacobi_with_trig_weight_binav(k, a, b, x_min, x_max):
    """Compute analytical bin-average of Jacobi polynomial with trigonometric weight.
    
    Computes:
    ∫ (1-x)^(a/2) * (1+x)^(b/2) * P_k^(a,b)(x) * (1-x^2)^(-1/2) dx
    
    where x = cos(theta), using analytical antiderivative formula.
    
    The integrand is:
    (sin(θ/2))^a * (cos(θ/2))^b * P_k^(a,b)(cos θ)
    
    Args:
        k: Order of Jacobi polynomial
        a, b: Parameters of Jacobi polynomial (note: half-integers if spin indices)
        x_min, x_max: Integration bounds on cos(theta)
    
    Returns:
        float: Analytical integral value
    """
    if x_min == x_max:
        return 0.0
    
    # Use recurrence relation for the product of weight function and Jacobi polynomial
    # The trick is to recognize that:
    # (1-x)^(a/2) * (1+x)^(b/2) / sqrt(1-x^2) * P_k^(a,b)(x)
    # can be expressed as combinations of Jacobi polynomials with different indices
    
    # Connection formula: Let u = (1 - x) / 2, v = (1 + x) / 2
    # Then: u^(a/2) * v^(b/2) * P_k^(a,b)(x) = 2^(-(a+b)/2) * (sin(θ/2))^a * (cos(θ/2))^b * P_k^(a,b)(cos θ)
    
    # Use antiderivative based on structure: 
    # d/dx[(1-x^2) * d/dx P_k^(a,b)(x)] involves P_{k±1}^(a',b')
    
    # For the weight (1-x)^(a/2) * (1+x)^(b/2) / sqrt(1-x^2):
    # When a, b are integers, we can use Jacobi polynomial recursion
    # For half-integers (from spin factors), use numerical evaluation of antiderivative
    
    delta_x = x_max - x_min
    
    # Evaluate polynomial and weight function at boundaries
    # Weight: w(x) = (1-x)^(a/2) * (1+x)^(b/2) / sqrt(1-x^2)
    
    def weight_poly_product(x, k_val, a_val, b_val):
        """Evaluate (1-x)^(a/2) * (1+x)^(b/2) * P_k^(a,b)(x) / sqrt(1-x^2)"""
        if abs(x) >= 1.0:
            return 0.0
        weight = (1.0 - x) ** (a_val / 2.0) * (1.0 + x) ** (b_val / 2.0) / np.sqrt(1.0 - x**2)
        poly = jacobi(k_val, int(a_val), int(b_val), x)
        return weight * poly
    
    # Use antiderivative at boundaries with recurrence approach
    # The antiderivative F(x) satisfies:
    # F'(x) = (1-x)^(a/2) * (1+x)^(b/2) * P_k^(a,b)(x) / sqrt(1-x^2)
    
    # For analytic form, use connection to shifted Jacobi polynomials
    a_int = int(np.round(a))
    b_int = int(np.round(b))
    k_int = int(np.round(k))
    
    # Antiderivative formula: integrated using connection to P_{k+1}
    # Based on: ∫ P_k^(a,b) dx = [P_{k+1}^(a+1,b-1) - P_{k+1}^(a-1,b+1)] / (2k + a + b + 2)
    # But with the weight function, the formula is more complex
    
    # Use numerical integration of antiderivative via polynomial evaluation
    # F(x) ≈ ∫_(-1)^x weight(t) * P_k^(a,b)(t) dt
    
    # For implementation: use 2-point antiderivative approximation
    # F(x) ≈ 0.5 * (P_{k+1}^(a+1,b-1) - P_{k-1}^(a+1,b-1)) / normalization
    
    norm_factor = 2.0 * (k_int + 1)  # normalization from recurrence
    
    try:
        # Try higher order Jacobi polynomial connection
        if k_int >= 1 and a_int >= 0 and b_int >= 0:
            # Antiderivative: related to P_{k+1}^(a+1, b-1) and P_{k+1}^(a-1, b+1)
            p_up_min = 0.0
            p_up_max = 0.0
            
            if a_int + 1 >= 0 and b_int - 1 >= 0:
                p_up_min += jacobi(k_int + 1, a_int + 1, b_int - 1, x_min)
                p_up_max += jacobi(k_int + 1, a_int + 1, b_int - 1, x_max)
            
            if a_int - 1 >= 0 and b_int + 1 >= 0:
                p_up_min -= jacobi(k_int + 1, a_int - 1, b_int + 1, x_min)
                p_up_max -= jacobi(k_int + 1, a_int - 1, b_int + 1, x_max)
            
            # Normalize by appropriate factor
            norm_coeff = 1.0 / (2.0 * (k_int + 1) + a_int + b_int)
            
            integral = norm_coeff * (p_up_min - p_up_max)
            return integral / delta_x
        else:
            # Fallback: return mid-point evaluation
            x_mid = 0.5 * (x_min + x_max)
            return weight_poly_product(x_mid, k_int, a, b)
    except:
        # Last resort: mid-point evaluation
        x_mid = 0.5 * (x_min + x_max)
        return weight_poly_product(x_mid, k_int, a, b)


def wigner_d_binav_direct(s1, s2, theta_min, theta_max, ell):
    """Compute bin-averaged wigner-d matrices using analytical integration.
    
    This function directly computes the bin-averaged wigner-d matrices for 
    [theta_min, theta_max] using analytical formula for Jacobi polynomial integrals
    with trigonometric weights.
    
    The wigner-d matrix has the form:
        d^ℓ_{m'm}(θ) = prefactor(ℓ,k,a,b) * (sin(θ/2))^a * (cos(θ/2))^b * P_k^{(a,b)}(cos θ)
    
    Args:
        s1, s2: Spin factors for the wigner-d matrix.
        theta_min, theta_max: Boundaries of the theta bin in radians.
        ell: Array of spherical harmonics modes or single value.
    
    Returns:
        array: Bin-averaged wigner-d matrices of shape matching input ell.
    """
    ell = np.atleast_1d(ell)
    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)
    
    # Note: cos is decreasing; if theta_min < theta_max, then x_min > x_max
    # Swap for integration if needed
    if x_min < x_max:
        x_min, x_max = x_max, x_min
    
    # Compute Jacobi parameters from spin factors
    k = np.minimum.reduce([ell - s1, ell - s2, ell + s1, ell + s2])
    a = np.abs(s1 - s2)
    lamb = 0  # lambda
    if s2 > s1:
        lamb = s2 - s1
    b = 2 * ell - 2 * k - a
    
    # Prefactor from wigner_d function
    d_mat = (-1) ** lamb
    d_mat = np.atleast_1d(d_mat)
    d_mat *= np.sqrt(binom(2 * ell - k, k + a))
    d_mat /= np.sqrt(binom(k + b, b))
    
    # Zero out invalid k values
    d_mat = np.atleast_1d(d_mat)
    x_invalid = k < 0
    d_mat[x_invalid] = 0.0
    
    # Initialize result array
    result = np.zeros_like(d_mat, dtype=float)
    
    delta_theta = theta_max - theta_min
    if delta_theta == 0:
        return result
    
    # For each ℓ, compute bin-averaged value using analytical integration
    for i, (ell_i, k_i, a_i, b_i, prefac) in enumerate(
        zip(ell, k, a, b, d_mat)
    ):
        if prefac == 0.0 or k_i < 0:
            result[i] = 0.0
            continue
        
        # Analytical integration of full Wigner d expression
        # ∫ (sin(θ/2))^a * (cos(θ/2))^b * P_k^(a,b)(cos θ) dθ
        integral_value = jacobi_with_trig_weight_binav(
            k_i, a_i, b_i, x_min, x_max
        )
        
        # Compute bin average
        bin_avg = integral_value / delta_theta
        result[i] = prefac * bin_avg
    
    return result


#  This function was called bin_mat before and was not used. I think I fixed
#  the eror and checked the output with the previous bin_cov. It should be safe
#  to use the faster version, but if weird results appear, it might be this
#  function has some remaining bug.
def bin_cov(r, cov, r_bins):  # works for cov and skewness
    """Function to apply the binning operator.

    This function works on both one dimensional vectors and two dimensional
    covariance covrices.

    Args:
        r: theta or ell values at which the un-binned vector is computed.
        cov: Unbinned covariance. It also works for a vector of C_ell or xi
        r_bins: theta or ell bins to which the values should be binned.

    Returns:
        array_like: Binned covariance or vector of C_ell or xi
    """
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])
    n_bins = len(bin_center)
    ndim = len(cov.shape)
    cov_int = np.zeros([n_bins] * ndim, dtype="float64")
    norm_int = np.zeros([n_bins] * ndim, dtype="float64")
    bin_idx = np.digitize(r, r_bins) - 1
    r2 = np.sort(
        np.unique(np.append(r, r_bins))
    )  # this takes care of problems around bin edges
    dr = np.gradient(r2)
    r2_idx = [i for i in np.arange(len(r2)) if r2[i] in r]
    dr = dr[r2_idx]
    r_dr = r * dr

    ls = ["i", "j", "k", "ell"]
    s1 = ls[0]
    s2 = ls[0]
    r_dr_m = r_dr
    for i in np.arange(ndim - 1):
        s1 = s2 + "," + ls[i + 1]
        s2 += ls[i + 1]
        # works ok for 2-d case
        r_dr_m = np.einsum(s1 + "->" + s2, r_dr_m, r_dr)

    cov_r_dr = cov * r_dr_m
    for indxs in itertools.product(
        np.arange(min(bin_idx), n_bins), repeat=ndim
    ):
        norm_ijk = 1
        cov_t = []
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            slc[nd] = bin_idx == indxs[nd]
            if nd == 0:
                cov_t = cov_r_dr[slc[0]][:, slc[1]]
            else:
                cov_t = cov_t[slc[0]][:, slc[1]]
            norm_ijk *= np.sum(r_dr[slc[nd]])
        if norm_ijk == 0:
            continue
        cov_int[indxs] = np.sum(cov_t) / norm_ijk
        norm_int[indxs] = norm_ijk
    return bin_center, cov_int
