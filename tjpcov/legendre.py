import numpy as np
from scipy.special import legendre_p_all


def get_bin_averaged_Pl(ell, x_edges):
    """
    Based on Eq. (65) of https://arxiv.org/abs/2012.08568,
    compute P_bar_l averaged over bins defined by x_edges (cos(theta_edges)).
    Return shape: (len(theta), len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1

    # Compute P_0 to P_{ell_max+1} using legendre_p_all
    # Shape: (ell_max+2, n_edges)
    p_all = legendre_p_all(ell_max + 1, x_edges)[0]

    # Get P_{ell+1} and P_{ell-1} by index
    p_lp1 = p_all[ell + 1]  # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]  # (len(ell), n_edges)

    # Difference: [P_{l+1}(x) - P_{l-1}(x)]
    vals = p_lp1 - p_lm1  # (len(ell), n_edges)

    # Take differences for each bin (note: x_edges is reversed)
    delta_vals = vals[:, :-1] - vals[:, 1:]  # (len(ell), len(theta))
    delta_x = x_edges[:-1] - x_edges[1:]  # (len(theta),)

    # Eq. (65): Divide by (2l+1) and delta_x in denominator
    p_bar = delta_vals / ((2 * ell[:, None] + 1) * delta_x)
    return p_bar.T  # Transpose to (len(theta), len(ell))


def get_bin_averaged_P2l(ell, x_edges):
    """
    Based on Eq. (B2) of https://arxiv.org/pdf/2012.08568,
    compute P^2_l averaged over bins defined by x_edges (cos(theta_edges)).
    Return shape: (len(theta), len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1

    # Compute P_0 to P_{ell_max+1} using legendre_p_all
    # Shape: (ell_max+2, n_edges)
    p_all = legendre_p_all(ell_max + 1, x_edges)[0]

    # Get Legendre polynomials by index
    p_l = p_all[ell]  # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]
    p_lp1 = p_all[ell + 1]

    # Compute each term in Eq. (B2)
    term1 = (ell + 2 / (2 * ell + 1))[:, None] * p_lm1
    term2 = (2 - ell)[:, None] * x_edges * p_l
    term3 = -(2 / (2 * ell + 1))[:, None] * p_lp1

    vals = term1 + term2 + term3

    # Compute bin-wise differences
    delta_vals = vals[:, :-1] - vals[:, 1:]
    delta_x = x_edges[:-1] - x_edges[1:]

    return (delta_vals / delta_x).T  # Transpose to (len(theta), len(ell))


def get_bin_averaged_G2l(ell, x_edges, sign):
    """
    Based on Eq. (B5) of https://arxiv.org/pdf/2012.08568,
    compute G_l,2 averaged over bins defined by x_edges (cos(theta_edges)),
    with sign = +1 for xi_plus and -1 for xi_minus.
    Return shape: (len(theta), len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1

    # Compute P_0 to P_{ell_max+1} using legendre_p_all
    # Shape: (ell_max+2, n_edges)
    p_all, dp_all = legendre_p_all(ell_max + 1, x_edges, diff_n=1)

    # Get Legendre polynomials and derivatives by index
    p_l = p_all[ell]  # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]
    p_lp1 = p_all[ell + 1]

    dp_l = dp_all[ell]  # (len(ell), n_edges)
    dp_lm1 = dp_all[ell - 1]

    # Compute each term in Eq. (B5)
    term1 = (
        -(ell * (ell - 1) / 2.0)[:, None]
        * (ell[:, None] + 2 / (2 * ell[:, None] + 1))
        * p_lm1
    )
    term2 = -(ell * (ell - 1) * (2 - ell) / 2.0)[:, None] * x_edges * p_l
    term3 = (ell * (ell - 1) / (2 * ell + 1))[:, None] * p_lp1
    term4 = (4 - ell[:, None]) * dp_l
    term5 = (ell[:, None] + 2) * (x_edges * dp_lm1 - p_lm1)

    term6_7 = (
        sign
        * 2.0
        * (
            ((ell[:, None] - 1) * (x_edges * dp_l - p_l))
            - (ell[:, None] + 2) * dp_lm1
        )
    )

    vals = term1 + term2 + term3 + term4 + term5 + term6_7

    # Compute bin-wise differences
    delta_vals = vals[:, :-1] - vals[:, 1:]
    delta_x = x_edges[:-1] - x_edges[1:]

    return (delta_vals / delta_x).T  # (len(theta), len(ell))


# --- Main functions ---
def get_legfactors_00_binav(ell_array, theta_edges):
    """
    Calculate the Legendre factors (2l + 1)/(4 pi) F_l^AB(theta)
    defined in Eq. (10) of https://arxiv.org/abs/2012.08568,
    for the 00 component, averaged over bins defined by theta_edges.
    Arguments:
        ell_array: Array of multipole moments (e.g., np.arange(ell_max))
        theta_edges: Array of theta bin boundaries (radians)
    Return value:
        (2l + 1)/(4 pi) F_l^AB(theta): (len(theta), len(ell_array))
    """
    x_edges = np.cos(theta_edges)

    # Handle ell < 1
    mask = ell_array < 1
    safe_ell = np.where(mask, 1, ell_array)

    Pl_bar = get_bin_averaged_Pl(safe_ell, x_edges)  # (len(theta), n_ell)

    factor = (2 * safe_ell + 1) / (4 * np.pi)
    result = factor * Pl_bar
    result[:, mask] = 0.0  # ell < 1 is 0

    return result


def get_legfactors_02_binav(ell_array, theta_edges):
    """
    Same as get_legfactors_00_binav but for the 02 component.
    """
    x_edges = np.cos(theta_edges)

    # Handle ell < 2
    mask = ell_array < 2
    safe_ell = np.where(mask, 1, ell_array)

    P2l_bar = get_bin_averaged_P2l(safe_ell, x_edges)  # (len(theta), n_ell)

    factor = (2 * safe_ell + 1) / (4 * np.pi * (safe_ell * (safe_ell + 1)))
    result = factor * P2l_bar
    result[:, mask] = 0.0  # ell < 2 is 0

    return result


def get_legfactors_22_binav(ell_array, theta_edges, sign):
    """
    Same as get_legfactors_00_binav but for the 22 component,
    with sign = +1 for xi_plus and -1 for xi_minus.
    """
    x_edges = np.cos(theta_edges)

    # Handle ell < 2
    mask = ell_array < 2
    safe_ell = np.where(mask, 1, ell_array)

    G2l_bar = get_bin_averaged_G2l(
        safe_ell, x_edges, sign=sign
    )  # (len(theta), n_ell)

    factor = (2 * safe_ell + 1) / (
        2 * np.pi * (safe_ell**2 * (safe_ell + 1) ** 2)
    )
    result = factor * G2l_bar
    result[:, mask] = 0.0  # ell < 2 is 0

    return result
