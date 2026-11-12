"""
Pytest configuration additions.

pytest loads ``conftest.py`` automatically before collecting the tests,
so the fixtures defined here can be requested by name (as test function
arguments) in any test module under ``tests/`` without importing them.
"""

import os

import healpy as hp
import numpy as np
import pyccl
import pytest
import sacc

INPUT_YML = "./tests/data/conf_covariance_clusters.yaml"
OUTDIR = "./tests/tmp/"


def cap_mask(area, nside=128):
    """HEALPix mask of a spherical cap with the given area in sr."""
    mask = np.zeros(hp.nside2npix(nside))
    theta = np.arccos(1 - area / (2 * np.pi))
    mask[hp.query_disc(nside, [0, 0, 1], theta)] = 1
    return mask


@pytest.fixture(name="save_cluster_sacc_data", scope="module")
def fixture_save_cluster_sacc_data() -> sacc.Sacc:
    """Return and save a Sacc object with cluster data."""
    survey_name = "NC_mock_redshift_richness"
    area = 5000.0
    s_count = sacc.Sacc()
    s_count.add_tracer("survey", survey_name, area)

    # Define bin information. The reference covariances were computed with
    # these edges (redshift and richness rounded to two decimals).
    z_edges = [0.2, 0.28, 0.35, 0.43, 0.5, 0.57, 0.65]
    log_rich_edges = np.log10([7.39, 16.14, 35.25, 77.0])
    bin_info = [
        {"name": f"bin_z_{i}", "lower": lo, "upper": hi}
        for i, (lo, hi) in enumerate(zip(z_edges[:-1], z_edges[1:]))
    ] + [
        {"name": f"bin_rich_{i}", "lower": lo, "upper": hi}
        for i, (lo, hi) in enumerate(
            zip(log_rich_edges[:-1], log_rich_edges[1:])
        )
    ]

    # Add bin tracers
    for bin_data in bin_info:
        s_count.add_tracer(
            "bin_z" if "bin_z" in bin_data["name"] else "bin_richness",
            bin_data["name"],
            bin_data["lower"],
            bin_data["upper"],
        )
    cc = sacc.standard_types.cluster_counts
    mlm = sacc.standard_types.cluster_mean_log_mass
    # Define cluster counts and mean log mass data
    data_points = [
        (cc, ("bin_rich_0", "bin_z_0"), 35142),
        (cc, ("bin_rich_0", "bin_z_1"), 6361),
        (cc, ("bin_rich_0", "bin_z_2"), 160),
        (cc, ("bin_rich_0", "bin_z_3"), 53273),
        (cc, ("bin_rich_0", "bin_z_4"), 9136),
        (cc, ("bin_rich_0", "bin_z_5"), 228),
        (cc, ("bin_rich_1", "bin_z_0"), 71499),
        (cc, ("bin_rich_1", "bin_z_1"), 11747),
        (cc, ("bin_rich_1", "bin_z_2"), 265),
        (cc, ("bin_rich_1", "bin_z_3"), 88108),
        (cc, ("bin_rich_1", "bin_z_4"), 13841),
        (cc, ("bin_rich_1", "bin_z_5"), 309),
        (cc, ("bin_rich_2", "bin_z_0"), 102306),
        (cc, ("bin_rich_2", "bin_z_1"), 15722),
        (cc, ("bin_rich_2", "bin_z_2"), 278),
        (cc, ("bin_rich_2", "bin_z_3"), 114798),
        (cc, ("bin_rich_2", "bin_z_4"), 16761),
        (cc, ("bin_rich_2", "bin_z_5"), 310),
        (mlm, ("bin_rich_0", "bin_z_0"), 13.4075),
        (mlm, ("bin_rich_0", "bin_z_1"), 13.7930),
        (mlm, ("bin_rich_0", "bin_z_2"), 14.3036),
        (mlm, ("bin_rich_0", "bin_z_3"), 13.4047),
        (mlm, ("bin_rich_0", "bin_z_4"), 13.7661),
        (mlm, ("bin_rich_0", "bin_z_5"), 14.3080),
        (mlm, ("bin_rich_1", "bin_z_0"), 13.3987),
        (mlm, ("bin_rich_1", "bin_z_1"), 13.7474),
        (mlm, ("bin_rich_1", "bin_z_2"), 14.2155),
        (mlm, ("bin_rich_1", "bin_z_3"), 13.3920),
        (mlm, ("bin_rich_1", "bin_z_4"), 13.7302),
        (mlm, ("bin_rich_1", "bin_z_5"), 14.2325),
        (mlm, ("bin_rich_2", "bin_z_0"), 13.3862),
        (mlm, ("bin_rich_2", "bin_z_1"), 13.7171),
        (mlm, ("bin_rich_2", "bin_z_2"), 14.1222),
        (mlm, ("bin_rich_2", "bin_z_3"), 13.3796),
        (mlm, ("bin_rich_2", "bin_z_4"), 13.6876),
        (mlm, ("bin_rich_2", "bin_z_5"), 14.1026),
    ]

    # Add data points
    # NOTE: tracer order must be (survey, richness, z)
    for data_type, (bin_richness_label, bin_z_label), value in data_points:
        s_count.add_data_point(
            data_type, (survey_name, bin_richness_label, bin_z_label), value
        )

    # Add covariance
    cov = [
        3.5142e4,
        6.361e3,
        1.6e2,
        5.3273e4,
        9.136e3,
        2.28e2,
        7.1499e4,
        1.1747e4,
        2.65e2,
        8.8108e4,
        1.3841e4,
        3.09e2,
        1.02306e5,
        1.5722e4,
        2.78e2,
        1.14798e5,
        1.6761e4,
        3.1e2,
        2.59949557e-6,
        2.90651791e-05,
        9.43896703e-04,
        1.68357584e-06,
        1.92845101e-05,
        7.04260444e-04,
        1.22804096e-06,
        1.48490066e-05,
        7.29613214e-04,
        9.67117735e-07,
        1.20824143e-05,
        6.20407091e-04,
        8.10018388e-07,
        1.03367315e-05,
        6.39281417e-04,
        7.02045729e-07,
        9.33062031e-06,
        6.01163806e-04,
    ]
    s_count.add_covariance(np.diag(cov))

    # Finalize and save
    s_count.to_canonical_order()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_dir, "tmp", "cluster_redshift_richness_sacc_data.fits"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s_count.save_fits(save_path, overwrite=True)
    return s_count


@pytest.fixture
def mock_cosmo():
    """Mock a CCL cosmology object shared by the cluster covariance tests."""
    Omg_c = 0.2640
    Omg_b = 0.0493
    h0 = 0.6736
    sigma8_value = 0.8111
    n_s_value = 0.9649
    w_0 = -1.0
    w_a = 0.0

    cosmo = pyccl.Cosmology(
        Omega_c=Omg_c,
        Omega_b=Omg_b,
        h=h0,
        sigma8=sigma8_value,
        n_s=n_s_value,
        w0=w_0,
        wa=w_a,
    )
    return cosmo
