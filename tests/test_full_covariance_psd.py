#!/usr/bin/python3
"""PSD and correlation checks of the full Gaussian + SSC + cNG covariance.

The full matrix is assembled from blocks validated earlier in the session:
  - SSC, cNG: the blocks saved by test_covariance_fourier_ssc_fsky.py and
    test_covariance_fourier_cNG_fsky.py after their CCL comparisons pass
    (tests/validated_blocks/).
  - Gaussian, two versions (each test runs for both):
      "nmt":  the DES NaMaster benchmark blocks (benchmarks/.../cov), which
              the NaMaster Gaussian tests compare against.
      "fsky": FourierGaussianFsky with the same fsky config as the SSC/cNG
              blocks (cheap; checked in test_covariance_gaussian_fsky.py).

Run it in the same pytest session as those tests, e.g.

    pytest tests/test_covariance_fourier_ssc_fsky.py \
        tests/test_covariance_fourier_cNG_fsky.py \
        tests/test_full_covariance_psd.py

Only the spin-0/E component of each spectrum is used. The checks need a
closed set of spectra: every block between them must exist. The tests are
skipped without validated SSC blocks; the cNG check is skipped without
validated cNG blocks (the cNG tests are skipped on GitHub Actions).
"""

import itertools
import os

import numpy as np
import pytest
import sacc
import yaml

from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky

ROOT = "./tests/benchmarks/32_DES_tjpcov_bm/"
INPUT_YML_FSKY = "./tests/data/conf_covariance_ssc_fsky.yaml"
VALIDATED_BLOCKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "validated_blocks"
)


# --- checks -------------------------------------------------------------


def correlation(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def assert_symmetric(cov, rtol=1e-10):
    scale = np.max(np.abs(cov))
    assert np.max(np.abs(cov - cov.T)) <= rtol * scale


def assert_psd(cov, rtol=1e-10, strict=False):
    """PSD (or, if strict, positive-definite) check.

    Done on the correlation matrix, since clustering and shear blocks differ
    by orders of magnitude and a raw eigenvalue tolerance would hide a
    negative eigenvalue in the small blocks.
    """
    assert np.all(np.isfinite(cov)), "non-finite entries"
    assert np.all(np.diag(cov) > 0), "non-positive variance"
    r = correlation(cov)
    assert_symmetric(r, rtol)
    # Cauchy-Schwarz: implied by PSD, but gives a readable failure
    worst = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    assert (
        np.abs(r[worst]) <= 1 + rtol
    ), f"|correlation| = {abs(r[worst]):.3g} at {worst}"
    eig = np.linalg.eigvalsh(r)
    if strict:
        np.linalg.cholesky(r)
        assert eig[0] > rtol * eig[-1], f"min eigenvalue {eig[0]:.3e}"
    else:
        assert eig[0] >= -rtol * eig[-1], f"min eigenvalue {eig[0]:.3e}"


# --- blocks -------------------------------------------------------------


def _load(fname_of, a, b):
    """File for block (a, b), or for (b, a) to be transposed."""
    for (x, y), transpose in (((a, b), False), ((b, a), True)):
        fname = fname_of(x, y)
        if os.path.isfile(fname):
            return fname, transpose
    return None, False


def nmt_block(a, b, nbpw):
    """Spin-0/E part of a DES NaMaster benchmark block, ordered (ell, cl)."""
    fname, transpose = _load(
        lambda x, y: os.path.join(
            ROOT, "cov", "cov_{}_{}_{}_{}.npz".format(*x, *y)
        ),
        a,
        b,
    )
    cov = np.load(fname)["cov_G"]
    n1, n2 = cov.shape[0] // nbpw, cov.shape[1] // nbpw
    cov = cov.reshape(nbpw, n1, nbpw, n2)[:, 0, :, 0]
    return cov.T if transpose else cov


def validated_block(term, a, b):
    fname, transpose = _load(
        lambda x, y: os.path.join(
            VALIDATED_BLOCKS_DIR,
            "{}_{}_{}_{}_{}.npz".format(term, *x, *y),
        ),
        a,
        b,
    )
    if fname is None:
        return None
    cov = np.load(fname)["cov"]
    return cov.T if transpose else cov


def has_blocks(term, subset):
    return all(
        validated_block(term, a, b) is not None
        for a, b in itertools.combinations_with_replacement(subset, 2)
    )


def largest_closed_set(spectra, term):
    """Largest set of spectra with every block of `term` available."""
    for n in range(len(spectra), 0, -1):
        for subset in itertools.combinations(spectra, n):
            if has_blocks(term, subset):
                return list(subset)
    return []


def assemble(get_block, subset):
    return np.block([[get_block(a, b) for b in subset] for a in subset])


# --- fixture ------------------------------------------------------------


@pytest.fixture(scope="module", params=["nmt", "fsky"])
def full_cov(request, tmp_path_factory):
    s = sacc.Sacc.load_fits(ROOT + "cls_cov.fits")
    spectra = [tuple(trs) for trs in s.get_tracer_combinations()]
    nbpw = s.get_ell_cl("cl_00", "DESgc__0", "DESgc__0")[0].size

    subset = largest_closed_set(spectra, "ssc")
    if not subset:
        pytest.skip(
            "No validated SSC blocks in tests/validated_blocks/: run "
            "test_covariance_fourier_ssc_fsky.py in the same session."
        )

    if request.param == "nmt":

        def gaussian(a, b):
            return nmt_block(a, b, nbpw)

    else:
        with open(INPUT_YML_FSKY) as f:
            config = yaml.safe_load(f)
        # Keep this test's files out of tests/tmp/
        config["tjpcov"]["outdir"] = str(tmp_path_factory.mktemp("gauss"))
        cov_fsky = FourierGaussianFsky(config)
        # This sacc has NaMaster (coupled) bandpower windows, non-zero at
        # every ell. Without window binning, FourierGaussianFsky derives
        # top-hat edges from them, which are not monotonic.
        if len(cov_fsky.get_binning_info()) == 3:
            pytest.skip(
                "FourierGaussianFsky cannot bin with NaMaster bandpower "
                "windows yet (needs the window binning from branch "
                "141-fsky-covariance-fix-bandpower)."
            )

        def gaussian(a, b):
            return cov_fsky.get_covariance_block(
                tracer_comb1=a, tracer_comb2=b, include_b_modes=False
            )

    terms = {
        "gaussian": request.param,
        "subset": subset,
        "G": assemble(gaussian, subset),
        "ssc": assemble(lambda a, b: validated_block("ssc", a, b), subset),
        "cng": None,
    }
    if has_blocks("cng", subset):
        terms["cng"] = assemble(
            lambda a, b: validated_block("cng", a, b), subset
        )
    return terms


# --- tests ----------------------------------------------------------------


def test_terms_symmetric(full_cov):
    for term in ["G", "ssc", "cng"]:
        if full_cov[term] is not None:
            assert_symmetric(full_cov[term])


def test_gaussian_positive_definite(full_cov):
    assert_psd(full_cov["G"], strict=True)


def test_ssc_psd(full_cov):
    # SSC is a positively weighted sum of outer products of the responses,
    # so exactly PSD but low rank. CCL integrates each (ell1, ell2) element
    # separately (~1e-4 relative accuracy), which leaves eigenvalues of
    # order -1e-7 relative to the largest; real block inconsistencies are
    # orders of magnitude larger.
    assert_psd(full_cov["ssc"], rtol=1e-5)


def test_gaussian_plus_ssc_positive_definite(full_cov):
    assert_psd(full_cov["G"] + full_cov["ssc"], strict=True)


def test_total_positive_definite(full_cov):
    if full_cov["cng"] is None:
        pytest.skip(
            "No validated cNG blocks for "
            f"{full_cov['subset']}: run test_covariance_fourier_cNG_fsky.py "
            "in the same session."
        )
    # cNG alone is not guaranteed PSD (2h/3h/4h terms), so it is only
    # checked as part of the total.
    ng = full_cov["ssc"] + full_cov["cng"]
    total = full_cov["G"] + ng
    assert_psd(total, strict=True)

    # Report what was checked: run with -s to see it
    print(
        f"\nGaussian: {full_cov['gaussian']}; spectra: {full_cov['subset']}"
        "\nmax non-Gaussian fraction of the variance: "
        f"{np.max(np.diag(ng) / np.diag(total)):.3g}"
    )
