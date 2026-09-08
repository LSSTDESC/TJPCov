"""Tests for the cluster_helpers module."""

from tjpcov.clusters_helpers import FFTHelper
import pytest
import pyccl as ccl


@pytest.fixture
def fft_helper(mock_cosmo):
    """Return an instance of FFTHelper."""
    return FFTHelper(mock_cosmo, z_min=0.1, z_max=1.0)


def test_fft_helper_initialization(fft_helper):
    """Test that FFTHelper initializes correctly."""
    assert fft_helper.cosmo is not None
    assert fft_helper.k_min == 1e-4
    assert fft_helper.k_max == 3
    assert fft_helper.N == 1024
    assert fft_helper.bias_fft == 1.4165


def test_set_fft_params(fft_helper):
    """Test that _set_fft_params sets the correct attributes."""
    assert hasattr(fft_helper, "r_min")
    assert hasattr(fft_helper, "r_max")
    assert hasattr(fft_helper, "G")
    assert hasattr(fft_helper, "L")
    assert hasattr(fft_helper, "k_grid")
    assert hasattr(fft_helper, "r_grid")
    assert hasattr(fft_helper, "idx_min")
    assert hasattr(fft_helper, "idx_max")
    assert hasattr(fft_helper, "pk_grid")
    assert hasattr(fft_helper, "fk_grid")

    # Check grid shapes
    assert fft_helper.k_grid.shape == (1024,)
    assert fft_helper.r_grid.shape == (1024,)
    assert fft_helper.pk_grid.shape == (1024,)
    assert fft_helper.fk_grid.shape == (1024,)


def test_two_fast_algorithm(fft_helper):
    """Test the two_fast_algorithm method."""
    # Mock the interpolation result
    result = fft_helper.two_fast_algorithm(z1=0.5, z2=1.0)
    assert isinstance(result, float)  # Ensure the result is a float


def test_I_ell_algorithm(fft_helper):
    """Test the _I_ell_algorithm method."""
    i = 10
    ratio = 0.5
    result = fft_helper._I_ell_algorithm(i, ratio)
    assert isinstance(result, complex)  # Ensure the result is a complex number


def test_two_fast_algorithm_interpolation_error(fft_helper):
    """Test two_fast_algorithm error on invalid interpolation."""
    # Force invalid interpolation by equating idx_min and idx_max
    fft_helper.idx_min = 0
    fft_helper.idx_max = 0

    with pytest.raises(Exception):
        fft_helper.two_fast_algorithm(z1=0.5, z2=1.0)
