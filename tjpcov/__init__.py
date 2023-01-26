#!/usr/bin/python3


def covariance_from_name(name):
    """Return the requested CovarianceBuilder child class.

    Args:
        name (str): Name of the class

    Returns:
        :class:`~tjpcov.covariance_builder.CovarianceBuilder` child class
    """
    # TODO: Make this automatic
    if name == "FourierGaussianNmt":
        from .covariance_fourier_gaussian_nmt import FourierGaussianNmt as Cov
    elif name == "FourierSSCHaloModel":
        from .covariance_fourier_ssc import FourierSSCHaloModel as Cov
    elif name == "ClusterCountsSSC":
        from .covariance_cluster_counts_ssc import ClusterCountsSSC as Cov
    elif name == "ClusterCountsGaussian":
        from .covariance_cluster_counts_gaussian import (
            ClusterCountsGaussian as Cov,
        )
    elif name == "FourierGaussianFsky":
        from .covariance_gaussian_fsky import FourierGaussianFsky as Cov
    elif name == "RealGaussianFsky":
        from .covariance_gaussian_fsky import RealGaussianFsky as Cov
    else:
        raise ValueError(f"Unknown covariance {name}")

    return Cov
