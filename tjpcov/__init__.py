# flake8: noqa
from .covariance_builder import CovarianceBuilder
from .covariance_fourier_gaussian_nmt import FourierGaussianNmt
from .covariance_fourier_ssc import FourierSSCHaloModel
from .covariance_gaussian_fsky import (
    FourierGaussianFsky,
    RealGaussianFsky,
)
from .wigner_transform import bin_cov, WignerTransform


def covariance_from_name(name):
    """Return the requested CovarianceBuilder child class.

    Args:
        name (str): Name of the class

    Returns:
        :class:`~tjpcov.covariance_builder.CovarianceBuilder` child class
    """

    def all_subclasses(cls):
        # Recursively find all subclasses (and their subclasses)
        # From https://stackoverflow.com/questions/3862310
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)]
        )

    subcs = all_subclasses(CovarianceBuilder)
    mappers = {m.__name__: m for m in subcs}
    if name in mappers:
        return mappers[name]
    else:
        raise ValueError(f"Unknown covariance {name}")
