from .tools import wigner_transform, bin_cov, parse
from .parser import read_config
from .covariance_base import CovarianceBase
from .covariance_fourier_gaussian_nmt import FourierGaussianNmtCovariance

def covariance_from_name(name):
    def all_subclasses(cls):
        # Recursively find all subclasses (and their subclasses)
        # From https://stackoverflow.com/questions/3862310
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)])
    subcs = all_subclasses(CovarianceBase)
    mappers = {m.__name__: m for m in subcs}
    if name in mappers:
        return mappers[name]
    else:
        raise ValueError(f"Unknown covariance {name}")
