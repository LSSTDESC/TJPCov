from .wigner_transform import wigner_transform, bin_cov
from .covariance_fourier_gaussian_nmt import CovarianceFourierGaussianNmt
from .covariance_fourier_ssc import FourierSSCHaloModel
from .covariance_gaussian_fsky import FourierGaussianFsky, RealGaussianFsky

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
