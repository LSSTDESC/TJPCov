from .covariance_io import CovarianceIO
from . import covariance_from_name
import numpy as np


class CovarianceCalculator(CovarianceIO):
    def __init__(self, config):
        super().__init__(config)

        self.cov_total = None
        self.covs = None

    def get_covariance_classes(self):
        if self.covs is None:
            # cov_type will be a list or string with the class names that you
            # will use to compute the different covariance terms
            cov_tbc = self.config['tjpcov'].get('cov_type', [])
            if isinstance(cov_tbc, str):
                cov_tbc = [cov_tbc]

            self.covs = {}
            space_types = []
            for covi in cov_tbc:
                covi = covariance_from_name(covi)
                # Check the cov_type has not been already requested (e.g. two
                # Gaussian contributions)
                if covi.cov_type in self.covs:
                    raise ValueError(f'Covariance type {covi.cov_type} ' +
                                     'already set. Make sure each type is ' +
                                     'requested only once.')

                # Check that you are not mixing Fourier and real space
                # covariances
                if len(space_types) > 0 and (covi.space_type not in
                                             space_types):
                    raise ValueError('Mixing configuration and Fourier space' +
                                     ' covariances.')

                space_types.append(covi.space_type)
                self.covs[covi.cov_type] = covi

        return self.covs

    def get_covariance(self):
        if self.cov_total is None:
            covs = self.get_covariance_classes()

            cov = []
            for ctype, cmat in covs:
                cov.append(cmat.get_covariance())

            self.cov_total = np.sum(cov)

        return self.cov_total
