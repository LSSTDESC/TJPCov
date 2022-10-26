import os

from . import covariance_from_name
from .covariance_io import CovarianceIO


class CovarianceCalculator:
    """
    Class meant to be use by the end user. This will read the configuration
    file which will contain information of what covariances are requested (by
    giving the Class names) and add all their contributions.
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config (dict or str): If dict, it returns the configuration
            dictionary directly. If string, it asumes a YAML file and parses
            it.
        """
        self.io = CovarianceIO(config)
        self.config = self.io.config

        self.cov_total = None
        self.cov_terms = None
        self.cov_classes = None

    def get_covariance_classes(self):
        """
        Return a dictionary with the covariance terms and instances of their
        corresponding classes.

        Returns
        -------
        classes (dict): Dictionary with keys the covariance types ('gauss',
        SSC', .. ) and values instances of the corresponding classes.
        """
        if self.cov_classes is None:
            # cov_type will be a list or string with the class names that you
            # will use to compute the different covariance terms
            cov_tbc = self.config["tjpcov"].get("cov_type", [])
            if isinstance(cov_tbc, str):
                cov_tbc = [cov_tbc]

            cov_classes = {}
            space_types = []
            for covi in cov_tbc:
                covi = covariance_from_name(covi)
                # Check the cov_type has not been already requested (e.g. two
                # Gaussian contributions)
                if covi.cov_type in cov_classes:
                    raise ValueError(
                        f"Covariance type {covi.cov_type} "
                        "already set. Make sure each type is "
                        "requested only once."
                    )

                # Check that you are not mixing Fourier and real space
                # covariances
                if len(space_types) > 0 and (
                    covi.space_type not in space_types
                ):
                    raise ValueError(
                        "Mixing configuration and Fourier space covariances."
                    )

                space_types.append(covi.space_type)
                cov_classes[covi.cov_type] = covi(self.config)

            self.cov_classes = cov_classes

        return self.cov_classes

    def get_covariance(self):
        """
        Return the covariance with all the requested contributions added up.

        Returns
        -------
            cov (array): Final covariance with all the requested contributions
            added up.
        """
        if self.cov_total is None:
            cov_terms = self.get_covariance_terms()

            self.cov_total = sum(cov_terms.values())

        return self.cov_total

    def get_covariance_terms(self):
        """
        Return a dictionary with keys the covariace types and values their
        covariance contributions.

        Returns
        -------
            dict: dictionary with keys the covariace types and values their
        covariance contributions.
        """
        if self.cov_terms is None:
            cov_classes = self.get_covariance_classes()

            cov_terms = {}
            for ctype, cmat in cov_classes.items():
                cov_terms[ctype] = cmat.get_covariance()

            self.cov_terms = cov_terms

        return self.cov_terms

    def create_sacc_cov(self, output="cls_cov.fits", save_terms=True):
        """
        Write the sacc file with the total covariance.

        Parameters
        ----------
            output (str): Filename. This will be joined to the outdir path
            specified in the configuration file.
            save_terms (bool): If true, save individual files for each of the
            requested contributions. The will have the covariance term (e.g.
            gauss) appended to the filename (before the extension, e.g.
            cls_cov_gauss.fits)
        """
        cov = self.get_covariance()
        self.io.create_sacc_cov(cov, output)

        if save_terms:
            cov_terms = self.get_covariance_terms()
            for term, cov in cov_terms.items():
                fname, ext = os.path.splitext(output)
                fname += f"_{term}{ext}"
                self.io.create_sacc_cov(cov, fname)
