import os

from . import covariance_from_name
from .covariance_io import CovarianceIO


class CovarianceCalculator:
    """Class for the end user that will compute all covariance terms.

    This will read the configuration file which will contain information of
    what covariances are requested (by giving the Class names) and add all
    their contributions.
    """

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        self.io = CovarianceIO(config)
        self.config = self.io.config

        self.cov_total = None
        self.cov_terms = None
        self.cov_classes = None

        use_mpi = self.config["tjpcov"].get("use_mpi", False)

        # This is only used in this class to save the output file only once.
        if use_mpi is True:
            try:
                import mpi4py.MPI
            except ImportError:
                raise ValueError("MPI option requires mpi4py to be installed")

            self.comm = mpi4py.MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = None
            self.size = None

    def get_covariance_classes(self):
        """Return a dictionary with the covariance classes initialized.

        Returns:
            dict: Dictionary with keys the covariance types ('gauss', SSC', ..
            ) and values instances of the corresponding classes.
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
                # Check the cov_type has not been already requested for the
                # same tracer types (e.g. two Gaussian contributions for ClxCl)
                if (covi.cov_type in cov_classes) and (
                    covi._tracer_types in cov_classes[covi.cov_type]
                ):
                    raise ValueError(
                        f"Covariance type {covi.cov_type} for "
                        "{covi._tracer_types} is already set. Make sure each "
                        "type is requested only once."
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
                if covi.cov_type not in cov_classes:
                    cov_classes[covi.cov_type] = {
                        covi._tracer_types: covi(self.config)
                    }
                else:
                    cov_classes[covi.cov_type].update(
                        {covi._tracer_types: covi(self.config)}
                    )

            self.cov_classes = cov_classes

        return self.cov_classes

    def get_covariance(self):
        """Return the covariance with all the contributions added up.

        Returns:
            array: Final covariance with all the requested contributions added
            up.
        """
        if self.cov_total is None:
            cov_terms = self.get_covariance_terms()

            # No need to do it only for rank == 0 since all Builder processes
            # have self.cov well defined and this is quite unexpensive.
            self.cov_total = sum(cov_terms.values())

        return self.cov_total

    def get_covariance_terms(self):
        """Return a dictionary with the covariance contributions.

        The dictionary has keys the covariace types and values their covariance
        contributions. We add all the contributions for different tracer types
        (e.g. ClxCl + ClxN + NxN). Since they are independent it is easy to
        recover each of them independently.

        Returns:
            dict: dictionary with keys the covariace types and values their
            covariance contributions.
        """
        if self.cov_terms is None:
            cov_classes = self.get_covariance_classes()

            cov_terms = {}
            for ctype, cov_dict in cov_classes.items():
                cov = []
                for cmat in cov_dict.values():
                    cov.append(cmat.get_covariance())

                # No need to do it only for rank == 0 since all Builder
                # processes have self.cov well defined and this is quite
                # unexpensive.
                cov_terms[ctype] = sum(cov)

            self.cov_terms = cov_terms

        return self.cov_terms

    def create_sacc_cov(self, output="cls_cov.fits", save_terms=True):
        """Write the sacc file with the total covariance.

        Args:
            output (str, optional): Filename. This will be joined to the outdir
                path specified in the configuration file. Default
                "cls_cov.fits"
            save_terms (bool, optional): If true, save individual files for
                each of the requested contributions. The will have the
                covariance term (e.g. gauss) appended to the filename (before
                the extension, e.g. cls_cov_gauss.fits)

        Returns:
            :obj:`sacc.sacc.Sacc`: The final sacc file with the covariance
            matrix included.
        """
        cov = self.get_covariance()

        # Only save the file with the root (rank = 0) process.
        if (self.rank is not None) and (self.rank != 0):
            return

        s = self.io.create_sacc_cov(cov, output)

        if save_terms:
            cov_terms = self.get_covariance_terms()
            for term, cov in cov_terms.items():
                fname, ext = os.path.splitext(output)
                fname += f"_{term}{ext}"
                self.io.create_sacc_cov(cov, fname)

        return s
