from .covariance_fourier_cNG import FouriercNGHaloModel


class FouriercNGHaloModelFsky(FouriercNGHaloModel):
    """Class to compute the CellxCell Halo Model cNG Covariance."""

    cov_type = "cNG"

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.fsky = self.config["GaussianFsky"].get("fsky", None)
        if self.fsky is None:
            raise ValueError(
                "You need to set fsky for FouriercNGHaloModelFsky"
            )

    def _get_fsky(self, tr=None):
        """Returns the fractional sky area from user input.

        Args:
            masks (:obj:`dict`): dictionary containing the survey
                tracers to obtain the survey mask (irrelevant in this case).
        Returns:
            - (:obj:`float`): fractional sky area.
        """
        return self.fsky
