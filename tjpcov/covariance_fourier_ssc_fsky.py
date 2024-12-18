import pyccl as ccl
from .covariance_fourier_ssc import FourierSSCHaloModel


class FourierSSCHaloModelFsky(FourierSSCHaloModel):
    """Class to compute the CellxCell Halo Model Super Sample Covariance
        with the fsky approximation.

    The SSC is computed in CCL with the "linear bias" approximation using
    :func:`pyccl.halos.halo_model.halomod_Tk3D_SSC_linear_bias`.
    """

    cov_type = "SSC"

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
                "You need to set fsky for FourierSSCHaloModelFsky"
            )

    def _get_sigma2_B(self, cosmo, a_arr, tr=None):
        """Returns the variance of the projected linear density field,
            for the fsky/disk approximation case.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a_arr (:obj:`float`, `array` or :obj:`None`): an array of
                scale factor values at which to evaluate
                the projected variance.
            tr (:obj:`dict`): dictionary containing the
                tracer name combinations.
        Returns:
            - (:obj:`float` or `array`): projected variance.
        """
        return ccl.sigma2_B_disc(cosmo, a_arr=a_arr, fsky=self.fsky)
