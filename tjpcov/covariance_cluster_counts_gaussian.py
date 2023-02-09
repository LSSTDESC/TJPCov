from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl


class ClusterCountsGaussian(CovarianceClusters):
    """Implementation of cluster covariance that calculates the gaussian
    (shot-noise) contribution to the autocorrelation of cluster counts (NxN).
    This class is able to compute the covariance for
    `_tracers_types = ("cluster", "cluster")`
    """

    cov_type = "gauss"
    _tracer_types = ("cluster", "cluster")

    def __init__(self, config):
        """Concrete implementation of covariance of cluster counts,
        specifically gaussian contribution (shot-noise) to the number count
        auto-correlation.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)

        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_gaussian(tracer_comb1, tracer_comb2)

    def _get_covariance_gaussian(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)

        Returns:
            float: Covariance for a single block
        """

        z_i, richness_i, z_j, richness_j = self._get_redshift_richness_bins(
            tracer_comb1, tracer_comb2
        )

        if richness_i != richness_j or z_i != z_j:
            return np.array(0)

        shot_noise = self.shot_noise(z_i, richness_i)

        cov_full = np.array(shot_noise)

        return cov_full

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """Compute a single covariance entry 'clusters_redshift_richness'
        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)
        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_gaussian(tracer_comb1, tracer_comb2)

    def shot_noise(self, z_i, lbd_i):
        """The covariance of number counts is a sum of a super sample
        covariance (SSC) term plus a gaussian diagonal term.  The diagonal
        term is also referred to as "shot noise" which we compute here.

        Args:
            z_i (int): redshift bin i
            lbd_i (int): richness bin i
        Returns:
            float: Gaussian covariance contribution
        """
        # Eqn B.7 or 1601.05779.pdf eqn 1
        def integrand(z):
            return (
                self.c
                * (ccl.comoving_radial_distance(self.cosmo, 1 / (1 + z)) ** 2)
                / (100 * self.h0 * ccl.h_over_h0(self.cosmo, 1 / (1 + z)))
                * self.mass_richness_integral(z, lbd_i, remove_bias=True)
                * self.observed_photo_z(z, z_i)
            )

        result = self._quad_integrate(
            integrand, self.z_lower_limit, self.z_upper_limit
        )
        return self.survey_area * result
