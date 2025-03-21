from .covariance_cluster_counts import CovarianceClusterCounts
from .clusters_helpers import extract_indices_rich_z
import numpy as np


class ClusterCountsGaussian(CovarianceClusterCounts):
    """Implementation of gaussian covariance term.

    This class calculates the gaussian (shot-noise) contribution to the
    autocorrelation of cluster counts (NxN).
    """

    cov_type = "gauss"

    def __init__(self, config):
        """Class to calculate the gaussian covariance of cluster counts

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
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_richness_1', 'bin_z_0')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_richness_0', 'bin_z_0')

        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_gaussian(tracer_comb1, tracer_comb2)

    def _get_covariance_gaussian(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
            ('survey', 'bin_richness_1', 'bin_z_0') or ('clusters_0_1',)
            tracer_comb2 (`tuple` of str): e.g.
            ('survey', 'bin_richness_0', 'bin_z_0') or ('clusters_0_0',)

        Returns:
            array_like: Covariance for a single block
        """
        # Extract richness and redshift indices for both tracer combinations
        richness_i, z_i = extract_indices_rich_z(tracer_comb1)
        richness_j, z_j = extract_indices_rich_z(tracer_comb2)

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
        """Returns the cluster shot noise contribution to the covariance.

        The covariance of number counts is a sum of a super sample
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
            return self.mass_richness_integral(
                z, lbd_i, remove_bias=True
            ) * self.comoving_volume_element(z, z_i, self.sigma_0)

        result = self._quad_integrate(
            integrand, self.z_lower_limit, self.z_upper_limit
        )
        return self.survey_area * result
