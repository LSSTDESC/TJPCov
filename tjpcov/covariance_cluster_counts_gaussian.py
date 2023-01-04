from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl
from scipy.integrate import romb


class ClusterCountsGaussian(CovarianceClusters):
    """Implementation of cluster covariance that calculates the autocorrelation
    of cluster counts (NxN).  This class is able to compute the covariance for
    `_tracers_types = ("cluster", "cluster")`
    """

    cov_type = "gauss"
    _tracer_types = ("cluster", "cluster")

    def __init__(self, config):
        """Concrete implementation of covariance of cluster counts,
        specifically gaussian contribution to the number count
        auto-correlation.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)
        self.romberg_num = 2**6 + 1

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)

        Returns:
            float: Covariance for a single block
        """
        return self._get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def _get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
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

        cov_full = np.zeros(
            (
                self.num_z_bins,
                self.num_z_bins,
                self.num_richness_bins,
                self.num_richness_bins,
            )
        )

        if richness_i != richness_j or z_i != z_j:
            return cov_full

        shot_noise = self.shot_noise(z_i, richness_i)

        cov_full[z_i, z_j, richness_i, richness_j] = shot_noise

        return cov_full
