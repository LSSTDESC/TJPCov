from .covariance_cluster_counts import CovarianceClusterCounts
import numpy as np


class ClusterCountsSSC(CovarianceClusterCounts):
    """Implementation of the SSC cluster covariance term.

    Calculates the sample variance contribution to the autocorrelation of
    cluster counts (NxN) following N. Ferreira 2019.
    """

    cov_type = "SSC"

    def __init__(self, config):
        """Class to calculate the SSC covariance of cluster counts

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.ssc_total = None  # initialise the full ssc covariance

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_1')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_0')
        Returns:
            float: Covariance for a single block
        """
        return self._get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def _get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_1')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_0')

        Returns:
            array_like: Covariance for a single block
        """

        richness_i = int(tracer_comb1[1].split("_")[-1])
        z_i = int(tracer_comb1[2].split("_")[-1])

        richness_j = int(tracer_comb2[1].split("_")[-1])
        z_j = int(tracer_comb2[2].split("_")[-1])

        # compute the full SSC covariance only once
        # (full covariance, for all the richness and redshift bins)
        # the code is almost fully vectorized (~30 times faster)
        if self.ssc_total is None:
            self.ssc_total = self.super_sample_covariance()

        # read the single entries of the total SSC covariance
        # to meet the format required by the code
        # ssc_total dim = [richness, richness, redshift, redshift]
        cov_full = np.array(self.ssc_total[richness_i, richness_j, z_i, z_j])

        return cov_full

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)
        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_cluster_counts(tracer_comb1, tracer_comb2)
