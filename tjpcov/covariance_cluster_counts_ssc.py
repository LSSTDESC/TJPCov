from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl
from scipy.integrate import romb


class ClusterCountsSSC(CovarianceClusters):
    """Implementation of cluster covariance that calculates the SSC
    contribution to the autocorrelation of cluster counts (NxN) following
    N. Ferreira 2019.  This class is able to compute the covariance for
    `_tracers_types = ("cluster", "cluster")`
    """

    cov_type = "SSC"
    _tracer_types = ("cluster", "cluster")

    def __init__(self, config):
        """Concrete implementation of covariance of cluster counts,
        specifically the SSC contribution to the number count auto-correlation.

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
            array_like: Covariance for a single block
        """

        z_i, richness_i, z_j, richness_j = self._get_redshift_richness_bins(
            tracer_comb1, tracer_comb2
        )

        # Create a redshift range grid
        z_low_limit = max(
            self.z_lower_limit, self.z_bins[z_i] - 4 * self.z_bin_spacing
        )
        z_upper_limit = min(
            self.z_upper_limit, self.z_bins[z_i + 1] + 6 * self.z_bin_spacing
        )
        z_range = np.linspace(z_low_limit, z_upper_limit, self.romberg_num)

        linear_growth_factor = np.array(
            ccl.growth_factor(self.cosmo, 1 / (1 + z_range))
        )

        comoving_volume_elements = np.array(
            [self.comoving_volume_element(z_ii, z_i) for z_ii in z_range]
        )

        mass_richness_prob_dist = np.array(
            [self.mass_richness_integral(z_ii, richness_i) for z_ii in z_range]
        )

        partial_SSC = np.array(
            [self.partial_SSC(z_ii, z_j, richness_j) for z_ii in z_range]
        )

        # Eqn 4.18
        super_sample_covariance = (
            partial_SSC
            * comoving_volume_elements
            * mass_richness_prob_dist
            * linear_growth_factor
        )

        redshift_spacing = (z_range[-1] - z_range[0]) / (self.romberg_num - 1)
        cov = (self.survey_area**2) * romb(
            super_sample_covariance, dx=redshift_spacing
        )

        cov_full = np.array(cov)

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
