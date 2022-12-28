from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl
from scipy.integrate import romb


class ClusterCounts(CovarianceClusters):
    """Implementation of cluster covariance that calculates the autocorrelation
    of cluster counts (NxN)
    """

    cov_type = "fourier"
    _reshape_order = "F"
    _tracer_types = ("cluster", "cluster")

    def __init__(self, config):
        """Concrete implementation of covariance of cluster counts, specifically
        the number count auto-correlation.

        Args:
            config: The configuration file path used to calculate the covariance
        """
        super().__init__(config)
        self.romberg_num = 2**6 + 1

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """This function returns the covariance block with the
        elements in the sacc file

        Args:
            tracer_comb1: _description_
            tracer_comb2: _description_

        Returns:
            _description_
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """This function returns the covariance block with the
        elements in the sacc file

        Args:
            tracer_comb1: _description_
            tracer_comb2: _description_

        Returns:
            _description_
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (_type_): e.g. ('clusters_0_0',)
            tracer_comb2 (_type_): e.g. ('clusters_0_1',)

        Returns:
            _description_
        """

        tracer_split1 = tracer_comb1[0].split("_")
        tracer_split2 = tracer_comb2[0].split("_")

        # Hack for now - until we decide on sorting for tracers in SACC, strip
        # 0's and take the remaining number, if you strip everything, default to
        # 0
        z_i = int(tracer_split1[1].lstrip("0") or 0)
        richness_i = int(tracer_split1[2].lstrip("0") or 0)
        z_j = int(tracer_split2[1].lstrip("0") or 0)
        richness_j = int(tracer_split2[2].lstrip("0") or 0)

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
            [
                self.comoving_volume_element(redshift, z_i)
                for redshift in z_range
            ]
        )

        mass_richness_prob_dist = np.array(
            [
                self.mass_richness_integral(redshift, richness_i)
                for redshift in z_range
            ]
        )

        partial_SSC = np.array(
            [
                self.partial_SSC(redshift, z_j, richness_j)
                for redshift in z_range
            ]
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

        shot_noise = 0
        if richness_i == richness_j and z_i == z_j:
            shot_noise = self.shot_noise(z_i, richness_i)

        cov_total = shot_noise + cov

        # TODO: store metadata in some header/log file
        return cov_total
