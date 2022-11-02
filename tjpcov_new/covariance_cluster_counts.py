from re import L
from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl

# Replace with CCL functions
from scipy.integrate import quad, romb


class CovarianceClusterCounts(CovarianceClusters):
    """Implementation of cluster covariance that calculates the autocorrelation of cluster counts (NxN)"""

    # Figure out
    cov_type = "fourier"
    _reshape_order = "F"

    def __init__(self, config):
        super().__init__(config)
        self.romberg_num = 2**6 + 1
        self.setup_vectors()
        self.eval_true_vec()
        self.eval_M1_true_vec()

    def setup_vectors(self):
        """
        Sets up the arrays according to number of redshift/richness bins.
        This should be refactored and removed, in favor of the element-by-element computations
        """
        self.Z1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.G1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.dV_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.M1_true_vec = np.zeros(
            (self.num_richness_bins, self.num_z_bins, self.romberg_num)
        )

    def get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """
        This function returns the covariance block with the elements in the sacc file
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """
        This function returns the covariance block with the elements in the sacc file
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'clusters_redshift_richness'

        Args:
            tracer_comb1 (_type_): e.g. ('clusters_0_0',)
            tracer_comb2 (_type_): e.g. ('clusters_0_1',)
        """
        tracer_split1 = tracer_comb1[0].split("_")
        tracer_split2 = tracer_comb2[0].split("_")

        z_i = int(tracer_split1[1])
        richness_i = int(tracer_split1[2])

        z_j = int(tracer_split2[1])
        richness_j = int(tracer_split2[2])

        dz = (self.Z1_true_vec[z_i, -1] - self.Z1_true_vec[z_i, 0]) / (
            self.romberg_num - 1
        )

        partial_vec = [
            self.partial2(self.Z1_true_vec[z_i, m], z_j, richness_j)
            for m in range(self.romberg_num)
        ]

        romb_vec = (
            partial_vec
            * self.dV_true_vec[z_i]
            * self.M1_true_vec[richness_i, z_i]
            * self.G1_true_vec[z_i]
        )

        cov = (self.survey_area**2) * romb(romb_vec, dx=dz)

        shot_noise = 0
        if richness_i == richness_j and z_i == z_j:
            shot_noise = self.shot_noise(z_i, richness_i)

        cov_total = shot_noise + cov

        # TODO: store metadata in some header/log file
        return cov_total

    def eval_true_vec(self):
        """Computes the -geometric- true vectors Z1, G1, dV for Cov_N_N."""

        for i in range(self.num_z_bins):

            z_low_limit = max(
                self.z_lower_limit, self.z_bins[i] - 4 * self.z_bin_range
            )
            z_upper_limit = min(
                self.z_upper_limit, self.z_bins[i + 1] + 6 * self.z_bin_range
            )

            self.Z1_true_vec[i] = np.linspace(
                z_low_limit, z_upper_limit, self.romberg_num
            )
            self.G1_true_vec[i] = ccl.growth_factor(
                self.cosmo, 1 / (1 + self.Z1_true_vec[i])
            )
            self.dV_true_vec[i] = [
                self.dV(self.Z1_true_vec[i, m], i)
                for m in range(self.romberg_num)
            ]

    def eval_M1_true_vec(self):
        """Pre computes the true vectors M1 for Cov_N_N."""

        print("evaluating M1_true_vec (this may take some time)...")

        for lbd in range(self.num_richness_bins):
            for z in range(self.num_z_bins):
                for m in range(self.romberg_num):
                    self.M1_true_vec[lbd, z, m] = self.integral_mass(
                        self.Z1_true_vec[z, m], lbd
                    )
