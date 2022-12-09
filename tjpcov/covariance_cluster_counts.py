from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl
from scipy.integrate import romb


class ClusterCounts(CovarianceClusters):
    """Implementation of cluster covariance that calculates 
    the autocorrelation of cluster counts (NxN)"""

    cov_type = "fourier"
    _reshape_order = "F"
    _tracer_types = ("cluster", "cluster")

    def __init__(self, config):
        super().__init__(config)
        self.romberg_num = 2**6 + 1

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """
        This function returns the covariance block with the 
        elements in the sacc file
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """
        This function returns the covariance block with the 
        elements in the sacc file
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

        # Hack for now - until we decide on sorting for 
        # tracers in SACC, strip 0's and take the remaining 
        # number, if you strip everything, default to 0
        z_i = int(tracer_split1[1].lstrip("0") or 0)
        richness_i = int(tracer_split1[2].lstrip("0") or 0)
        z_j = int(tracer_split2[1].lstrip("0") or 0)
        richness_j = int(tracer_split2[2].lstrip("0") or 0)

        # Compute geometric values based on redshift bin
        Z1_true = self.calc_Z1(z_i)
        G1_true = self.calc_G1(Z1_true)
        dV_true = self.calc_dV(Z1_true, z_i)
        M1_true = self.calc_M1(Z1_true, richness_i)

        dz = (Z1_true[-1] - Z1_true[0]) / (self.romberg_num - 1)

        partial_vec = np.array(
            [
                self.partial2(Z1_true[m], z_j, richness_j)
                for m in range(self.romberg_num)
            ]
        )
        romb_vec = partial_vec * dV_true * M1_true * G1_true

        cov = (self.survey_area**2) * romb(romb_vec, dx=dz)

        shot_noise = 0
        if richness_i == richness_j and z_i == z_j:
            shot_noise = self.shot_noise(z_i, richness_i)

        cov_total = shot_noise + cov

        # TODO: store metadata in some header/log file
        return cov_total

    def calc_Z1(self, z_i):
        z_low_limit = max(
            self.z_lower_limit, self.z_bins[z_i] - 4 * self.z_bin_range
        )
        z_upper_limit = min(
            self.z_upper_limit, self.z_bins[z_i + 1] + 6 * self.z_bin_range
        )

        return np.linspace(z_low_limit, z_upper_limit, self.romberg_num)

    def calc_G1(self, Z1_true_vec):
        return np.array(ccl.growth_factor(self.cosmo, 1 / (1 + Z1_true_vec)))

    def calc_dV(self, Z1_true_vec, z_i):
        return np.array(
            [self.dV(Z1_true_vec[m], z_i) for m in range(self.romberg_num)]
        )

    def calc_M1(self, Z1_true_vec, richness_i):
        M1_true = np.zeros(self.romberg_num)

        for m in range(self.romberg_num):
            M1_true[m] = self.integral_mass(Z1_true_vec[m], richness_i)

        return M1_true
