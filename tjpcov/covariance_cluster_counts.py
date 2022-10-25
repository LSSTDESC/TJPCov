from .covariance_clusters import CovarianceClusters
import numpy as np
import pyccl as ccl

# Replace with CCL functions
from scipy.integrate import quad, romb


class CovarianceClusterCounts(CovarianceClusters):
    """_summary_

    Args:
        CovarianceClusters (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Figure out
    cov_type = "fourier"
    _reshape_order = "F"

    def __init__(self, config):
        super().__init__(config)

        self.romberg_num = 2**6 + 1
        self.Z1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.G1_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.dV_true_vec = np.zeros((self.num_z_bins, self.romberg_num))
        self.M1_true_vec = np.zeros(
            (self.num_richness_bins, self.num_z_bins, self.romberg_num)
        )

        # TODO this should be moved to evaluate 1 entry at a time, so we can use parallelization
        # Computes the geometric true vectors
        self.eval_true_vec()
        # Pre computes the true vectors M1 for Cov_N_N
        self.eval_M1_true_vec()

    def get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """_summary_

        Args:
            tracer_comb1 (_type_): _description_
            tracer_comb2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """_summary_

        Args:
            tracer_comb1 (_type_): _description_
            tracer_comb2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.get_covariance_cluster_counts(tracer_comb1, tracer_comb2)

    def get_covariance_cluster_counts(self, tracer_comb1, tracer_comb2):
        """Cluster counts covariance
        Args:
            bin_z_i (float or ?array): tomographic bins in z_i or z_j
            bin_lbd_i (float or ?array): bins of richness (usually log spaced)
        Returns:
            float: Covariance at given bins
        """
        # Compute a single covariance entry 'clusters_redshift_richness' e.g.
        # tracer_comb1 = ('clusters_0_0',)
        # tracer_comb2 = ('clusters_0_0',)
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
        """Computes the -geometric- true vectors Z1, G1, dV for Cov_N_N.
        Args:
            (int) romb_num: controls romb integral precision.
                        Typically 10**6 + 1
        Returns:
            (array) Z1_true_vec
            (array) G1_true_vec
            (array) dV_true_vec
        """

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
        """Pre computes the true vectors M1 for Cov_N_N.
        Args:
            (int) romb_num: controls romb integral precision.
                        Typically 10**6 + 1
        """

        print("evaluating M1_true_vec (this may take some time)...")

        for lbd in range(self.num_richness_bins):
            for z in range(self.num_z_bins):
                for m in range(self.romberg_num):
                    self.M1_true_vec[lbd, z, m] = self.integral_mass(
                        self.Z1_true_vec[z, m], lbd
                    )


class MassRichnessRelation(object):
    """
    Helper class to hold different mass richness relations
    """

    @staticmethod
    def MurataCostanzi(ln_true_mass, richness_bin, richness_bin_next, h0):
        """
        Define lognormal mass-richness relation
        (leveraging paper from Murata et. alli - ArxIv 1707.01907 and Costanzi et al ArxIv 1810.09456v1)

        Args:
            ln_true_mass: ln(true mass)
            richness_bin: ith richness bin
            richness_bin_next: i+1th richness bin
            h0:
        Returns:
            The probability that the true mass ln(ln_true_mass) is observed within
            the bins richness_bin and richness_bin_next
        """

        alpha = 3.207  # Murata
        beta = 0.75  # Costanzi
        sigma_zero = 2.68  # Costanzi
        q = 0.54  # Costanzi
        m_pivot = 3.0e14 / h0  # in solar masses , Murata and Costanzi use it

        sigma_lambda = sigma_zero + q * (ln_true_mass - np.log(m_pivot))
        average = alpha + beta * (ln_true_mass - np.log(m_pivot))

        def integrand(richness):
            return (
                (1.0 / richness)
                * np.exp(
                    -((np.log(richness) - average) ** 2.0)
                    / (2.0 * sigma_lambda**2.0)
                )
                / (np.sqrt(2.0 * np.pi) * sigma_lambda)
            )

        return quad(integrand, richness_bin, richness_bin_next)[0]
