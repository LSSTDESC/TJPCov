from .covariance_builder import CovarianceBuilder
from .cluster_covariance_base import ClusterCovarianceBase
from .clusters_helpers import (
    FFTHelper,
    mass_func_map,
)
import numpy as np
import pyccl as ccl
from sacc import standard_types


class ClusterMass(ClusterCovarianceBase, CovarianceBuilder):
    """Calculate the covariance of cluster mass measurements.

    This class is able to compute the covariance for
    `_tracers_types = ("cluster_mean_log_mass", "cluster_mean_log_mass")`
    """

    space_type = "Fourier"
    cov_type = "gauss"
    _tracer_types = (
        standard_types.cluster_mean_log_mass,
        standard_types.cluster_mean_log_mass,
    )

    def __init__(self, config):
        """Class to calculate the covariance of cluster mass measurements.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        sacc_file = self.io.get_sacc_file()
        if (
            standard_types.cluster_mean_log_mass
            not in sacc_file.get_data_types()
        ):
            raise ValueError(
                "Cluster mass covariance was requested but cluster mass data"
                + " points were not included in the sacc file."
            )

        self.overdensity_delta = 200

        cosmo = self.get_cosmology()
        self.load_cluster_parameters()
        self.load_from_cosmology(cosmo)
        self.load_from_sacc(sacc_file)
        self.fft_helper = FFTHelper(
            cosmo, self.z_lower_limit, self.z_upper_limit
        )
        self.covariance_block_data_type = standard_types.cluster_mean_log_mass

    def load_cluster_parameters(self):
        """Load cluster parameters from the configuration file."""
        self._load_cluster_parameters()

    def _get_covariance_block_for_sacc(
        self, tracer_comb1, tracer_comb2, **kwargs
    ):
        """Compute a single covariance entry 'cluster_mean_log_mass'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_1')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_0')

        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_gaussian(tracer_comb1, tracer_comb2)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        """Compute a single covariance entry 'cluster_mean_log_mass'

        Args:
            tracer_comb1 (`tuple` of str): e.g. ('clusters_0_0',)
            tracer_comb2 (`tuple` of str): e.g. ('clusters_0_1',)
        Returns:
            array_like: Covariance for a single block
        """
        return self._get_covariance_gaussian(tracer_comb1, tracer_comb2)

    def _get_covariance_gaussian(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance entry 'cluster_mean_log_mass'

        Args:
            tracer_comb1 (`tuple` of str): e.g.
            ('survey', 'bin_richness_1', 'bin_z_0') or ('clusters_0_1',)
            tracer_comb2 (`tuple` of str): e.g.
            ('survey', 'bin_richness_0', 'bin_z_0') or ('clusters_0_0',)

        Returns:
            float: Covariance for a single block
        """

        richness_i, z_i = self.extract_indices_rich_z(tracer_comb1)
        richness_j, z_j = self.extract_indices_rich_z(tracer_comb2)

        if richness_i != richness_j or z_i != z_j:
            return np.array(0)

        # REMARK: Currently, the covariance for cluster mass is just the
        # standard deviation within the bin.  By the time the data gets to
        # TJPCov we only have a point estimate of the mean mass within the bin
        # so we cannot calculate the standard deviation.
        #
        # Since this standard deviation is just a temporary error estimate,
        # take the standard deviation from the existing sacc file covariance
        # and return it (don't change it).  Eventually we will calculate it.
        s = self.io.get_sacc_file()
        ix1 = s.indices(
            data_type=standard_types.cluster_mean_log_mass,
            tracers=tracer_comb1,
        )
        ix2 = s.indices(
            data_type=standard_types.cluster_mean_log_mass,
            tracers=tracer_comb2,
        )
        return s.covariance.covmat[np.ix_(ix1, ix2)].flatten()
