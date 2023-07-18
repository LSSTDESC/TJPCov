from .covariance_builder import CovarianceBuilder
from .clusters_helpers import FFTHelper
import numpy as np
import pyccl as ccl
from sacc import standard_types


class ClusterMass(CovarianceBuilder):
    """Implementation of cluster covariance that calculates the covriance
    of cluster weak lensing mass measurements.
    This class is able to compute the covariance for
    `_tracers_types = ("cluster_mean_log_mass", "cluster_mean_log_mass")`
    """

    space_type = "Fourier"
    cov_type = "gauss"
    _tracer_types = (
        standard_types.cluster_mean_log_mass,
        standard_types.cluster_mean_log_mass,
    )

    def __init__(self, config, min_halo_mass=1e13):
        """Constructor for the base class, used to pass through config options
        for covariance calculation.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
            min_halo_mass (float, optional): Minimum halo mass.
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
        self.h0 = float(self.config["parameters"].get("h"))
        self.load_from_sacc(sacc_file, min_halo_mass)

        cosmo = self.get_cosmology()
        self.load_from_cosmology(cosmo)
        self.fft_helper = FFTHelper(
            cosmo, self.z_lower_limit, self.z_upper_limit
        )

    def load_from_cosmology(self, cosmo):
        """Values used by the covariance calculation that come from a CCL
        cosmology object.  Derived attributes from the cosmology are set here.

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
        """
        self.cosmo = cosmo
        mass_def = ccl.halos.MassDef200m()
        self.c = ccl.physical_constants.CLIGHT / 1000
        self.mass_func = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)

    def load_from_sacc(self, sacc_file, min_halo_mass):
        """Cluster covariance has special parameters set in the SACC file. This
        informs the code that the data to calculate the cluster covariance is
        there.  We set extract those values from the sacc file here, and set
        the attributes here.

        Args:
            sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object, already
            loaded.
        """

        z_tracer_type = "bin_z"
        survey_tracer_type = "survey"
        richness_tracer_type = "bin_richness"

        survey_tracer = [
            x
            for x in sacc_file.tracers.values()
            if x.tracer_type == survey_tracer_type
        ]
        if len(survey_tracer) == 0:
            self.survey_tracer_nm = ""
            self.survey_area = 4 * np.pi
            print(
                "Survey tracer not provided in sacc file.\n"
                + "We will use the default value."
            )
        else:
            self.survey_area = survey_tracer[0].sky_area * (np.pi / 180) ** 2

        # Setup redshift bins
        z_bins = [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == z_tracer_type
        ]
        self.num_z_bins = len(z_bins)
        self.z_min = z_bins[0].lower
        self.z_max = z_bins[-1].upper
        self.z_bins = np.round(
            np.linspace(self.z_min, self.z_max, self.num_z_bins + 1), 2
        )
        self.z_bin_spacing = (self.z_max - self.z_min) / self.num_z_bins
        self.z_lower_limit = max(0.02, self.z_bins[0] - 4 * self.z_bin_spacing)
        # Set upper limit to be 40% higher than max redshift
        self.z_upper_limit = self.z_bins[-1] + 0.4 * self.z_bins[-1]

        # Setup richness bins
        richness_bins = [
            v
            for v in sacc_file.tracers.values()
            if v.tracer_type == richness_tracer_type
        ]
        self.num_richness_bins = len(richness_bins)
        self.min_richness = 10 ** richness_bins[0].lower
        self.max_richness = 10 ** richness_bins[-1].upper
        self.richness_bins = np.round(
            np.logspace(
                np.log10(self.min_richness),
                np.log10(self.max_richness),
                self.num_richness_bins + 1,
            ),
            2,
        )

        self.min_mass = np.log(min_halo_mass)
        self.max_mass = np.log(1e16)

    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        """Build full matrix from blocks.  Uses a combination data type and
        tracer combinations to place data blocks in the covariance matrix.

        Args:
            blocks (list): List of blocks
            tracers_cov (list): List of tracer combinations corresponding to
                each block in blocks. They must have the same order

        Returns:
            array: Covariance matrix for all combinations in the sacc file.
        """
        blocks = iter(blocks)

        s = self.io.get_sacc_file()
        ndim = s.mean.size

        cov_full = np.zeros((ndim, ndim))

        print("Building the covariance: placing blocks in their place")
        for tracer_comb1, tracer_comb2 in tracers_cov:
            print(tracer_comb1, tracer_comb2)

            cov_ij = next(blocks)
            ix1 = s.indices(
                data_type=standard_types.cluster_mean_log_mass,
                tracers=tracer_comb1,
            )
            ix2 = s.indices(
                data_type=standard_types.cluster_mean_log_mass,
                tracers=tracer_comb2,
            )

            cov_full[np.ix_(ix1, ix2)] = cov_ij
            cov_full[np.ix_(ix2, ix1)] = cov_ij.T

        return cov_full

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
                ('survey', 'bin_z_0', 'bin_richness_1')
            tracer_comb2 (`tuple` of str): e.g.
                ('survey', 'bin_z_0', 'bin_richness_0')

        Returns:
            float: Covariance for a single block
        """

        z_i = int(tracer_comb1[1].split("_")[-1])
        richness_i = int(tracer_comb1[2].split("_")[-1])

        z_j = int(tracer_comb2[1].split("_")[-1])
        richness_j = int(tracer_comb2[2].split("_")[-1])

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
