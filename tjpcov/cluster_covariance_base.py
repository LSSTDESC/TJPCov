"""Shared cosmology/SACC loading behaviour for cluster covariance classes."""

import pyccl as ccl
import numpy as np
from .clusters_helpers import mass_func_map


class ClusterCovarianceBase:
    """Base class to provide cosmology/SACC loading shared by cluster
    covariance classes.
    """

    def load_from_sacc(self, sacc_file):
        """Extract and compute attributes from a SACC file.

        Args:
            sacc_file (:obj: `sacc.sacc.Sacc`): SACC file object,
            already loaded.

        Returns:
            dict: A dictionary containing all computed attributes.
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
            survey_area = 4 * np.pi
            print(
                "Survey tracer not provided in sacc file.\n"
                + "We will use the default value.",
                flush=True,
            )
        else:
            survey_area = survey_tracer[0].sky_area * (np.pi / 180) ** 2

        # Setup redshift bins
        z_bins = sorted(
            [
                v
                for v in sacc_file.tracers.values()
                if v.tracer_type == z_tracer_type
            ],
            key=lambda z: z.lower,
        )
        num_z_bins = len(z_bins)
        z_min = np.min([zbin.lower for zbin in z_bins])
        z_max = np.max([zbin.upper for zbin in z_bins])
        z_bins = np.array(
            [round(z_bins[0].lower, 2)]
            + [round(zbin.upper, 2) for zbin in z_bins]
        )
        z_bin_spacing = (z_max - z_min) / num_z_bins
        z_lower_limit = max(0.02, z_bins[0] - 4 * z_bin_spacing)
        z_upper_limit = (
            z_bins[-1] + 0.4 * z_bins[-1]
        )  # Set upper limit to be 40% higher than max redshift

        # Setup richness bins
        richness_bins = sorted(
            [
                v
                for v in sacc_file.tracers.values()
                if v.tracer_type == richness_tracer_type
            ],
            key=lambda rich: rich.lower,
        )
        num_richness_bins = len(richness_bins)
        min_richness = 10 ** np.min([rbin.lower for rbin in richness_bins])
        max_richness = 10 ** np.max([rbin.upper for rbin in richness_bins])
        richness_bins = np.array(
            [10 ** richness_bins[0].lower]
            + [10**rbin.upper for rbin in richness_bins]
        )
        richness_bins = np.round(richness_bins, 2)

        sacc_meta_dict = {
            "survey_area": survey_area,
            "num_z_bins": num_z_bins,
            "z_min": z_min,
            "z_max": z_max,
            "z_bins": z_bins,
            "z_bin_spacing": z_bin_spacing,
            "z_lower_limit": z_lower_limit,
            "z_upper_limit": z_upper_limit,
            "num_richness_bins": num_richness_bins,
            "min_richness": min_richness,
            "max_richness": max_richness,
            "richness_bins": richness_bins,
        }
        for key, value in sacc_meta_dict.items():
            setattr(self, key, value)
        # Return all computed attributes as a dictionary
        return sacc_meta_dict

    def extract_indices_rich_z(self, tracer_comb):
        """Extract richness and redshift indices from a tracer combination."""
        if len(tracer_comb) == 1:
            # Handle input type 2: ('clusters_0_1',)
            parts = tracer_comb[0].split("_")
            richness = int(parts[-2])  # Second-to-last part is richness
            z = int(parts[-1])  # Last part is redshift
        else:
            # Handle input type 1: ('survey', 'bin_richness_1', 'bin_z_0')
            richness = None
            z = None
            for part in tracer_comb:
                if part.startswith("bin_richness_") or part.startswith(
                    "bin_rich_"
                ):  # Handle both prefixes
                    richness = int(part.split("_")[-1])
                elif part.startswith("bin_z_"):
                    z = int(part.split("_")[-1])
            if richness is None or z is None:
                raise ValueError(
                    "Could not extract richness or z from tracer combination: "
                    f"{tracer_comb}"
                )
        return richness, z

    def load_from_cosmology(self, cosmo):
        """Load parameters from a CCL cosmology object.

        Derived attributes from the cosmology are set here.

        Args:
            cosmo (:obj:`pyccl.Cosmology`): Input cosmology
        """
        self.cosmo = cosmo
        self.c = ccl.physical_constants.CLIGHT / 1000
        self.h0 = float(self.config["parameters"].get("h"))

    def _load_cluster_parameters(self):
        """Load cluster parameters from the configuration file."""
        mass_func_name = self.config["mor_parameters"].get("mass_func")
        self.mass_def = self.config["mor_parameters"].get("mass_def")
        self.min_halo_ln_mass = np.log(
            float(self.config["mor_parameters"].get("min_halo_mass"))
        )
        self.max_halo_ln_mass = np.log(
            float(self.config["mor_parameters"].get("max_halo_mass"))
        )
        if mass_func_name not in mass_func_map:
            raise ValueError(f"Invalid mass function: {mass_func_name}")

        # Create the mass definition, mass function, and halo bias objects
        self.mass_func = mass_func_map[mass_func_name](mass_def=self.mass_def)
