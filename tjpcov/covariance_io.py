import os
import warnings
from datetime import datetime

import jinja2
import sacc
import yaml


class CovarianceIO:
    """Class to handle the file input/output of the covariances.

    This class does not compute anything and it is initialized inside the
    CovarianceBuilder and CovarianceCalculator classes.
    """

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        if not isinstance(config, (dict, str)):
            raise TypeError(
                "config must be of type dict or str, given" + f"{type(config)}"
            )
        self.sacc_file = None

        if isinstance(config, str):
            config = self.get_dict_from_yaml(config)

        self.config = config

        if "tjpcov" not in self.config.keys():
            raise ValueError("tjpcov section not found in configuration.")

        if not isinstance(self.config["tjpcov"], dict):
            raise ValueError("tjpcov section must be a dictionary.")

        if "outdir" not in self.config["tjpcov"].keys():
            warnings.warn(
                "outdir not found in the tjpcov configuration, "
                + "defaulting to the working directory."
            )

        self.outdir = config["tjpcov"].get("outdir", "./")
        os.makedirs(self.outdir, exist_ok=True)

    @staticmethod
    def get_dict_from_yaml(filename):
        """Parse a configuration file.

        Args:
            filename (str): The config file to parse. Should be YAML formatted.

        Return:
            dict: The raw config file as a dictionary.
        """
        with open(filename, "r") as fp:
            config_str = jinja2.Template(fp.read()).render()

        config = yaml.load(config_str, Loader=yaml.Loader)

        return config

    def create_sacc_cov(self, cov, output="cls_cov.fits", overwrite=False):
        """Write created cov to a new sacc object.

        Args:
            output (str, optional): filename output. Defaults to "cls_cov.fits"
            overwrite (bool, optional): True if you want to overwrite an
                existing file. If False, it will not overwrite the file but
                will append the UTC time to the output to avoid losing the
                computed covariance. Defaults to False.

        Returns:
            :obj:`sacc.sacc.Sacc`: The final sacc file with the covariance
            matrix included.
        """
        output_file_nm = os.path.join(self.outdir, output)

        sacc_clone = self.get_sacc_file().copy()
        sacc_clone.add_covariance(cov, overwrite=True)

        if not os.path.isfile(output_file_nm) or overwrite:
            sacc_clone.save_fits(output_file_nm, overwrite=overwrite)
            return sacc_clone

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_file_nm = output_file_nm + f"_{timestamp}"
        warnings.warn(
            f"Output file {output} already exists. "
            "Appending the UTC time to the filename to avoid "
            "losing the covariance computation. Writing sacc "
            f"file to {output_file_nm}"
        )

        sacc_clone.save_fits(output_file_nm, overwrite=overwrite)

        return sacc_clone

    def get_sacc_file(self):
        """Return the input sacc file."""

        if self.sacc_file is None:
            self.sacc_file = self._load_sacc_from_config()

        return self.sacc_file

    def _load_sacc_from_config(self):
        sacc_file = self.config["tjpcov"].get("sacc_file")
        if "sacc_file" not in self.config["tjpcov"].keys():
            raise ValueError(
                "sacc_file not found in the tjpcov configuration."
            )
        if not isinstance(sacc_file, str):
            raise ValueError(
                "sacc_file entry in the config file must be a string."
            )

        sacc_file = sacc.Sacc.load_fits(sacc_file)
        return sacc_file
