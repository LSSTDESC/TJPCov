import os
import warnings
from datetime import datetime

import jinja2
import sacc
import yaml


class CovarianceIO:
    """Class to handle the input/output of the covariances.

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
        self.config = self._read_config(config)
        self.sacc_file = None

        # Output directory where to save all the time consuming calculations
        self.outdir = self.config["tjpcov"].get("outdir", "./")
        os.makedirs(self.outdir, exist_ok=True)

    def _read_config(self, config):
        """Return the configuration dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.

        Return:
            dict: Configuration dictionary
        """
        if isinstance(config, dict):
            pass
        elif isinstance(config, str):
            config = self._parse(config)
        else:
            raise ValueError(
                "config must be of type dict or str, given" + f"{type(config)}"
            )

        return config

    @staticmethod
    def _parse(filename):
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
        output = os.path.join(self.get_outdir(), output)

        s = self.get_sacc_file().copy()
        s.add_covariance(cov, overwrite=True)

        if os.path.isfile(output) and (not overwrite):
            date = datetime.utcnow()
            timestamp = date.strftime("%Y%m%d%H%M%S")
            output_new = output + f"_{timestamp}"
            warnings.warn(
                f"Output file {output} already exists. "
                "Appending the UTC time to the filename to avoid "
                "losing the covariance computation. Writing sacc "
                "file to {output_new}"
            )
            output = output_new

        s.save_fits(output, overwrite=overwrite)

        return s

    def get_outdir(self):
        """Return the path to save the sacc file."""
        return self.outdir

    def get_sacc_file(self):
        """Return the input sacc file."""
        if self.sacc_file is None:
            sacc_file = self.config["tjpcov"].get("sacc_file")
            if isinstance(sacc_file, sacc.Sacc):
                self.sacc_file = sacc_file
            elif isinstance(sacc_file, str):
                self.sacc_file = sacc.Sacc.load_fits(sacc_file)
            else:
                raise ValueError(
                    "sacc_file must be a sacc.Sacc or str, "
                    + f"given {type(sacc_file)}"
                )

        return self.sacc_file
