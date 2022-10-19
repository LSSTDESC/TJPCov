import pyccl as ccl
import pickle
import sacc
import numpy as np
import warnings
import yaml
import jinja2
import importlib
import os
from datetime import datetime

class CovarianceIO():
    def __init__(self, config):
        """
        CovarianceIO class for TJPCov.

        This is the class that handles the input/output of the covariances.
        It does not compute anything.

        Parameters
        ----------
        config (dict or str):
        """
        self.config = self._read_config(config)
        self.sacc_file = None

        # Output directory where to save all the time consuming calculations
        self.outdir = self.config['tjpcov'].get('outdir', './')
        os.makedirs(self.outdir, exist_ok=True)

    def _read_config(self, config):
        if isinstance(config, dict):
            pass
        elif isinstance(config, str):
            config = self._parse(config)
        else:
            raise ValueError("config must be of type dict or str, given" +
                             f"{type(config)}")

        return config

    @staticmethod
    def _parse(filename):
        """
        Parse a configuration file.

        Parameters
        ----------
        filename : str
            The config file to parse. Should be YAML formatted.

        Returns
        -------
        config: dict
            The raw config file as a dictionary.
        """

        with open(filename, 'r') as fp:
            config_str = jinja2.Template(fp.read()).render()
        config = yaml.load(config_str, Loader=yaml.Loader)

        return config

    def create_sacc_cov(self, cov, output='cls_cov.fits', overwrite=False):
        """
        Write created cov to a new sacc object

        Parameters:
        ----------
        output (str): filename output.
        overwrite (bool): True if you want to overwrite an existing file. If
        False, it will not overwrite the file but will append the UTC time to
        the output to avoid losing the computed covariance.

        Returns:
        -------
        None

        """
        output = os.path.join(self.get_outdir(), output)

        s = self.get_sacc_file().copy()
        s.add_covariance(cov)

        if os.path.isfile(output) and (not overwrite):
            date = datetime.utcnow()
            timestamp = date.strftime("%Y%m%d%H%M%S")
            output_new = output + f'_{timestamp}'
            warnings.warn(f"Output file {output} already exists. " +
                          "Appending the UTC time to the filename to avoid " +
                          "losing the covariance computation. Writing sacc " +
                          "file to {output_new}")
            output = output_new

        s.save_fits(output, overwrite=overwrite)

        return s

    def get_outdir(self):
        return self.outdir

    def get_sacc_file(self):
        if self.sacc_file is None:
            sacc_file = self.config['tjpcov'].get('sacc_file')
            if isinstance(sacc_file, sacc.Sacc):
                self.sacc_file = sacc_file
            elif isinstance(sacc_file, str):
                self.sacc_file = sacc.Sacc.load_fits(sacc_file)
            else:
                raise ValueError("sacc_file must be a sacc.Sacc or str, " +
                                 f"given {type(sacc_file)}")

        return self.sacc_file

    def get_sacc_with_concise_dtypes(self):
        """
        Return a copy of the sacc file with concise data types

        Parameters:

        Returns:
        --------
            sacc_data (Sacc): Data Sacc instance with concise data types
        """

        s = self.get_sacc_file().copy()
        dtypes = s.get_data_types()

        dt_long = []
        for dt in dtypes:
            if len(dt.split('_')) > 2:
                dt_long.append(dt)

        for dt in dt_long:
            pd = sacc.parse_data_type_name(dt)

            if pd.statistic != 'cl':
                raise ValueError(f'data_type {dt} not recognized. Is it a Cell?')

            if pd.subtype is None:
                dc = 'cl_00'
            elif pd.subtype == 'e':
                dc = 'cl_0e'
            elif pd.subtype == 'b':
                dc = 'cl_0b'
            elif pd.subtype == 'ee':
                dc = 'cl_ee'
            elif pd.subtype == 'bb':
                dc = 'cl_bb'
            elif pd.subtype == 'eb':
                dc = 'cl_eb'
            elif pd.subtype == 'be':
                dc = 'cl_be'
            else:
                raise ValueError(f'Data type subtype {pd.subtype} not recognized')


            # Change the data_type to its concise versio
            for dp in s.get_data_points(dt):
                dp.data_type = dc

        return s
