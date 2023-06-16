import os
import warnings
import numpy as np
import pyccl as ccl 

from .covariance_builder import CovarianceFourier
# from tjpcov.tools import GlobalLock


class CovarianceFourierNG(CovarianceFourier):
    """Class to compute the covariance matrix in Fourier space for the
    Connected (Non-Gaussian) case. It inherits from CovarianceFourier.
    uses CCL modified version for the NG covariance.
    """
    
    cov_type="ng"
    
    def __init__(self, config): 
        """_summary_

        Args:
            config (dict or str): TODO complete usual description
        """
        super().__init__(config)
        
        self.ng_conf = self.config.get("NG", {})
        
    def get_covariance_block(
        self, 
        tracer_comb1, 
        tracer_comb2, 
        integration_method=None, 
        include_b_modes=True,
        _TODO_=None
    ):
        """_summary_

        Args:
            tracer_comb1 (_type_): _description_
            tracer_comb2 (_type_): _description_
            integration_method (_type_, optional): _description_. Defaults to None.
            include_b_modes (bool, optional): _description_. Defaults to True.
        
        Returns:
            array: _description_
        """
        fname = "ng_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)
        fname = os.path.join(self.io.outdir, fname)
        # TODO: check case cov already exists and clobber=False
        if os.path.isfile(fname): # TODO and not clobber: 
            print(f"Loading saved covariance {fname}")
            cov = np.load(fname)["cov"]
            return cov
        
        
        cosmo = self.get_cosmology() # TODO: not accepting cached cosmo as in nmt
        
        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2
        
        
        pass
    
    def _compute_all_blocks(self, **kwargs):
        """Compute all the covariance blocks for the NG covariance.
        
        Args:
            **kwargs: Arbitrary keyword arguments.
        
        Returns:   
            list: List of all the independent covariance blocks. 
        """
        pass 
    