import pyccl as ccl
import sacc
from tjpcov import wigner_transform, bin_cov, parse
import numpy as np


class CovarianceCalculator():
    def __init__ (self, 
                        cosmo_fn = None, 
                        sacc_fn_xi = None, # real space
                        sacc_fn_cl = None, # harmonic space
                        window_fn = None):
        """
        Is reading from a single yaml a good option?

        Parameters
        ----------
        C : CCL Cosmology object
            Fiducial Cosmology used in calculations
        sacc : sacc file
            sacc object containing the tracers and binning to be used in covariance calculation
        window : None, dict, float
            If None, used window function specified in the sacc file
            if dict, the keys are sacc tracer names and values are either HealSparse inverse variance
            maps 
            if float it is assumed to be f_sky value
        """    
        if isinstance(cosmo_fn, str): 
            try:
                cosmo = ccl.Cosmology.read_yaml(cosmo_fn)
            except Exception as err:
                print(f"Error in ccl cosmology loading from yaml \n{err}")
        else:
            cosmo = cosmo_fn

        if isinstance(sacc_fn_cl, str):
            try:
                cl_data =sacc.Sacc.load_fits(sacc_fn_cl)
            except Exception as err:
                print(f"Error in sacc loading from yaml \n{err}")
        else:
            cl_data = sacc_fn_cl # fao assert here

        if isinstance(sacc_fn_xi, str):
            try:
                xi_data = sacc.Sacc.load_fits(sacc_fn_xi)
            except Exception as err:
                print(f"Error in sacc loading from yaml \n{err}")
        else:
            xi_data = sacc_fn_xi #fao assert here




        self.cosmo = cosmo
        self.cl_data = cl_data
        self.xi_data = xi_data
        self.window_fn = window_fn # Define a windown handler



        #get_ell_theta
        #WT

    def print_setup(self):
        """
        Prints current setup
        """
        cosmo = self.cosmo
        ell  = self.ell



    def wt_setup(ell, theta):
        """
        Set this up once before the covariance evaluations

        Parameters:
        -----------
        ell (array): array of multipoles
        theta ( array): array of theta in degrees

        Returns:
        --------
        """
        # ell = two_point_data.metadata['ell']
        # theta_rad = two_point_data.metadata['th']*d2r
        # get_ell_theta()


        WT_factors={}
        WT_factors['lens','source']=(0,2)
        WT_factors['source','lens']=(2,0) #same as (0,2)
        WT_factors['source','source']={'plus':(2,2),'minus':(2,-2)}
        WT_factors['lens','lens']=(0,0)

        ell = np.array(ell)
        if not np.alltrue(ell>1):
            # fao check warnings in WT for ell < 2
            print("Removing ell=1 for Wigner Transformation")
            ell = ell[(ell>1)]

        WT_kwargs={'l': ell,
                   'theta': theta*d2r,
                   's1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}


        WT=wigner_transform(**WT_kwargs)
        return WT



    def get_cov_WT_spin(tracer_comb=None):
        """
        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------

        """
    #     tracers=tuple(i.split('_')[0] for i in tracer_comb)
        tracers=[]
        for i in tracer_comb:
            if 'lens' in i:
                tracers+=['lens']
            if 'src' in i:
                tracers+=['source']
        return WT_factors[tuple(tracers)]


