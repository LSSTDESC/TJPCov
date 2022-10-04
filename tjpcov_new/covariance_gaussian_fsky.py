from . import wigner_transform, bin_cov
from .covariance_builder import CovarianceFourier, CovarianceReal
import numpy as np
import os
import warnings
import pyccl as ccl

class CovarianceFourierGaussianFsky(CovarianceFourier):
    # TODO: Improve this class to use the sacc file information or
    # configuration given in the yaml file. Kept like this for now to check I
    # don't break the tests during the refactoring.
    cov_type = 'gauss'
    _reshape_order = 'F'

    def __init__(self, config):
        super().__init__(config)

        self.fsky = self.config['GaussianFsky'].get('fsky', None)
        if self.fsky is None:
            raise ValueError('You need to set fsky for FourierGaussianFsky')

    def get_binning_info(self, binning='linear'):
        """
        Get the ells for bins given the sacc object

        Parameters:
        -----------
        binning (str): Binning type.

        Returns:
        --------
        ell (array): All the ells covered
        ell_eff (array): The effective ells
        ell_edges (array): The bandpower edges
        """
        # TODO: This should be obtained from the sacc file or the input
        # configuration. Check how it is done in TXPipe:
        # https://github.com/LSSTDESC/TXPipe/blob/a9dfdb7809ac7ed6c162fd3930c643a67afcd881/txpipe/covariance.py#L23
        ell_eff = self.get_ell_eff()
        nbpw = ell_eff.size

        ellb_min, ellb_max = ell_eff.min(), ell_eff.max()
        if binning == 'linear':
            del_ell = (ell_eff[1:] - ell_eff[:-1])[0]

            ell_min = ellb_min - del_ell/2
            ell_max = ellb_max + del_ell/2

            ell_delta = (ell_max-ell_min)//nbpw
            ell_edges = np.arange(ell_min, ell_max+1, ell_delta)
            ell = np.arange(ell_min, ell_max + ell_delta - 2)
        else:
            raise NotImplementedError(f'Binning {binning} not implemented yet')

        return ell, ell_eff, ell_edges

    def get_covariance_block(self, tracer_comb1=None, tracer_comb2=None,
                             binned=True, for_real=False):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Returns:
        --------
            final:  unbinned covariance for C_ell
            final_b : binned covariance
        """
        cosmo = self.get_cosmology()
        ell, ell_bins, ell_edges = self.get_binning_info()

        ccl_tracers, tracer_Noise = self.get_tracer_info()

        cl = {}
        cl[13] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell)
        cl[24] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell)
        cl[14] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell)
        cl[23] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell)

        SN = {}
        SN[13] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[0] else 0
        SN[24] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[1] else 0
        SN[14] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[1] else 0
        SN[23] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[0] else 0


        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell))  # placeholder
        coupling_mat[1423] = np.eye(len(ell))  # placeholder

        cov = {}
        cov[1324] = np.outer(cl[13]+SN[13], cl[24]+SN[24])*coupling_mat[1324]
        cov[1423] = np.outer(cl[14]+SN[14], cl[23]+SN[23])*coupling_mat[1423]

        cov = cov[1423]+cov[1324]

        if for_real:
            # If it is to compute the real space covariance, return the
            # covariance before binning or normalizing
            return cov

        norm = (2*ell+1)*np.gradient(ell)*self.fsky
        cov /= norm

        if binned:
            lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

        return cov


class CovarianceRealGaussianFsky(CovarianceReal):
    def __init__(self, config):
        super().__init__(config)
        self.WT = None
        self.fourier = CovarianceFourierGaussianFsky(config)
        self.fsky = self.fourier.fsky

    def get_covariance_block(self, tracer_comb1=None, tracer_comb2=None,
                              ccl_tracers=None, tracer_Noise=None,
                              xi_plus_minus1='plus', xi_plus_minus2='plus',
                              binned=True):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Returns:
        --------
            final:  unbinned covariance for C_ell
            final_b : binned covariance
        """

        ell, ell_bins, ell_edges = self.fourier.get_binning_info()
        # TODO: Copied from the __init__ in main.py. This should be
        # changed! Kept for the tests
        # th_list = self.set_ell_theta(2.5, 250., 20, do_xi=True)
        theta, theta_bins, theta_edges = self.get_binning_info()

        cov = self.fourier.get_covariance_block(tracer_comb1, tracer_comb2,
                                                for_real=True)
        norm = np.pi*4*self.fsky

        cov /= norm

        if self.WT is None:  # class modifier of WT initialization
            print("Preparing WT...")

            self.WT = self.wt_setup(ell, theta)
            print("Done!")

        s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
        s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1[xi_plus_minus1]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2[xi_plus_minus2]
        th, cov = self.WT.projected_covariance2(l_cl=ell, s1_s2=s1_s2_1,
                                                s1_s2_cross=s1_s2_2,
                                                cl_cov=cov)
        if binned:
            d2r = np.pi/180
            thb, cov = bin_cov(r=th/d2r, r_bins=theta_edges, cov=cov)

        return cov

    def wt_setup(self, ell, theta):
        """
        Set this up once before the covariance evaluations

        Parameters:
        -----------
        ell (array): array of multipoles
        theta ( array): array of theta in degrees

        Returns:
        --------
        """
        WT_factors = {}
        WT_factors['lens', 'source'] = (0, 2)
        WT_factors['source', 'lens'] = (2, 0)  # same as (0,2)
        WT_factors['source', 'source'] = {'plus': (2, 2), 'minus': (2, -2)}
        WT_factors['lens', 'lens'] = (0, 0)

        self.WT_factors = WT_factors

        ell = np.array(ell)
        if not np.alltrue(ell > 1):
            # fao check warnings in WT for ell < 2
            print("Removing ell=1 for Wigner Transformation")
            ell = ell[(ell > 1)]

        WT_kwargs = {'l': ell,
                     'theta': theta * np.pi / 180,
                     's1_s2': [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)]}

        WT = wigner_transform(**WT_kwargs)
        return WT

    def get_cov_WT_spin(self, tracer_comb=None):
        """
        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------
        WT_factors:

        """
        tracers = []
        for i in tracer_comb:
            if 'lens' in i:
                tracers += ['lens']
            if 'src' in i:
                tracers += ['source']
        return self.WT_factors[tuple(tracers)]

    def get_binning_info(self, binning='log', do_xi=False):
        """
        Get the theta for bins given the sacc object

        Parameters:
        -----------
        binning (str): Binning type.

        Returns:
        --------
        theta (array): All the thetas covered
        theta_eff (array): The effective thetas
        theta_edges (array): The bandpower edges
        """
        # TODO: This should be obtained from the sacc file or the input
        # configuration. Check how it is done in TXPipe:
        # https://github.com/LSSTDESC/TXPipe/blob/a9dfdb7809ac7ed6c162fd3930c643a67afcd881/txpipe/covariance.py#L23

        theta_bins = self.get_theta_eff()
        nbpw = theta_bins.size

        thetab_min, thetab_max = theta_bins.min(), theta_bins.max()
        if binning == 'log':
            # assuming log bins
            del_theta = (theta_bins[1:]/theta_bins[:-1]).mean()
            theta_min = thetab_min-del_theta/2
            theta_max = thetab_min+del_theta/2

            th_min = theta_min/60  # in degrees
            th_max = theta_max/60
            theta_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                    nbpw+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), nbpw*30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, nbpw*30)

            theta = np.unique(np.sort(np.append(th, th2)))
            theta_bins = 0.5 * (theta_edges[1:] + theta_edges[:-1])
        else:
            raise NotImplementedError(f'Binning {binning} not implemented yet')

        return theta, theta_bins, theta_edges
