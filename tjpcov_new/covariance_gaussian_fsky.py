from . import wigner_transform, bin_cov
from .covariance_builder import CovarianceFourier, CovarianceReal
import numpy as np
import os
import warnings
import pyccl as ccl

class FourierGaussianFsky(CovarianceFourier):
    # TODO: Improve this class to use the sacc file information or
    # configuration given in the yaml file. Kept like this for now to check I
    # don't break the tests during the refactoring.
    def __init__(self, config):
        super.__init__(config)

        self.fsky = self.config['GaussianFsky'].get('fsky', None)
        if self.fsky is None:
            raise ValueError('You need to set fsky for FourierGaussianFsky')

    def get_covariance_block(self, tracer_comb1=None, tracer_comb2=None,
                             ccl_tracers=None, tracer_Noise=None, binned=True,
                             for_real=False):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Returns:
        --------
            final:  unbinned covariance for C_ell
            final_b : binned covariance
        """
        cosmo = self.get_cosmology()
        ell, ell_bins, ell_edges = self.get_ell_theta()

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

        if binned and (ell_edges is not None):
            lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

        return cov

    def set_ell_theta(self, ang_min, ang_max, n_ang,
                      ang_scale='linear', do_xi=False):
        """
        Utility for return custom theta/ell bins (outside sacc)

        Parameters:
        -----------
        ang_min (int, float):
            if do_xi, ang is assumed to be theta (arcmin)
            if do_xi == False,  ang is assumed to be ell
        Returns:
        --------
            (theta, theta_edges ) (degrees):
        """
        # FIXME:
        # Use sacc is passing this
        if not do_xi:
            ang_delta = (ang_max-ang_min)//n_ang
            ang_edges = np.arange(ang_min, ang_max+1, ang_delta)
            ang = np.arange(ang_min, ang_max + ang_delta - 2)

        if do_xi:
            th_min = ang_min/60  # in degrees
            th_max = ang_max/60
            n_th_bins = n_ang
            ang_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                    n_th_bins+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), n_th_bins*30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, n_th_bins*30)

            ang = np.unique(np.sort(np.append(th, th2)))
            ang_bins = 0.5 * (ang_edges[1:] + ang_edges[:-1])

            return ang, ang_bins, ang_edges  # TODO FIXIT

        return ang, ang_edges

    def get_ell_theta(self, ang_scale='linear', do_xi=False):
        """
        Get ell or theta for bins given the sacc object
        For now, presuming only log and linear bins

        Parameters:
        -----------

        Returns:
        --------
        """
        ang_name = "ell" if not do_xi else 'theta'

        # assuming same ell for all bins:
        sacc_file = self.get_sacc_file()
        data_type = sacc_file.get_data_types()[0]
        tracer_comb = sacc_file.get_tracer_combinations()[0]
        ang_bins = sacc_file.get_tag(ang_name, data_type=data_type,
                                     tracers=tracer_comb)

        ang_bins = np.array(ang_bins)

        angb_min, angb_max = ang_bins.min(), ang_bins.max()
        if ang_name == 'theta':
            # assuming log bins
            del_ang = (ang_bins[1:]/ang_bins[:-1]).mean()
            ang_scale = 'log'
            assert 1 == 1

        elif ang_name == 'ell':
            # assuming linear bins
            del_ang = (ang_bins[1:] - ang_bins[:-1])[0]
            ang_scale = 'linear'

        ang, ang_edges = self.set_ell_theta(angb_min-del_ang/2,
                                            angb_max+del_ang/2,
                                            len(ang_bins),
                                            ang_scale=ang_scale, do_xi=do_xi)
        # Sanity check
        if ang_scale == 'linear':
            assert np.allclose((ang_edges[1:]+ang_edges[:-1])/2, ang_bins), \
                "differences in produced ell/theta"
        return ang, ang_bins, ang_edges


class RealGaussianFsky(CovarianceReal, FourierGaussianFsky):
    def __init__(self, config):
        super().__init__(config)
        self.WT = None

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

        ell, ell_bins, ell_edges = self.get_ell_theta()
        # TODO: Copied from the __init__ in main.py. This should be
        # changed! Kept for the tests
        th_list = self.set_ell_theta(2.5, 250., 20, do_xi=True)
        theta,  theta_bins, theta_edges,  = th_list

        cov = super().get_covariance_block(tracer_comb1, tracer_comb2,
                                        ccl_tracers, tracer_Noise,
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
                     'theta': theta*d2r,
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
