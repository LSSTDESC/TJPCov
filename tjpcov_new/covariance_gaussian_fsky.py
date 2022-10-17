from . import bin_cov
from .covariance_builder import CovarianceFourier, CovarianceProjectedReal
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

    def get_covariance_block(self, tracer_comb1, tracer_comb2,
                             include_b_modes=True, for_real=False, lmax=None):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Parameters
        ----------
            tracer_comb 1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb 2 (list): List of the pair of tracer names of C_ell^2
            include_b_modes (bool): If True, return the full SSC with zeros in
            for B-modes (if any). If False, return the non-zero block. This
            option cannot be modified through the configuration file to avoid
            breaking the compatibility with the NaMaster covariance.
            for_real (bool): If True, returns the covariance before
            normalization and binning. It requires setting lmax.
            lmax (int): Maximum ell up to which to compute the covariance

        Returns:
        --------
            cov (array): The covariance
        """
        cosmo = self.get_cosmology()
        if for_real:
            if lmax is None:
                raise ValueError("You need to set lmax if for_real is True")
            else:
                ell = np.arange(lmax + 1)
        else:
            # binning information not need for Real
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

        # TODO: Maybe it's a better approximation just to use the ell_effective
        lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

        # Include the B-modes if requested (e.g. needed to build the full
        # covariance)
        if include_b_modes:
            nbpw = lb.size
            ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
            ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
            cov_full = np.zeros((nbpw, ncell1, nbpw, ncell2))
            cov_full[:, 0, :, 0] = cov
            cov_full = cov_full.reshape((nbpw * ncell1, nbpw * ncell2),
                                        order=self._reshape_order)
            cov = cov_full

        return cov


class CovarianceRealGaussianFsky(CovarianceProjectedReal):
    cov_type = 'gauss'
    _reshape_order = 'F'
    # Set the fourier attribute to None and set it later in the __init__
    fourier = None
    def __init__(self, config):
        super().__init__(config)
        # Note that the sacc file that the Fourier class will read is in real
        # space and you cannot use the methods that depend on a Fourier space
        # sacc file.
        self.fourier = CovarianceFourierGaussianFsky(config)
        self.fsky = self.fourier.fsky

    def _get_fourier_block(self, tracer_comb1, tracer_comb2):
        """
        Return the Fourier covariance block for two pair of tracers.

        Parameters
        ----------
        tracer_comb 1 (list): List of the pair of tracer names of C_ell^1
        tracer_comb 2 (list): List of the pair of tracer names of C_ell^2

        Returns:
        --------
        cov (array): Covariance matrix
        """
        # For now we just use the EE block which should be dominant over the
        # EB, BE and BB pieces when projecting to real space
        cov = self.fourier.get_covariance_block(tracer_comb1, tracer_comb2,
                                                for_real=True, lmax=self.lmax)
        norm = np.pi*4*self.fsky

        return cov / norm
