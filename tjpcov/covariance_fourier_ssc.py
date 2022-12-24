import os

import healpy as hp
import numpy as np
import pyccl as ccl

from .covariance_builder import CovarianceFourier


class FourierSSCHaloModel(CovarianceFourier):
    """Class to compute the CellxCell Halo Model Super Sample Covariance.

    The SSC is computed in CCL with the "linear bias" approximation using
    :func:`pyccl.halos.halo_model.halomod_Tk3D_SSC_linear_bias`.
    """

    cov_type = "SSC"

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.ssc_conf = self.config.get("SSC", {})

    def get_covariance_block(
        self,
        tracer_comb1,
        tracer_comb2,
        integration_method=None,
        include_b_modes=True,
    ):
        """Compute a single SSC covariance matrix for a given pair of C_ell.

        If outdir is set, it will save the covariance to a file called
        ssc_tr1_tr2_tr3_tr4.npz. This file will be read and its output returned
        if found.

        Blocks of the B-modes are assumed 0 so far.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            integration_method (str, optional): integration method to be
                used for the Limber integrals. Possibilities: 'qag_quad' (GSL's
                qag method backed up by quad when it fails) and 'spline'
                (the integrand is splined and then integrated analytically). If
                given, it will take priority over the specified in the
                configuration file through config['SSC']['integration_method'].
                Elsewise, it will use 'qag_quad'.
            include_b_modes (bool, optional): If True, return the full SSC with
                zeros in for B-modes (if any). If False, return the non-zero
                block. This option cannot be modified through the configuration
                file to avoid breaking the compatibility with the NaMaster
                covariance. Defaults to True.

        Returns:
            array:  Super sample covariance matrix for a pair of C_ell.
        """
        fname = "ssc_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)
        fname = os.path.join(self.io.outdir, fname)
        if os.path.isfile(fname):
            cf = np.load(fname)
            return cf["cov" if include_b_modes else "cov_nob"]

        if integration_method is None:
            integration_method = self.ssc_conf.get(
                "integration_method", "qag_quad"
            )

        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2

        cosmo = self.get_cosmology()
        mass_def = ccl.halos.MassDef200m()
        hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
        hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
        nfw = ccl.halos.HaloProfileNFW(
            ccl.halos.ConcentrationDuffy08(mass_def), fourier_analytic=True
        )
        hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)

        # Get range of redshifts. z_min = 0 for compatibility with the limber
        # integrals
        sacc_file = self.io.get_sacc_file()
        z_max = []
        for i in range(4):
            tr_sacc = sacc_file.tracers[tr[i + 1]]
            z = tr_sacc.z
            # z, nz = tr_sacc.z, tr_sacc.nz
            # z_min.append(z[np.where(nz > 0)[0][0]])
            # z_max.append(z[np.where(np.cumsum(nz)/np.sum(nz) > 0.999)[0][0]])
            z_max.append(z.max())

        z_max = np.min(z_max)

        # Array of a.
        # Use the a's in the pk spline
        na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        a, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)
        a = a[1 / a < z_max + 1]

        bias1 = self.bias_lens.get(tr[1], 1)
        bias2 = self.bias_lens.get(tr[2], 1)
        bias3 = self.bias_lens.get(tr[3], 1)
        bias4 = self.bias_lens.get(tr[4], 1)

        ccl_tracers, _ = self.get_tracer_info()

        isnc1 = isinstance(ccl_tracers[tr[1]], ccl.NumberCountsTracer)
        isnc2 = isinstance(ccl_tracers[tr[2]], ccl.NumberCountsTracer)
        isnc3 = isinstance(ccl_tracers[tr[3]], ccl.NumberCountsTracer)
        isnc4 = isinstance(ccl_tracers[tr[4]], ccl.NumberCountsTracer)

        tk3D = ccl.halos.halomod_Tk3D_SSC_linear_bias(
            cosmo=cosmo,
            hmc=hmc,
            prof=nfw,
            bias1=bias1,
            bias2=bias2,
            bias3=bias3,
            bias4=bias4,
            is_number_counts1=isnc1,
            is_number_counts2=isnc2,
            is_number_counts3=isnc3,
            is_number_counts4=isnc4,
        )

        masks = self.get_masks_dict(tr, {})
        # TODO: Optimize this, avoid computing the mask_wl for all blocks.
        # Note that this is correct for same footprint cross-correlations. In
        # case of multisurvey analyses this approximation might break.
        m12 = masks[1] * masks[2]
        m34 = masks[3] * masks[4]
        area = hp.nside2pixarea(hp.npix2nside(m12.size))

        alm = hp.map2alm(m12)
        blm = hp.map2alm(m34)

        mask_wl = hp.alm2cl(alm, blm)
        mask_wl *= 2 * np.arange(mask_wl.size) + 1
        mask_wl /= np.sum(m12) * np.sum(m34) * area**2

        # TODO: Allow using fsky instead of the masks?
        sigma2_B = ccl.sigma2_B_from_mask(cosmo, a=a, mask_wl=mask_wl)

        ell = self.get_ell_eff()
        cov_ssc = ccl.covariances.angular_cl_cov_SSC(
            cosmo,
            cltracer1=ccl_tracers[tr[1]],
            cltracer2=ccl_tracers[tr[2]],
            cltracer3=ccl_tracers[tr[3]],
            cltracer4=ccl_tracers[tr[4]],
            ell=ell,
            tkka=tk3D,
            sigma2_B=(a, sigma2_B),
            integration_method=integration_method,
        )

        nbpw = ell.size
        ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
        ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
        cov_full = np.zeros((nbpw, ncell1, nbpw, ncell2))
        cov_full[:, 0, :, 0] = cov_ssc
        cov_full = cov_full.reshape((nbpw * ncell1, nbpw * ncell2))

        np.savez_compressed(fname, cov=cov_full, cov_nob=cov_ssc)

        if not include_b_modes:
            return cov_ssc

        return cov_full
