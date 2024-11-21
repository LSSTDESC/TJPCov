import os

# import healpy as hp
import numpy as np
import pyccl as ccl

from .covariance_builder import CovarianceFourier


class FouriercNGHaloModel(CovarianceFourier):
    """Class to compute the CellxCell Halo Model cNG Covariance."""

    cov_type = "cNG"

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        self.cNG_conf = self.config.get("cNG", {})

    def get_covariance_block(
        self,
        tracer_comb1,
        tracer_comb2,
        integration_method=None,
        include_b_modes=True,
    ):
        """Compute a single cNG covariance matrix for a given pair of C_ell.

        If outdir is set, it will save the covariance to a file called
        cng_tr1_tr2_tr3_tr4.npz. This file will be read and its output returned
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
                configuration file through config['cNG']['integration_method'].
                Elsewise, it will use 'qag_quad'.
            include_b_modes (bool, optional): If True, return the full cNG with
                zeros in for B-modes (if any). If False, return the non-zero
                block. This option cannot be modified through the configuration
                file to avoid breaking the compatibility with the NaMaster
                covariance. Defaults to True.

        Returns:
            array:  Super sample covariance matrix for a pair of C_ell.
        """
        fname = "cng_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)
        fname = os.path.join(self.io.outdir, fname)
        if os.path.isfile(fname):
            cf = np.load(fname)
            return cf["cov" if include_b_modes else "cov_nob"]

        if integration_method is None:
            integration_method = self.cNG_conf.get(
                "integration_method", "qag_quad"
            )

        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2

        cosmo = self.get_cosmology()
        mass_def = ccl.halos.MassDef200m
        hmf = ccl.halos.MassFuncTinker08(mass_def=mass_def)
        hbf = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
        cM = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
        nfw = ccl.halos.HaloProfileNFW(
            mass_def=mass_def, concentration=cM, fourier_analytic=True
        )
        hmc = ccl.halos.HMCalculator(
            mass_function=hmf, halo_bias=hbf, mass_def=mass_def
        )

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
        # Cut the array for efficiency
        sel = 1 / a < z_max + 1
        # Include the next node so that z_max is in the range
        sel[np.sum(~sel) - 1] = True
        a = a[sel]

        bias1 = self.bias_lens.get(tr[1], 1)
        bias2 = self.bias_lens.get(tr[2], 1)
        bias3 = self.bias_lens.get(tr[3], 1)
        bias4 = self.bias_lens.get(tr[4], 1)

        ccl_tracers, _ = self.get_tracer_info()

        masks = self.get_masks_dict(tr, {})
        # TODO: This should be unified with the other classes in
        # CovarianceBuilder.
        fsky = np.mean(masks[1] * masks[2] * masks[3] * masks[4])

        # Tk3D = b1*b2*b3*b4 * T_234h (NFW) + T_1h (HOD)

        if lk_arr is None:
            lk_arr = cosmo.get_pk_spline_lk()
        if a_arr is None:
            a_arr = cosmo.get_pk_spline_a()


        tkk = ccl.halos.halomod_trispectrum_1h(cosmo, hmc, np.exp(lk_arr),
                                               a_arr, prof=nfw)

        tkk += ccl.halos.halomod_trispectrum_2h_22(cosmo, hmc, np.exp(lk_arr),
                                                   a_arr, prof=nfw)

        tkk += ccl.halos.halomod_trispectrum_2h_13(cosmo, hmc, np.exp(lk_arr),
                                                   a_arr, prof=nfw)

        tkk += ccl.halos.halomod_trispectrum_3h(cosmo, hmc, np.exp(lk_arr),
                                                a_arr, prof=nfw)

        tkk *= bias1 * bias2 * bias3 * bias4

        # TODO: Commented out for now to avoid having to specify HOD params
        # tkk += halomod_trispectrum_4h(cosmo, hmc, np.exp(lk_arr), a_arr,
        #                               prof=prof,
        #                               prof2=prof2,
        #                               prof3=prof3,
        #                               prof4=prof4,
        #                               p_of_k_a=None)

        tk3D = ccl.tk3d.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk)


        ell = self.get_ell_eff()
        cov_cng = ccl.covariances.angular_cl_cov_cNG(
            cosmo,
            tracer1=ccl_tracers[tr[1]],
            tracer2=ccl_tracers[tr[2]],
            tracer3=ccl_tracers[tr[3]],
            tracer4=ccl_tracers[tr[4]],
            ell=ell,
            t_of_kk_a=tk3D,
            integration_method=integration_method,
            fsky=fsky,
        )

        nbpw = ell.size
        ncell1 = self.get_tracer_comb_ncell(tracer_comb1)
        ncell2 = self.get_tracer_comb_ncell(tracer_comb2)
        cov_full = np.zeros((nbpw, ncell1, nbpw, ncell2))
        cov_full[:, 0, :, 0] = cov_cng
        cov_full = cov_full.reshape((nbpw * ncell1, nbpw * ncell2))

        np.savez_compressed(fname, cov=cov_full, cov_nob=cov_cng)

        if not include_b_modes:
            return cov_cng

        return cov_full
