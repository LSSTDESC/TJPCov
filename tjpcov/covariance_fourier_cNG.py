import os

# import healpy as hp
import numpy as np
import pyccl as ccl

from .clusters_helpers import get_halo_model
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

        self.HOD_dict = {
            "log10Mmin_0": None,
            "log10Mmin_p": None,
            "siglnM_0": None,
            "siglnM_p": None,
            "log10M0_0": None,
            "log10M0_p": None,
            "log10M1_0": None,
            "log10M1_p": None,
            "alpha_0": None,
            "alpha_p": None,
            "fc_0": None,
            "fc_p": None,
            "bg_0": None,
            "bg_p": None,
            "bmax_0": None,
            "bmax_p": None,
            "a_pivot": None,
            "ns_independent": None,
        }

        for key in self.HOD_dict.keys():
            self.HOD_dict[key] = self.config["HOD"].get(key, None)
            if self.HOD_dict[key] is None:
                raise ValueError(
                    "You need to set "
                    + key
                    + " in the HOD header for cNG calculation"
                )

        # Cache of the trispectrum terms, shared by many blocks
        self.tk_cache = {}

    def _get_tk3D(self, a_arr, lk_arr, isnc, bias):
        """Return the cNG trispectrum for a combination of tracers.

        Tk3D = bias * T_234h (NFW) + T_1h (HOD for number counts tracers).
        The 2-, 3- and 4-halo terms are computed once for each a and k grid
        and the 1-halo term once for each combination of profiles.

        Args:
            a_arr (array): Scale factors
            lk_arr (array): Natural logarithm of the wavenumbers in 1/Mpc
            isnc (tuple): If each of the 4 tracers is number counts
            bias (float): Product of the linear bias of the 4 tracers

        Returns:
            :class:`~pyccl.tk3d.Tk3D`: Connected trispectrum
        """
        cosmo = self.get_cosmology()
        hmc, cM, nfw = get_halo_model(self.config)
        mass_def = hmc.mass_def
        k = np.exp(lk_arr)
        grid = (a_arr.tobytes(), lk_arr.tobytes())

        if (grid, "234h") not in self.tk_cache:
            args = (cosmo, hmc, k, a_arr)
            H = ccl.halos
            self.tk_cache[(grid, "234h")] = (
                H.pk_4pt.halomod_trispectrum_2h_22(
                    *args, prof=nfw, separable_growth=True
                )
                + H.halomod_trispectrum_2h_13(*args, prof=nfw)
                + H.halomod_trispectrum_3h(
                    *args, prof=nfw, separable_growth=True
                )
                + H.halomod_trispectrum_4h(
                    *args, prof=nfw, separable_growth=True
                )
            )

        if (grid, isnc) not in self.tk_cache:
            hod = ccl.halos.HaloProfileHOD(
                mass_def=mass_def, concentration=cM, **self.HOD_dict
            )
            # Choose using the HOD with or without number_counts = True
            # based on the results of isnc
            p = [hod if nc else nfw for nc in isnc]
            p2pt = [
                (
                    ccl.halos.profiles_2pt.Profile2ptHOD()
                    if isnc[i] and isnc[i + 1]
                    else ccl.halos.profiles_2pt.Profile2pt()
                )
                for i in (0, 2)
            ]
            self.tk_cache[(grid, isnc)] = ccl.halos.halomod_trispectrum_1h(
                cosmo,
                hmc,
                k,
                a_arr,
                prof=p[0],
                prof2=p[1],
                prof3=p[2],
                prof4=p[3],
                prof12_2pt=p2pt[0],
                prof34_2pt=p2pt[1],
            )

        tkk = bias * self.tk_cache[(grid, "234h")]
        tkk += self.tk_cache[(grid, isnc)]
        return ccl.tk3d.Tk3D(
            a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk, is_logt=False
        )

    def _get_fsky(self, tr=None):
        """Returns the fractional sky area from the mean survey masks.

        Args:
            masks (:obj:`dict`): dictionary containing the survey
                tracers to obtain the survey mask.
        Returns:
            - (:obj:`float`): fractional sky area.
        """
        masks = self.get_masks_dict(tr, {})
        # Masks can be float32 maps, which CCL does not accept as a double
        fsky = float(np.mean(masks[1] * masks[2] * masks[3] * masks[4]))
        return fsky

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
            array:  Connected NG covariance matrix for a pair of C_ell.
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

        # Get range of redshifts. z_min = 0 for compatibility with the limber
        # integrals
        sacc_file = self.io.get_sacc_file()
        z_max = []
        for i in range(4):
            tr_sacc = sacc_file.tracers[tr[i + 1]]
            z = tr_sacc.z
            z_max.append(z.max())
        # Divide by zero errors happen when default a_arr used for 1h term

        z_max = np.min(z_max)

        # Array of a.
        # Use the a's in the pk spline
        na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)
        # Cut the array for efficiency
        sel = 1 / a_arr < z_max + 1
        # Include the next node so that z_max is in the range
        sel[np.sum(~sel) - 1] = True
        a_arr = a_arr[sel]

        # Array of k. The P(k) spline one by default, which can have > 1000
        # nodes. config['cNG']['nk'] sets a coarser grid to speed up the
        # trispectrum calculation, which scales as nk^2.
        lk_arr = cosmo.get_pk_spline_lk()
        if self.cNG_conf.get("nk") is not None:
            lk_arr = np.linspace(lk_arr[0], lk_arr[-1], self.cNG_conf["nk"])

        bias = np.prod([self.bias_lens.get(tr[i], 1) for i in range(1, 5)])

        ccl_tracers, _ = self.get_tracer_info()

        # TODO: This should be unified with the other classes in
        # CovarianceBuilder.
        fsky = self._get_fsky(tr=tr)

        isnc = tuple(
            (sacc_file.tracers[tr[i]].quantity == "galaxy_density")
            or ("lens" in tr[i])
            for i in range(1, 5)
        )
        tk3D = self._get_tk3D(a_arr, lk_arr, isnc, bias)
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
