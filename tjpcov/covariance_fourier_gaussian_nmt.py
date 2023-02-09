import os
import warnings

import numpy as np
import pyccl as ccl
import pymaster as nmt

from .covariance_builder import CovarianceFourier


class FourierGaussianNmt(CovarianceFourier):
    """Class to compute the Gaussian CellxCell covariance with NaMaster.

    This class uses the Narrow Kernel Approximation. It can also use the
    Toeplitz approximation.
    """

    cov_type = "gauss"

    def __init__(self, config):
        """Initialize the class with a config file or dictionary.

        Args:
            config (dict or str): If dict, it returns the configuration
                dictionary directly. If string, it asumes a YAML file and
                parses it.
        """
        super().__init__(config)

        # For NaMaster you need to pass the masks
        self.mask_files = self.config["tjpcov"].get("mask_file")
        self.mask_names = self.config["tjpcov"].get("mask_names")

        # Binning info is only needed if workspaces are not passed
        self.binning_info = self.config["tjpcov"].get("binning_info", None)

        # nside is needed if mask_files is a hdf5 file
        self.nside = self.config["tjpcov"].get("nside", None)

        # Read NaMaster specific options
        self.nmt_conf = self.config.get("NaMaster", {})
        for k in ["f", "w", "cw"]:
            if k not in self.nmt_conf:
                self.nmt_conf[k] = {}

        # Read cache from input file. It will update the cache passed as an
        # argument of the different methods
        self.cache = self.config.get("cache", {})

    def _compute_all_blocks(self, **kwargs):
        """Compute all the independent covariance blocks.

        Args:
            **kwargs: The arguments to pass to get_covariance_block. See its
                documentation.

        Returns:
            list: List of all the independent covariance blocks.
        """
        # We are redefining this method to improve performance by parallelizing
        # first the blocks that produce independent workspaces and covariance
        # workspaces. That way two processes will not be computing the same
        # (cov)workspace at the same time.

        ccl_tracers, tracer_Noise, tracer_Noise_coupled = self.get_tracer_info(
            return_noise_coupled=True
        )

        # Make a list of all pair of tracer combinations needed to compute the
        # independent workspaces
        trs_wsp = self.get_list_of_tracers_for_wsp()
        # Now the tracers for covariance workspaces (without trs_wsp)
        trs_cwsp = self.get_list_of_tracers_for_cov_wsp(remove_trs_wsp=True)

        # Make a list of all remaining combinations
        tracers_cov = self.get_list_of_tracers_for_cov_without_trs_wsp_cwsp()

        # Save blocks and the corresponding tracers, as comm.gather does not
        # return the blocks in the original order.
        blocks = []
        tracers_blocks = []
        print("Computing independent covariance blocks")
        print("Computing the blocks for independent workspaces")
        for tracer_comb1, tracer_comb2 in self._split_tasks_by_rank(trs_wsp):
            print(tracer_comb1, tracer_comb2)
            cov = self.get_covariance_block_for_sacc(
                tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, **kwargs
            )
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))

        if self.comm:
            self.comm.Barrier()

        print("Computing the blocks for independent covariance workspaces")
        for tracer_comb1, tracer_comb2 in self._split_tasks_by_rank(trs_cwsp):
            print(tracer_comb1, tracer_comb2)
            cov = self.get_covariance_block_for_sacc(
                tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, **kwargs
            )
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))

        if self.comm:
            self.comm.Barrier()

        print("Computing the remaining blocks")
        # Now loop over the remaining tracers
        for tracer_comb1, tracer_comb2 in self._split_tasks_by_rank(
            tracers_cov
        ):
            print(tracer_comb1, tracer_comb2)
            cov = self.get_covariance_block_for_sacc(
                tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, **kwargs
            )
            blocks.append(cov)
            tracers_blocks.append((tracer_comb1, tracer_comb2))

        return blocks, tracers_blocks

    def get_cl_for_cov(self, clab, nlab, ma, mb, w, nl_is_cp):
        """Computes the coupled Cell that goes into the covariance matrix.

        Args:
            clab (array): Fiducial Cell for the tracers a and b, used in the
                covariance estimation
            nlab (array): Coupled noise for the tracers a and b
            ma (array): Mask of the field a
            mb (array): Mask of the field b
            w (:obj:`pymaster.workspaces.NmtWorkspace`): NmtWorkspace of the
            fields a and b nl_is_cp (bool): True if nlab is coupled. False
            otherwise.

        Returns:
            array:  Coupled Cell with signal and noise
        """
        mean_mamb = np.mean(ma * mb)
        if mean_mamb == 0:
            cl_cp = np.zeros_like(nlab)
        elif nl_is_cp:
            cl_cp = (w.couple_cell(clab) + nlab) / mean_mamb
        else:
            cl_cp = (w.couple_cell(clab) + np.mean(ma) * nlab) / mean_mamb

        return cl_cp

    def get_covariance_block(
        self,
        tracer_comb1,
        tracer_comb2,
        use_coupled_noise=True,
        coupled=False,
        cache=None,
    ):
        """Compute a single covariance matrix for a given pair of C_ell.

        If outdir is set, it will save the covariance to a file called
        cov_tr1_tr2_tr3_tr4.npz. This file will be read and its output
        returned if found.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
            use_coupled_noise (bool, optional): If True, use the coupled noise.
                Note that if noise is provided via the cache arg, this will
                be used and assumed to be coupled if this option is True.
                Defaults to True.
            coupled (bool, optional): True to return the coupled Gaussian
                covariance (default False)
            cache (dict): Dictionary with the corresponding noise, masks,
                fields, workspaces and covariance workspaces. It accepts noise
                (keys: 'SN13', 'SN23', 'SN14', 'SN24'), masks (keys: 'm1',
                'm2', 'm3', 'm4'), fields (keys: 'f1', 'f2', 'f3', 'f4'),
                workspaces (keys: 'w13', 'w23', 'w14', 'w24', 'w12', 'w34'),
                the covariance workspace (key: 'cw') and a NmtBin (key:
                'bins').

        Returns:
            array: Gaussian covariance matrix for a pair of C_ell.
        """
        if coupled:
            raise NotImplementedError(
                "Computing coupled covariance matrix not implemented yet"
            )
            fname = "covcp_{}_{}_{}_{}.npz".format(
                *tracer_comb1, *tracer_comb2
            )
        else:
            fname = "cov_{}_{}_{}_{}.npz".format(*tracer_comb1, *tracer_comb2)

        fname = os.path.join(self.io.outdir, fname)
        if os.path.isfile(fname):
            print(f"Loading saved covariance {fname}")
            cov = np.load(fname)["cov"]
            return cov

        if cache is None:
            cache = {}
        cache.update(self.cache)

        if "bins" in cache:
            bins = cache["bins"]
            if (self.binning_info is not None) and (
                bins is not self.binning_info
            ):
                raise ValueError(
                    "Binning passed through cache is not the "
                    "same as the one passed during "
                    "initialization."
                )
        else:
            bins = self.binning_info

        # Get nbpw and ell arrays. Doing all this stuff because the window
        # function in the sacc file might be wrong.
        nbpw = self.get_nbpw()
        nell = self.get_nell(bins, self.nside, cache)

        ell = np.arange(nell)

        if "cosmo" in cache:
            cosmo = cache["cosmo"]
        else:
            cosmo = self.get_cosmology()

        tr = {}
        tr[1], tr[2] = tracer_comb1
        tr[3], tr[4] = tracer_comb2

        ncell = {}
        ncell[12] = self.get_tracer_comb_ncell(tracer_comb1)
        ncell[34] = self.get_tracer_comb_ncell(tracer_comb2)
        ncell[13] = self.get_tracer_comb_ncell((tr[1], tr[3]))
        ncell[24] = self.get_tracer_comb_ncell((tr[2], tr[4]))
        ncell[14] = self.get_tracer_comb_ncell((tr[1], tr[4]))
        ncell[23] = self.get_tracer_comb_ncell((tr[2], tr[3]))

        s = self.get_tracers_spin_dict(tr)

        ccl_tracers, tracer_Noise, tracer_Noise_coupled = self.get_tracer_info(
            return_noise_coupled=True
        )

        # Fiducial cl
        cl = {}
        # Noise (coupled or not)
        SN = {}
        if not use_coupled_noise:
            warnings.warn(
                "Computing the coupled noise from the uncoupled "
                "noise. This assumes the noise is white"
            )

        # Loop over the 4 different field combinations and fill the cl and
        # noise dictionaries
        for i in [13, 24, 14, 23]:
            i1, i2 = [int(j) for j in str(i)]
            key = f"cl{i}"
            if key in cache:
                cl[i] = cache[key]
            else:
                cl[i] = np.zeros((ncell[i], ell.size))
                cl[i][0] = ccl.angular_cl(
                    cosmo, ccl_tracers[tr[i1]], ccl_tracers[tr[i2]], ell
                )

            # Noise
            auto = tr[i1] == tr[i2]
            key = f"SN{i}"
            if key in cache:
                SN[i] = cache[key]
            else:
                SN[i] = np.zeros((ncell[i], ell.size))
                SN[i][0] = SN[i][-1] = np.ones_like(ell)
                if use_coupled_noise:
                    nl_cp = tracer_Noise_coupled[tr[i1]] if auto else 0
                    if nl_cp is None:
                        raise ValueError(
                            "Requested use_coupled_noise but "
                            "tracer_Noise_coupled is None for "
                            f"tracer {tr[i1]}. This could "
                            "mean that it does not have the "
                            "n_ell_coupled metadata information"
                            " in the sacc file."
                        )
                    SN[i] *= nl_cp
                else:
                    SN[i] *= tracer_Noise[tr[i1]] if auto else 0
                if s[i1] == 2:
                    SN[i][0, :2] = SN[i][-1, :2] = 0

        if (
            np.any(cl[13])
            or np.any(cl[24])
            or np.any(cl[14])
            or np.any(cl[23])
        ):

            # TODO: Modify depending on how TXPipe caches things
            # Mask, mask_names, field and workspaces dictionaries
            mn = self.get_mask_names_dict(tr)
            m = self.get_masks_dict(tr, cache)
            f = self.get_fields_dict(tr, cache, masks=m)
            w = self.get_workspaces_dict(tr, bins, cache, masks=m, fields=f)

            # TODO; Allow input options as output folder, if recompute, etc.
            if "cw" in cache:
                cw = cache["cw"]
            else:
                cw = self.get_covariance_workspace(
                    f[1],
                    f[2],
                    f[3],
                    f[4],
                    mn[1],
                    mn[2],
                    mn[3],
                    mn[4],
                    lmax=int(ell[-1]),
                    **self.nmt_conf["cw"],
                )

            cl_cov = {}
            cl_cov[13] = self.get_cl_for_cov(
                cl[13], SN[13], m[1], m[3], w[13], nl_is_cp=use_coupled_noise
            )
            cl_cov[23] = self.get_cl_for_cov(
                cl[23], SN[23], m[2], m[3], w[23], nl_is_cp=use_coupled_noise
            )
            cl_cov[14] = self.get_cl_for_cov(
                cl[14], SN[14], m[1], m[4], w[14], nl_is_cp=use_coupled_noise
            )
            cl_cov[24] = self.get_cl_for_cov(
                cl[24], SN[24], m[2], m[4], w[24], nl_is_cp=use_coupled_noise
            )

            cov = nmt.gaussian_covariance(
                cw,
                s[1],
                s[2],
                s[3],
                s[4],
                cl_cov[13],
                cl_cov[14],
                cl_cov[23],
                cl_cov[24],
                w[12],
                w[34],
                coupled,
            )
        else:
            size1 = ncell[12] * nbpw
            size2 = ncell[34] * nbpw
            cov = np.zeros((size1, size2))

        np.savez_compressed(fname, cov=cov)

        return cov

    def get_covariance_workspace(
        self, f1, f2, f3, f4, m1, m2, m3, m4, **kwargs
    ):
        """Return the covariance workspace of the fields f1, f2, f3, f4.

        Args:
            f1 (:obj:`pymaster.field.NmtField`):  Field 1
            f2 (:obj:`pymaster.field.NmtField`):  Field 2
            f3 (:obj:`pymaster.field.NmtField`):  Field 3
            f4 (:obj:`pymaster.field.NmtField`):  Field 4
            m1 (str): Mask name assotiated to the field 1
            m2 (str): Mask name assotiated to the field 2
            m3 (str): Mask name assotiated to the field 3
            m4 (str): Mask name assotiated to the field 4
            **kwargs:  Extra arguments to pass to
                pymaster.compute_coupling_coefficients. In addition, if
                recompute=True is passed, the cw will be recomputed even if
                found in the disk.

        Returns:
             :obj:`pymaster.covariance.NmtCovarianceWorkspace`: Covariance
             Workspace of the fields f1, f2, f3, f4
        """
        outdir = self.io.outdir
        spins = {
            m1: f1.fl.spin,
            m2: f2.fl.spin,
            m3: f3.fl.spin,
            m4: f4.fl.spin,
        }

        # Any other symmetry?
        combinations = [
            (m1, m2, m3, m4),
            (m2, m1, m3, m4),
            (m1, m2, m4, m3),
            (m2, m1, m4, m3),
            (m3, m4, m1, m2),
            (m4, m3, m1, m2),
            (m3, m4, m2, m1),
            (m4, m3, m2, m1),
        ]

        # Currently, outdir will be always not None. If not specified, it
        # will be the current directory. I leave this for now since we might
        # want to disable the option of writing files in the future for small
        # fast runs in the future.
        if outdir is not None:
            fnames = []
            isfiles = []
            for mn1, mn2, mn3, mn4 in combinations:
                s1, s2, s3, s4 = [spins[mi] for mi in [mn1, mn2, mn3, mn4]]
                f = f"cw{s1}{s2}{s3}{s4}__{mn1}__{mn2}__{mn3}__{mn4}.fits"
                f = os.path.join(outdir, f)
                if f not in fnames:
                    fnames.append(f)
                    isfiles.append(os.path.isfile(fnames[-1]))
        else:
            fnames = [None]
            isfiles = [None]

        cw = nmt.NmtCovarianceWorkspace()
        if "recompute" in kwargs:
            recompute = kwargs.pop("recompute")
        else:
            recompute = False
        if recompute or (not np.any(isfiles)):
            cw.compute_coupling_coefficients(f1, f2, f3, f4, **kwargs)
            # Recheck that the file has not been written by other proccess
            if fnames[0] and not os.path.isfile(fnames[0]):
                cw.write_to(fnames[0])
            for fn, isf in zip(fnames[1:], isfiles[1:]):
                # This will only be run if they don't exist or recompute = True
                # Recheck the file exist in case other process has removed it
                # in the mean time.
                if isf and os.path.isfile(fn):
                    # Remove old covariance workspace if you have recomputed it
                    os.remove(fn)
        else:
            ix = isfiles.index(True)
            cw.read_from(fnames[ix])

        return cw

    def get_fields_dict(self, tracer_names, cache=None, masks=None, **kwargs):
        """Return a dictionary with the fields assotiated to the given tracers.

        Args:
            tracer_names (dict):  Dictionary of the tracer names of the same
                form as mask_name. It has to be given as {1: name1, 2: name2,
                3: name3, 4: name4}, where 12 and 34 are the pair of tracers
                that go into the first and second Cell you are computing the
                covariance for; i.e. <Cell^12 Cell^34>.
            cache (dict): Dictionary with cached variables. It will use the
                cached field if found. The keys must be 'f1', 'f2', 'f3' or
                'f4' and the values the corresponding NmtFields.
            masks (dict): Dictionary of the masks of the fields correlated with
                keys 1, 2, 3 or 4 and values the loaded masks.
            **kwargs: Arguments to pass to NaMaster when computing the
                field. They will override the ones passed in the configuration
                file through nmt_conf['f'].

        Returns:
            dict: Dictionary with the masks assotiated to the fields to be
            correlated.
        """
        mask_names = self.get_mask_names_dict(tracer_names)
        if masks is None:
            masks = self.get_masks_dict(tracer_names, cache)
        if cache is None:
            cache = {}
        spins = self.get_tracers_spin_dict(tracer_names)
        nmt_conf = self.nmt_conf["f"].copy()
        nmt_conf.update(kwargs)
        f = {}
        f_by_mask_name = {}
        for i in [1, 2, 3, 4]:
            key = f"f{i}"
            if key in cache:
                f[i] = cache[key]
            else:
                # We add the spin to make sure we distinguish fields of
                # different types even though they share the same mask
                k = mask_names[i] + str(spins[i])
                if k not in f_by_mask_name:
                    f_by_mask_name[k] = nmt.NmtField(
                        masks[i], None, spin=spins[i], **nmt_conf
                    )
                f[i] = f_by_mask_name[k]

        return f

    def get_list_of_tracers_for_wsp(self):
        """Return the tracers needed to compute the independent workspaces.

        Returns:
            list of str: List of tracers needed to compute the independent
            workspaces.
        """
        sacc_file = self.io.get_sacc_file()
        tracers = sacc_file.get_tracer_combinations()

        fnames = []
        tracers_out = []
        for i, trs1 in enumerate(tracers):
            s1, s2 = self.get_tracer_comb_spin(trs1)
            mn1, mn2 = [self.mask_names[tri] for tri in trs1]

            for trs2 in tracers[i:]:
                s3, s4 = self.get_tracer_comb_spin(trs2)
                mn3, mn4 = [self.mask_names[tri] for tri in trs2]

                fname1 = f"w{s1}{s2}__{mn1}__{mn2}.fits"
                fname2 = f"w{s3}{s4}__{mn3}__{mn4}.fits"

                if (fname1 in fnames) or (fname2 in fnames):
                    continue

                fnames.append(fname1)
                fnames.append(fname2)

                tracers_out.append((trs1, trs2))

        return tracers_out

    def get_list_of_tracers_for_cov_wsp(self, remove_trs_wsp=False):
        """Return the tracers to compute the independent covariance workspaces.

        Args:
            remove_trs_wsp (bool, optional): If True, remove the tracer
                combinations from used to generate the workspaces independently
                (i.e the output of get_list_of_tracers_for_wsp). Defaults to
                False.

        Returns:
            list of str: List of tracers needed to compute the independent
            covariance workspaces.
        """
        sacc_file = self.io.get_sacc_file()
        tracers = sacc_file.get_tracer_combinations()

        fnames = []
        tracers_out = []
        for i, trs1 in enumerate(tracers):
            s1, s2 = self.get_tracer_comb_spin(trs1)
            mn1, mn2 = [self.mask_names[tri] for tri in trs1]
            for trs2 in tracers[i:]:
                s3, s4 = self.get_tracer_comb_spin(trs2)
                mn3, mn4 = [self.mask_names[tri] for tri in trs2]

                fname = f"cw{s1}{s2}{s3}{s4}__{mn1}__{mn2}__{mn3}__{mn4}.fits"
                if fname not in fnames:
                    fnames.append(fname)
                    tracers_out.append((trs1, trs2))

        if remove_trs_wsp:
            trs_wsp = self.get_list_of_tracers_for_wsp()
            for trs in trs_wsp:
                tracers_out.remove(trs)

        return tracers_out

    def get_list_of_tracers_for_cov_without_trs_wsp_cwsp(self):
        """Return the remaining covariance tracers combinations.

        It will remove the tracer combinations used to compute the workspaces
        and covariance workspaces.

        Returns:
            list of str: List of independent tracers combinations.
        """
        tracers_out = self.get_list_of_tracers_for_cov()

        trs_cwsp = self.get_list_of_tracers_for_cov_wsp()
        for trs in trs_cwsp:
            tracers_out.remove(trs)

        return tracers_out

    def get_nell(self, bins=None, nside=None, cache=None):
        """Return the number of ells for the fiducial Cells.

        If the sacc file stored bandpowers are wrong. You will need to pass one
        of the other arguments.

        Args:
            bins ( pymaster.NmtBin): NmtBin instance with the desired
                binning.
            nside (int): Healpy map nside.
            cache (dict): Dictionary with cached variables. It will use the
                cached workspaces to read the bandpower windows.

        Returns:
            int: Number of ells for the fidicual Cells points; i.e. lmax or
            3*nside
        """
        # Extracting the workspace from the cache first to use it later in a
        # easy way.
        if (cache is not None) and ("workspaces" in cache):
            w = list(list(cache["workspaces"].values())[0].values())[0]
        else:
            w = None

        if isinstance(w, nmt.NmtWorkspace):
            # We don't want to read the workspace just to get the nell
            bpw = w.get_bandpower_windows()
            nell = bpw.shape[-1]
        elif bins is not None:
            nell = bins.lmax + 1
        else:
            s = self.io.get_sacc_file()
            try:
                dtype = s.get_data_types()[0]
                tracers = s.get_tracer_combinations(data_type=dtype)[0]
                ix = s.indices(data_type=dtype, tracers=tracers)
                bpw = s.get_bandpower_windows(ix)
                if bpw is None:
                    raise ValueError
                nell = bpw.nell
            except ValueError as e:
                # If the window functions are wrong. Do magic
                warnings.warn(
                    "The window functions in the sacc file are wrong:"
                )
                warnings.warn(str(e))
                if "binning/ell_max" in s.metadata:
                    warnings.warn(
                        "Trying to circunvent this error: we will use"
                        "nell = lmax + 1 as given in the metadata"
                    )
                    nell = s.metadata["binning/ell_max"] + 1
                    if nside is not None and nell > 3 * nside:
                        warnings.warn(
                            "lmax is larger than 3*nside. We will use "
                            "nell = 3*nside"
                        )
                        nell = 3 * nside
                elif nside is not None:
                    warnings.warn(
                        "Trying to circunvent this error: we will try"
                        "with nell = 3*nside"
                    )
                    nell = 3 * nside
                else:
                    raise ValueError(
                        "nside, NmtBin or NmtWorkspace instances "
                        "must be passed"
                    )
        return nell

    def get_workspace(self, f1, f2, m1, m2, bins, **kwargs):
        """Return the workspace of the fields f1, f2.

        Args:
            f1 (:obj:`pymaster.field.NmtField`):  Field 1
            f2 (:obj:`pymaster.field.NmtField`):  Field 2
            m1 (str): Mask name assotiated to the field 1
            m2 (str): Mask name assotiated to the field 2
            bins (:obj:`pymaster.bins.NmtBin`):  NmtBin instance
            mask_names (dict): Dictionary with tracer names as key and maks
                names as values.
            **kwargs:  Extra arguments to pass to w.compute_coupling_matrix.
                In addition, if recompute=True is passed, the cw will be
                recomputed even if found in the disk.

        Returns:
             :obj:`pymaster.covariance.NmtCovarianceWorkspace`: Covariance
             Workspace of the fields f1, f2, f3, f4
        """
        if not isinstance(bins, nmt.NmtBin):
            raise ValueError(
                "You must pass a NmtBin instance through the "
                "cache or at initialization"
            )

        outdir = self.io.outdir
        s1, s2 = f1.fl.spin, f2.fl.spin

        # Currently, outdir will be always not None. If not specified, it
        # will be the current directory. I leave this for now since we might
        # want to disable the option of writing files in the future for small
        # fast runs in the future.
        if outdir is not None:
            fname = os.path.join(outdir, f"w{s1}{s2}__{m1}__{m2}.fits")
            isfile = os.path.isfile(fname)

            # The workspace of m1 x m2 and m2 x m1 is the same.
            fname2 = os.path.join(outdir, f"w{s2}{s1}__{m2}__{m1}.fits")
            isfile2 = os.path.isfile(fname2)
        else:
            fname = isfile = fname2 = isfile2 = None

        w = nmt.NmtWorkspace()
        if "recompute" in kwargs:
            recompute = kwargs.pop("recompute")
        else:
            recompute = False

        if recompute or ((not isfile) and (not isfile2)):
            w.compute_coupling_matrix(f1, f2, bins, **kwargs)
            # Recheck that the file has not been written by other proccess
            if fname and not os.path.isfile(fname):
                w.write_to(fname)
            # Check if the other files exist. Recheck in case other process has
            # removed it in the mean time.
            if isfile2 and os.path.isfile(fname2):
                # Remove the other to avoid later confusions.
                os.remove(fname2)
        elif isfile:
            w.read_from(fname)
        else:
            w.read_from(fname2)

        return w

    def get_workspaces_dict(
        self, tracer_names, bins, cache=None, fields=None, masks=None, **kwargs
    ):
        """Return a dictionary with the workspaces for the given tracers.

        Args:
            tracer_names (dict):  Dictionary of the masks names assotiated to
                the fields to be correlated. It has to be given as {1: name1,
                2: name2, 3: name3, 4: name4}, where 12 and 34 are the pair of
                tracers that go into the first and second Cell you are
                computing the covariance for; i.e. <Cell^12 Cell^34>.
            bins (:obj:`pymaster.bins.NmtBin`): NmtBin instance with the
            desired binning.
            cache (dict): Dictionary with cached variables. It will use the
                cached field if found. The keys must be 'w12', 'w34', 'w13',
                'w23', 'w14' or 'w24' and the values the corresponding
                NmtWorkspaces. Alternatively, you can pass a dictionary with
                keys as (mask_name1, mask_name2).
            field (dict): Dictionary of the NmtFields of the fields correlated
                with keys 1, 2, 3 or 4 and values the NmtFields.
                masks (dict): Dictionary of the masks of the fields correlated
                with keys 1, 2, 3 or 4 and values the loaded masks.
            **kwargs: Arguments to pass to NaMaster when computing the
                workspace. They will override the ones passed in the
                configuration file through nmt_conf['w'].

        Returns:
            dict:  Dictionary with the workspaces assotiated to the different
            field combinations needed for the covariance. Its keys are 13, 23,
            14, 24, 12, 34; with values the corresponding
            :obj:`pymaster.workspaces.NmtWorkspaces`.
        """
        mask_names = self.get_mask_names_dict(tracer_names)
        if masks is None:
            masks = self.get_masks_dict(tracer_names, cache)
        if fields is None:
            fields = self.get_fields_dict(tracer_names, cache, masks=masks)
        if cache is None:
            cache = {}

        nmt_conf = self.nmt_conf["w"].copy()
        nmt_conf.update(kwargs)

        w = {}
        w_by_mask_name = {}
        # 12 and 34 must be first to avoid asigning them None if their maks do
        # not overlap
        for i in [12, 34, 13, 23, 14, 24]:
            i1, i2 = [int(j) for j in str(i)]
            # Workspace
            key = f"w{i}"
            if key in cache:
                w[i] = cache[key]
            else:
                s1, s2 = fields[i1].fl.spin, fields[i2].fl.spin
                # In this case you have to check for m1 x m2 and m2 x m1
                k = (mask_names[i1], mask_names[i2])
                sk = "".join(sorted(f"{s1}{s2}"))

                # Look for the workspaces of appropriate spin combinations
                cache_wsp = cache.get("workspaces", None)
                if cache_wsp is not None:
                    if sk in cache_wsp:
                        cache_wsp = cache_wsp[sk]
                    elif sk[::-1] in cache_wsp:
                        cache_wsp = cache_wsp[sk[::-1]]

                if sk not in w_by_mask_name:
                    w_by_mask_name_s = {}
                    w_by_mask_name[sk] = w_by_mask_name_s
                else:
                    w_by_mask_name_s = w_by_mask_name[sk]

                if k in w_by_mask_name_s:
                    w[i] = w_by_mask_name_s[k]
                elif k[::-1] in w_by_mask_name_s:
                    w[i] = w_by_mask_name_s[k[::-1]]
                elif (i not in [12, 34]) and (
                    np.mean(masks[i1] * masks[i2]) == 0
                ):
                    # w13, w23, w14, w24 are needed to couple the theoretical
                    # Cell and are not needed if the masks do not overlap.
                    # However, w12 and w34 are needed for
                    # nmt.gaussian_covariance, which will complain if they are
                    # None
                    w_by_mask_name_s[k] = None
                    w[i] = w_by_mask_name_s[k]
                elif (cache_wsp is not None) and (
                    (k in cache_wsp) or k[::-1] in cache_wsp
                ):
                    if k in cache_wsp:
                        fname = cache_wsp[k]
                    else:
                        fname = cache_wsp[k[::-1]]
                    wsp = nmt.NmtWorkspace()
                    wsp.read_from(fname)
                    w[i] = w_by_mask_name_s[k] = wsp
                else:
                    w_by_mask_name_s[k] = self.get_workspace(
                        fields[i1],
                        fields[i2],
                        mask_names[i1],
                        mask_names[i2],
                        bins,
                        **nmt_conf,
                    )
                    w[i] = w_by_mask_name_s[k]

        return w
