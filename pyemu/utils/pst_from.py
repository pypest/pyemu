from __future__ import print_function, division
import os
from pathlib import Path
from datetime import datetime
import shutil
import inspect
import warnings
import platform
import numpy as np
import pandas as pd
import pyemu
import time
from ..pyemu_warnings import PyemuWarning
import copy
import string


# the tolerable percent difference (100 * (max - min)/mean)
# used when checking that constant and zone type parameters are in fact constant (within
# a given zone)
DIRECT_PAR_PERCENT_DIFF_TOL = 1.0


def _get_datetime_from_str(sdt):
    # could be expanded if someone is feeling clever.
    if isinstance(sdt, str):
        PyemuWarning(
            "Assuming passed reference start date time is "
            "year first str {0}".format(sdt)
        )
        sdt = pd.to_datetime(sdt, yearfirst=True)
    assert isinstance(sdt, pd.Timestamp), "Error interpreting start_datetime"
    return sdt


def _check_var_len(var, n, fill=None):
    if not isinstance(var, list):
        var = [var]
    if fill is not None:
        if fill == "first":
            fill = var[0]
        elif fill == "last":
            fill = var[-1]
    nv = len(var)
    if nv < n:
        var.extend([fill for _ in range(n - nv)])
    return var


class PstFrom(object):
    """construct high-dimensional PEST(++) interfaces with all the bells and whistles

    Args:
        original_d (`str`): the path to a complete set of model input and output files
        new_d (`str`): the path to where the model files and PEST interface files will be copied/built
        longnames (`bool`): flag to use longer-than-PEST-likes parameter and observation names.  Default is True
        remove_existing (`bool`): flag to destroy any existing files and folders in `new_d`.  Default is False
        spatial_reference (varies): an object that faciliates geo-locating model cells based on index.  Default is None
        zero_based (`bool`): flag if the model uses zero-based indices, Default is True
        start_datetime (`str`): a string that can be case to a datatime instance the represents the starting datetime
            of the model
        tpl_subfolder (`str`): option to write template files to a subfolder within ``new_d``.
            Default is False (write template files to ``new_d``).

        chunk_len (`int`): the size of each "chunk" of files to spawn a multiprocessing.Pool member to process.
            On windows, beware setting this much smaller than 50 because of the overhead associated with
            spawning the pool.  This value is added to the call to `apply_list_and_array_pars`. Default is 50

    Note:
        This is the way...

    Example::

        pf = PstFrom("path_to_model_files","new_dir_with_pest_stuff",start_datetime="1-1-2020")
        pf.add_parameters("hk.dat")
        pf.add_observations("heads.csv")
        pf.build_pst("pest.pst")
        pe = pf.draw(100)
        pe.to_csv("prior.csv")

    """

    def __init__(
        self,
        original_d,
        new_d,
        longnames=True,
        remove_existing=False,
        spatial_reference=None,
        zero_based=True,
        start_datetime=None,
        tpl_subfolder=None,
        chunk_len=50,
    ):

        self.original_d = Path(original_d)
        self.new_d = Path(new_d)
        self.original_file_d = None
        self.mult_file_d = None
        self.tpl_d = self.new_d
        if tpl_subfolder is not None:
            self.tpl_d = Path(self.new_d, tpl_subfolder)
        self.remove_existing = bool(remove_existing)
        self.zero_based = bool(zero_based)
        self._spatial_reference = spatial_reference
        self._spatial_ref_xarray = None
        self._spatial_ref_yarray = None
        self.spatial_reference = None
        if start_datetime is not None:
            start_datetime = _get_datetime_from_str(start_datetime)
        self.start_datetime = start_datetime
        self.geostruct = None
        self.par_struct_dict = {}
        # self.par_struct_dict_l = {}

        self.mult_files = []
        self.org_files = []

        self.par_dfs = []
        self.obs_dfs = []
        self.py_run_file = "forward_run.py"
        self.mod_command = "python {0}".format(self.py_run_file)
        self.pre_py_cmds = []
        self.pre_sys_cmds = []  # a list of preprocessing commands to add to
        # the forward_run.py script commands are executed with os.system()
        # within forward_run.py.
        self.mod_py_cmds = []
        self.mod_sys_cmds = []
        self.post_py_cmds = []
        self.post_sys_cmds = []  # a list of post-processing commands to add to
        # the forward_run.py script. Commands are executed with os.system()
        # within forward_run.py.
        self.extra_py_imports = []
        self.tmp_files = []

        self.tpl_filenames, self.input_filenames = [], []
        self.ins_filenames, self.output_filenames = [], []

        self.longnames = bool(longnames)
        self.logger = pyemu.Logger("PstFrom.log", echo=True)

        self.logger.statement("starting PstFrom process")

        self._prefix_count = {}

        self.get_xy = None
        self.add_pars_callcount = 0
        self.ijwarned = {}
        self.initialize_spatial_reference()

        self._setup_dirs()
        self._parfile_relations = []
        self._pp_facs = {}
        self.pst = None
        self._function_lines_list = []  # each function is itself a list of lines
        self.direct_org_files = []
        self.ult_ubound_fill = 1.0e30
        self.ult_lbound_fill = -1.0e30
        self.chunk_len = int(chunk_len)

    @property
    def parfile_relations(self):
        """build up a container of parameter file information.  Called
        programmatically..."""
        if isinstance(self._parfile_relations, list):
            pr = pd.concat(self._parfile_relations, ignore_index=True)
        else:
            pr = self._parfile_relations
        # quick checker
        for name, g in pr.groupby("model_file"):
            if g.sep.nunique() > 1:
                self.logger.warn(
                    "separator mismatch for {0}, seps passed {1}"
                    "".format(name, [s for s in g.sep.unique()])
                )
            if g.fmt.nunique() > 1:
                self.logger.warn(
                    "format mismatch for {0}, fmt passed {1}"
                    "".format(name, [f for f in g.fmt.unique()])
                )
            # if ultimate parameter bounds have been set for only one instance
            # of the model file we need to pass this through to all
            ubound = g.apply(
                lambda x: pd.Series(
                    {
                        k: v
                        for n, c in enumerate(x.use_cols)
                        for k, v in [["ubound{0}".format(c), x.upper_bound[n]]]
                    }
                )
                if x.use_cols is not None
                else pd.Series({k: v for k, v in [["ubound", x.upper_bound]]}),
                axis=1,
            )
            if ubound.nunique(0, False).gt(1).any():
                ub_min = ubound.min().fillna(self.ult_ubound_fill).to_dict()
                pr.loc[g.index, "upper_bound"] = g.use_cols.apply(
                    lambda x: [ub_min["ubound{0}".format(c)] for c in x]
                    if x is not None
                    else ub_min["ubound"]
                )
            # repeat for lower bounds
            lbound = g.apply(
                lambda x: pd.Series(
                    {
                        k: v
                        for n, c in enumerate(x.use_cols)
                        for k, v in [["lbound{0}".format(c), x.lower_bound[n]]]
                    }
                )
                if x.use_cols is not None
                else pd.Series({k: v for k, v in [["lbound", x.lower_bound]]}),
                axis=1,
            )
            if lbound.nunique(0, False).gt(1).any():
                lb_max = lbound.max().fillna(self.ult_lbound_fill).to_dict()
                pr.loc[g.index, "lower_bound"] = g.use_cols.apply(
                    lambda x: [lb_max["lbound{0}".format(c)] for c in x]
                    if x is not None
                    else lb_max["lbound"]
                )
        pr["zero_based"] = self.zero_based
        return pr

    def _generic_get_xy(self, args, **kwargs):
        i, j = self.parse_kij_args(args, kwargs)
        return i, j

    def _dict_get_xy(self, arg, **kwargs):
        if isinstance(arg, list):
            arg = tuple(arg)
        xy = self._spatial_reference.get(arg, None)
        if xy is None:
            arg_len = None
            try:
                arg_len = len(arg)
            except Exception as e:
                self.logger.lraise(
                    "Pstfrom._dict_get_xy() error getting xy from arg:'{0}' - no len support".format(
                        arg
                    )
                )
            if arg_len == 1:
                xy = self._spatial_reference.get(arg[0], None)
            elif arg_len == 2 and arg[0] == 0:
                xy = self._spatial_reference.get(arg[1], None)
            elif arg_len == 2 and arg[1] == 0:
                xy = self._spatial_reference.get(arg[0], None)
            else:
                self.logger.lraise(
                    "Pstfrom._dict_get_xy() error getting xy from arg:'{0}'".format(arg)
                )
        if xy is None:
            self.logger.lraise(
                "Pstfrom._dict_get_xy() error getting xy from arg:'{0}' - still None".format(
                    arg
                )
            )
        return xy[0], xy[1]

    def _flopy_sr_get_xy(self, args, **kwargs):
        i, j = self.parse_kij_args(args, kwargs)
        if all([ij is None for ij in [i, j]]):
            return i, j
        else:
            return (
                self._spatial_reference.xcentergrid[i, j],
                self._spatial_reference.ycentergrid[i, j],
            )

    def _flopy_mg_get_xy(self, args, **kwargs):
        i, j = self.parse_kij_args(args, kwargs)
        if all([ij is None for ij in [i, j]]):
            return i, j
        else:
            if self._spatial_ref_xarray is None:
                self._spatial_ref_xarray = self._spatial_reference.xcellcenters
                self._spatial_ref_yarray = self._spatial_reference.ycellcenters

            return (self._spatial_ref_xarray[i, j], self._spatial_ref_yarray[i, j])

    def parse_kij_args(self, args, kwargs):
        """parse args into kij indices.  Called programmatically"""
        if len(args) >= 2:
            ij_id = None
            if "ij_id" in kwargs:
                ij_id = kwargs["ij_id"]
            if ij_id is not None:
                i, j = [args[ij] for ij in ij_id]
            else:
                if not self.ijwarned[self.add_pars_callcount]:
                    self.logger.warn(
                        (
                            "get_xy() warning: position of i and j in index_cols "
                            "not specified, assume (i,j) are final two entries in "
                            "index_cols."
                        )
                    )
                    self.ijwarned[self.add_pars_callcount] = True
                # assume i and j are the final two entries in index_cols
                i, j = args[-2], args[-1]
        else:
            if not self.ijwarned[self.add_pars_callcount]:
                self.logger.warn(
                    (
                        "get_xy() warning: need locational information "
                        "(e.g. i,j) to generate xy, "
                        "insufficient index cols passed to interpret: {}"
                        ""
                    ).format(str(args))
                )
                self.ijwarned[self.add_pars_callcount] = True
            i, j = None, None
        return i, j

    def initialize_spatial_reference(self):
        """process the spatial reference argument.  Called programmatically"""
        if self._spatial_reference is None:
            self.get_xy = self._generic_get_xy
        elif hasattr(self._spatial_reference, "xcentergrid") and hasattr(
            self._spatial_reference, "ycentergrid"
        ):
            self.get_xy = self._flopy_sr_get_xy
        elif hasattr(self._spatial_reference, "xcellcenters") and hasattr(
            self._spatial_reference, "ycellcenters"
        ):
            # support modelgrid style cell locs
            self._spatial_reference.xcentergrid = self._spatial_reference.xcellcenters
            self._spatial_reference.ycentergrid = self._spatial_reference.ycellcenters
            self.get_xy = self._flopy_mg_get_xy
        elif isinstance(self._spatial_reference, dict):
            self.logger.statement("dictionary-based spatial reference detected...")
            self.get_xy = self._dict_get_xy
        else:
            self.logger.lraise(
                "initialize_spatial_reference() error: " "unsupported spatial_reference"
            )
        self.spatial_reference = self._spatial_reference

    def write_forward_run(self):
        """write the forward run script.  Called by build_pst()"""
        # update python commands with system style commands
        for alist, ilist in zip(
            [self.pre_py_cmds, self.mod_py_cmds, self.post_py_cmds],
            [self.pre_sys_cmds, self.mod_sys_cmds, self.post_sys_cmds],
        ):
            if ilist is None:
                continue

            if not isinstance(ilist, list):
                ilist = [ilist]
            for cmd in ilist:
                new_sys_cmd = "pyemu.os_utils.run(r'{0}')\n".format(cmd)
                if new_sys_cmd in alist:
                    self.logger.warn(
                        "sys_cmd '{0}' already in sys cmds, skipping...".format(
                            new_sys_cmd
                        )
                    )
                else:
                    self.logger.statement("forward_run line:{0}".format(new_sys_cmd))
                    alist.append(new_sys_cmd)

        with open(self.new_d / self.py_run_file, "w") as f:
            f.write(
                "import os\nimport multiprocessing as mp\nimport numpy as np"
                + "\nimport pandas as pd\n"
            )
            f.write("import pyemu\n")
            for ex_imp in self.extra_py_imports:
                f.write("import {0}\n".format(ex_imp))

            for func_lines in self._function_lines_list:
                f.write("\n")
                f.write("# function added thru PstFrom.add_py_function()\n")
                for func_line in func_lines:
                    f.write(func_line)
                f.write("\n")
            f.write("def main():\n")
            f.write("\n")
            s = "    "
            for tmp_file in self.tmp_files:
                f.write(s + "try:\n")
                f.write(s + "   os.remove(r'{0}')\n".format(tmp_file))
                f.write(s + "except Exception as e:\n")
                f.write(
                    s + "   print(r'error removing tmp file:{0}')\n".format(tmp_file)
                )
            for line in self.pre_py_cmds:
                f.write(s + line + "\n")
            for line in self.mod_py_cmds:
                f.write(s + line + "\n")
            for line in self.post_py_cmds:
                f.write(s + line + "\n")
            f.write("\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    mp.freeze_support()\n    main()\n\n")

    def _pivot_par_struct_dict(self):
        struct_dict = {}
        for gs, gps in self.par_struct_dict.items():
            par_dfs = []
            for _, l in gps.items():
                df = pd.concat(l)
                if "timedelta" in df.columns:
                    df.loc[:, "y"] = 0  #
                    df.loc[:, "x"] = df.timedelta.apply(lambda x: x.days)
                par_dfs.append(df)
            struct_dict[gs] = par_dfs
        return struct_dict

    def build_prior(
        self, fmt="ascii", filename=None, droptol=None, chunk=None, sigma_range=6
    ):
        """Build the prior parameter covariance matrix

        Args:
            fmt (`str`): the file format to save to.  Default is "ASCII", can be "binary", "coo", or "none"
            filename (`str`): the filename to save the cov to
            droptol (`float`): absolute value of prior cov entries that are smaller than `droptol` are treated as
                zero.
            chunk (`int`): number of entries to write to binary/coo at once.  Default is None (write all elements at once
            sigma_range (`int`): number of standard deviations represented by parameter bounds.  Default is 6 (99%
                confidence).  4 would be approximately 95% confidence bounds

        Returns:
            `pyemu.Cov`: the prior parameter covariance matrix

        Note:
            This method processes parameters by group names

            For really large numbers of parameters (>30K), this method
            will cause memory errors.  Luckily, in most cases, users
            only want this matrix to generate a prior parameter ensemble
            and the `PstFrom.draw()` is a better choice...

        """
        struct_dict = self._pivot_par_struct_dict()
        self.logger.log("building prior covariance matrix")
        if len(struct_dict) > 0:
            cov = pyemu.helpers.geostatistical_prior_builder(
                self.pst, struct_dict=struct_dict, sigma_range=sigma_range
            )
        else:
            cov = pyemu.Cov.from_parameter_data(self.pst, sigma_range=sigma_range)

        if filename is None:
            filename = self.pst.filename.with_suffix(".prior.cov")
        if fmt != "none":
            self.logger.statement(
                "saving prior covariance matrix to file {0}".format(filename)
            )
        if fmt == "ascii":
            cov.to_ascii(filename)
        elif fmt == "binary":
            cov.to_binary(filename, droptol=droptol, chunk=chunk)
        elif fmt == "uncfile":
            cov.to_uncfile(filename)
        elif fmt == "coo":
            cov.to_coo(filename, droptol=droptol, chunk=chunk)
        self.logger.log("building prior covariance matrix")
        return cov

    def draw(self, num_reals=100, sigma_range=6, use_specsim=False, scale_offset=True):
        """Draw a parameter ensemble from the distribution implied by the initial parameter values in the
        control file and the prior parameter covariance matrix.

        Args:
            num_reals (`int`): the number of realizations to draw
            sigma_range (`int`): number of standard deviations represented by parameter bounds.  Default is 6 (99%
                confidence).  4 would be approximately 95% confidence bounds
            use_specsim (`bool`): flag to use spectral simulation for grid-scale pars (highly recommended).
                Default is False
            scale_offset (`bool`): flag to apply scale and offset to parameter bounds before calculating prior variance.
                Dfault is True.  If you are using non-default scale and/or offset and you get an exception during
                draw, try changing this value to False.

        Returns:
            `pyemu.ParameterEnsemble`: a prior parameter ensemble

        Note:
            This method draws by parameter group

            If you are using grid-style parameters, please use spectral simulation (`use_specsim=True`)

        """
        self.logger.log("drawing realizations")
        if self.pst.npar_adj == 0:
            self.logger.warn("no adjustable parameters, nothing to draw...")
            return
        # precondition {geostruct:{group:df}} dict to {geostruct:[par_dfs]}
        struct_dict = self._pivot_par_struct_dict()
        # list for holding grid style groups
        gr_pe_l = []
        if use_specsim:
            if not pyemu.geostats.SpecSim2d.grid_is_regular(
                self.spatial_reference.delr, self.spatial_reference.delc
            ):
                self.logger.lraise(
                    "draw() error: can't use spectral simulation with irregular grid"
                )
            self.logger.log("spectral simulation for grid-scale pars")
            # loop over geostructures defined in PestFrom object
            # (setup through add_parameters)
            for geostruct, par_df_l in struct_dict.items():
                par_df = pd.concat(par_df_l)  # force to single df
                if (
                    "i" in par_df.columns and par_df.partype[0] == "grid"
                ):  # need 'i' and 'j' for specsim
                    # grid par slicer
                    grd_p = pd.notna(par_df.i)  # & (par_df.partype == 'grid') &
                else:
                    grd_p = np.array([0])
                # if there are grid pars (also grid pars with i,j info)
                if grd_p.sum() > 0:
                    # select pars to use specsim for
                    gr_df = par_df.loc[grd_p]
                    gr_df = gr_df.astype({"i": int, "j": int})  # make sure int
                    # (won't be if there were nans in concatenated df)
                    if len(gr_df) > 0:
                        # get specsim object for this geostruct
                        ss = pyemu.geostats.SpecSim2d(
                            delx=self.spatial_reference.delr,
                            dely=self.spatial_reference.delc,
                            geostruct=geostruct,
                        )
                        # specsim draw (returns df)
                        gr_pe1 = ss.grid_par_ensemble_helper(
                            pst=self.pst,
                            gr_df=gr_df,
                            num_reals=num_reals,
                            sigma_range=sigma_range,
                            logger=self.logger,
                        )
                        # append to list of specsim drawn pars
                        gr_pe_l.append(gr_pe1)
                        # rebuild struct_dict entry for this geostruct
                        # to not include specsim pars
                        struct_dict[geostruct] = []
                        # loop over all in list associated with geostruct
                        for p_df in par_df_l:
                            # if pars are not in the specsim pars just created
                            # assign them to this struct_dict entry
                            # needed if none specsim pars are linked to same geostruct
                            if not p_df.index.isin(gr_df.index).all():
                                struct_dict[geostruct].append(p_df)
            self.logger.log("spectral simulation for grid-scale pars")
        # draw remaining pars based on their geostruct
        self.logger.log("Drawing non-specsim pars")
        pe = pyemu.helpers.geostatistical_draws(
            self.pst,
            struct_dict=struct_dict,
            num_reals=num_reals,
            sigma_range=sigma_range,
            scale_offset=scale_offset,
        )
        self.logger.log("Drawing non-specsim pars")
        if len(gr_pe_l) > 0:
            gr_par_pe = pd.concat(gr_pe_l, axis=1)
            pe.loc[:, gr_par_pe.columns] = gr_par_pe.values
        # par_ens = pyemu.ParameterEnsemble(pst=self.pst, df=pe)
        self.logger.log("drawing realizations")
        return pe

    def build_pst(self, filename=None, update=False, version=1):
        """Build control file from i/o files in PstFrom object.
        Warning: This builds a pest control file from scratch, overwriting
        anything already in self.pst object and anything already writen to `filename`

        Args:
            filename (`str`): the filename to save the control file to.
                If None, the name is formed from the `PstFrom.original_d`
                ,the orginal directory name from which the forward model
                was extracted.  Default is None.
                The control file is saved in the `PstFrom.new_d` directory.
            update (`bool`) or (str): flag to add to existing Pst object and
                rewrite. If string {'pars', 'obs'} just update respective
                components of Pst. Default is False - build from PstFrom
                components.
            version (`int`): control file version to write, Default is 1
        Note:
            This builds a pest control file from scratch, overwriting anything already
            in self.pst object and anything already writen to `filename`

            The new pest control file is assigned an NOPTMAX value of 0

        """

        par_data_cols = pyemu.pst_utils.pst_config["par_fieldnames"]
        obs_data_cols = pyemu.pst_utils.pst_config["obs_fieldnames"]
        if update:
            if self.pst is None:
                self.logger.warn(
                    "Can't update Pst object not initialised. "
                    "Setting update to False"
                )
                update = False
            else:
                if filename is None:
                    filename = get_filepath(self.new_d, self.pst.filename)
        else:
            if filename is None:
                filename = Path(self.new_d, self.original_d.name).with_suffix(".pst")
        filename = get_filepath(self.new_d, filename)

        # if os.path.dirname(filename) in ["", "."]:
        #    filename = os.path.join(self.new_d, filename)

        if update:
            pst = self.pst
            if update is True:
                update = {"pars": False, "obs": False}
            elif isinstance(update, str):
                update = {update: True}
            elif isinstance(update, (set, list)):
                update = {s: True for s in update}
            uupdate = True
        else:
            update = {"pars": False, "obs": False}
            uupdate = False
            pst = pyemu.Pst(filename, load=False)

        if "pars" in update.keys() or not uupdate:
            if len(self.par_dfs) > 0:
                # parameter data from object
                par_data = pd.concat(self.par_dfs).loc[:, par_data_cols]
                # info relating parameter multiplier files to model input files
                parfile_relations = self.parfile_relations
                parfile_relations.to_csv(self.new_d / "mult2model_info.csv")
                if not any(
                    ["apply_list_and_array_pars" in s for s in self.pre_py_cmds]
                ):
                    self.pre_py_cmds.insert(
                        0,
                        "pyemu.helpers.apply_list_and_array_pars("
                        "arr_par_file='mult2model_info.csv',chunk_len={0})".format(
                            self.chunk_len
                        ),
                    )
            else:
                par_data = pyemu.pst_utils._populate_dataframe(
                    [], pst.par_fieldnames, pst.par_defaults, pst.par_dtype
                )
            pst.parameter_data = par_data
            # pst.template_files = self.tpl_filenames
            # pst.input_files = self.input_filenames
            pst.model_input_data = pd.DataFrame(
                {"pest_file": self.tpl_filenames, "model_file": self.input_filenames},
                index=self.tpl_filenames,
            )

        if "obs" in update.keys() or not uupdate:
            if len(self.obs_dfs) > 0:
                obs_data = pd.concat(self.obs_dfs).loc[:, obs_data_cols]
            else:
                obs_data = pyemu.pst_utils._populate_dataframe(
                    [], pst.obs_fieldnames, pst.obs_defaults, pst.obs_dtype
                )
                obs_data.loc[:, "obsnme"] = []
                obs_data.index = []
            obs_data.sort_index(inplace=True)
            pst.observation_data = obs_data
            # pst.instruction_files = self.ins_filenames
            # pst.output_files = self.output_filenames
            pst.model_output_data = pd.DataFrame(
                {"pest_file": self.ins_filenames, "model_file": self.output_filenames},
                index=self.ins_filenames,
            )
        if not uupdate:
            pst.model_command = self.mod_command

        pst.prior_information = pst.null_prior
        pst.control_data.noptmax = 0
        self.pst = pst

        self.pst.write(filename, version=version)
        self.write_forward_run()
        pst.try_parse_name_metadata()
        return pst

    def _setup_dirs(self):
        self.logger.log("setting up dirs")
        if not os.path.exists(self.original_d):
            self.logger.lraise("original_d '{0}' not found" "".format(self.original_d))
        if not os.path.isdir(self.original_d):
            self.logger.lraise(
                "original_d '{0}' is not a directory" "".format(self.original_d)
            )
        if self.new_d.exists():
            if self.remove_existing:
                self.logger.log("removing existing new_d '{0}'" "".format(self.new_d))
                shutil.rmtree(self.new_d)
                time.sleep(0.0001)
                self.logger.log("removing existing new_d '{0}'" "".format(self.new_d))
            else:
                self.logger.lraise(
                    "new_d '{0}' already exists "
                    "- use remove_existing=True"
                    "".format(self.new_d)
                )

        self.logger.log(
            "copying original_d '{0}' to new_d '{1}'"
            "".format(self.original_d, self.new_d)
        )
        shutil.copytree(self.original_d, self.new_d)
        self.logger.log(
            "copying original_d '{0}' to new_d '{1}'"
            "".format(self.original_d, self.new_d)
        )

        self.original_file_d = self.new_d / "org"
        if self.original_file_d.exists():
            self.logger.lraise(
                "'org' subdir already exists in new_d '{0}'" "".format(self.new_d)
            )
        self.original_file_d.mkdir(exist_ok=True)

        self.mult_file_d = self.new_d / "mult"
        if self.mult_file_d.exists():
            self.logger.lraise(
                "'mult' subdir already exists in new_d '{0}'" "".format(self.new_d)
            )
        self.mult_file_d.mkdir(exist_ok=True)

        self.tpl_d.mkdir(exist_ok=True)

        self.logger.log("setting up dirs")

    def _par_prep(
        self,
        filenames,
        index_cols,
        use_cols,
        fmts=None,
        seps=None,
        skip_rows=None,
        c_char=None,
    ):

        # todo: cast str column names, index_cols and use_cols to lower if str?
        # todo: check that all index_cols and use_cols are the same type
        file_dict = {}
        fmt_dict = {}
        sep_dict = {}
        skip_dict = {}
        (
            filenames,
            fmts,
            seps,
            skip_rows,
            index_cols,
            use_cols,
        ) = self._prep_arg_list_lengths(
            filenames, fmts, seps, skip_rows, index_cols, use_cols
        )
        storehead = None
        if index_cols is not None:
            for filename, sep, fmt, skip in zip(filenames, seps, fmts, skip_rows):
                # cast to pathlib.Path instance
                # input file path may or may not include original_d
                # input_filepath = get_filepath(self.original_d, filename)
                rel_filepath = get_relative_filepath(self.original_d, filename)
                dest_filepath = self.new_d / rel_filepath

                # data file in dest_ws/org/ folder
                org_file = self.original_file_d / rel_filepath.name

                self.logger.log("loading list {0}".format(dest_filepath))
                df, storehead, _ = self._load_listtype_file(
                    rel_filepath, index_cols, use_cols, fmt, sep, skip, c_char
                )
                # Currently just passing through comments in header (i.e. before the table data)
                stkeys = np.array(
                    sorted(storehead.keys())
                )  # comments line numbers as sorted array
                if (
                    stkeys.size > 0 and stkeys.min() == 0
                ):  # TODO pass comment_char through to par_file_rel so mid-table comments can be preserved
                    skip = 1 + np.sum(np.diff(stkeys) == 1)
                # # looping over model input filenames
                if fmt.lower() == "free":
                    if sep is None:
                        sep = " "
                        if rel_filepath.suffix.lower() == ".csv":
                            sep = ","
                if df.columns.is_integer():
                    hheader = False
                else:
                    hheader = df.columns

                self.logger.statement(
                    "loaded list '{0}' of shape {1}" "".format(dest_filepath, df.shape)
                )
                # TODO BH: do we need to be careful of the format of the model
                #  files? -- probs not necessary for the version in
                #  original_file_d - but for the eventual product model file,
                #  it might be format sensitive - yuck
                # Update, BH: I think the `original files` saved can always
                # be comma delim --they are never directly used
                # as model inputs-- as long as we pass the required model
                # input file format (and sep), right?
                # write orig version of input file to `org` (e.g.) dir

                # make any subfolders if they don't exist
                # org_path = Path(self.original_file_d, rel_file_path.parent)
                # org_path.mkdir(exist_ok=True)

                if len(storehead) != 0:
                    kwargs = {}
                    if "win" in platform.platform().lower():
                        kwargs = {"line_terminator": "\n"}
                    with open(org_file, "w") as fp:
                        lc = 0
                        fr = 0
                        for key in sorted(storehead.keys()):
                            if key > lc:
                                self.logger.warn(
                                    "Detected mid-table comment "
                                    "on line {0} tabular model file, "
                                    "comment will be lost"
                                    "".format(key + 1)
                                )
                                lc += 1
                                continue
                                # TODO if we want to preserve mid-table comments,
                                #  these lines might help - will also need to
                                #  pass comment_char through so it can be
                                #  used by the apply methods
                                # to = key - lc
                                # df.iloc[fr:to].to_csv(
                                #     fp, sep=',', mode='a', header=hheader, # todo - presence of header may cause an issue with this
                                #     **kwargs)
                                # lc += to - fr
                                # fr = to
                            fp.write(storehead[key])
                            fp.flush()
                            lc += 1
                        if lc < len(df):
                            df.iloc[fr:].to_csv(
                                fp,
                                sep=",",
                                mode="a",
                                header=hheader,
                                index=False,
                                **kwargs,
                            )
                else:
                    df.to_csv(
                        org_file,
                        index=False,
                        sep=",",
                        header=hheader,
                    )
                file_dict[rel_filepath] = df
                fmt_dict[rel_filepath] = fmt
                sep_dict[rel_filepath] = sep
                skip_dict[rel_filepath] = skip
                self.logger.log("loading list {0}".format(dest_filepath))

            # check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i + 1, len(fnames)):
                    if file_dict[fnames[i]].shape[1] != file_dict[fnames[j]].shape[1]:
                        self.logger.lraise(
                            "shape mismatch for array types, '{0}' "
                            "shape {1} != '{2}' shape {3}".format(
                                fnames[i],
                                file_dict[fnames[i]].shape[1],
                                fnames[j],
                                file_dict[fnames[j]].shape[1],
                            )
                        )
        else:  # load array type files
            # loop over model input files
            for input_filena, sep, fmt, skip in zip(filenames, seps, fmts, skip_rows):
                # cast to pathlib.Path instance
                # input file path may or may not include original_d
                input_filena = get_filepath(self.original_d, input_filena)
                if fmt.lower() == "free":
                    # cast to string to work with pathlib objects
                    if input_filena.suffix.lower() == ".csv":
                        if sep is None:
                            sep = ","
                else:
                    # TODO - or not?
                    raise NotImplementedError(
                        "Only free format array " "par files currently supported"
                    )
                # file path relative to model workspace
                rel_filepath = input_filena.relative_to(self.original_d)
                dest_filepath = self.new_d / rel_filepath
                self.logger.log("loading array {0}".format(dest_filepath))
                if not dest_filepath.exists():
                    self.logger.lraise(
                        "par filename '{0}' not found ".format(dest_filepath)
                    )
                # read array type input file
                arr = np.loadtxt(dest_filepath, delimiter=sep, ndmin=2)
                self.logger.log("loading array {0}".format(dest_filepath))
                self.logger.statement(
                    "loaded array '{0}' of shape {1}".format(input_filena, arr.shape)
                )
                # save copy of input file to `org` dir
                # make any subfolders if they don't exist
                np.savetxt(self.original_file_d / rel_filepath.name, arr)
                file_dict[rel_filepath] = arr
                fmt_dict[rel_filepath] = fmt
                sep_dict[rel_filepath] = sep
                skip_dict[rel_filepath] = skip
            # check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i + 1, len(fnames)):
                    if file_dict[fnames[i]].shape != file_dict[fnames[j]].shape:
                        self.logger.lraise(
                            "shape mismatch for array types, '{0}' "
                            "shape {1} != '{2}' shape {3}"
                            "".format(
                                fnames[i],
                                file_dict[fnames[i]].shape,
                                fnames[j],
                                file_dict[fnames[j]].shape,
                            )
                        )
        return (
            index_cols,
            use_cols,
            file_dict,
            fmt_dict,
            sep_dict,
            skip_dict,
            storehead,
        )

    def _next_count(self, prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        else:
            self._prefix_count[prefix] += 1

        return self._prefix_count[prefix]

    def add_py_function(
        self, file_name, call_str=None, is_pre_cmd=True, function_name=None
    ):
        """add a python function to the forward run script

        Args:
            file_name (`str`): a python source file
            call_str (`str`): the call string for python function in
                `file_name`.
                `call_str` will be added to the forward run script, as is.
            is_pre_cmd (`bool` or `None`): flag to include `call_str` in
                PstFrom.pre_py_cmds.  If False, `call_str` is
                added to PstFrom.post_py_cmds instead. If passed as `None`,
                then the function `call_str` is added to the forward run
                script but is not called.  Default is True.
            function_name (`str`): DEPRECATED, used `call_str`
        Returns:
            None

        Note:
            `call_str` is expected to reference standalone a function
            that contains all the imports it needs or these imports
            should have been added to the forward run script through the
            `PstFrom.py_imports` list.

            This function adds the `call_str` call to the forward
            run script (either as a pre or post command). It is up to users
            to make sure `call_str` is a valid python function call
            that includes the parentheses and requisite arguments

            This function expects "def " + `function_name` to be flushed left
            at the outer most indentation level

        Example::

            pf = PstFrom()
            # add the function "mult_well_function" from the script file "preprocess.py" as a
            # command to run before the model is run
            pf.add_py_function("preprocess.py",
                               "mult_well_function(arg1='userarg')",
                               is_pre_cmd = True)
            # add the post processor function "made_it_good" from the script file "post_processors.py"
            pf.add_py_function("post_processors.py","make_it_good(()",is_pre_cmd=False)


        """
        if function_name is not None:
            warnings.warn(
                "add_py_function(): 'function_name' argument is deprecated, "
                "use 'call_str' instead",
                DeprecationWarning,
            )
            if call_str is None:
                call_str = function_name
        if call_str is None:
            self.logger.lraise(
                "add_py_function(): No function call string passed in arg " "'call_str'"
            )
        if not os.path.exists(file_name):
            self.logger.lraise(
                "add_py_function(): couldnt find python source file '{0}'".format(
                    file_name
                )
            )
        if "(" not in call_str or ")" not in call_str:
            self.logger.lraise(
                "add_py_function(): call_str '{0}' missing paretheses".format(call_str)
            )
        function_name = call_str[
            : call_str.find("(")
        ]  # strip to first occurance of '('
        func_lines = []
        search_str = "def " + function_name + "("
        abet_set = set(string.ascii_uppercase)
        abet_set.update(set(string.ascii_lowercase))
        with open(file_name, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    self.logger.lraise(
                        "add_py_function(): EOF while searching for function '{0}'".format(
                            search_str
                        )
                    )
                if line.startswith(
                    search_str
                ):  # case sens and no strip since 'def' should be flushed left
                    func_lines.append(line)
                    while True:
                        line = f.readline()
                        if line == "":
                            break
                        if line[0] in abet_set:
                            break
                        func_lines.append(line)
                    break

        self._function_lines_list.append(func_lines)
        if is_pre_cmd is True:
            self.pre_py_cmds.append(call_str)
        elif is_pre_cmd is False:
            self.post_py_cmds.append(call_str)
        else:
            self.logger.warn(
                "add_py_function() command: {0} is not being called directly".format(
                    call_str
                )
            )

    def _process_array_obs(
        self,
        out_filename,
        ins_filename,
        prefix,
        ofile_sep,
        ofile_skip,
        longnames,
        zone_array,
    ):
        """private method to setup observations for an array-style file

        Args:
            out_filename (`str`): the output array file
            ins_filename (`str`): the instruction file to create
            prefix (`str`): the prefix to add to the observation names and also to use as the
                observation group name.
            ofile_sep (`str`): the separator in the output file.  This is currently just a
                placeholder arg, only whitespace-delimited files are supported
            ofile_skip (`int`): number of header and/or comment lines to skip at the
                top of the output file
            longnames (`bool`): flag to allow longer observations names
            zone_array (numpy.ndarray): an integer array used to identify positions to skip in the
                output array

        Returns:
            None

        Note:
            This method is called programmatically by `PstFrom.add_observations()`

        """
        if ofile_sep is not None:
            self.logger.lrase(
                "array obs are currently only supported for whitespace delim"
            )
        if not os.path.exists(self.new_d / out_filename):
            self.logger.lraise(
                "array obs output file '{0}' not found".format(out_filename)
            )
        if len(prefix) == 0 and self.longnames:
            prefix = Path(out_filename).stem
        f_out = open(self.new_d / out_filename, "r")
        f_ins = open(self.new_d / ins_filename, "w")
        f_ins.write("pif ~\n")
        iline = 0
        if ofile_skip is not None:
            ofile_skip = int(ofile_skip)
            f_ins.write("l{0}\n".format(ofile_skip))
            # read the output file forward
            for _ in range(ofile_skip):
                f_out.readline()
                iline += 1
        onames, ovals = [], []
        iidx = 0
        for line in f_out:
            raw = line.split()
            f_ins.write("l1 ")
            for jr, r in enumerate(raw):

                try:
                    fr = float(r)
                except Exception as e:
                    self.logger.lraise(
                        "array obs error casting: '{0}' on line {1} to a float: {2}".format(
                            r, iline, str(e)
                        )
                    )

                zval = None
                if zone_array is not None:
                    try:
                        zval = zone_array[iidx, jr]
                    except Exception as e:
                        self.logger.lraise(
                            "array obs error getting zone value for i,j {0},{1} in line {2}: {3}".format(
                                iidx, jr, iline, str(e)
                            )
                        )
                    if zval <= 0:
                        f_ins.write(" !dum! ")
                        if jr < len(raw) - 1:
                            f_ins.write(" w ")
                        continue

                if longnames:
                    oname = "arrobs_{0}_i:{1}_j:{2}".format(prefix, iidx, jr)
                    if zval is not None:
                        oname += "_zone:{0}".format(zval)
                else:
                    oname = "{0}_{1}_{2}".format(prefix, iidx, jr)
                    if zval is not None:
                        z_str = "_{0}".format(zval)
                        if len(oname) + len(z_str) < 20:
                            oname += z_str
                    if len(oname) > 20:
                        self.logger.lraise(
                            "array obs name too long: '{0}'".format(oname)
                        )
                f_ins.write(" !{0}! ".format(oname))
                if jr < len(raw) - 1:
                    f_ins.write(" w ")
            f_ins.write("\n")
            iline += 1
            iidx += 1

    def add_observations(
        self,
        filename,
        insfile=None,
        index_cols=None,
        use_cols=None,
        use_rows=None,
        prefix="",
        ofile_skip=None,
        ofile_sep=None,
        rebuild_pst=False,
        obsgp=None,
        zone_array=None,
        includes_header=True,
    ):
        """
        Add values in output files as observations to PstFrom object

        Args:
            filename (`str`): model output file name(s) to set up
                as observations. By default filename should give relative
                loction from top level of pest template directory
                (`new_d` as passed to `PstFrom()`).
            insfile (`str`): desired instructions file filename
            index_cols (`list`-like or `int`): columns to denote are indices for obs
            use_cols (`list`-like or `int`): columns to set up as obs. If None,
                and `index_cols` is not None (i.e list-syle obs assumed),
                observations will be set up for all columns in `filename` that
                are not in `index_cols`.
            use_rows (`list`-like or `int`): select only specific row of file for obs
            prefix (`str`): prefix for obsnmes
            ofile_skip (`int`): number of lines to skip in model output file
            ofile_sep (`str`): delimiter in output file.
                If `None`, the delimiter is eventually governed by the file
                extension (`,` for .csv).
            rebuild_pst (`bool`): (Re)Construct PstFrom.pst object after adding
                new obs
            obsgp (`str` of `list`-like): observation group name(s). If type
                `str` (or list of len == 1) and `use_cols` is None (i.e. all
                non-index cols are to  be set up as obs), the same group name
                will be mapped to all obs in call. If None the obs group name
                will be derived from the base of the constructed observation
                name. If passed as `list` (and len(`list`) = `n` > 1), the
                entries in obsgp will be interpreted to explicitly define the
                grouped for the first `n` cols in `use_cols`, any remaining
                columns will default to None and the base of the observation
                name will be used. Default is None.
            zone_array (`np.ndarray`): array defining spatial limits or zones
                for array-style observations. Default is None
            includes_header (`bool`): flag indicating that the list file includes a
                header row.  Default is True.

        Returns:
            `Pandas.DataFrame`: dataframe with info for new observations

        Note:
            This is the main entry for adding observations to the pest interface

            If `index_cols` and `use_cols` are both None, then it is assumed that
            array-style observations are being requested.  In this case,
            `filenames` must be only one filename.

            `zone_array` is only used for array-style observations.  Zone values
            less than or equal to zero are skipped (using the "dum" option)


        Example::

            # setup observations for the 2nd thru 5th columns of the csv file
            # using the first column as the index
            df = pf.add_observations("heads.csv",index_col=0,use_cols=[1,2,3,4],
                                     ofile_sep=",")
            # add array-style observations, skipping model cells with an ibound
            # value less than or equal to zero
            df = pf.add_observations("conce_array.dat,index_col=None,use_cols=None,
                                     zone_array=ibound)


        """
        use_cols_psd = copy.copy(use_cols)  # store passed use_cols argument
        if insfile is None:  # setup instruction file name
            insfile = "{0}.ins".format(filename)
        self.logger.log("adding observations from output file " "{0}".format(filename))
        # precondition arguments
        (
            filenames,
            fmts,
            seps,
            skip_rows,
            index_cols,
            use_cols,
        ) = self._prep_arg_list_lengths(
            filename,
            index_cols=index_cols,
            use_cols=use_cols,
            fmts=None,
            seps=ofile_sep,
            skip_rows=ofile_skip,
        )
        # array style obs, if both index_cols and use_cols are None (default)
        if index_cols is None and use_cols is None:
            if not isinstance(filenames, str):
                if len(filenames) > 1:
                    self.logger.lraise(
                        "only a single filename can be used for array-style observations"
                    )
                filenames = filenames[0]
            self.logger.log(
                "adding observations from array output file '{0}'".format(filenames)
            )
            # Setup obs for array style output, build and write instruction file
            self._process_array_obs(
                filenames,
                insfile,
                prefix,
                ofile_sep,
                ofile_skip,
                self.longnames,
                zone_array,
            )

            # Add obs from ins file written by _process_array_obs()
            new_obs = self.add_observations_from_ins(
                ins_file=self.new_d / insfile, out_file=self.new_d / filename
            )
            # Try to add an observation group name -- should default to `obgnme`
            # TODO: note list style default to base of obs name, here array default to `obgnme`
            if obsgp is not None:  # if a group name is passed
                new_obs.loc[:, "obgnme"] = obsgp
            elif prefix is not None and len(prefix) != 0:  # if prefix is passed
                new_obs.loc[:, "obgnme"] = prefix
            # else will default to `obgnme`
            self.logger.log(
                "adding observations from array output file '{0}'".format(filenames)
            )
            if rebuild_pst:
                if self.pst is not None:
                    self.logger.log("Adding obs to control file " "and rewriting pst")
                    self.build_pst(filename=self.pst.filename, update="obs")
                else:
                    self.build_pst(filename=self.pst.filename, update=False)
                    self.logger.warn(
                        "pst object not available, " "new control file will be written"
                    )
            return new_obs

        # list style obs
        self.logger.log(
            "adding observations from tabular output file " "'{0}'".format(filenames)
        )
        # -- will end up here if either of index_cols or use_cols is not None
        df, storehead, inssep = self._load_listtype_file(
            filenames, index_cols, use_cols, fmts, seps, skip_rows
        )
        if inssep != ',':
            inssep = seps
        else:
            inssep = [inssep]
        # rectify df?
        # if iloc[0] are strings and index_cols are ints,
        # can we assume that there were infact column headers?
        if all(isinstance(c, str) for c in df.iloc[0]) and all(
            isinstance(a, int) for a in index_cols
        ):
            index_cols = df.iloc[0][index_cols].to_list()  # redefine index_cols
            if use_cols is not None:
                use_cols = df.iloc[0][use_cols].to_list()  # redefine use_cols
            df = (
                df.rename(columns=df.iloc[0].to_dict())
                .drop(0)
                .reset_index(drop=True)
                .apply(pd.to_numeric, errors="ignore")
            )
        # Select all non index cols if use_cols is None
        if use_cols is None:
            use_cols = df.columns.drop(index_cols).tolist()
        # Currently just passing through comments in header (i.e. before the table data)
        lenhead = 0
        stkeys = np.array(
            sorted(storehead.keys())
        )  # comments line numbers as sorted array
        if stkeys.size > 0 and stkeys.min() == 0:
            lenhead += 1 + np.sum(np.diff(stkeys) == 1)
        new_obs_l = []
        for filename, sep in zip(filenames, inssep):  # should only ever be one but hey...
            self.logger.log(
                "building insfile for tabular output file {0}" "".format(filename)
            )
            # Build dataframe from output file df for use in insfile
            df_temp = _get_tpl_or_ins_df(
                df,
                prefix,
                typ="obs",
                index_cols=index_cols,
                use_cols=use_cols,
                longnames=self.longnames,
            )
            df.loc[:, "idx_str"] = df_temp.idx_strs
            # Select only certain rows if requested
            if use_rows is not None:
                if isinstance(use_rows, str):
                    if use_rows not in df.idx_str:
                        self.logger.warn(
                            "can't find {0} in generated observation idx_str. "
                            "setting up obs for all rows instead"
                            "".format(use_rows)
                        )
                        use_rows = None
                elif isinstance(use_rows, int):
                    use_rows = [use_rows]
                use_rows = [r for r in use_rows if r <= len(df)]
                use_rows = df.iloc[use_rows].idx_str.unique()

            # Construct ins_file from df
            # first rectify group name with number of columns
            ncol = len(use_cols)
            fill = True  # default fill=True means that the groupname will be
            # derived from the base of the observation name
            # if passed group name is a string or list with len < ncol
            # and passed use_cols was None or of len > len(obsgp)
            if obsgp is not None:
                if use_cols_psd is None:  # no use_cols defined (all are setup)
                    if len([obsgp] if isinstance(obsgp, str) else obsgp) == 1:
                        # only 1 group provided, assume passed obsgp applys
                        # to all use_cols
                        fill = "first"
                    else:
                        # many obs groups passed, assume last will fill if < ncol
                        fill = "last"
                # else fill will be set to True (base of obs name will be used)
            else:
                obsgp = True  # will use base of col
            obsgp = _check_var_len(obsgp, ncol, fill=fill)
            df_ins = pyemu.pst_utils.csv_to_ins_file(
                df.set_index("idx_str"),
                ins_filename=self.new_d / insfile,
                only_cols=use_cols,
                only_rows=use_rows,
                marker="~",
                includes_header=includes_header,
                includes_index=False,
                prefix=prefix,
                longnames=self.longnames,
                head_lines_len=lenhead,
                sep=sep,
                gpname=obsgp,
            )
            self.logger.log(
                "building insfile for tabular output file {0}" "".format(filename)
            )
            new_obs = self.add_observations_from_ins(
                ins_file=self.new_d / insfile, out_file=self.new_d / filename
            )
            if "obgnme" in df_ins.columns:
                new_obs.loc[:, "obgnme"] = df_ins.loc[new_obs.index, "obgnme"]
            new_obs_l.append(new_obs)
        new_obs = pd.concat(new_obs_l)
        self.logger.log(
            "adding observations from tabular output file " "'{0}'".format(filenames)
        )
        if rebuild_pst:
            if self.pst is not None:
                self.logger.log("Adding obs to control file " "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update="obs")
            else:
                pstname = Path(self.new_d, self.original_d.name)
                self.logger.warn(
                    "pst object not available, "
                    f"new control file will be written with filename {pstname}"
                )
                self.build_pst(filename=None, update=False)

        return new_obs

    def add_observations_from_ins(
        self, ins_file, out_file=None, pst_path=None, inschek=True
    ):
        """add new observations to a control file from an existing instruction file

        Args:
            ins_file (`str`): instruction file with exclusively new
               observation names. N.B. if `ins_file` just contains base
               filename string (i.e. no directory name), the path to PEST
               directory will be automatically appended.
            out_file (`str`): model output file.  If None, then
               ins_file.replace(".ins","") is used. Default is None.
               If `out_file` just contains base filename string
               (i.e. no directory name), the path to PEST directory will be
               automatically appended.
            pst_path (`str`): the path to append to the instruction file and
               out file in the control file.  If not None, then any existing
               path in front of the template or ins file is split off and
               pst_path is prepended.  If python is being run in a directory
               other than where the control file will reside, it is useful
               to pass `pst_path` as `.`. Default is None
            inschek (`bool`): flag to try to process the existing output file
               using the `pyemu.InstructionFile` class.  If successful,
               processed outputs are used as obsvals

        Returns:
            `pandas.DataFrame`: the data for the new observations that were
               added

        Note:
            populates the new observation information with default values

        Example::

            pf = pyemu.PstFrom("temp","template")
            pf.add_observations_from_ins(os.path.join("template","new_obs.dat.ins"),
                                 pst_path=".")

        """
        # lifted almost completely from `Pst().add_observation()`
        if os.path.dirname(ins_file) in ["", "."]:
            # if insfile is passed as just a filename,
            # append pest directory name
            ins_file = self.new_d / ins_file
            pst_path = "."  # reset and new assumed pst_path
        # else:
        # assuming that passed insfile is the full path to file from current location
        if not os.path.exists(ins_file):
            self.logger.lraise(
                "ins file not found: {0}, {1}" "".format(os.getcwd(), ins_file)
            )
        if out_file is None:
            out_file = str(ins_file).replace(".ins", "")
        elif os.path.dirname(out_file) in ["", "."]:
            out_file = self.new_d / out_file
        if ins_file == out_file:
            self.logger.lraise("ins_file == out_file, doh!")

        # get the obs names in the instructions file
        self.logger.log(
            "adding observation from instruction file '{0}'".format(ins_file)
        )
        obsnme = pyemu.pst_utils.parse_ins_file(ins_file)

        sobsnme = set(obsnme)
        if len(self.obs_dfs) > 0:
            sexist = pd.concat(self.obs_dfs).obsnme
        else:
            sexist = []
        sexist = set(sexist)  # todo need to check this here?
        sint = sobsnme.intersection(sexist)
        if len(sint) > 0:
            self.logger.lraise(
                "the following obs instruction file {0} are already in the "
                "control file:{1}".format(ins_file, ",".join(sint))
            )

        # find "new" obs that are not already in the control file
        new_obsnme = sobsnme - sexist
        if len(new_obsnme) == 0:
            self.logger.lraise(
                "no new observations found in instruction file {0}".format(ins_file)
            )

        # extend observation_data
        new_obsnme = np.sort(list(new_obsnme))
        new_obs_data = pyemu.pst_utils._populate_dataframe(
            new_obsnme,
            pyemu.pst_utils.pst_config["obs_fieldnames"],
            pyemu.pst_utils.pst_config["obs_defaults"],
            pyemu.pst_utils.pst_config["obs_dtype"],
        )
        new_obs_data.loc[new_obsnme, "obsnme"] = new_obsnme
        new_obs_data.index = new_obsnme

        # need path relative to where control file
        ins_file_pstrel = Path(ins_file).relative_to(self.new_d)
        out_file_pstrel = Path(out_file).relative_to(self.new_d)
        if pst_path is not None:
            ins_file_pstrel = pst_path / ins_file_pstrel
            out_file_pstrel = pst_path / out_file_pstrel
        self.ins_filenames.append(ins_file_pstrel)
        self.output_filenames.append(out_file_pstrel)
        # add to temporary files to be removed at start of forward run
        self.tmp_files.append(out_file_pstrel)
        df = None
        if inschek:
            # df = pst_utils._try_run_inschek(ins_file,out_file,cwd=cwd)
            # ins_file = os.path.join(cwd, ins_file)
            # out_file = os.path.join(cwd, out_file)
            df = pyemu.pst_utils.try_process_output_file(
                ins_file=ins_file, output_file=out_file
            )
        if df is not None:
            # print(self.observation_data.index,df.index)
            new_obs_data.loc[df.index, "obsval"] = df.obsval
        self.obs_dfs.append(new_obs_data)
        self.logger.log(
            "adding observation from instruction file '{0}'".format(ins_file)
        )
        return new_obs_data

    def add_parameters(
        self,
        filenames,
        par_type,
        zone_array=None,
        dist_type="gaussian",
        sigma_range=4.0,
        upper_bound=1.0e10,
        lower_bound=1.0e-10,
        transform="log",
        par_name_base="p",
        index_cols=None,
        use_cols=None,
        pargp=None,
        pp_space=10,
        use_pp_zones=False,
        num_eig_kl=100,
        spatial_reference=None,
        geostruct=None,
        datetime=None,
        mfile_fmt="free",
        mfile_skip=None,
        ult_ubound=None,
        ult_lbound=None,
        rebuild_pst=False,
        alt_inst_str="inst",
        comment_char=None,
        par_style="multiplier",
    ):
        """
        Add list or array style model input files to PstFrom object.
        This method is the main entry point for adding parameters to the
        pest interface

        Args:
            filenames (`str`): Model input filenames to parameterize
            par_type (`str`): One of `grid` - for every element, `constant` - for single
                parameter applied to every element, `zone` - for zone-based
                parameterization (only for array-style) or `pilotpoint` - for
                pilot-point base parameterization of array style input files.
                Note `kl` not yet implemented # TODO
            zone_array (`np.ndarray`): array defining spatial limits or zones
                for parameterization.
            dist_type: not yet implemented # TODO
            sigma_range: not yet implemented # TODO
            upper_bound (`float`): PEST parameter upper bound # TODO support different ubound,lbound,transform if multiple use_col
            lower_bound (`float`): PEST parameter lower bound
            transform (`str`): PEST parameter transformation.  Must be either "log","none" or "fixed.  The "tied" transform
                must be used after calling `PstFrom.build_pst()`.
            par_name_base (`str`): basename for parameters that are set up
            index_cols (`list`-like): if not None, will attempt to parameterize
                expecting a tabular-style model input file. `index_cols`
                defines the unique columns used to set up pars. If passed as a
                list of `str`, stings are expected to denote the columns
                headers in tabular-style parameter files; if `i` and `j` in
                list, these columns will be used to define spatial position for
                spatial correlations (if required). WARNING: If passed as list
                of `int`, `i` and `j` will be assumed to be in last two entries
                in the list. Can be passed as a dictionary using the keys
                `i` and `j` to explicitly speficy the columns that relate to
                model rows and columns to be identified and processed to x,y.
            use_cols (`list`-like or `int`): for tabular-style model input file,
                defines the columns to be parameterised
            pargp (`str`): Parameter group to assign pars to. This is PESTs
                pargp but is also used to gather correlated parameters set up
                using multiple `add_parameters()` calls (e.g. temporal pars)
                with common geostructs.
            pp_space (`int`,`str` or `pd.DataFrame`): Spatial pilot point information.
                If `int` it is the spacing in rows and cols of where to place pilot points.
                If `pd.DataFrame`, then this arg is treated as a prefined set of pilot points
                and in this case, the dataframe must have "name", "x", "y", and optionally "zone" columns.
                If `str`, then an attempt is made to load a dataframe from a csv file (if `pp_space` ends with ".csv"),
                 shapefile (if `pp_space` ends with ".shp") or from a pilot points file.  If `pp_space` is None,
                 an integer spacing of 10 is used.  Default is None
            use_pp_zones (`bool`): a flag to use the greater-than-zero values
                in the zone_array as pilot point zones.
                If False, zone_array values greater than zero are treated as a
                single zone.  This argument is only used if `pp_space` is None or `int`. Default is False.
            num_eig_kl: TODO - impliment with KL pars
            spatial_reference (`pyemu.helpers.SpatialReference`): If different
                spatial reference required for pilotpoint setup.
                If None spatial reference passed to `PstFrom()` will be used
                for pilot-points
            geostruct (`pyemu.geostats.GeoStruct()`): For specifying correlation
                geostruct for pilot-points and par covariance.
            datetime (`str`): optional %Y%m%d string or datetime object for
                setting up temporally correlated pars. Where datetime is passed
                 correlation axis for pars will be set to timedelta.
            mfile_fmt (`str`): format of model input file - this will be preserved
            mfile_skip (`int` or `str`): header in model input file to skip
                when reading and reapply when writing. Can optionally be `str` in which case `mf_skip` will be treated
                as a `comment_char`.
            ult_ubound (`float`): Ultimate upper bound for model input
                parameter once all mults are applied - ensure physical model par vals. If not passed,
                it is set to 1.0e+30
            ult_lbound (`float`): Ultimate lower bound for model input
                parameter once all mults are applied.  If not passed, it is set to
                1.0e-30 for log transform and -1.0e+30 for non-log transform
            rebuild_pst (`bool`): (Re)Construct PstFrom.pst object after adding
                new parameters
            alt_inst_str (`str`): Alternative to default `inst` string in
                parameter names
            comment_char (`str`): option to skip comment lines in model file.
                This is not additive with `mfile_skip` option.
                Warning: currently comment lines within list-like tabular data
                will be lost.
            par_style (`str`): either "multiplier" or "direct" where the former setups
                up a multiplier parameter process against the existing model input
                array and the former setups a template file to write the model
                input file directly.  Default is "multiplier".

        Returns:
            `pandas.DataFrame`: dataframe with info for new parameters

        Example::

            # setup grid-scale direct parameters for an array of numbers
            df = pf.add_parameters("hk.dat",par_type="grid",par_style="direct")
            # setup pilot point multiplier parameters for an array of numbers
            # with a pilot point being set in every 5th active model cell
            df = pf.add_parameters("recharge.dat",par_type="pilotpoint",pp_space=5,
                                   zone_array="ibound.dat")
            # setup a single multiplier parameter for the 4th column
            # of a column format (list type) file
            df = pf.add_parameters("wel_list_1.dat",par_type="constant",
                                   index_cols=[0,1,2],use_cols=[3])

        """
        # TODO need more support for temporal pars?
        #  - As another partype using index_cols or an additional time_cols

        # TODO support passing par_file (i,j)/(x,y) directly where information
        #  is not contained in model parameter file - e.g. no i,j columns
        self.add_pars_callcount += 1
        self.ijwarned[self.add_pars_callcount] = False
        # this keeps denormal values for creeping into the model input arrays
        if ult_ubound is None:
            ult_ubound = self.ult_ubound_fill
        if ult_lbound is None:
            ult_lbound = self.ult_lbound_fill

        if transform.lower().strip() not in ["none", "log", "fixed"]:
            self.logger.lraise(
                "unrecognized transform ('{0}'), should be in ['none','log','fixed']".format(
                    transform
                )
            )

        if transform == "fixed" and geostruct is not None:
            self.logger.lraise(
                "geostruct is not 'None', cant draw values for fixed pars"
            )

        # some checks for direct parameters
        par_style = par_style.lower()
        if par_style not in ["multiplier", "direct"]:
            self.logger.lraise(
                "add_parameters(): unrecognized 'style': {0}, should be either 'multiplier' or 'direct'".format(
                    par_style
                )
            )
        if isinstance(filenames, str) or isinstance(filenames, Path):
            filenames = [filenames]
        # data file paths relative to the pest parent directory
        filenames = [
            get_relative_filepath(self.original_d, filename) for filename in filenames
        ]
        if len(filenames) == 0:
            self.logger.lraise("add_parameters(): filenames is empty")
        if par_style == "direct":
            if len(filenames) != 1:
                self.logger.lraise(
                    "add_parameters(): 'filenames' arg for 'direct' style must contain "
                    + "one and only one filename, not {0} files".format(len(filenames))
                )
            if filenames[0] in self.direct_org_files:
                self.logger.lraise(
                    "add_parameters(): original model input file '{0}' ".format(
                        filenames[0]
                    )
                    + " already used for 'direct' parameterization"
                )
            else:
                self.direct_org_files.append(filenames[0])
        # Default par data columns used for pst
        par_data_cols = pyemu.pst_utils.pst_config["par_fieldnames"]
        self.logger.log(
            "adding {0} type {1} style parameters for file(s) {2}".format(
                par_type, par_style, [str(f) for f in filenames]
            )
        )
        if geostruct is not None:
            if geostruct.sill != 1.0:  #  and par_style != "multiplier": #TODO !=?
                self.logger.warn(
                    "geostruct sill != 1.0"  # for 'multiplier' style parameters"
                )
            if geostruct.transform != transform:
                self.logger.warn(
                    "0) Inconsistency between " "geostruct transform and partrans."
                )
                self.logger.warn(
                    "1) Setting geostruct transform to " "{0}".format(transform)
                )
                if geostruct not in self.par_struct_dict.keys():
                    # safe to just reset transform
                    geostruct.transform = transform
                else:
                    self.logger.warn("2) This will create a new copy of " "geostruct")
                    # to avoid flip flopping transform need to make a new geostruct
                    geostruct = copy.copy(geostruct)
                    geostruct.transform = transform
                self.logger.warn(
                    "-) Better to pass an appropriately " "transformed geostruct"
                )

        # Get useful variables from arguments passed
        # if index_cols passed as a dictionary that maps i,j information
        idx_cols = index_cols
        ij_in_idx = None
        xy_in_idx = None
        if isinstance(index_cols, dict):
            # only useful if i and j are in keys .... or xy?
            # TODO: or datetime str?
            keys = np.array([k.lower() for k in index_cols.keys()])
            idx_cols = [index_cols[k] for k in keys]
            if any(all(a in keys for a in aa) for aa in [["i", "j"], ["x", "y"]]):
                if all(ij in keys for ij in ["i", "j"]):
                    o_idx = np.argsort(keys)
                    ij_in_idx = o_idx[np.searchsorted(keys[o_idx], ["i", "j"])]
                if all(xy in keys for xy in ["x", "y"]):
                    o_idx = np.argsort(keys)
                    xy_in_idx = o_idx[np.searchsorted(keys[o_idx], ["x", "y"])]
            else:
                self.logger.lraise(
                    "If passing `index_cols` as type == dict, "
                    "keys need to contain [`i` and `j`] or "
                    "[`x` and `y`]"
                )

        (
            index_cols,
            use_cols,
            file_dict,
            fmt_dict,
            sep_dict,
            skip_dict,
            headerlines,  # needed for direct pars (need to be in tpl)
        ) = self._par_prep(
            filenames,
            idx_cols,
            use_cols,
            fmts=mfile_fmt,
            skip_rows=mfile_skip,
            c_char=comment_char,
        )
        if datetime is not None:  # convert and check datetime
            # TODO: something needed here to allow a different relative point.
            datetime = _get_datetime_from_str(datetime)
            if self.start_datetime is None:
                self.logger.warn(
                    "NO START_DATEIME PROVIDED, ASSUMING PAR "
                    "DATETIME IS START {}".format(datetime)
                )
                self.start_datetime = datetime
            assert (
                datetime >= self.start_datetime
            ), "passed datetime is earlier than start_datetime {0}, {1}".format(
                datetime, self.start_datetime
            )
            t_offest = datetime - self.start_datetime

        # Pull out and construct name-base for parameters
        if isinstance(par_name_base, str):
            par_name_base = [par_name_base]
        # if `use_cols` is passed check number of base names is the same as cols
        if len(par_name_base) == 1:
            pass
        elif use_cols is not None and len(par_name_base) == len(use_cols):
            pass
        else:
            self.logger.lraise(
                "par_name_base should be a string, "
                "single-element container, or container of "
                "len use_cols, not '{0}'"
                "".format(str(par_name_base))
            )

        # otherewise, things get tripped up in the ensemble/cov stuff
        if pargp is not None:
            if isinstance(pargp, list):
                pargp = [pg.lower() for pg in pargp]
            else:
                pargp = pargp.lower()
        par_name_base = [pnb.lower() for pnb in par_name_base]

        if self.longnames:  # allow par names to be long... fine for pestpp
            fmt = "_{0}".format(alt_inst_str) + ":{0}"
            chk_prefix = "_{0}".format(alt_inst_str)  # add `instance` identifier
        else:
            fmt = "{0}"  # may not be so well supported
            chk_prefix = ""
        # increment name base if already passed
        for i in range(len(par_name_base)):
            par_name_base[i] += fmt.format(
                self._next_count(par_name_base[i] + chk_prefix)
            )

        # multiplier file name will be taken first par group, if passed
        # (the same multipliers will apply to all pars passed in this call)
        # Remove `:` for filenames

        par_name_store = par_name_base[0].replace(":", "")  # for os filename

        # Define requisite filenames
        if par_style == "multiplier":
            mlt_filename = "{0}_{1}.csv".format(par_name_store, par_type)
            # pst input file (for tpl->in pair) is multfile (in mult dir)
            in_fileabs = self.mult_file_d / mlt_filename
            # pst input file (for tpl->in pair) is multfile (in mult dir)
            in_filepst = in_fileabs.relative_to(self.new_d)
            tpl_filename = self.tpl_d / (mlt_filename + ".tpl")
        else:
            mlt_filename = np.NaN
            # absolute path to org/datafile
            in_fileabs = self.original_file_d / filenames[0].name
            # pst input file (for tpl->in pair) is orgfile (in org dir)
            # relative path to org/datafile (relative to dest model workspace):
            in_filepst = in_fileabs.relative_to(self.new_d)
            tpl_filename = self.tpl_d / (filenames[0].name + ".tpl")

        pp_filename = None  # setup placeholder variables
        fac_filename = None

        # Process model parameter files to produce appropriate pest pars
        if index_cols is not None:  # Assume list/tabular type input files
            # ensure inputs are provided for all required cols
            ncol = len(use_cols)
            ult_lbound = _check_var_len(ult_lbound, ncol)
            ult_ubound = _check_var_len(ult_ubound, ncol)
            pargp = _check_var_len(pargp, ncol)
            lower_bound = _check_var_len(lower_bound, ncol, fill="first")
            upper_bound = _check_var_len(upper_bound, ncol, fill="first")
            if len(use_cols) != len(ult_lbound) != len(ult_ubound):
                self.logger.lraise(
                    "mismatch in number of columns to use {0} "
                    "and number of ultimate lower {0} or upper "
                    "{1} par bounds defined"
                    "".format(len(use_cols), len(ult_lbound), len(ult_ubound))
                )

            self.logger.log(
                "writing list-based template file '{0}'".format(tpl_filename)
            )
            # Generate tabular type template - also returns par data
            # relative file paths are in file_dict as Path instances (kludgey)
            dfs = [file_dict[Path(filename)] for filename in filenames]
            get_xy = None
            if (
                par_type.startswith("grid") or par_type.startswith("p")
            ) and geostruct is not None:
                get_xy = self.get_xy
            df = write_list_tpl(
                filenames,
                dfs,
                par_name_base,
                tpl_filename=tpl_filename,
                par_type=par_type,
                suffix="",
                index_cols=index_cols,
                use_cols=use_cols,
                zone_array=zone_array,
                gpname=pargp,
                longnames=self.longnames,
                ij_in_idx=ij_in_idx,
                xy_in_idx=xy_in_idx,
                get_xy=get_xy,
                zero_based=self.zero_based,
                input_filename=in_fileabs,
                par_style=par_style,
                headerlines=headerlines,  # only needed for direct pars
            )
            assert (
                np.mod(len(df), len(use_cols)) == 0.0
            ), "Parameter dataframe wrong shape for number of cols {0}" "".format(
                use_cols
            )
            # variables need to be passed to each row in df
            lower_bound = np.tile(lower_bound, int(len(df) / ncol))
            upper_bound = np.tile(upper_bound, int(len(df) / ncol))
            self.logger.log(
                "writing list-based template file '{0}'".format(tpl_filename)
            )
        else:  # Assume array type parameter file
            self.logger.log(
                "writing array-based template file '{0}'".format(tpl_filename)
            )
            shp = file_dict[list(file_dict.keys())[0]].shape
            # ARRAY constant, zones or grid (cell-by-cell)
            if par_type in {"constant", "zone", "grid"}:
                self.logger.log(
                    "writing template file "
                    "{0} for {1}".format(tpl_filename, par_name_base)
                )
                # Generate array type template - also returns par data
                df = write_array_tpl(
                    name=par_name_base[0],
                    tpl_filename=tpl_filename,
                    suffix="",
                    par_type=par_type,
                    zone_array=zone_array,
                    shape=shp,
                    longnames=self.longnames,
                    get_xy=self.get_xy,
                    fill_value=1.0,
                    gpname=pargp,
                    input_filename=in_fileabs,
                    par_style=par_style,
                )
                self.logger.log(
                    "writing template file"
                    " {0} for {1}".format(tpl_filename, par_name_base)
                )
            # ARRAY PILOTPOINT setup
            elif par_type in {
                "pilotpoints",
                "pilot_points",
                "pilotpoint",
                "pilot_point",
                "pilot-point",
                "pilot-points",
            }:
                if par_style == "direct":
                    self.logger.lraise(
                        "pilot points not supported for 'direct' par_style"
                    )
                # Setup pilotpoints for array type par files
                self.logger.log("setting up pilot point parameters")
                # finding spatial references for for setting up pilot points
                if spatial_reference is None:
                    # if none passed with add_pars call
                    self.logger.statement(
                        "No spatial reference " "(containing cell spacing) passed."
                    )
                    if self.spatial_reference is not None:
                        # using global sr on PstFrom object
                        self.logger.statement(
                            "OK - using spatial reference " "in parent object."
                        )
                        spatial_reference = self.spatial_reference
                    else:
                        # uhoh
                        self.logger.lraise(
                            "No spatial reference in parent "
                            "object either. "
                            "Can't set-up pilotpoints"
                        )
                # check that spatial reference lines up with the original array dimensions
                structured = False
                if not isinstance(spatial_reference, dict):
                    structured = True
                    for mod_file, ar in file_dict.items():
                        orgdata = ar.shape
                        assert orgdata[0] == spatial_reference.nrow, (
                            "Spatial reference nrow not equal to original data nrow for\n"
                            + os.path.join(
                                *os.path.split(self.original_file_d)[1:], mod_file
                            )
                        )
                        assert orgdata[1] == spatial_reference.ncol, (
                            "Spatial reference ncol not equal to original data ncol for\n"
                            + os.path.join(
                                *os.path.split(self.original_file_d)[1:], mod_file
                            )
                        )
                # (stolen from helpers.PstFromFlopyModel()._pp_prep())
                # but only settting up one set of pps at a time
                pp_dict = {0: par_name_base}
                pp_filename = "{0}pp.dat".format(par_name_store)
                # pst inputfile (for tpl->in pair) is
                # par_name_storepp.dat table (in pst ws)
                in_filepst = pp_filename
                tpl_filename = self.tpl_d / (pp_filename + ".tpl")
                # tpl_filename = get_relative_filepath(self.new_d, tpl_filename)
                pp_locs = None
                if pp_space is None:  # default spacing if not passed
                    self.logger.warn("pp_space is None, using 10...\n")
                    pp_space = 10
                else:
                    if isinstance(pp_space, float):
                        pp_space = int(pp_space)
                    elif isinstance(pp_space, int):
                        pass
                    elif isinstance(pp_space, str):

                        if pp_space.lower().strip().endswith(".csv"):
                            self.logger.statement(
                                "trying to load pilot point location info from csv file '{0}'".format(
                                    self.new_d / Path(pp_space)
                                )
                            )
                            pp_locs = pd.read_csv(self.new_d / pp_space)

                        elif pp_space.lower().strip().endswith(".shp"):
                            self.logger.statement(
                                "trying to load pilot point location info from shapefile '{0}'".format(
                                    self.new_d / Path(pp_space)
                                )
                            )
                            pp_locs = pyemu.pp_utils.pilot_points_from_shapefile(
                                str(self.new_d / Path(pp_space))
                            )
                        else:
                            self.logger.statement(
                                "trying to load pilot point location info from pilot point file '{0}'".format(
                                    self.new_d / Path(pp_space)
                                )
                            )
                            pp_locs = pyemu.pp_utils.pp_file_to_dataframe(
                                self.new_d / pp_space
                            )
                        self.logger.statement(
                            "pilot points found in file '{0}' will be transferred to '{1}' for parameterization".format(
                                pp_space, pp_filename
                            )
                        )
                    elif isinstance(pp_space, pd.DataFrame):
                        pp_locs = pp_space
                    else:
                        self.logger.lraise(
                            "unrecognized 'pp_space' value, should be int, csv file, pp file or dataframe, not '{0}'".format(
                                type(pp_space)
                            )
                        )
                    if pp_locs is not None:
                        cols = pp_locs.columns.tolist()
                        if "name" not in cols:
                            self.logger.lraise("'name' col not found in pp dataframe")
                        if "x" not in cols:
                            self.logger.lraise("'x' col not found in pp dataframe")
                        if "y" not in cols:
                            self.logger.lraise("'y' col not found in pp dataframe")
                        if "zone" not in cols:
                            self.logger.warn(
                                "'zone' col not found in pp dataframe, adding generic zone"
                            )
                            pp_locs.loc[:, "zone"] = 1
                        elif zone_array is not None:
                            # check that all the zones in the pp df are in the zone array
                            missing = []
                            for uz in pp_locs.zone.unique():
                                if int(uz) not in zone_array:
                                    missing.append(str(uz))
                            if len(missing) > 0:
                                self.logger.lraise(
                                    "the following pp zone values were not found in the zone array: {0}".format(
                                        ",".join(missing)
                                    )
                                )

                            for uz in np.unique(zone_array):
                                if uz < 1:
                                    continue
                                if uz not in pp_locs.zone.values:

                                    missing.append(str(uz))
                            if len(missing) > 0:
                                self.logger.warn(
                                    "the following zones don't have any pilot points:{0}".format(
                                        ",".join(missing)
                                    )
                                )

                if not structured and pp_locs is None:
                    self.logger.lraise(
                        "pilot point type parameters with an unstructured grid requires 'pp_space' "
                        "contain explict pilot point information"
                    )

                if not structured and zone_array is not None:
                    # self.logger.lraise("'zone_array' not supported for unstructured grids and pilot points")
                    if "zone" not in pp_locs.columns:
                        self.logger.lraise(
                            "'zone' not found in pp info dataframe and 'zone_array' passed"
                        )
                    uvals = np.unique(zone_array)
                    zvals = set([int(z) for z in pp_locs.zone.tolist()])
                    missing = []
                    for uval in uvals:
                        if int(uval) not in zvals and int(uval) != 0:
                            missing.append(str(int(uval)))
                    if len(missing) > 0:
                        self.logger.warn(
                            "the following values in the zone array were not found in the pp info: {0}".format(
                                ",".join(missing)
                            )
                        )

                if geostruct is None:  # need a geostruct for pilotpoints

                    # can use model default, if provided
                    if self.geostruct is None:  # but if no geostruct passed...
                        if not structured:
                            self.logger.lraise(
                                "pilot point type parameters with an unstructured grid requires an"
                                " explicit `geostruct` arg be passed to either PstFrom or add_parameters()"
                            )
                        self.logger.warn(
                            "pp_geostruct is None,"
                            "using ExpVario with contribution=1 "
                            "and a=(pp_space*max(delr,delc))"
                        )
                        # set up a default - could probably do something better if pp locs are passed
                        if not isinstance(pp_space, int):
                            space = 10
                        else:
                            space = pp_space
                        pp_dist = space * float(
                            max(
                                spatial_reference.delr.max(),
                                spatial_reference.delc.max(),
                            )
                        )
                        v = pyemu.geostats.ExpVario(contribution=1.0, a=pp_dist)
                        pp_geostruct = pyemu.geostats.GeoStruct(
                            variograms=v, name="pp_geostruct", transform=transform
                        )
                    else:
                        pp_geostruct = self.geostruct
                        if pp_geostruct.transform != transform:
                            self.logger.warn(
                                "0) Inconsistency between "
                                "pp_geostruct transform and "
                                "partrans."
                            )
                            self.logger.warn(
                                "1) Setting pp_geostruct transform "
                                "to {0}".format(transform)
                            )
                            self.logger.warn(
                                "2) This will create a new copy of " "pp_geostruct"
                            )
                            self.logger.warn(
                                "3) Better to pass an appropriately "
                                "transformed geostruct"
                            )
                            pp_geostruct = copy.copy(pp_geostruct)
                            pp_geostruct.transform = transform
                else:
                    pp_geostruct = geostruct

                if pp_locs is None:
                    # Set up pilot points

                    df = pyemu.pp_utils.setup_pilotpoints_grid(
                        sr=spatial_reference,
                        ibound=zone_array,
                        use_ibound_zones=use_pp_zones,
                        prefix_dict=pp_dict,
                        every_n_cell=pp_space,
                        pp_dir=self.new_d,
                        tpl_dir=self.tpl_d,
                        shapename=str(self.new_d / "{0}.shp".format(par_name_store)),
                        longnames=self.longnames,
                    )
                else:
                    df = pyemu.pp_utils.pilot_points_to_tpl(
                        pp_locs,
                        tpl_filename,
                        par_name_base[0],
                        longnames=self.longnames,
                    )
                    df.loc[:, "pargp"] = par_name_base[0]

                df.set_index("parnme", drop=False, inplace=True)
                # df includes most of the par info for par_dfs and also for
                # relate_parfiles
                self.logger.statement(
                    "{0} pilot point parameters created".format(df.shape[0])
                )
                # should be only one group at a time
                pargp = df.pargp.unique()
                self.logger.statement("pilot point 'pargp':{0}".format(",".join(pargp)))
                self.logger.log("setting up pilot point parameters")

                # Calculating pp factors
                pg = pargp[0]
                # this reletvively quick
                ok_pp = pyemu.geostats.OrdinaryKrige(pp_geostruct, df)
                # build krige reference information on the fly - used to help
                # prevent unnecessary krig factor calculation
                pp_info_dict = {
                    "pp_data": ok_pp.point_data.loc[:, ["x", "y", "zone"]],
                    "cov": ok_pp.point_cov_df,
                    "zn_ar": zone_array,
                    "sr": spatial_reference,
                }
                fac_processed = False
                for facfile, info in self._pp_facs.items():  # check against
                    # factors already calculated
                    if (
                        info["pp_data"].equals(pp_info_dict["pp_data"])
                        and info["cov"].equals(pp_info_dict["cov"])
                        and np.array_equal(info["zn_ar"], pp_info_dict["zn_ar"])
                    ):
                        if type(info["sr"]) == type(spatial_reference):
                            if isinstance(spatial_reference, dict):
                                if len(info["sr"]) != len(spatial_reference):
                                    continue
                        else:
                            continue

                        fac_processed = True  # don't need to re-calc same factors
                        fac_filename = facfile  # relate to existing fac file
                        break
                if not fac_processed:
                    # TODO need better way of naming sequential fac_files?
                    self.logger.log("calculating factors for pargp={0}".format(pg))
                    fac_filename = self.new_d / "{0}pp.fac".format(par_name_store)
                    var_filename = fac_filename.with_suffix(".var.dat")
                    self.logger.statement(
                        "saving krige variance file:{0}".format(var_filename)
                    )
                    self.logger.statement(
                        "saving krige factors file:{0}".format(fac_filename)
                    )
                    # store info on pilotpoints
                    self._pp_facs[fac_filename] = pp_info_dict
                    # this is slow (esp on windows) so only want to do this
                    # when required
                    if structured:
                        ok_pp.calc_factors_grid(
                            spatial_reference,
                            var_filename=var_filename,
                            zone_array=zone_array,
                            num_threads=10,
                        )
                        ok_pp.to_grid_factors_file(fac_filename)
                    else:
                        # put the sr dict info into a df
                        # but we only want to use the n
                        if zone_array is not None:
                            for zone in np.unique(zone_array):
                                if int(zone) == 0:
                                    continue

                                data = []
                                for node, (x, y) in spatial_reference.items():
                                    if zone_array[0, node] == zone:
                                        data.append([node, x, y])
                                if len(data) == 0:
                                    continue
                                node_df = pd.DataFrame(data, columns=["node", "x", "y"])
                                ok_pp.calc_factors(
                                    node_df.x,
                                    node_df.y,
                                    num_threads=1,
                                    pt_zone=zone,
                                    idx_vals=node_df.node.apply(np.int),
                                )
                            ok_pp.to_grid_factors_file(
                                fac_filename, ncol=len(spatial_reference)
                            )
                        else:
                            data = []
                            for node, (x, y) in spatial_reference.items():
                                data.append([node, x, y])
                            node_df = pd.DataFrame(data, columns=["node", "x", "y"])
                            ok_pp.calc_factors(node_df.x, node_df.y, num_threads=10)
                            ok_pp.to_grid_factors_file(
                                fac_filename, ncol=node_df.shape[0]
                            )

                    self.logger.log("calculating factors for pargp={0}".format(pg))
            # TODO - other par types - JTW?
            elif par_type == "kl":
                self.logger.lraise("array type 'kl' not implemented")
            else:
                self.logger.lraise(
                    "unrecognized 'par_type': '{0}', "
                    "should be in "
                    "['constant','zone','grid','pilotpoints',"
                    "'kl'"
                )
            self.logger.log(
                "writing array-based template file " "'{0}'".format(tpl_filename)
            )

        if datetime is not None:
            # add time info to par_dfs
            df["datetime"] = datetime
            df["timedelta"] = t_offest
        # accumulate information that relates mult_files (set-up here and
        # eventually filled by PEST) to actual model files so that actual
        # model input file can be generated
        # (using helpers.apply_list_and_array_pars())
        zone_filename = None
        if zone_array is not None and zone_array.ndim < 3:
            # zone_filename = tpl_filename.replace(".tpl",".zone")
            zone_filename = Path(str(tpl_filename).replace(".tpl", ".zone"))
            self.logger.statement(
                "saving zone array {0} for tpl file {1}".format(
                    zone_filename, tpl_filename
                )
            )
            np.savetxt(zone_filename, zone_array, fmt="%4d")
            zone_filename = zone_filename.name

        relate_parfiles = []
        for mod_file in file_dict.keys():
            mult_dict = {
                "org_file": Path(self.original_file_d.name, mod_file.name),
                "model_file": mod_file,
                "use_cols": use_cols,
                "index_cols": index_cols,
                "fmt": fmt_dict[mod_file],
                "sep": sep_dict[mod_file],
                "head_rows": skip_dict[mod_file],
                "upper_bound": ult_ubound,
                "lower_bound": ult_lbound,
            }
            if par_style == "multiplier":
                mult_dict["mlt_file"] = Path(self.mult_file_d.name, mlt_filename)

            if pp_filename is not None:
                # if pilotpoint need to store more info
                assert fac_filename is not None, "missing pilot-point input filename"
                mult_dict["fac_file"] = os.path.relpath(fac_filename, self.new_d)
                mult_dict["pp_file"] = pp_filename
                mult_dict["pp_fill_value"] = 1.0
                mult_dict["pp_lower_limit"] = 1.0e-10
                mult_dict["pp_upper_limit"] = 1.0e10
            if zone_filename is not None:
                mult_dict["zone_file"] = zone_filename
            relate_parfiles.append(mult_dict)
        relate_pars_df = pd.DataFrame(relate_parfiles)
        # store on self for use in pest build etc
        self._parfile_relations.append(relate_pars_df)

        # add cols required for pst.parameter_data
        df.loc[:, "partype"] = par_type
        df.loc[:, "partrans"] = transform
        df.loc[:, "parubnd"] = upper_bound
        df.loc[:, "parlbnd"] = lower_bound
        # df.loc[:,"tpl_filename"] = tpl_filename

        # store tpl --> in filename pair
        self.tpl_filenames.append(get_relative_filepath(self.new_d, tpl_filename))
        self.input_filenames.append(in_filepst)
        for file_name in file_dict.keys():
            # store mult --> original file pairs
            self.org_files.append(file_name)
            self.mult_files.append(mlt_filename)

        # add pars to par_data list BH: is this what we want?
        # - BH: think we can get away with dropping duplicates?
        missing = set(par_data_cols) - set(df.columns)
        for field in missing:  # fill missing pst.parameter_data cols with defaults
            df[field] = pyemu.pst_utils.pst_config["par_defaults"][field]
        df = df.drop_duplicates(subset="parnme")  # drop pars that appear multiple times
        # df = df.loc[:, par_data_cols]  # just storing pst required cols
        # - need to store more for cov builder (e.g. x,y)
        # TODO - check when self.par_dfs gets used
        #  if constructing struct_dict here....
        #  - possibly not necessary to store
        self.par_dfs.append(df)
        # pivot df to list of df per par group in this call
        # (all groups will be related to same geostruct)
        # TODO maybe use different marker to denote a relationship between pars
        #  at the moment relating pars using common geostruct and pargp but may
        #  want to reserve pargp for just PEST
        if "covgp" not in df.columns:
            gp_dict = {g: [d] for g, d in df.groupby("pargp")}
        else:
            gp_dict = {g: [d] for g, d in df.groupby("covgp")}
        # df_list = [d for g, d in df.groupby('pargp')]
        if geostruct is not None and (
            par_type.lower() not in ["constant", "zone"] or datetime is not None
        ):
            # relating pars to geostruct....
            if geostruct not in self.par_struct_dict.keys():
                # add new geostruct
                self.par_struct_dict[geostruct] = gp_dict
                # self.par_struct_dict_l[geostruct] = list(gp_dict.values())
            else:
                # append group to appropriate key associated with this geostruct
                # this is how pars setup with different calls are collected
                # so their correlations can be tracked
                for gp, gppars in gp_dict.items():
                    # if group not already set up
                    if gp not in self.par_struct_dict[geostruct].keys():
                        # update dict entry with new {key:par} pair
                        self.par_struct_dict[geostruct].update({gp: gppars})
                    else:
                        # if gp already assigned to this geostruct append par
                        # list to approprate group key
                        self.par_struct_dict[geostruct][gp].extend(gppars)
                # self.par_struct_dict_l[geostruct].extend(list(gp_dict.values()))
        else:  # TODO some rules for if geostruct is not passed....
            if "x" in df.columns:
                pass
                #  TODO warn that it looks like spatial pars but no geostruct?
            # if self.geostruct is not None:
            #     geostruct = self.geostruct
            # elif pp_geostruct is not None:
            #     geostruct = pp_geostruct
            # else:
            #     TODO - do we need an error or warning and define a default?
            #     options:
            # if spatial_reference is None:
            #     spatial_reference = self.spatial_reference  # TODO placeholder for now. but this needs improving, sr and self.sr might be None
            # dist = 10 * float(
            #             max(spatial_reference.delr.max(),
            #                 spatial_reference.delc.max()))
            # v = pyemu.geostats.ExpVario(contribution=1.0, a=dist)
            # geostruct = pyemu.geostats.GeoStruct(
            #     variograms=v)
            # temporal default:
            # v = pyemu.geostats.ExpVario(contribution=1.0, a=180.0)  # 180 correlation length
            # geostruct = pyemu.geostats.GeoStruct(
            #     variograms=v)

        self.logger.log(
            "adding {0} type {1} style parameters for file(s) {2}".format(
                par_type, par_style, [str(f) for f in filenames]
            )
        )

        if rebuild_pst:  # may want to just update pst and rebuild
            # (with new relations)
            if self.pst is not None:
                self.logger.log("Adding pars to control file " "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update="pars")
            else:
                self.build_pst(filename=self.pst.filename, update=False)
                self.logger.warn(
                    "pst object not available, " "new control file will be written"
                )

    def _load_listtype_file(
        self, filename, index_cols, use_cols, fmt=None, sep=None, skip=None, c_char=None
    ):
        if isinstance(filename, list):
            assert len(filename) == 1
            filename = filename[0]  # should only ever be one passed
        if isinstance(fmt, list):
            assert len(fmt) == 1
            fmt = fmt[0]
        if isinstance(sep, list):
            assert len(sep) == 1
            sep = sep[0]
        if isinstance(skip, list):
            assert len(skip) == 1
            skip = skip[0]

        # trying to use use_cols and index_cols to work out whether to
        # read header from csv.
        # either use_cols or index_cols could still be None
        # -- case of both being None should already have been caught
        # but index_cols could still be None...

        check_args = [a for a in [index_cols, use_cols] if a is not None]
        # `a` should be list if it is not None
        if all(isinstance(a[0], str) for a in check_args):
            # index_cols can be from header str
            header = 0  # will need to read a header
        elif all(isinstance(a[0], int) for a in check_args):
            # index_cols are column numbers in input file
            header = None
        else:
            if len(check_args) > 1:
                #  implies neither are None but they either both are not str,int
                #  or are different
                self.logger.lraise(
                    "unrecognized type for index_cols or use_cols "
                    "should be str or int and both should be of the "
                    "same type, not {0} or {1}".format(
                        *[str(type(a[0])) for a in check_args]
                    )
                )
            else:
                # implies not correct type
                self.logger.lraise(
                    "unrecognized type for either index_cols or use_cols "
                    "should be str or int, not {0}".format(type(check_args[0][0]))
                )

        # checking no overlap between index_cols and use_cols
        if len(check_args) > 1:
            si = set(index_cols)
            su = set(use_cols)

            i = si.intersection(su)
            if len(i) > 0:
                self.logger.lraise(
                    "use_cols also listed in " "index_cols: {0}".format(str(i))
                )

        file_path = self.new_d / filename
        if not os.path.exists(file_path):
            self.logger.lraise("par/obs filename '{0}' not found " "".format(file_path))
        self.logger.log("reading list {0}".format(file_path))
        if fmt.lower() == "free":
            if sep is None:
                sep = "\s+"
                if Path(filename).suffix == ".csv":
                    sep = ","
        else:
            # TODO support reading fixed-format
            #  (based on value of fmt passed)
            #  ... or not?
            self.logger.warn(
                "0) Only reading free format list par " "files currently supported."
            )
            self.logger.warn("1) Assuming safe to read as whitespace " "delim.")
            self.logger.warn("2) Desired format string will still " "be passed through")
            sep = "\s+"
        try:
            # read each input file
            if skip > 0 or c_char is not None:
                if c_char is None:
                    with open(file_path, "r") as fp:
                        storehead = {
                            lp: line for lp, line in enumerate(fp) if lp < skip
                        }
                else:
                    with open(file_path, "r") as fp:
                        storehead = {
                            lp: line
                            for lp, line in enumerate(fp)
                            if lp < skip or line.strip().startswith(c_char)
                        }
            else:
                storehead = {}
        except TypeError:
            c_char = skip
            skip = None
            with open(file_path, "r") as fp:
                storehead = {
                    lp: line
                    for lp, line in enumerate(fp)
                    if line.strip().startswith(c_char)
                }
        df = pd.read_csv(
            file_path,
            comment=c_char,
            sep=sep,
            skiprows=skip,
            header=header,
            low_memory=False,
        )
        self.logger.log("reading list {0}".format(file_path))
        # ensure that column ids from index_col is in input file
        missing = []
        for index_col in index_cols:
            if index_col not in df.columns:
                missing.append(index_col)
            # df.loc[:, index_col] = df.loc[:, index_col].astype(np.int) # TODO int? why?
        if len(missing) > 0:
            self.logger.lraise(
                "the following index_cols were not "
                "found in file '{0}':{1}"
                "".format(file_path, str(missing))
            )
        # ensure requested use_cols are in input file
        if use_cols is not None:
            for use_col in use_cols:
                if use_col not in df.columns:
                    missing.append(use_cols)
        if len(missing) > 0:
            self.logger.lraise(
                "the following use_cols were not found "
                "in file '{0}':{1}"
                "".format(file_path, str(missing))
            )
        return df, storehead, sep

    def _prep_arg_list_lengths(
        self,
        filenames,
        fmts=None,
        seps=None,
        skip_rows=None,
        index_cols=None,
        use_cols=None,
    ):
        """
        Private wrapper function to align filenames, formats, delimiters,
        reading options and setup columns for passing sequentially to
        load_listtype
        Args:
            filenames (`str`) or (`list`): names for files ot eventually read
            fmts (`str`) or (`list`): of column formaters for input file.
                If `None`, free-formatting is assumed
            seps (`str`) or (`list`): column separator free formatter files.
                If `None`, a list of `None`s is returned and the delimiter
                is eventually governed by the file extension (`,` for .csv)
            skip_rows (`str`) or (`list`): Number of rows in file header to not
                form part of the dataframe
            index_cols (`int`) or (`list`): Columns in tabular file to use as indicies
            use_cols (`int`) or (`list`): Columns in tabular file to
                use as par or obs cols
        Returns:
            algined lists of:
            filenames, fmts, seps, skip_rows, index_cols, use_cols
            for squentially passing to `_load_listtype_file()`

        """
        if not isinstance(filenames, list):
            filenames = [filenames]
        if fmts is None:
            fmts = ["free" for _ in filenames]
        if not isinstance(fmts, list):
            fmts = [fmts]
        if len(fmts) != len(filenames):
            self.logger.warn(
                "Discrepancy between number of filenames ({0}) "
                "and number of formatter strings ({1}). "
                "Will repeat first ({2})"
                "".format(len(filenames), len(fmts), fmts[0])
            )
            fmts = [fmts[0] for _ in filenames]
        fmts = ["free" if fmt is None else fmt for fmt in fmts]
        if seps is None:
            seps = [None for _ in filenames]
        if not isinstance(seps, list):
            seps = [seps]
        if len(seps) != len(filenames):
            self.logger.warn(
                "Discrepancy between number of filenames ({0}) "
                "and number of seps defined ({1}). "
                "Will repeat first ({2})"
                "".format(len(filenames), len(seps), seps[0])
            )
            seps = [seps[0] for _ in filenames]
        if skip_rows is None:
            skip_rows = [None for _ in filenames]
        if not isinstance(skip_rows, list):
            skip_rows = [skip_rows]
        if len(skip_rows) != len(filenames):
            self.logger.warn(
                "Discrepancy between number of filenames ({0}) "
                "and number of skip_rows defined ({1}). "
                "Will repeat first ({2})"
                "".format(len(filenames), len(skip_rows), skip_rows[0])
            )
            skip_rows = [skip_rows[0] for _ in filenames]
        skip_rows = [0 if s is None else s for s in skip_rows]

        if index_cols is None and use_cols is not None:
            self.logger.lraise(
                "index_cols is None, but use_cols is not ({0})" "".format(str(use_cols))
            )

        if index_cols is not None:
            if not isinstance(index_cols, list):
                index_cols = [index_cols]
        if use_cols is not None:
            if not isinstance(use_cols, list):
                use_cols = [use_cols]
        return filenames, fmts, seps, skip_rows, index_cols, use_cols


def write_list_tpl(
    filenames,
    dfs,
    name,
    tpl_filename,
    index_cols,
    par_type,
    use_cols=None,
    suffix="",
    zone_array=None,
    gpname=None,
    longnames=False,
    get_xy=None,
    ij_in_idx=None,
    xy_in_idx=None,
    zero_based=True,
    input_filename=None,
    par_style="multiplier",
    headerlines=None,
):
    """Write template files for a list style input.

    Args:
        filenames (`str` of `container` of `str`): original input filenames
        dfs (`pandas.DataFrame` or `container` of pandas.DataFrames): pandas
            representations of input file.
        name (`str` or container of str): parameter name prefixes.
            If more that one column to be parameterised, must be a container
            of strings providing the prefix for the parameters in the
            different columns.
        tpl_filename (`str`): Path (from current execution directory)
            for desired template file
        index_cols (`list`): column names to use as indices in tabular
            input dataframe
        par_type (`str`): 'constant','zone', or 'grid' used in parname
            generation. If `constant`, one par is set up for each `use_cols`.
            If `zone`, one par is set up for each zone for each `use_cols`.
            If `grid`, one par is set up for every unique index combination
            (from `index_cols`) for each `use_cols`.
        use_cols (`list`): Columns in tabular input file to paramerterise.
            If None, pars are set up for all columns apart from index cols.
        suffix (`str`): Optional par name suffix
        zone_array (`np.ndarray`): Array defining zone divisions.
            If not None and `par_type` is `grid` or `zone` it is expected that
            `index_cols` provide the indicies for
            querying `zone_array`. Therefore, array dimension should equal
            `len(index_cols)`.
        longnames (`boolean`): Specify is pars will be specified without the
            12 char restriction - recommended if using Pest++.
        get_xy (`pyemu.PstFrom` method): Can be specified to get real-world xy
            from `index_cols` passed (to assist correlation definition)
        ij_in_idx (`list` or `array`): defining which `index_cols` contain i,j
        xy_in_idx (`list` or `array`): defining which `index_cols` contain x,y
        zero_based (`boolean`): IMPORTANT - pass as False if `index_cols`
            are NOT zero-based indicies (e.g. MODFLOW row/cols).
            If False 1 with be subtracted from `index_cols`.
        input_filename (`str`): Path to input file (paired with tpl file)
        par_style (`str`): either 'direct' or 'multiplier'

    Returns:
        `pandas.DataFrame`: dataframe with info for the new parameters

    Note:
        This function is called by `PstFrom` programmatically

    """
    # get dataframe with autogenerated parnames based on `name`, `index_cols`,
    # `use_cols`, `suffix` and `par_type`
    if par_style == "direct":
        df_tpl = _write_direct_df_tpl(
            filenames[0],
            tpl_filename,
            dfs[0],
            name,
            index_cols,
            par_type,
            use_cols=use_cols,
            suffix=suffix,
            gpname=gpname,
            zone_array=zone_array,
            longnames=longnames,
            get_xy=get_xy,
            ij_in_idx=ij_in_idx,
            xy_in_idx=xy_in_idx,
            zero_based=zero_based,
            headerlines=headerlines,
        )
    else:
        df_tpl = _get_tpl_or_ins_df(
            dfs,
            name,
            index_cols,
            par_type,
            use_cols=use_cols,
            suffix=suffix,
            gpname=gpname,
            zone_array=zone_array,
            longnames=longnames,
            get_xy=get_xy,
            ij_in_idx=ij_in_idx,
            xy_in_idx=xy_in_idx,
            zero_based=zero_based,
        )

    for col in use_cols:  # corellations flagged using pargp
        df_tpl["covgp{0}".format(col)] = df_tpl.loc[:, "pargp{0}".format(col)].values
    # needs modifying if colocated pars in same group
    if par_type == "grid" and "x" in df_tpl.columns:
        if df_tpl.duplicated(["x", "y"]).any():
            # may need to use a different grouping for parameter correlations
            # where parameter x and y values are the same but pars are not
            # correlated (e.g. 2d correlation but different layers)
            # - this will only work if `index_cols` contains a third dimension.
            if len(index_cols) > 2:
                third_d = index_cols.copy()
                if xy_in_idx is not None:
                    for idx in xy_in_idx:
                        third_d.pop(idx)
                elif ij_in_idx is not None:
                    for idx in ij_in_idx:
                        third_d.pop(idx)
                else:  # if xy_in_idx and ij_in_idx ar None
                    # then parse_kij assumes that i is at idx[-2] and j at idx[-1]
                    third_d.pop()  # pops -1
                    third_d.pop()  # pops -2
                PyemuWarning(
                    "Coincidently located pars in list-like file, "
                    "attempting to separate pars based on `index_cols` "
                    "passed - using index_col[{0}] for third dimension"
                    "".format(third_d[-1])
                )
                for col in use_cols:
                    df_tpl["covgp{0}".format(col)] = df_tpl.loc[
                        :, "covgp{0}".format(col)
                    ].str.cat(
                        pd.DataFrame(df_tpl.sidx.to_list()).iloc[:, 0].astype(str),
                        "_cov",
                    )
            else:
                PyemuWarning(
                    "Coincidently located pars in list-like file. "
                    "Likely to cause issues building par cov or "
                    "drawing par ensemble. Can be resolved by passing "
                    "an additional `index_col` as a basis for "
                    "splitting colocated correlations (e.g. Layer)"
                )
    # pull out par details where multiple `use_cols` are requested
    parnme = list(df_tpl.loc[:, use_cols].values.flatten())
    pargp = list(
        df_tpl.loc[:, ["pargp{0}".format(col) for col in use_cols]].values.flatten()
    )

    covgp = list(
        df_tpl.loc[:, ["covgp{0}".format(col) for col in use_cols]].values.flatten()
    )
    df_tpl = df_tpl.drop(
        [col for col in df_tpl.columns if str(col).startswith("covgp")], axis=1
    )
    df_par = pd.DataFrame(
        {"parnme": parnme, "pargp": pargp, "covgp": covgp}, index=parnme
    )

    parval_cols = [c for c in df_tpl.columns if "parval1" in str(c)]
    parval = list(df_tpl.loc[:, [pc for pc in parval_cols]].values.flatten())

    if (
        par_type == "grid" and "x" in df_tpl.columns
    ):  # TODO work out if x,y needed for constant and zone pars too
        df_par["x"], df_par["y"] = np.concatenate(
            df_tpl.apply(lambda r: [[r.x, r.y] for _ in use_cols], axis=1).values
        ).T
    if not longnames:
        too_long = df_par.loc[df_par.parnme.apply(lambda x: len(x) > 12), "parnme"]
        if too_long.shape[0] > 0:
            raise Exception(
                "write_list_tpl() error: the following parameter "
                "names are too long:{0}"
                "".format(",".join(list(too_long)))
            )
    for use_col in use_cols:
        df_tpl.loc[:, use_col] = df_tpl.loc[:, use_col].apply(
            lambda x: "~  {0}  ~".format(x)
        )
    if par_style == "multiplier":
        pyemu.helpers._write_df_tpl(
            filename=tpl_filename, df=df_tpl, sep=",", tpl_marker="~"
        )

        if input_filename is not None:
            df_in = df_tpl.copy()
            df_in.loc[:, use_cols] = 1.0
            df_in.to_csv(input_filename)
    df_par.loc[:, "tpl_filename"] = tpl_filename
    df_par.loc[:, "input_filename"] = input_filename
    df_par.loc[:, "parval1"] = parval
    return df_par


def _write_direct_df_tpl(
    in_filename,
    tpl_filename,
    df,
    name,
    index_cols,
    typ,
    use_cols=None,
    suffix="",
    zone_array=None,
    longnames=False,
    get_xy=None,
    ij_in_idx=None,
    xy_in_idx=None,
    zero_based=True,
    gpname=None,
    headerlines=None,
):

    """
    Private method to auto-generate parameter or obs names from tabular
    model files (input or output) read into pandas dataframes
    Args:
        tpl_filename (`str` ): template filename
        df (`pandas.DataFrame`): DataFrame of list type input file
        name (`str`): Parameter name prefix
        index_cols (`str` or `list`): columns of dataframes to use as indicies
        typ (`str`): 'constant','zone', or 'grid' used in parname generation.
            If `constant`, one par is set up for each `use_cols`.
            If `zone`, one par is set up for each zone for each `use_cols`.
            If `grid`, one par is set up for every unique index combination
            (from `index_cols`) for each `use_cols`.
        use_cols (`list`): Columns to parameterise. If None, pars are set up
            for all columns apart from index cols
        suffix (`str`): Optional par name suffix.
        zone_array (`np.ndarray`): Array defining zone divisions.
            If not None and `par_type` is `grid` or `zone` it is expected that
            `index_cols` provide the indicies for querying `zone_array`.
            Therefore, array dimension should equal `len(index_cols)`.
        longnames (`boolean`): Specify is obs/pars will be specified without the
            20/12 char restriction - recommended if using Pest++.
        get_xy (`pyemu.PstFrom` method): Can be specified to get real-world xy
            from `index_cols` passed (to include in obs/par name)
        ij_in_idx (`list` or `array`): defining which `index_cols` contain i,j
        xy_in_idx (`list` or `array`): defining which `index_cols` contain x,y
        zero_based (`boolean`): IMPORTANT - pass as False if `index_cols`
            are NOT zero-based indicies (e.g. MODFLOW row/cols).
            If False 1 with be subtracted from `index_cols`.

    Returns:
        pandas.DataFrame with paranme and pargroup define for each `use_col`

    Note:
        This function is called by `PstFrom` programmatically

    """
    # TODO much of this duplicates what is in _get_tpl_or_ins_df() -- could posssibly be consolidated
    # work out the union of indices across all dfs

    sidx = []

    didx = df.loc[:, index_cols].apply(lambda x: tuple(x), axis=1)
    sidx.extend(didx)

    df_ti = pd.DataFrame({"sidx": sidx}, columns=["sidx"])
    # get some index strings for naming
    if longnames:
        j = "_"
        fmt = "{0}|{1}"
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["idx{0}".format(i) for i in range(len(index_cols))]
    else:
        fmt = "{1:3}"
        j = ""
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["{0}".format(i) for i in range(len(index_cols))]

    if not zero_based:
        df_ti.loc[:, "sidx"] = df_ti.sidx.apply(
            lambda x: tuple(xx - 1 if isinstance(xx, int) else xx for xx in x)
        )
    df_ti.loc[:, "idx_strs"] = df_ti.sidx.apply(
        lambda x: j.join([fmt.format(iname, xx) for xx, iname in zip(x, inames)])
    ).str.replace(" ", "")

    df_ti.loc[:, "idx_strs"] = df_ti.idx_strs.str.replace(":", "")
    df_ti.loc[:, "idx_strs"] = df_ti.idx_strs.str.replace("|", ":")

    if get_xy is not None:
        if xy_in_idx is not None:
            # x and y already in index cols
            df_ti[["x", "y"]] = pd.DataFrame(df_ti.sidx.to_list()).iloc[:, xy_in_idx]
        else:
            df_ti.loc[:, "xy"] = df_ti.sidx.apply(get_xy, ij_id=ij_in_idx)
            df_ti.loc[:, "x"] = df_ti.xy.apply(lambda x: x[0])
            df_ti.loc[:, "y"] = df_ti.xy.apply(lambda x: x[1])

    if use_cols is None:
        use_cols = [c for c in df_ti.columns if c not in index_cols]

    direct_tpl_df = df.copy()
    for iuc, use_col in enumerate(use_cols):
        if not isinstance(name, str):
            nname = name[iuc]
            # if zone type, find the zones for each index position
        else:
            nname = name
        if zone_array is not None and typ in ["zone", "grid"]:
            if zone_array.ndim != len(index_cols):
                raise Exception(
                    "get_tpl_or_ins_df() error: "
                    "zone_array.ndim "
                    "({0}) != len(index_cols)({1})"
                    "".format(zone_array.ndim, len(index_cols))
                )
            df_ti.loc[:, "zval"] = df_ti.sidx.apply(lambda x: zone_array[x])

        if gpname is None or gpname[iuc] is None:
            ngpname = nname
        else:
            if not isinstance(gpname, str):
                ngpname = gpname[iuc]
            else:
                ngpname = gpname
        df_ti.loc[:, "pargp{}".format(use_col)] = ngpname
        if typ == "constant":
            # one par for entire use_col column
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}_{2}".format(
                    nname, use_col, "direct"
                )
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

            _check_diff(df.loc[:, use_col].values, in_filename)
            df_ti.loc[:, "parval1_{0}".format(use_col)] = df.loc[:, use_col][0]

        elif typ == "zone":
            # one par for each zone
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}_{2}".format(
                    nname, use_col, "direct"
                )
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x)
                    )
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

            # todo check that values are constant within zones and
            #  assign parval1
            raise NotImplementedError(
                "list-based direct zone-type parameters not implemented"
            )

        elif typ == "grid":
            # one par for each index
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}_{2}".format(
                    nname, use_col, "direct"
                )
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x)
                    )
                df_ti.loc[:, use_col] += "_" + df_ti.idx_strs
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)

            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                df_ti.loc[:, use_col] += df_ti.idx_strs
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

            df_ti.loc[:, "parval1_{0}".format(use_col)] = df.loc[:, use_col].values

        else:
            raise Exception(
                "get_tpl_or_ins_df() error: "
                "unrecognized 'typ', if not 'obs', "
                "should be 'constant','zone', "
                "or 'grid', not '{0}'".format(typ)
            )

        if not longnames:
            if df_ti.loc[:, use_col].apply(lambda x: len(x)).max() > 12:
                too_long = df_ti.loc[:, use_col].apply(lambda x: len(x)) > 12
                print(too_long)
                raise ValueError("_write_direct_df_tpl(): couldnt form short par names")

        direct_tpl_df.loc[:, use_col] = (
            df_ti.loc[:, use_col].apply(lambda x: "~ {0} ~".format(x)).values
        )
    # nasty assumption to distiguish if we need to write headers
    if isinstance(direct_tpl_df.columns[0], str):
        header = True
    else:
        header = False
    pyemu.helpers._write_df_tpl(
        tpl_filename, direct_tpl_df, index=False, header=header, headerlines=headerlines
    )

    return df_ti


def _get_tpl_or_ins_df(
    dfs,
    name,
    index_cols,
    typ,
    use_cols=None,
    suffix="",
    zone_array=None,
    longnames=False,
    get_xy=None,
    ij_in_idx=None,
    xy_in_idx=None,
    zero_based=True,
    gpname=None,
):
    """
    Private method to auto-generate parameter or obs names from tabular
    model files (input or output) read into pandas dataframes
    Args:
        filenames (`str` or `list` of str`): filenames
        dfs (`pandas.DataFrame` or `list`): DataFrames (can be list of DataFrames)
            to set up parameters or observations
        name (`str`): Parameter name or Observation name prefix
        index_cols (`str` or `list`): columns of dataframes to use as indicies
        typ (`str`): 'obs' to set up observation names or,
            'constant','zone', or 'grid' used in parname generation.
            If `constant`, one par is set up for each `use_cols`.
            If `zone`, one par is set up for each zone for each `use_cols`.
            If `grid`, one par is set up for every unique index combination
            (from `index_cols`) for each `use_cols`.
        use_cols (`list`): Columns to parameterise. If None, pars are set up
            for all columns apart from index cols. Not used if `typ`==`obs`.
        suffix (`str`): Optional par name suffix. Not used if `typ`==`obs`.
        zone_array (`np.ndarray`): Only used for paremeters (`typ` != `obs`).
            Array defining zone divisions.
            If not None and `par_type` is `grid` or `zone` it is expected that
            `index_cols` provide the indicies for querying `zone_array`.
            Therefore, array dimension should equal `len(index_cols)`.
        longnames (`boolean`): Specify is obs/pars will be specified without the
            20/12 char restriction - recommended if using Pest++.
        get_xy (`pyemu.PstFrom` method): Can be specified to get real-world xy
            from `index_cols` passed (to include in obs/par name)
        ij_in_idx (`list` or `array`): defining which `index_cols` contain i,j
        xy_in_idx (`list` or `array`): defining which `index_cols` contain x,y
        zero_based (`boolean`): IMPORTANT - pass as False if `index_cols`
            are NOT zero-based indicies (e.g. MODFLOW row/cols).
            If False 1 with be subtracted from `index_cols`.=

    Returns:
        if `typ`==`obs`: pandas.DataFrame with index strings for setting up obs
        names when passing through to
        pyemu.pst_utils.csv_to_ins_file(df.set_index('idx_str')

        else: pandas.DataFrame with paranme and pargroup define for each `use_col`

    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    if not isinstance(dfs, list):
        dfs = list(dfs)

    # work out the union of indices across all dfs
    if typ != "obs":
        sidx = set()
        for df in dfs:
            didx = set(df.loc[:, index_cols].apply(lambda x: tuple(x), axis=1))
            sidx.update(didx)
    else:
        # order matters for obs
        sidx = []
        for df in dfs:
            didx = df.loc[:, index_cols].values
            aidx = [i for i in didx if i not in sidx]
            sidx.extend(aidx)

    df_ti = pd.DataFrame({"sidx": list(sidx)}, columns=["sidx"])
    # get some index strings for naming
    if longnames:
        j = "_"
        # fmt = "{0}|{1}"
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["idx{0}".format(i) for i in range(len(index_cols))]
        # full formatter string
        fmt = j.join([f"{iname}|{{{i}}}" for i, iname in enumerate(inames)])
    else:
        # fmt = "{1:3}"
        j = ""
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["{0}".format(i) for i in range(len(index_cols))]
        # full formatter string
        fmt = j.join([f"{{{i}:3}}" for i, iname in enumerate(inames)])
    if (
        not zero_based
    ):  # only if indices are ints (trying to support strings as par ids)
        df_ti.loc[:, "sidx"] = df_ti.sidx.apply(
            lambda x: tuple(xx - 1 if isinstance(xx, int) else xx for xx in x)
        )

    df_ti.loc[:, "idx_strs"] = df_ti.sidx.apply(lambda x: fmt.format(*x)).str.replace(
        " ", ""
    )
    df_ti.loc[:, "idx_strs"] = df_ti.idx_strs.str.replace(":", "", regex=False)
    df_ti.loc[:, "idx_strs"] = df_ti.idx_strs.str.replace("|", ":", regex=False)

    if get_xy is not None:
        if xy_in_idx is not None:
            # x and y already in index cols
            df_ti[["x", "y"]] = pd.DataFrame(df_ti.sidx.to_list()).iloc[:, xy_in_idx]
        else:
            df_ti.loc[:, "xy"] = df_ti.sidx.apply(get_xy, ij_id=ij_in_idx)
            df_ti.loc[:, "x"] = df_ti.xy.apply(lambda x: x[0])
            df_ti.loc[:, "y"] = df_ti.xy.apply(lambda x: x[1])

    if typ == "obs":
        return df_ti  #################### RETURN if OBS

    if use_cols is None:
        use_cols = [c for c in df_ti.columns if c not in index_cols]
        # if direct, we have more to deal with...
    for iuc, use_col in enumerate(use_cols):

        if not isinstance(name, str):
            nname = name[iuc]
            # if zone type, find the zones for each index position
        else:
            nname = name
        if zone_array is not None and typ in ["zone", "grid"]:
            if zone_array.ndim != len(index_cols):
                raise Exception(
                    "get_tpl_or_ins_df() error: "
                    "zone_array.ndim "
                    "({0}) != len(index_cols)({1})"
                    "".format(zone_array.ndim, len(index_cols))
                )
            df_ti.loc[:, "zval"] = df_ti.sidx.apply(lambda x: zone_array[x])

        if gpname is None or gpname[iuc] is None:
            ngpname = nname
        else:
            if not isinstance(gpname, str):
                ngpname = gpname[iuc]
            else:
                ngpname = gpname
        df_ti.loc[:, "pargp{}".format(use_col)] = ngpname
        df_ti.loc[:, "parval1_{0}".format(use_col)] = 1.0
        if typ == "constant":
            # one par for entire use_col column
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}".format(nname, use_col)
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

        elif typ == "zone":
            # one par for each zone
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}".format(nname, use_col)
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x)
                    )
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

        elif typ == "grid":
            # one par for each index
            if longnames:
                df_ti.loc[:, use_col] = "{0}_usecol:{1}".format(nname, use_col)
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x)
                    )
                df_ti.loc[:, use_col] += "_" + df_ti.idx_strs
                if suffix != "":
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)

            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                df_ti.loc[:, use_col] += df_ti.idx_strs
                if suffix != "":
                    df_ti.loc[:, use_col] += suffix

        else:
            raise Exception(
                "get_tpl_or_ins_df() error: "
                "unrecognized 'typ', if not 'obs', "
                "should be 'constant','zone', "
                "or 'grid', not '{0}'".format(typ)
            )

        if not longnames:
            if df_ti.loc[:, use_col].apply(lambda x: len(x)).max() > 12:
                too_long = df_ti.loc[:, use_col].apply(lambda x: len(x)) > 12
                print(too_long)
                raise ValueError("_get_tpl_or_ins_df(): couldnt form short par names")
    return df_ti


def write_array_tpl(
    name,
    tpl_filename,
    suffix,
    par_type,
    zone_array=None,
    gpname=None,
    shape=None,
    longnames=False,
    fill_value=1.0,
    get_xy=None,
    input_filename=None,
    par_style="multiplier",
):
    """
    write a template file for a 2D array.

     Args:
        name (`str`): the base parameter name
        tpl_filename (`str`): the template file to write - include path
        suffix (`str`): suffix to append to par names
        par_type (`str`): type of parameter
        zone_array (`numpy.ndarray`): an array used to skip inactive cells. Values less than 1 are
            not parameterized and are assigned a value of fill_value. Default is None.
        gpname (`str`): pargp filed in dataframe
        shape (`tuple`): dimensions of array to write
        longnames (`bool`): Use parnames > 12 char
        fill_value:
        get_xy:
        input_filename:
        par_style (`str`): either 'direct' or 'multiplier'

    Returns:
        df (`pandas.DataFrame`): a dataframe with parameter information

    Note:
        This function is called by `PstFrom` programmatically

    """

    if shape is None and zone_array is None:
        raise Exception(
            "write_array_tpl() error: must pass either zone_array " "or shape"
        )
    elif shape is not None and zone_array is not None:
        if shape != zone_array.shape:
            raise Exception(
                "write_array_tpl() error: passed "
                "shape {0} != zone_array.shape {1}".format(shape, zone_array.shape)
            )
    elif shape is None:
        shape = zone_array.shape
    if len(shape) != 2:
        raise Exception(
            "write_array_tpl() error: shape '{0}' not 2D" "".format(str(shape))
        )

    par_style = par_style.lower()
    if par_style == "direct":
        if not os.path.exists(input_filename):
            raise Exception(
                "write_grid_tpl() error: couldn't find input file "
                + " {0}, which is required for 'direct' par_style".format(
                    input_filename
                )
            )
        org_arr = np.loadtxt(input_filename, ndmin=2)
        if par_type == "grid":
            pass
        elif par_type == "constant":
            _check_diff(org_arr, input_filename)
        elif par_type == "zone":
            for zval in np.unique(zone_array):
                if zval < 1:
                    continue
                zone_org_arr = org_arr.copy()
                zone_org_arr[zone_array != zval] = np.NaN
                _check_diff(zone_org_arr, input_filename, zval)
    elif par_style == "multiplier":
        org_arr = np.ones(shape)
    else:
        raise Exception(
            "write_grid_tpl() error: unrecognized 'par_style' {0} ".format(par_style)
            + "should be 'direct' or 'multiplier'"
        )

    def constant_namer(i, j):
        if longnames:
            pname = "{0}_const_{1}".format(par_style, name)
            if suffix != "":
                pname += "_{0}".format(suffix)
        else:
            pname = "{1}{2}".format(par_style[0], name, suffix)
            if len(pname) > 12:
                raise Exception("constant par name too long:" "{0}".format(pname))
        return pname

    def zone_namer(i, j):
        zval = 1
        if zone_array is not None:
            zval = zone_array[i, j]
        if longnames:
            pname = "{0}_{1}_zone:{2}".format(par_style, name, zval)
            if suffix != "":
                pname += "_{0}".format(suffix)
        else:
            pname = "{1}_zn{2}".format(par_style[0], name, zval)
            if len(pname) > 12:
                raise Exception("zone par name too long:{0}".format(pname))
        return pname

    def grid_namer(i, j):
        if longnames:
            pname = "{0}_{1}_i:{2}_j:{3}".format(par_style, name, i, j)
            if get_xy is not None:
                pname += "_x:{0:0.2f}_y:{1:0.2f}".format(*get_xy([i, j]))
            if zone_array is not None:
                pname += "_zone:{0}".format(zone_array[i, j])
            if suffix != "":
                pname += "_{0}".format(suffix)
        else:
            pname = "{1}{2:03d}{3:03d}".format(par_style[0], name, i, j)
            if len(pname) > 12:
                raise Exception("grid pname too long:{0}".format(pname))
        return pname

    if par_type == "constant":
        namer = constant_namer
    elif par_type == "zone":
        namer = zone_namer
    elif par_type == "grid":
        namer = grid_namer
    else:
        raise Exception(
            "write_array_tpl() error: unsupported par_type"
            ", options are 'constant', 'zone', or 'grid', not"
            "'{0}'".format(par_type)
        )

    parnme = []
    org_par_val_dict = {}
    xx, yy, ii, jj = [], [], [], []
    with open(tpl_filename, "w") as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zone_array is not None and zone_array[i, j] < 1:
                    pname = " {0} ".format(fill_value)
                else:
                    if get_xy is not None:
                        x, y = get_xy([i, j])
                        xx.append(x)
                        yy.append(y)
                    ii.append(i)
                    jj.append(j)

                    pname = namer(i, j)
                    parnme.append(pname)
                    org_par_val_dict[pname] = org_arr[i, j]
                    pname = " ~   {0}    ~".format(pname)

                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    df.loc[:, "parval1"] = df.parnme.apply(lambda x: org_par_val_dict[x])
    if par_type == "grid":
        df.loc[:, "i"] = ii
        df.loc[:, "j"] = jj
        if get_xy is not None:
            df.loc[:, "x"] = xx
            df.loc[:, "y"] = yy
    if gpname is None:
        gpname = name
    df.loc[:, "pargp"] = "{0}_{1}_{2}".format(
        gpname, suffix.replace("_", ""), par_style
    ).rstrip("_")
    df.loc[:, "tpl_filename"] = tpl_filename
    df.loc[:, "input_filename"] = input_filename
    if input_filename is not None:
        arr = np.ones(shape)
        np.savetxt(input_filename, arr, fmt="%2.1f")

    return df


def _check_diff(org_arr, input_filename, zval=None):
    percent_diff = 100.0 * np.abs(
        (np.nanmax(org_arr) - np.nanmin(org_arr)) / np.nanmean(org_arr)
    )

    if percent_diff > DIRECT_PAR_PERCENT_DIFF_TOL:
        message = "_check_diff() error: direct par for file '{0}'".format(
            input_filename
        ) + "exceeds tolerance for percent difference: {0} > {1}".format(
            percent_diff, DIRECT_PAR_PERCENT_DIFF_TOL
        )
        if zval is not None:
            message += " in zone {0}".format(zval)
        raise Exception(message)


def get_filepath(folder, filename):
    """Return a path to a file within a folder,
    without repeating the folder in the output path,
    if the input filename (path) already contains the folder."""
    filename = Path(filename)
    folder = Path(folder)
    if folder not in filename.parents:
        filename = folder / filename
    return filename


def get_relative_filepath(folder, filename):
    """Like :func:`~pyemu.utils.pst_from.get_filepath`, except
    return path for filename relative to folder.
    """
    return get_filepath(folder, filename).relative_to(folder)
