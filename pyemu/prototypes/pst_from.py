
from __future__ import print_function, division
import os
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


def _get_datetime_from_str(sdt):
    # could be expanded if someone is feeling clever.
    if isinstance(sdt, str):
        PyemuWarning("Assuming passed reference start date time is "
                         "year first str {0}".format(sdt))
        sdt = pd.to_datetime(sdt, yearfirst=True)
    assert isinstance(sdt, pd.Timestamp), ("Error interpreting start_datetime")
    return sdt


# noinspection PyProtectedMember
class PstFrom(object):
    """

    Args:
        original_d:
        new_d:
        longnames:
        remove_existing:
        spatial_reference:
        zero_based:
        start_datetime:
    """
    # TODO auto ins setup from list style output (or array) and use_cols etc
    # TODO reals draw
    # TODO poss move/test some of the flopy/modflow specific setup apply
    #  methods to/in gw_utils. - save reinventing the setup/apply methods
    def __init__(self, original_d, new_d, longnames=True,
                 remove_existing=False, spatial_reference=None,
                 zero_based=True, start_datetime=None):  # TODO geostruct?

        self.original_d = original_d
        self.new_d = new_d
        self.original_file_d = None
        self.mult_file_d = None
        self.remove_existing = bool(remove_existing)
        self.zero_based = bool(zero_based)
        self._spatial_reference = spatial_reference
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

        self.initialize_spatial_reference()

        self._setup_dirs()
        self._parfile_relations = []
        self._pp_facs = {}
        self.pst = None

    @property
    def parfile_relations(self):
        if isinstance(self._parfile_relations, list):
            pr = pd.concat(self._parfile_relations,
                           ignore_index=True)
        else:
            pr = self._parfile_relations
        # quick checker
        for name, g in pr.groupby('model_file'):
            if g.sep.nunique() > 1:
                self.logger.warn(
                    "separator mismatch for {0}, seps passed {1}"
                    "".format(name, [s for s in g.sep.unique()]))
            if g.fmt.nunique() > 1:
                self.logger.warn(
                    "format mismatch for {0}, fmt passed {1}"
                    "".format(name, [f for f in g.fmt.unique()]))
            # if ultimate parameter bounds have been set for only one instance
            # of the model file we need to pass this through to all
            ubound = g.apply(
                lambda x: pd.Series(
                    {k: v for n, c in enumerate(x.use_cols)
                     for k, v in [['ubound{0}'.format(c), x.upper_bound[n]]]})
                if x.use_cols is not None
                else pd.Series(
                    {k: v for k, v in [['ubound', x.upper_bound]]}), axis=1)
            if ubound.nunique(0, False).gt(1).any():
                if ubound.nunique(0, False).gt(2).any():
                    # more than one upper bound set
                    self.logger.lraise(
                        "different upper bounds requested for same par for {0}"
                        "".format(name))
                else:
                    # one set - the rest are None - need to replace None
                    # with set values
                    # df with set values
                    fil = ubound.apply(lambda x:
                                       pd.Series([None]) if x.isna().all()
                                       else x[x.notna()].values).T
                    self.logger.warn("Upper bound for par passed for some but "
                                     "not all instances, will set NA to "
                                     "passed values\n{}".format(fil))
                    # replace Nones in list in Series with passed values
                    pr.loc[g.index, 'upper_bound'] = g.use_cols.apply(
                        lambda x: [fil[0].loc['ubound{0}'.format(c)] for c in x]
                        if x is not None else fil[0].loc['ubound'])
            # repeat for lower bounds
            lbound = g.apply(
                lambda x: pd.Series(
                    {k: v for n, c in enumerate(x.use_cols)
                     for k, v in [['lbound{0}'.format(c), x.lower_bound[n]]]})
                if x.use_cols is not None
                else pd.Series(
                    {k: v for k, v in [['lbound', x.lower_bound]]}), axis=1)
            if lbound.nunique(0, False).gt(1).any():
                if lbound.nunique(0, False).gt(2).any():
                    self.logger.lraise(
                        "different lower bounds requested for same par for {0}"
                        "".format(name))
                else:
                    fil = lbound.apply(lambda x:
                                       pd.Series([None]) if x.isna().all()
                                       else x[x.notna()].values).T
                    self.logger.warn("Lower bound for par passed for some but "
                                     "not all instances, will set NA to "
                                     "passed values\n{}".format(fil))
                    pr.loc[g.index, 'lower_bound'] = g.use_cols.apply(
                        lambda x: [fil[0].loc['lbound{0}'.format(c)] for c in x]
                        if x is not None else fil[0].loc['lbound'])
        pr['zero_based'] = self.zero_based
        return pr

    def _generic_get_xy(self, *args):
        if len(args) == 3:  # kij
            return float(args[1]), float(args[2])
        elif len(args) == 2:  # ij
            return float(args[0]), float(args[1])
        else:
            return 0.0, 0.0

    def _flopy_sr_get_xy(self, *args):
        i, j = self.parse_kij_args(*args)
        return (self._spatial_reference.xcentergrid[i, j],
                self._spatial_reference.ycentergrid[i, j])

    def _flopy_mg_get_xy(self, *args):
        i, j = self.parse_kij_args(*args)
        return (self._spatial_reference.xcellcenters[i, j],
                self._spatial_reference.ycellcenters[i, j])

    def parse_kij_args(self, *args):
        # TODO - deal with args not being ordered k,i,j
        #  perhaps support mapping e.g. {'i':, 'j':,...}
        if len(args) >= 2:  # kij
            i, j = args[-2], args[-1]

        #elif len(args) == 2:  # ij
        #    i, j = args[0], args[1]
        else:
            self.logger.lraise(("get_xy() error: wrong number of args, "
                                "should be 3 (kij) or 2 (ij)"
                                ", not '{0}'").format(str(args)))
        # if not self.zero_based:
        #     # TODO: check this,
        #     # I think index is already being adjusted in write_list_tpl()
        #     # and write_array_tpl() should always be zero_based (I think)
        #     i -= 1
        #     j -= 1
        return i, j

    def initialize_spatial_reference(self):
        if self._spatial_reference is None:
            self.get_xy = self._generic_get_xy
        elif (hasattr(self._spatial_reference, "xcentergrid") and
              hasattr(self._spatial_reference, "ycentergrid")):
            self.get_xy = self._flopy_sr_get_xy
        elif (hasattr(self._spatial_reference, "xcellcenters") and
              hasattr(self._spatial_reference, "ycellcenters")):
            # support modelgrid style cell locs
            self._spatial_reference.xcentergrid = self._spatial_reference.xcellcenters
            self._spatial_reference.xcentergrid = self._spatial_reference.xcellcenters
            self.get_xy = self._flopy_mg_get_xy
        else:
            self.logger.lraise("initialize_spatial_reference() error: "
                               "unsupported spatial_reference")
        self.spatial_reference = self._spatial_reference

    def write_forward_run(self):
        # update python commands with system style commands
        for alist, ilist in zip(
                [self.pre_py_cmds, self.mod_py_cmds, self.post_py_cmds],
                [self.pre_sys_cmds, self.mod_sys_cmds, self.post_sys_cmds]):
            if ilist is None:
                continue

            if not isinstance(ilist,list):
                ilist = [ilist]
            for cmd in ilist:
                self.logger.statement("forward_run line:{0}".format(cmd))
                alist.append("pyemu.os_utils.run(r'{0}')\n".format(cmd))

        with open(os.path.join(self.new_d, self.py_run_file),
                  'w') as f:
            f.write(
                "import os\nimport multiprocessing as mp\nimport numpy as np" + \
                "\nimport pandas as pd\n")
            f.write("import pyemu\n")
            f.write("def main():\n")
            f.write("\n")
            s = "    "
            for ex_imp in self.extra_py_imports:
                f.write(s + 'import {0}\n'.format(ex_imp))
            for tmp_file in self.tmp_files:
                f.write(s + "try:\n")
                f.write(s + "   os.remove('{0}')\n".format(tmp_file))
                f.write(s + "except Exception as e:\n")
                f.write(s + "   print('error removing tmp file:{0}')\n".format(
                    tmp_file))
            for line in self.pre_py_cmds:
                f.write(s + line + '\n')
            for line in self.mod_py_cmds:
                f.write(s + line + '\n')
            for line in self.post_py_cmds:
                f.write(s + line + '\n')
            f.write("\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    mp.freeze_support()\n    main()\n\n")

    def _pivot_par_struct_dict(self):
        struct_dict = {}
        for gs, gps in self.par_struct_dict.items():
            par_dfs = []
            for _, l in gps.items():
                df = pd.concat(l)
                if 'timedelta' in df.columns:
                    df.loc[:, "y"] = 0  #
                    df.loc[:, "x"] = df.timedelta.apply(lambda x: x.days)
                par_dfs.append(df)
            struct_dict[gs] = par_dfs
        return struct_dict
    
    def build_prior(self, fmt="ascii", filename=None, droptol=None, chunk=None,
                    sigma_range=6):
        """

        Args:
            fmt:
            filename:
            droptol:
            chunk:
            sigma_range:

        Returns:

        """
        struct_dict = self._pivot_par_struct_dict()
        self.logger.log("building prior covariance matrix")
        if len(struct_dict) > 0:
            cov = pyemu.helpers.geostatistical_prior_builder(self.pst,
                                                             struct_dict=struct_dict,
                                                             sigma_range=sigma_range)
        else:
            cov = pyemu.Cov.from_parameter_data(self.pst, sigma_range=sigma_range)

        if filename is None:
            filename = self.pst.filename.replace('.pst', ".prior.cov")
        if fmt != "none":
            self.logger.statement("saving prior covariance matrix to file {0}".format(filename))
        if fmt == 'ascii':
            cov.to_ascii(filename)
        elif fmt == 'binary':
            cov.to_binary(filename, droptol=droptol, chunk=chunk)
        elif fmt == 'uncfile':
            cov.to_uncfile(filename)
        elif fmt == 'coo':
            cov.to_coo(filename, droptol=droptol, chunk=chunk)
        self.logger.log("building prior covariance matrix")
        return cov

    def draw(self, num_reals=100, sigma_range=6, use_specsim=False,
             scale_offset=True):
        """

        Args:
            num_reals:
            sigma_range:
            use_specsim:
            scale_offset:

        Returns:

        """
        self.logger.log("drawing realizations")
        # precondition {geostruct:{group:df}} dict to {geostruct:[par_dfs]}
        struct_dict = self._pivot_par_struct_dict()
        # list for holding grid style groups
        gr_pe_l = []
        if use_specsim:
            if not pyemu.geostats.SpecSim2d.grid_is_regular(
                    self.spatial_reference.delr, self.spatial_reference.delc):
                self.logger.lraise("draw() error: can't use spectral simulation with irregular grid")
            self.logger.log("spectral simulation for grid-scale pars")
            # loop over geostructures defined in PestFrom object
            # (setup through add_parameters)
            for geostruct, par_df_l in struct_dict.items():
                par_df = pd.concat(par_df_l)  # force to single df
                grd_p = (par_df.partype == 'grid')  # grid par slicer
                # if there are grid pars (also grid pars with i,j info)
                if grd_p.sum() > 0 and 'i' in par_df.columns:
                    # select pars to use specsim for
                    gr_df = par_df.loc[grd_p & pd.notna(par_df.i)]
                    gr_df = gr_df.astype({'i': int, 'j': int})  # make sure int
                    # (won't be if there were nans in concatenated df)
                    if len(gr_df) > 0:
                        # get specsim object for geostruct
                        ss = pyemu.geostats.SpecSim2d(
                            delx=self.spatial_reference.delr,
                            dely=self.spatial_reference.delc,
                            geostruct=geostruct)
                        # specsim draw (returns df)
                        gr_pe1 = ss.grid_par_ensemble_helper(
                            pst=self.pst, gr_df=gr_df, num_reals=num_reals,
                            sigma_range=sigma_range, logger=self.logger)
                        gr_pe_l.append(gr_pe1)  # append to list
                        # drop these pars as already drawn
                        trimmed = []
                        for p_df in par_df_l:
                            if not p_df.index.isin(gr_df.index).all():
                                trimmed.append(p_df)
                        # redefine struct_dict entry to not include spec sim par
                        struct_dict[geostruct] = trimmed

            self.logger.log("spectral simulation for grid-scale pars")
        # draw remaining pars based on their geostruct
        pe = pyemu.helpers.geostatistical_draws(
            self.pst, struct_dict=struct_dict, num_reals=num_reals,
            sigma_range=sigma_range, scale_offset=scale_offset)._df
        if len(gr_pe_l) > 0:
            gr_par_pe = pd.concat(gr_pe_l, axis=1)
            pe.loc[:, gr_par_pe.columns] = gr_par_pe.values
        par_ens = pyemu.ParameterEnsemble(pst=self.pst, df=pe)
        self.logger.log("drawing realizations")
        return par_ens

    def build_pst(self, filename=None, update=False):
        """Build control file from i/o files in PstFrom object.
        Warning: This builds a pest control file from scratch
            - overwriting anything already in self.pst object and
            anything already writen to `filename`

        Args:
            filename (`str`): the filename to save the control file to.
                If None, the name is formed from the `PstFrom.original_d`
                --- the orginal directory name from which the forward model
                was extracted.  Default is None.
                The control file is saved in the `PstFrom.new_d` directory.
            update (bool) or (str): flag to add to existing Pst object and
                rewrite. If string {'pars', 'obs'} just update respective
                components of Pst. Default is False - build from PstFrom
                components.
        Note:
            This builds a pest control file from scratch
            - overwriting anything already in self.pst object and
            anything already writen to `filename`
        """
        par_data_cols = pyemu.pst_utils.pst_config["par_fieldnames"]
        obs_data_cols = pyemu.pst_utils.pst_config["obs_fieldnames"]
        if update:
            if self.pst is None:
                self.logger.warn("Can't update Pst object not initialised. "
                                 "Setting update to False")
                update = False
            else:
                if filename is None:
                    filename = os.path.join(self.new_d, self.pst.filename)
        else:
            if filename is None:
                filename = os.path.join(self.new_d, self.original_d)
        if os.path.dirname(filename) in ['', '.']:
            filename = os.path.join(self.new_d, filename)

        if update:
            pst = self.pst
            if update is True:
                update = {'pars': False, 'obs': False}
            elif isinstance(update, str):
                update = {str: True}
            elif isinstance(update, (set, list)):
                update = {s: True for s in update}
            uupdate = True
        else:
            update = {'pars': False, 'obs': False}
            uupdate = False
            pst = pyemu.Pst(filename, load=False)

        if 'pars' in update.keys() or not uupdate:
            if len(self.par_dfs) > 0:
                # parameter data from object
                par_data = pd.concat(self.par_dfs).loc[:, par_data_cols]
                # info relating parameter multiplier files to model input files
                parfile_relations = self.parfile_relations
                parfile_relations.to_csv(os.path.join(self.new_d,
                                                      'mult2model_info.csv'))
                if not any(["apply_list_and_array_pars" in s
                            for s in self.pre_py_cmds]):
                    self.pre_py_cmds.append(
                        "pyemu.helpers.apply_list_and_array_pars("
                        "arr_par_file='mult2model_info.csv')")
            else:
                par_data = pyemu.pst_utils._populate_dataframe(
                    [], pst.par_fieldnames, pst.par_defaults, pst.par_dtype)
            pst.parameter_data = par_data
            pst.template_files = self.tpl_filenames
            pst.input_files = self.input_filenames

        if 'obs' in update.keys() or not uupdate:
            if len(self.obs_dfs) > 0:
                obs_data = pd.concat(self.obs_dfs).loc[:, obs_data_cols]
            else:
                obs_data = pyemu.pst_utils._populate_dataframe(
                    [], pst.obs_fieldnames, pst.obs_defaults, pst.obs_dtype)
                obs_data.loc[:, "obsnme"] = []
                obs_data.index = []
            obs_data.sort_index(inplace=True)
            pst.observation_data = obs_data
            pst.instruction_files = self.ins_filenames
            pst.output_files = self.output_filenames
        if not uupdate:
            pst.model_command = self.mod_command

        pst.prior_information = pst.null_prior
        self.pst = pst
        self.pst.write(filename)
        self.write_forward_run()
        return pst

    def _setup_dirs(self):
        self.logger.log("setting up dirs")
        if not os.path.exists(self.original_d):
            self.logger.lraise("original_d '{0}' not found"
                               "".format(self.original_d))
        if not os.path.isdir(self.original_d):
            self.logger.lraise("original_d '{0}' is not a directory"
                               "".format(self.original_d))
        if os.path.exists(self.new_d):
            if self.remove_existing:
                self.logger.log("removing existing new_d '{0}'"
                                "".format(self.new_d))
                shutil.rmtree(self.new_d)
                time.sleep(0.0001)
                self.logger.log("removing existing new_d '{0}'"
                                "".format(self.new_d))
            else:
                self.logger.lraise("new_d '{0}' already exists "
                                   "- use remove_existing=True"
                                   "".format(self.new_d))

        self.logger.log("copying original_d '{0}' to new_d '{1}'"
                        "".format(self.original_d, self.new_d))
        shutil.copytree(self.original_d, self.new_d)
        self.logger.log("copying original_d '{0}' to new_d '{1}'"
                        "".format(self.original_d, self.new_d))

        self.original_file_d = os.path.join(self.new_d, "org")
        if os.path.exists(self.original_file_d):
            self.logger.lraise("'org' subdir already exists in new_d '{0}'"
                               "".format(self.new_d))
        os.makedirs(self.original_file_d)

        self.mult_file_d = os.path.join(self.new_d, "mult")
        if os.path.exists(self.mult_file_d):
            self.logger.lraise("'mult' subdir already exists in new_d '{0}'"
                               "".format(self.new_d))
        os.makedirs(self.mult_file_d)

        self.logger.log("setting up dirs")

    def _par_prep(self, filenames, index_cols, use_cols, fmts=None, seps=None,
                  skip_rows=None):

        # todo: cast str column names, index_cols and use_cols to lower if str?
        # todo: check that all index_cols and use_cols are the same type
        file_dict = {}
        fmt_dict = {}
        sep_dict = {}
        skip_dict = {}
        (filenames, fmts, seps, skip_rows,
         index_cols, use_cols) = self._prep_arg_list_lengths(
            filenames, fmts, seps, skip_rows, index_cols, use_cols)
        if index_cols is not None:
            for filename, sep, fmt, skip in zip(filenames, seps, fmts,
                                                skip_rows):
                df, storehead = self._load_listtype_file(
                    filename, index_cols, use_cols, fmt, sep, skip)
                file_path = os.path.join(self.new_d, filename)
                # # looping over model input filenames
                if fmt.lower() == 'free':
                    if sep is None:
                        sep = ' '
                        if filename.lower().endswith(".csv"):
                            sep = ','
                if df.columns.is_integer():
                    hheader = False
                else:
                    hheader = 0

                self.logger.statement("loaded list '{0}' of shape {1}"
                                      "".format(file_path, df.shape))
                # TODO BH: do we need to be careful of the format of the model
                #  files? -- probs not necessary for the version in
                #  original_file_d - but for the eventual product model file,
                #  it might be format sensitive - yuck
                # Update, BH: I think the `original files` saved can always
                # be comma delim --they are never directly used 
                # as model inputs-- as long as we pass the required model 
                # input file format (and sep), right?  
                # write orig version of input file to `org` (e.g.) dir

                if len(storehead) != 0:
                    kwargs = {}
                    if "win" in platform.platform().lower():
                        kwargs = {"line_terminator": "\n"}
                    with open(os.path.join(
                            self.original_file_d, filename, 'w')) as fp:
                        fp.write('\n'.join(storehead))
                        fp.flush()
                        df.to_csv(fp, sep=',', mode='a', header=hheader,
                                  **kwargs)
                else:
                    df.to_csv(os.path.join(self.original_file_d, filename),
                              index=False, sep=',', header=hheader)
                file_dict[filename] = df
                fmt_dict[filename] = fmt
                sep_dict[filename] = sep
                skip_dict[filename] = skip
                self.logger.log("loading list {0}".format(file_path))

            # check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1, len(fnames)):
                    if (file_dict[fnames[i]].shape[1] !=
                            file_dict[fnames[j]].shape[1]):
                        self.logger.lraise(
                            "shape mismatch for array types, '{0}' "
                            "shape {1} != '{2}' shape {3}".
                            format(fnames[i], file_dict[fnames[i]].shape[1],
                                   fnames[j], file_dict[fnames[j]].shape[1]))
        else:  # load array type files
            # loop over model input files
            for filename, sep, fmt, skip in zip(filenames, seps, fmts,
                                                skip_rows):
                if fmt.lower() == 'free':
                    if filename.lower().endswith(".csv"):
                        if sep is None:
                            sep = ','
                else:
                    # TODO - or not?
                    raise NotImplementedError("Only free format array "
                                              "par files currently supported")
                file_path = os.path.join(self.new_d, filename)
                self.logger.log("loading array {0}".format(file_path))
                if not os.path.exists(file_path):
                    self.logger.lraise("par filename '{0}' not found ".
                                       format(file_path))
                # read array type input file 
                arr = np.loadtxt(os.path.join(self.new_d, filename),
                                 delimiter=sep)
                self.logger.log("loading array {0}".format(file_path))
                self.logger.statement("loaded array '{0}' of shape {1}".
                                      format(filename, arr.shape))
                # save copy of input file to `org` dir
                np.savetxt(os.path.join(self.original_file_d, filename), arr)
                file_dict[filename] = arr
                fmt_dict[filename] = fmt
                sep_dict[filename] = sep
                skip_dict[filename] = skip
            # check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1, len(fnames)):
                    if (file_dict[fnames[i]].shape !=
                            file_dict[fnames[j]].shape):
                        self.logger.lraise(
                            "shape mismatch for array types, '{0}' "
                            "shape {1} != '{2}' shape {3}"
                            "".format(fnames[i], file_dict[fnames[i]].shape,
                                      fnames[j], file_dict[fnames[j]].shape))
        return index_cols, use_cols, file_dict, fmt_dict, sep_dict, skip_dict

    def _next_count(self,prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        else:
            self._prefix_count[prefix] += 1

        return self._prefix_count[prefix]

    def add_observations(self, filename, insfile=None,
                         index_cols=None, use_cols=None,
                         use_rows=None, prefix='', ofile_skip=None,
                         ofile_sep=None, rebuild_pst=False):
        """
        Add list style outputs as observation files to PstFrom object

        Args:
            filename (`str`): path to model output file name to set up
                as observations
            insfile (`str`): desired instructions file filename
            index_cols (`list`-like or `int`): columns to denote are indices for obs
            use_cols (`list`-like or `int`): columns to set up as obs
            use_rows (`list`-like or `int`): select only specific row of file for obs
            prefix (`str`): prefix for obsnmes
            ofile_skip (`int`): number of lines to skip in model output file
            ofile_sep (`str`): delimiter in output file
            rebuild_pst (`bool`): (Re)Construct PstFrom.pst object after adding
                new obs

        Returns: DataFrame of new observations

        """
        if insfile is None:
            insfile = "{0}.ins".format(filename)
        self.logger.log("adding observations from tabular output file")
        # precondition arguments
        (filenames, fmts, seps, skip_rows,
         index_cols, use_cols) = self._prep_arg_list_lengths(
            filename, index_cols=index_cols, use_cols=use_cols,
            fmts=None, seps=ofile_sep, skip_rows=ofile_skip)
        # load model output file
        df, storehead = self._load_listtype_file(
            filenames, index_cols, use_cols, fmts, seps, skip_rows)
        new_obs_l = []
        for filename, sep in zip(filenames, seps):  # should only ever be one but hey...
            self.logger.log("building insfile for tabular output file {0}"
                            "".format(filename))
            df_temp = _get_tpl_or_ins_df(df, prefix, typ='obs',
                                         index_cols=index_cols,
                                         use_cols=use_cols,
                                         longnames=self.longnames)
            df.loc[:, 'idx_str'] = df_temp.idx_strs
            if use_rows is not None:
                if isinstance(use_rows, str):
                    if use_rows not in df.idx_str:
                        self.logger.warn(
                            "can't find {0} in generated observation idx_str. "
                            "setting up obs for all rows instead"
                            "".format(use_rows))
                        use_rows = None
                elif isinstance(use_rows, int):
                    use_rows = [use_rows]
                use_rows = [r for r in use_rows if r <= len(df)]
                use_rows = df.iloc[use_rows].unique()
            # construct ins_file from df
            df_ins = pyemu.pst_utils.csv_to_ins_file(
                df.set_index('idx_str'),
                ins_filename=os.path.join(
                    self.new_d, insfile),
                only_cols=use_cols, only_rows=use_rows, marker='~',
                includes_header=True, includes_index=False, prefix=prefix,
                longnames=True, head_lines_len=len(storehead), sep=sep)
            self.logger.log("building insfile for tabular output file {0}"
                            "".format(filename))
            new_obs = self.add_observations_from_ins(
                ins_file=insfile, out_file=os.path.join(self.new_d, filename))
            new_obs_l.append(new_obs)
        new_obs = pd.concat(new_obs_l)
        # TODO obs group names
        self.logger.log("adding observations from tabular output file")
        if rebuild_pst:
            if self.pst is not None:
                self.logger.log("Adding pars to control file "
                                "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update='obs')
            else:
                self.build_pst(filename=self.pst.filename, update=False)
                self.logger.warn("pst object not available, "
                                 "new control file will be written")
        return new_obs

    def add_observations_from_ins(self, ins_file, out_file=None, pst_path=None,
                                  inschek=True):
        """ add new observations to a control file

         Args:
             ins_file (`str`): instruction file with exclusively new
                observation names
             out_file (`str`): model output file.  If None, then 
                ins_file.replace(".ins","") is used. Default is None
             pst_path (`str`): the path to append to the instruction file and 
                out file in the control file.  If not None, then any existing 
                path in front of the template or in file is split off and 
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

             pst = pyemu.Pst(os.path.join("template", "my.pst"))
             pst.add_observations(os.path.join("template","new_obs.dat.ins"), 
                                  pst_path=".")
             pst.write(os.path.join("template", "my_new.pst")

         """
        # lifted almost completely from `Pst().add_observation()`
        if os.path.dirname(ins_file) in ['', '.']:
            ins_file = os.path.join(self.new_d, ins_file)
            pst_path = '.'
        if not os.path.exists(ins_file):
            self.logger.lraise("ins file not found: {0}, {1}"
                               "".format(os.getcwd(), ins_file))
        if out_file is None:
            out_file = ins_file.replace(".ins", "")
        if ins_file == out_file:
            self.logger.lraise("ins_file == out_file, doh!")

        # get the parameter names in the template file
        self.logger.log(
            "adding observation from instruction file '{0}'".format(ins_file))
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
                "control file:{1}".
                format(ins_file, ','.join(sint)))

        # find "new" obs that are not already in the control file
        new_obsnme = sobsnme - sexist
        if len(new_obsnme) == 0:
            self.logger.lraise(
                "no new observations found in instruction file {0}".format(
                    ins_file))

        # extend observation_data
        new_obsnme = np.sort(list(new_obsnme))
        new_obs_data = pyemu.pst_utils._populate_dataframe(
            new_obsnme, pyemu.pst_utils.pst_config["obs_fieldnames"],
            pyemu.pst_utils.pst_config["obs_defaults"],
            pyemu.pst_utils.pst_config["obs_dtype"])
        new_obs_data.loc[new_obsnme, "obsnme"] = new_obsnme
        new_obs_data.index = new_obsnme
        # cwd = '.'
        if pst_path is not None:
            # cwd = os.path.join(*os.path.split(ins_file)[:-1])
            ins_file_pstrel = os.path.join(pst_path,
                                           os.path.split(ins_file)[-1])
            out_file_pstrel = os.path.join(pst_path,
                                           os.path.split(out_file)[-1])
        self.ins_filenames.append(ins_file_pstrel)
        self.output_filenames.append(out_file_pstrel)
        # add to temporary files to be removed at start of forward run
        self.tmp_files.append(out_file_pstrel)
        df = None
        if inschek:
            # df = pst_utils._try_run_inschek(ins_file,out_file,cwd=cwd)
            # ins_file = os.path.join(cwd, ins_file)
            # out_file = os.path.join(cwd, out_file)
            df = pyemu.pst_utils.try_process_output_file(ins_file=ins_file,
                                                         output_file=out_file)
        if df is not None:
            # print(self.observation_data.index,df.index)
            new_obs_data.loc[df.index, "obsval"] = df.obsval
        self.obs_dfs.append(new_obs_data)
        self.logger.log(
            "adding observation from instruction file '{0}'".format(ins_file))
        return new_obs_data

    def add_parameters(self, filenames, par_type, zone_array=None,
                       dist_type="gaussian", sigma_range=4.0,
                       upper_bound=1.0e10, lower_bound=1.0e-10,
                       transform="log", par_name_base="p", index_cols=None,
                       use_cols=None, pargp=None, pp_space=10,
                       use_pp_zones=False, num_eig_kl=100,
                       spatial_reference=None, geostruct=None,
                       datetime=None, mfile_fmt='free', mfile_skip=None,
                       ult_ubound=None, ult_lbound=None, rebuild_pst=False,
                       alt_inst_str='inst'):
        """
        Add list or array style model input files to PstFrom object.
        This method

        Args:
            filenames (`str`): Model input filenames to parameterize
            par_type (`str`): One of `grid` - for every element,
                `constant` - for single parameter applied to every element,
                `zone` - for zone-based parameterization (only for array-style) or
                `pilotpoint` - for pilot-point base parameterization of array
                    style input files.
                Note `kl` not yet implemented # TODO
            zone_array (`np.ndarray`): array defining spatial limits or zones
                for parameterization.
            dist_type: not yet implemented # TODO
            sigma_range: not yet implemented # TODO
            upper_bound (`float`): PEST parameter upper bound # TODO support different ubound,lbound,transform if multiple use_col
            lower_bound (`float`): PEST parameter lower bound
            transform (`str`): PEST parameter transformation
            par_name_base (`str`): basename for parameters that are set up
            index_cols (`list`-like): if not None, will attempt to parameterize
                expecting a tabular-style model input file. `index_cols`
                defines the unique columns used to set up pars
            use_cols (`list`-like or `int`): for tabular-style model input file,
                defines the columns to be parameterised
            pargp (`str`): Parameter group to assign pars to. This is PESTs
                pargp but is also used to gather correlated parameters set up
                using multiple `add_parameters()` calls (e.g. temporal pars)
                with common geostructs.
            pp_space (`int`): Spacing between pilot point parameters
            use_pp_zones (`bool`): a flag to use the greater-than-zero values
                in the zone_array as pilot point zones.
                If False, zone_array values greater than zero are treated as a
                single zone.  Default is False.
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
            mfile_skip (`int`): header in model input file to skip when reading
                and reapply when writing
            ult_ubound (`float`): Ultimate upper bound for model input
                parameter once all mults are applied - ensure physical model par vals
            ult_lbound (`float`): Ultimate lower bound for model input
                parameter once all mults are applied
            rebuild_pst (`bool`): (Re)Construct PstFrom.pst object after adding
                new parameters
            alt_inst_str (`str`): Alternative to default `inst` string in
                parameter names
        """
        # TODO need to support temporal pars?
        #  - As another partype using index_cols or an additional time_cols
        #  - for now, can support if passed from separate input files with datetime arg
        # Default par data columns used for pst
        par_data_cols = pyemu.pst_utils.pst_config["par_fieldnames"]
        self.logger.log("adding parameters for file(s) "
                        "{0}".format(str(filenames)))

        # Get useful variables from arguments passed
        (index_cols, use_cols, file_dict,
         fmt_dict, sep_dict, skip_dict) = self._par_prep(
            filenames, index_cols, use_cols, fmts=mfile_fmt,
            skip_rows=mfile_skip)
        if datetime is not None:  # convert and check datetime
            datetime = _get_datetime_from_str(datetime)
            if self.start_datetime is None:
                self.logger.warn("NO START_DATEIME PROVIDED, ASSUMING PAR "
                                 "DATETIME IS START {}".format(datetime))
                self.start_datetime = datetime
            assert datetime >= self.start_datetime, (
                "passed datetime is earlier than start_datetime {0}, {1}".
                    format(datetime, self.start_datetime))
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
            self.logger.lraise("par_name_base should be a string, "
                               "single-element container, or container of "
                               "len use_cols, not '{0}'"
                               "".format(str(par_name_base)))
        if self.longnames:  # allow par names to be long... fine for pestpp
            fmt = "_{0}".format(alt_inst_str) + ":{0}"
            chk_prefix = "_{0}".format(alt_inst_str)  # add `instance` identifier
        else:
            fmt = "{0}"  # may not be so well supported
            chk_prefix = ""
        # increment name base if already passed
        for i in range(len(par_name_base)):
            par_name_base[i] += fmt.format(
                self._next_count(par_name_base[i] + chk_prefix))
        # multiplier file name will be taken first par group, if passed
        # (the same multipliers will apply to all pars passed in this call)
        # Remove `:` for filenames
        par_name_store = par_name_base[0].replace(':', '')  # for os filename

        # Define requisite filenames
        mlt_filename = "{0}_{1}.csv".format(par_name_store, par_type)
        # pst input file (for tpl->in pair) is multfile (in mult dir)
        in_filepst = os.path.relpath(os.path.join(
            self.mult_file_d, mlt_filename), self.new_d)
        tpl_filename = mlt_filename + ".tpl"
        pp_filename = None  # setup placeholder variables
        fac_filename = None

        def _check_var_len(var, n, fill=None):
            if not isinstance(var, list):
                var = [var]
            if fill is not None:
                if fill == 'first':
                    fill = var[0]
                elif fill == 'last':
                    fill = var[-1]
            nv = len(var)
            if nv < n:
                var.extend([fill for _ in range(n-nv)])
            return var

        # Process model parameter files to produce appropriate pest pars
        if index_cols is not None:  # Assume list/tabular type input files
            # ensure inputs are provided for all required cols
            ncol = len(use_cols)
            ult_lbound = _check_var_len(ult_lbound, ncol)
            ult_ubound = _check_var_len(ult_ubound, ncol)
            pargp = _check_var_len(pargp, ncol)
            lower_bound = _check_var_len(lower_bound, ncol, fill='first')
            upper_bound = _check_var_len(upper_bound, ncol, fill='first')
            transform = _check_var_len(transform, ncol, fill='first')
            if len(use_cols) != len(ult_lbound) != len(ult_ubound):
                self.logger.lraise("mismatch in number of columns to use {0} "
                                   "and number of ultimate lower {0} or upper "
                                   "{1} par bounds defined"
                                   "".format(len(use_cols), len(ult_lbound),
                                             len(ult_ubound)))

            self.logger.log(
                "writing list-based template file '{0}'".format(tpl_filename))
            # Generate tabular type template - also returns par data
            df = write_list_tpl(
                file_dict.values(), par_name_base,
                tpl_filename=os.path.join(self.new_d, tpl_filename),
                par_type=par_type, suffix='', index_cols=index_cols,
                use_cols=use_cols, zone_array=zone_array, gpname=pargp,
                longnames=self.longnames, get_xy=self.get_xy,
                zero_based=self.zero_based,
                input_filename=os.path.join(self.mult_file_d, mlt_filename))
            assert np.mod(len(df), len(use_cols)) == 0., (
                "Parameter dataframe wrong shape for number of cols {0}"
                "".format(use_cols))
            # variables need to be passed to each row in df
            lower_bound = np.tile(lower_bound, int(len(df)/ncol))
            upper_bound = np.tile(upper_bound, int(len(df)/ncol))
            transform = np.tile(transform, int(len(df)/ncol))
            self.logger.log(
                "writing list-based template file '{0}'".format(tpl_filename))
        else:  # Assume array type parameter file
            self.logger.log(
                "writing array-based template file '{0}'".format(tpl_filename))
            shp = file_dict[list(file_dict.keys())[0]].shape
            # ARRAY constant, zones or grid (cell-by-cell)
            if par_type in {"constant", "zone", "grid"}:
                self.logger.log(
                    "writing template file "
                    "{0} for {1}".format(tpl_filename, par_name_base))
                # Generate array type template - also returns par data
                df = write_array_tpl(
                    name=par_name_base[0],
                    tpl_filename=os.path.join(self.new_d, tpl_filename),
                    suffix='', par_type=par_type, zone_array=zone_array,
                    shape=shp, longnames=self.longnames, get_xy=self.get_xy,
                    fill_value=1.0, gpname=pargp,
                    input_filename=os.path.join(self.mult_file_d,
                                                mlt_filename))
                self.logger.log(
                    "writing template file"
                    " {0} for {1}".format(tpl_filename, par_name_base))
            # ARRAY PILOTPOINT setup
            elif par_type in {"pilotpoints", "pilot_points",
                              "pilotpoint", "pilot_point",
                              "pilot-point", "pilot-points"}:
                # Setup pilotpoints for array type par files
                self.logger.log("setting up pilot point parameters")
                # finding spatial references for for setting up pilot points
                if spatial_reference is None:
                    # if none passed with add_pars call
                    self.logger.statement("No spatial reference "
                                          "(containing cell spacing) passed.")
                    if self.spatial_reference is not None:
                        # using global sr on PestFrom object
                        self.logger.statement("OK - using spatial reference "
                                              "in parent object.")
                        spatial_reference = self.spatial_reference
                    else:
                        # uhoh
                        self.logger.lraise("No spatial reference in parent "
                                           "object either. "
                                           "Can't set-up pilotpoints")
                # (stolen from helpers.PstFromFlopyModel()._pp_prep())
                # but only settting up one set of pps at a time
                pp_dict = {0: par_name_base}
                pp_filename = "{0}pp.dat".format(par_name_store)
                # pst inputfile (for tpl->in pair) is
                # par_name_storepp.dat table (in pst ws)
                in_filepst = pp_filename
                tpl_filename = pp_filename + ".tpl"
                if pp_space is None:  # default spacing if not passed
                    self.logger.warn("pp_space is None, using 10...\n")
                    pp_space = 10
                if geostruct is None:  # need a geostruct for pilotpoints
                    # can use model default, if provided
                    if self.geostruct is None:  # but if no geostruct passed...
                        self.logger.warn("pp_geostruct is None,"
                                         "using ExpVario with contribution=1 "
                                         "and a=(pp_space*max(delr,delc))")
                        # set up a default
                        pp_dist = pp_space * float(
                            max(spatial_reference.delr.max(),
                                spatial_reference.delc.max()))
                        v = pyemu.geostats.ExpVario(contribution=1.0, a=pp_dist)
                        pp_geostruct = pyemu.geostats.GeoStruct(
                            variograms=v, name="pp_geostruct", transform="log")
                    else:
                        pp_geostruct = self.geostruct
                else:
                    pp_geostruct = geostruct
                # Set up pilot points
                df = pyemu.pp_utils.setup_pilotpoints_grid(
                    sr=spatial_reference,
                    ibound=zone_array,
                    use_ibound_zones=use_pp_zones,
                    prefix_dict=pp_dict,
                    every_n_cell=pp_space,
                    pp_dir=self.new_d,
                    tpl_dir=self.new_d,
                    shapename=os.path.join(
                        self.new_d, "{0}.shp".format(par_name_store)),
                    longnames=self.longnames)
                df.set_index('parnme', drop=False, inplace=True)
                # df includes most of the par info for par_dfs and also for
                # relate_parfiles
                self.logger.statement("{0} pilot point parameters created".
                                      format(df.shape[0]))
                # should be only one group at a time
                pargp = df.pargp.unique()
                self.logger.statement("pilot point 'pargp':{0}".
                                      format(','.join(pargp)))
                self.logger.log("setting up pilot point parameters")

                # Calculating pp factors
                pg = pargp[0]
                # this reletvively quick
                ok_pp = pyemu.geostats.OrdinaryKrige(pp_geostruct, df)
                # build krige reference information on the fly - used to help
                # prevent unnecessary krig factor calculation
                pp_info_dict = {
                    'pp_data': ok_pp.point_data.loc[:, ['x', 'y', 'zone']],
                    'cov': ok_pp.point_cov_df,
                    'zn_ar': zone_array}
                fac_processed = False
                for facfile, info in self._pp_facs.items():  # check against
                    # factors already calculated
                    if (info['pp_data'].equals(pp_info_dict['pp_data']) and
                            info['cov'].equals(pp_info_dict['cov']) and
                            np.array_equal(info['zn_ar'],
                                           pp_info_dict['zn_ar'])):
                        fac_processed = True  # don't need to re-calc same factors
                        fac_filename = facfile  # relate to existing fac file
                        break
                if not fac_processed:
                    # TODO need better way of naming squential fac_files?
                    self.logger.log(
                        "calculating factors for pargp={0}".format(pg))
                    fac_filename = os.path.join(
                        self.new_d, "{0}pp.fac".format(par_name_store))
                    var_filename = fac_filename.replace(".fac", ".var.dat")
                    self.logger.statement(
                        "saving krige variance file:{0}".format(var_filename))
                    self.logger.statement(
                        "saving krige factors file:{0}".format(fac_filename))
                    # store info on pilotpoints
                    self._pp_facs[fac_filename] = pp_info_dict
                    # this is slow (esp on windows) so only want to do this
                    # when required
                    ok_pp.calc_factors_grid(spatial_reference,
                                            var_filename=var_filename,
                                            zone_array=zone_array,
                                            num_threads=10)
                    ok_pp.to_grid_factors_file(fac_filename)
                    self.logger.log(
                        "calculating factors for pargp={0}".format(pg))
            # TODO - other par types - JTW?
            elif par_type == "kl":
                self.logger.lraise("array type 'kl' not implemented")
            else:
                self.logger.lraise("unrecognized 'par_type': '{0}', "
                                   "should be in "
                                   "['constant','zone','grid','pilotpoints',"
                                   "'kl'")
            self.logger.log("writing array-based template file "
                            "'{0}'".format(tpl_filename))

        if datetime is not None:
            # add time info to par_dfs
            df['datetime'] = datetime
            df['timedelta'] = t_offest
        # accumulate information that relates mult_files (set-up here and
        # eventually filled by PEST) to actual model files so that actual
        # model input file can be generated
        # (using helpers.apply_list_and_array_pars())
        relate_parfiles = []
        for mod_file in file_dict.keys():
            mult_dict = {
                "org_file": os.path.join(
                    *os.path.split(self.original_file_d)[1:], mod_file),
                "mlt_file": os.path.join(
                    *os.path.split(self.mult_file_d)[1:], mlt_filename),
                "model_file": mod_file,
                "use_cols": use_cols,
                "index_cols": index_cols,
                "fmt": fmt_dict[mod_file],
                "sep": sep_dict[mod_file],
                "head_rows": skip_dict[mod_file],
                "upper_bound": ult_ubound,
                "lower_bound": ult_lbound}
            if pp_filename is not None:
                # if pilotpoint need to store more info
                assert fac_filename is not None, (
                    "missing pilot-point input filename")
                mult_dict["fac_file"] = os.path.relpath(fac_filename,
                                                        self.new_d)
                mult_dict['pp_file'] = pp_filename
            relate_parfiles.append(mult_dict)
        relate_pars_df = pd.DataFrame(relate_parfiles)
        # store on self for use in pest build etc
        self._parfile_relations.append(relate_pars_df)

        # add cols required for pst.parameter_data
        df.loc[:, "partype"] = par_type
        df.loc[:, "partrans"] = transform
        df.loc[:, "parubnd"] = upper_bound
        df.loc[:, "parlbnd"] = lower_bound
        #df.loc[:,"tpl_filename"] = tpl_filename

        # store tpl --> in filename pair
        self.tpl_filenames.append(tpl_filename)
        self.input_filenames.append(in_filepst)
        for file_name in file_dict.keys():
            # store mult --> original file pairs
            self.org_files.append(file_name)
            self.mult_files.append(mlt_filename)

        # add pars to par_data list BH: is this what we want?
        # - BH: think we can get away with dropping duplicates?
        missing = set(par_data_cols) - set(df.columns)
        for field in missing:  # fill missing pst.parameter_data cols with defaults
            df[field] = pyemu.pst_utils.pst_config['par_defaults'][field]
        df = df.drop_duplicates()  # drop pars that appear multiple times
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
        gp_dict = {g: [d] for g, d in df.groupby('pargp')}
        # df_list = [d for g, d in df.groupby('pargp')]
        if geostruct is not None:
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
                        # if pargp already assigned to this geostruct append par
                        # list to approprate group key
                        self.par_struct_dict[geostruct][gp].extend(gppars)
                # self.par_struct_dict_l[geostruct].extend(list(gp_dict.values()))
        else:  # TODO some rules for if geostruct is not passed....
            if 'x' in df.columns:
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

        self.logger.log("adding parameters for file(s) "
                        "{0}".format(str(filenames)))

        if rebuild_pst:  # may want to just update pst and rebuild
            # (with new relations)
            if self.pst is not None:
                self.logger.log("Adding pars to control file "
                                "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update='pars')
            else:
                self.build_pst(filename=self.pst.filename, update=False)
                self.logger.warn("pst object not available, "
                                 "new control file will be written")

    def _load_listtype_file(self, filename, index_cols, use_cols,
                            fmt=None, sep=None, skip=None):
        if isinstance(filename, list):
            assert len(filename) == 1
            filename = filename[0]
        if isinstance(fmt, list):
            assert len(fmt) == 1
            fmt = fmt[0]
        if isinstance(sep, list):
            assert len(sep) == 1
            sep = sep[0]
        if isinstance(skip, list):
            assert len(skip) == 1
            skip = skip[0]
        if isinstance(index_cols[0], str) and isinstance(use_cols[0], str):
            # index_cols can be from header str
            header = 0  # will need to read a header
        elif isinstance(index_cols[0], int) and isinstance(use_cols[0], int):
            # index_cols are column numbers in input file
            header = None
        else:
            self.logger.lraise("unrecognized type for index_cols or use_cols "
                               "should be str or int and both should be of the "
                               "same type, not {0} or {1}".
                               format(str(type(index_cols)),
                                      str(type(use_cols))))
        itype = type(index_cols)
        utype = type(use_cols)
        if itype != utype:
            self.logger.lraise("index_cols type '{0} != use_cols "
                               "type '{1}'".
                               format(str(itype), str(utype)))

        si = set(index_cols)
        su = set(use_cols)

        i = si.intersection(su)
        if len(i) > 0:
            self.logger.lraise("use_cols also listed in "
                               "index_cols: {0}".format(str(i)))

        file_path = os.path.join(self.new_d, filename)
        if not os.path.exists(file_path):
            self.logger.lraise("par filename '{0}' not found "
                               "".format(file_path))
        self.logger.log("reading list {0}".format(file_path))
        if fmt.lower() == 'free':
            if sep is None:
                sep = "\s+"
                if filename.lower().endswith(".csv"):
                    sep = ','
        else:
            # TODO support reading fixed-format
            #  (based on value of fmt passed)
            #  ... or not?
            self.logger.warn("0) Only reading free format list par "
                             "files currently supported.")
            self.logger.warn("1) Assuming safe to read as whitespace "
                             "delim.")
            self.logger.warn("2) Desired format string will still "
                             "be passed through")
            sep = '\s+'
        # read each input file
        if skip > 0:
            with open(file_path, 'r') as fp:
                storehead = [next(fp) for _ in range(skip)]
        else:
            storehead = []
        df = pd.read_csv(file_path, header=header, skiprows=skip, sep=sep)
        self.logger.log("reading list {0}".format(file_path))
        # ensure that column ids from index_col is in input file
        missing = []
        for index_col in index_cols:
            if index_col not in df.columns:
                missing.append(index_col)
            # df.loc[:, index_col] = df.loc[:, index_col].astype(np.int) # TODO int? why?
        if len(missing) > 0:
            self.logger.lraise("the following index_cols were not "
                               "found in file '{0}':{1}"
                               "".format(file_path, str(missing)))
        # ensure requested use_cols are in input file
        for use_col in use_cols:
            if use_col not in df.columns:
                missing.append(use_cols)
        if len(missing) > 0:
            self.logger.lraise("the following use_cols were not found "
                               "in file '{0}':{1}"
                               "".format(file_path, str(missing)))

        return df, storehead

    def _prep_arg_list_lengths(self, filenames, fmts=None, seps=None,
                               skip_rows=None, index_cols=None, use_cols=None):
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
            fmts = ['free' for _ in filenames]
        if not isinstance(fmts, list):
            fmts = [fmts]
        if len(fmts) != len(filenames):
            self.logger.warn("Discrepancy between number of filenames ({0}) "
                             "and number of formatter strings ({1}). "
                             "Will repeat first ({2})"
                             "".format(len(filenames), len(fmts), fmts[0]))
            fmts = [fmts[0] for _ in filenames]
        fmts = ['free' if fmt is None else fmt for fmt in fmts]
        if seps is None:
            seps = [None for _ in filenames]
        if not isinstance(seps, list):
            seps = [seps]
        if len(seps) != len(filenames):
            self.logger.warn("Discrepancy between number of filenames ({0}) "
                             "and number of seps defined ({1}). "
                             "Will repeat first ({2})"
                             "".format(len(filenames), len(seps), seps[0]))
            seps = [seps[0] for _ in filenames]
        if skip_rows is None:
            skip_rows = [None for _ in filenames]
        if not isinstance(skip_rows, list):
            skip_rows = [skip_rows]
        if len(skip_rows) != len(filenames):
            self.logger.warn("Discrepancy between number of filenames ({0}) "
                             "and number of skip_rows defined ({1}). "
                             "Will repeat first ({2})"
                             "".format(len(filenames), len(skip_rows),
                                       skip_rows[0]))
            skip_rows = [skip_rows[0] for _ in filenames]
        skip_rows = [0 if s is None else s for s in skip_rows]

        if index_cols is None and use_cols is not None:
            self.logger.lraise("index_cols is None, but use_cols is not ({0})"
                               "".format(str(use_cols)))

        if index_cols is not None:
            if not isinstance(index_cols, list):
                index_cols = [index_cols]
            if not isinstance(use_cols, list):
                use_cols = [use_cols]
        return filenames, fmts, seps, skip_rows, index_cols, use_cols


def write_list_tpl(dfs, name, tpl_filename, index_cols, par_type,
                   use_cols=None, suffix='', zone_array=None, gpname=None,
                   longnames=False, get_xy=None, zero_based=True,
                   input_filename=None):
    """ Write template files for a list style input.

    Args:
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
            from `index_cols` passed (to include in par name)
        zero_based (`boolean`): IMPORTANT - pass as False if `index_cols`
            are NOT zero-based indicies (e.g. MODFLOW row/cols).
            If False 1 with be subtracted from `index_cols`.
        input_filename (`str`): Path to input file (paired with tpl file)

    Returns:

    """
    # get dataframe with autogenerated parnames based on `name`, `indx_cols`,
    # `use_cols`, `suffix` and `par_type`
    df_tpl = _get_tpl_or_ins_df(dfs, name, index_cols, par_type,
                                use_cols=use_cols, suffix=suffix, gpname=gpname,
                                zone_array=zone_array, longnames=longnames,
                                get_xy=get_xy, zero_based=zero_based)
    parnme = list(df_tpl.loc[:, use_cols].values.flatten())
    pargp = list(
        df_tpl.loc[:, ["pargp{0}".format(col)
                       for col in use_cols]].values.flatten())
    df_par = pd.DataFrame({"parnme": parnme, "pargp": pargp}, index=parnme)
    if par_type == 'grid' and 'x' in df_tpl.columns:  # TODO work out if x,y needed for constant and zone pars too
        df_par['x'], df_par['y'] = np.concatenate(
            df_tpl.apply(lambda r: [[r.x, r.y] for _ in use_cols],
                         axis=1).values).T
    if not longnames:
        too_long = df_par.loc[df_par.parnme.apply(lambda x: len(x) > 12),
                              "parnme"]
        if too_long.shape[0] > 0:
            raise Exception("write_list_tpl() error: the following parameter "
                            "names are too long:{0}"
                            "".format(','.join(list(too_long))))
    for use_col in use_cols:
        df_tpl.loc[:, use_col] = df_tpl.loc[:, use_col].apply(
            lambda x: "~  {0}  ~".format(x))
    pyemu.helpers._write_df_tpl(filename=tpl_filename, df=df_tpl, sep=',',
                                tpl_marker='~')

    if input_filename is not None:
        df_in = df_tpl.copy()
        df_in.loc[:, use_cols] = 1.0
        df_in.to_csv(input_filename)
    df_par.loc[:, "tpl_filename"] = tpl_filename
    df_par.loc[:, "input_filename"] = input_filename
    return df_par


def _get_tpl_or_ins_df(dfs, name, index_cols, typ, use_cols=None,
                       suffix='', zone_array=None, longnames=False, get_xy=None,
                       zero_based=True, gpname=None):
    """
    Private method to auto-generate parameter or obs names from tabular
    model files (input or output) read into pandas dataframes
    Args:
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
        zero_based (`boolean`): IMPORTANT - pass as False if `index_cols`
            are NOT zero-based indicies (e.g. MODFLOW row/cols).
            If False 1 with be subtracted from `index_cols`.

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
    if typ != 'obs':
        sidx = set()
        for df in dfs:
            didx = set(df.loc[:, index_cols].apply(
                lambda x: tuple(x), axis=1))
            sidx.update(didx)
    else:
        # order matters for obs
        sidx = []
        for df in dfs:
            didx = df.loc[:, index_cols].apply(
                lambda x: tuple(x), axis=1).values
            aidx = [i for i in didx if i not in sidx]
            sidx.extend(aidx)

    df_ti = pd.DataFrame({"sidx": list(sidx)}, columns=["sidx"])
    # get some index strings for naming
    if longnames:
        j = '_'
        fmt = "{0}:{1}"
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["idx{0}".format(i) for i in range(len(index_cols))]
    else:
        fmt = "{1:3}"
        j = ''
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["{0}".format(i) for i in range(len(index_cols))]

    if not zero_based:
        # TODO: need to be careful here potential to have two
        #  conflicting/compounding `zero_based` actions
        #  by default we pass PestFrom zero_based object to this method
        #  so if not zero_based will subtract 1 from idx here...
        #  ----the get_xy method also -= 1 (checkout changes to get_xy())
        df_ti.loc[:, "sidx"] = df_ti.sidx.apply(
            lambda x: tuple(xx - 1 for xx in x))
    df_ti.loc[:, "idx_strs"] = df_ti.sidx.apply(
        lambda x: j.join([fmt.format(iname, xx)
                          for xx, iname in zip(x, inames)])).str.replace(' ', '')

    if get_xy is not None:
        # TODO need to be more flexible with index_cols
        #   cant just assume index_cols will be k,i,j (if 3) and i,j (if 2)
        df_ti.loc[:, 'xy'] = df_ti.sidx.apply(lambda x: get_xy(*x))
        df_ti.loc[:, 'x'] = df_ti.xy.apply(lambda x: x[0])
        df_ti.loc[:, 'y'] = df_ti.xy.apply(lambda x: x[1])

    if typ == 'obs':
        return df_ti  #################### RETURN if OBS
    # else
    # use all non-index columns if use_cols not passed
    if use_cols is None:
        use_cols = [c for c in df_ti.columns if c not in index_cols]
    for iuc, use_col in enumerate(use_cols):
        if not isinstance(name, str):
            nname = name[iuc]
            # if zone type, find the zones for each index position
        else:
            nname = name
        if zone_array is not None and typ in ["zone", "grid"]:
            if zone_array.ndim != len(index_cols):
                raise Exception("get_tpl_or_ins_df() error: "
                                "zone_array.ndim "
                                "({0}) != len(index_cols)({1})"
                                "".format(zone_array.ndim,
                                          len(index_cols)))
            df_ti.loc[:, "zval"] = df_ti.sidx.apply(
                lambda x: zone_array[x])
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
                df_ti.loc[:, use_col] = "{0}_use_col:{1}".format(
                    nname, use_col)
                if suffix != '':
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_ti.loc[:, use_col] += suffix

        elif typ == "zone":
            # one par for each zone
            if longnames:
                df_ti.loc[:, use_col] = "{0}_use_col:{1}".format(
                    nname, use_col)
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x))
                if suffix != '':
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_ti.loc[:, use_col] += suffix

        elif typ == "grid":
            # one par for each index
            if longnames:
                df_ti.loc[:, use_col] = "{0}_use_col:{1}".format(
                    nname, use_col)
                if zone_array is not None:
                    df_ti.loc[:, use_col] += df_ti.zval.apply(
                        lambda x: "_zone:{0}".format(x))
                df_ti.loc[:, use_col] += '_' + df_ti.idx_strs
                if suffix != '':
                    df_ti.loc[:, use_col] += "_{0}".format(suffix)

            else:
                df_ti.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                df_ti.loc[:, use_col] += df_ti.idx_strs
                if suffix != '':
                    df_ti.loc[:, use_col] += suffix

        else:
            raise Exception("get_tpl_or_ins_df() error: "
                            "unrecognized 'typ', if not 'obs', "
                            "should be 'constant','zone', "
                            "or 'grid', not '{0}'".format(typ))
    return df_ti


def write_array_tpl(name, tpl_filename, suffix, par_type, zone_array=None,
                    gpname=None, shape=None, longnames=False, fill_value=1.0,
                    get_xy=None, input_filename=None):
    """
    write a template file for a 2D array.
     Args:
        name (`str`): the base parameter name
        tpl_filename (`str`): the template file to write - include path
        suffix (`str`): suffix to append to par names
        par_type (`str`): type of parameter
        zone_array (`numpy.ndarray`): an array used to skip inactive cells.
            Values less than 1 are
            not parameterized and are assigned a value of fill_value.
            Default is None.
        gpname (`str`): pargp filed in dataframe
        shape (`tuple`): dimensions of array to write
        longnames (`bool`): Use parnames > 12 char
        fill_value:
        get_xy:
        input_filename:

    Returns:
        df (`pandas.DataFrame`): a dataframe with parameter information
    """

    if shape is None and zone_array is None:
        raise Exception("write_array_tpl() error: must pass either zone_array "
                        "or shape")
    elif shape is not None and zone_array is not None:
        if shape != zone_array.shape:
            raise Exception("write_array_tpl() error: passed "
                            "shape != zone_array.shape")
    elif shape is None:
        shape = zone_array.shape
    if len(shape) != 2:
        raise Exception("write_array_tpl() error: shape '{0}' not 2D"
                        "".format(str(shape)))

    def constant_namer(i, j):
        if longnames:
            pname = "const_{0}".format(name)
            if suffix != '':
                pname += "_{0}".format(suffix)
        else:
            pname = "{0}{1}".format(name, suffix)
            if len(pname) > 12:
                raise ("constant par name too long:"
                       "{0}".format(pname))
        return pname

    def zone_namer(i, j):
        zval = 1
        if zone_array is not None:
            zval = zone_array[i, j]
        if longnames:
            pname = "{0}_zone:{1}".format(name, zval)
            if suffix != '':
                pname += "_{0}".format(suffix)
        else:

            pname = "{0}_zn{1}".format(name, zval)
            if len(pname) > 12:
                raise ("zone par name too long:{0}".format(pname))
        return pname

    def grid_namer(i, j):
        if longnames:
            pname = "{0}_i:{1}_j:{2}".format(name, i, j)
            if get_xy is not None:
                pname += "_x:{0:0.2f}_y:{1:0.2f}".format(*get_xy(i, j))
            if zone_array is not None:
                pname += "_zone:{0}".format(zone_array[i, j])
            if suffix != '':
                pname += "_{0}".format(suffix)
        else:
            pname = "{0}{1:03d}{2:03d}".format(name, i, j)
            if len(pname) > 12:
                raise ("grid pname too long:{0}".format(pname))
        return pname

    if par_type == "constant":
        namer = constant_namer
    elif par_type == "zone":
        namer = zone_namer
    elif par_type == "grid":
        namer = grid_namer
    else:
        raise Exception("write_array_tpl() error: unsupported par_type"
                        ", options are 'constant', 'zone', or 'grid', not"
                        "'{0}'".format(par_type))

    parnme = []
    xx, yy, ii, jj = [], [], [], []
    with open(tpl_filename, 'w') as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zone_array is not None and zone_array[i, j] < 1:
                    pname = " {0} ".format(fill_value)
                else:
                    if get_xy is not None:
                        x, y = get_xy(i, j)
                        xx.append(x)
                        yy.append(y)
                    ii.append(i)
                    jj.append(j)

                    pname = namer(i, j)
                    parnme.append(pname)
                    pname = " ~   {0}    ~".format(pname)
                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    if par_type == 'grid':
        df.loc[:, 'i'] = ii
        df.loc[:, 'j'] = jj
        if get_xy is not None:
            df.loc[:, 'x'] = xx
            df.loc[:, 'y'] = yy
    if gpname is None:
        gpname = name
    df.loc[:, "pargp"] = "{0}_{1}".format(
        gpname, suffix.replace('_', '')).rstrip('_')
    df.loc[:, "tpl_filename"] = tpl_filename
    df.loc[:, "input_filename"] = input_filename
    if input_filename is not None:
        arr = np.ones(shape)
        np.savetxt(input_filename, arr, fmt="%2.1f")

    return df


