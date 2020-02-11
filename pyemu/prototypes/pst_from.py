
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
from ..pyemu_warnings import PyemuWarning


class PstFrom(object):
    # TODO auto ins setup from list style output (or array) and use_cols etc
    # TODO pilotoint style par set-up etc
    # TODO prior builder + reals draw
    # TODO poss move/test some of the flopy/modflow specific setup apply
    #  methods to/in gw_utils. - save reinventing the setup/apply methods
    def __init__(self, original_d, new_d, longnames=True,
                 remove_existing=False, spatial_reference=None,
                 zero_based=True):  # TODO geostruct?

        self.original_d = original_d
        self.new_d = new_d
        self.original_file_d = None
        self.mult_file_d = None
        self.remove_existing = bool(remove_existing)
        self.zero_based = bool(zero_based)
        self._spatial_reference = spatial_reference

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
        if len(args) == 3:  # kij
            i, j = args[1], args[2]

        elif len(args) == 2:  # ij
            i, j = args[0], args[1]
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
            self.get_xy = self._flopy_mg_get_xy
        else:
            self.logger.lraise("initialize_spatial_reference() error: "
                               "unsupported spatial_reference")

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
                alist.append("pyemu.os_utils.run('{0}')\n".format(cmd))

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

    def build_prior(self):
        pass

    def draw(self):
        pass

    # def _init_pst(self, tpl_files=None, in_files=None,
    #               ins_files=None, out_files=None):
    #     """Initialise a pest control file object through i/o files
    # 
    #     Args:
    #         tpl_files:
    #         in_files:
    #         ins_files:
    #         out_files:
    # 
    #     Returns:
    # 
    #     """
    #     #  TODO: amend so that pst can be built from PstFrom components
    #     tpl_files, in_files, ins_files, out_files = (
    #         [] if arg is None else arg for arg in
    #         [tpl_files, in_files, ins_files, out_files])
    #     # borrowing from PstFromFlopyModel() method:
    #     self.logger.statement("changing dir in to {0}".format(self.new_d))
    #     os.chdir(self.new_d)
    #     try:
    #         self.logger.log("instantiating control file from i/o files")
    #         self.logger.statement(
    #             "tpl files: {0}".format(",".join(tpl_files)))
    #         self.logger.statement(
    #             "ins files: {0}".format(",".join(ins_files)))
    #         pst = pyemu.Pst.from_io_files(tpl_files=tpl_files,
    #                                       in_files=in_files,
    #                                       ins_files=ins_files,
    #                                       out_files=out_files)
    #         self.logger.log("instantiating control file from i/o files")
    #     except Exception as e:
    #         os.chdir("..")
    #         self.logger.lraise("error build Pst:{0}".format(str(e)))
    #     os.chdir('..')
    #     self.pst = pst

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
                par_data = pd.concat(self.par_dfs)
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
            pst.input_files = [os.path.join(self.mult_file_d, f)
                               for f in self.input_filenames]

        if 'obs' in update.keys() or not uupdate:
            if len(self.obs_dfs) > 0:
                obs_data = pd.concat(self.obs_dfs)
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
        # load list type files
        if index_cols is not None:
            if not isinstance(index_cols, list):
                index_cols = [index_cols]
            if isinstance(index_cols[0], str):
                # index_cols can be from header str
                header = 0
            elif isinstance(index_cols[0], int):
                # index_cols are column numbers in input file
                header = None
            else:
                self.logger.lraise("unrecognized type for index_cols, "
                                   "should be str or int, not {0}".
                                   format(str(type(index_cols[0]))))
            if use_cols is not None:
                if not isinstance(use_cols, list):
                    use_cols = [use_cols]
                if isinstance(use_cols[0], str):
                    header = 0
                elif isinstance(use_cols[0], int):
                    header = None
                else:
                    self.logger.lraise("unrecognized type for use_cols, "
                                       "should be str or int, not {0}".
                                       format(str(type(use_cols[0]))))

                itype = type(index_cols[0])
                utype = type(use_cols[0])
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

            for filename, sep, fmt, skip in zip(filenames, seps, fmts,
                                                skip_rows):
                file_path = os.path.join(self.new_d, filename)
                # looping over model input filenames
                if fmt.lower() == 'free':
                    if sep is None:
                        delim_whitespace = True
                        sep = ' '
                        if filename.lower().endswith(".csv"):
                            delim_whitespace = False
                            sep = ','
                    else:
                        delim_whitespace = False
                    self.logger.log("loading list {0}".format(file_path))
                    if not os.path.exists(file_path):
                        self.logger.lraise("par filename '{0}' not found "
                                           "".format(file_path))
                    # read each input file
                    if skip > 0:
                        with open(file_path, 'r') as fp:
                            storehead = [next(fp) for _ in range(skip)]
                    else:
                        storehead = []
                    df = pd.read_csv(file_path, header=header, skiprows=skip,
                                     delim_whitespace=delim_whitespace)
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
                    delim_whitespace = True
                    # read each input file
                    if skip > 0:
                        with open(file_path, 'r') as fp:
                            storehead = [next(fp) for _ in range(skip)]
                    else:
                        storehead = []
                    df = pd.read_csv(file_path, header=header, skiprows=skip,
                                     delim_whitespace=delim_whitespace)

                # ensure that column ids from index_col is in input file
                missing = []
                for index_col in index_cols:
                    if index_col not in df.columns:
                        missing.append(index_col)
                    df.loc[:, index_col] = df.loc[:, index_col].astype(np.int)
                if len(missing) > 0:
                    self.logger.lraise("the following index_cols were not "
                                       "found in file '{0}':{1}"
                                       "".format(file_path, str(missing)))
                # ensure requested use_cols are in input file
                for use_col in use_cols:
                    if use_col not in df.columns:
                        missing.append(use_col)
                if len(missing) > 0:
                    self.logger.lraise("the following use_cols were not found "
                                       "in file '{0}':{1}"
                                       "".format(file_path, str(missing)))
                hheader = header
                if hheader is None:
                    hheader = False
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

        # if tpl_filename is None:
        #     tpl_filename = os.path.split(filenames[0])[-1] + "{0}.tpl".\
        #         format(self._next_count(os.path.split(filenames[0])[-1]))
        # if tpl_filename in self.tpl_filenames:
        #     self.logger.lraise("tpl_filename '{0}' already listed".format(tpl_filename))
        #
        # self.tpl_filenames.append(tpl_filename)
        # mult_file = os.path.join("mult",tpl_filename.replace(".tpl",""))
        # self.output_filenames.append(mult_file)
        #
        # for filename in file_dict.keys():
        #     self.mult_files.append(mult_file)
        #     self.org_files.append(os.path.join("org",filename))

        return index_cols, use_cols, file_dict, fmt_dict, sep_dict, skip_dict

    def _next_count(self,prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        else:
            self._prefix_count[prefix] += 1

        return self._prefix_count[prefix]
    
    def add_observations(self, ins_file, out_file=None, pst_path=None,
                         inschek=True, rebuild_pst=False):
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
        cwd = '.'
        if pst_path is not None:
            cwd = os.path.join(*os.path.split(ins_file)[:-1])
            ins_file = os.path.join(pst_path, os.path.split(ins_file)[-1])
            out_file = os.path.join(pst_path, os.path.split(out_file)[-1])
        self.ins_filenames.append(ins_file)
        self.output_filenames.append(out_file)
        # add to temporary files to be removed at start of forward run
        self.tmp_files.append(out_file)
        df = None
        if inschek:
            # df = pst_utils._try_run_inschek(ins_file,out_file,cwd=cwd)
            ins_file = os.path.join(cwd, ins_file)
            out_file = os.path.join(cwd, out_file)
            df = pyemu.pst_utils.try_process_output_file(ins_file=ins_file,
                                                         output_file=out_file)
        if df is not None:
            # print(self.observation_data.index,df.index)
            new_obs_data.loc[df.index, "obsval"] = df.obsval
        self.obs_dfs.append(new_obs_data)
        self.logger.log(
            "adding observation from instruction file '{0}'".format(ins_file))
        if rebuild_pst:
            if self.pst is not None:
                self.logger.log("Adding pars to control file "
                                "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update='obs')
            else:
                self.build_pst(filename=self.pst.filename, update=False)
                self.logger.warn("pst object not available, "
                                 "new control file will be written")
        return new_obs_data
    
    def add_parameters(self, filenames, par_type, zone_array=None,
                       dist_type="gaussian", sigma_range=4.0,
                       upper_bound=1.0e10, lower_bound=1.0e-10,
                       transform="log", par_name_base="p", index_cols=None,
                       use_cols=None, pp_space=10, num_eig_kl=100,
                       spatial_reference=None, mfile_fmt='free',
                       ult_ubound=None, ult_lbound=None, rebuild_pst=False):
        """Add list or array style model input files to PstFrom object.
        This method

        Args: TODO - obvs doc string once this settles down.
            filenames:
            par_type:
            zone_array:
            dist_type:
            sigma_range:
            upper_bound:
            lower_bound:
            transform:
            par_name_base:
            index_cols:
            use_cols:
            pp_space:
            num_eig_kl:
            spatial_reference:

        Returns:

        """
        self.logger.log("adding parameters for file(s) "
                        "{0}".format(str(filenames)))
        (index_cols, use_cols, file_dict, 
         fmt_dict, sep_dict, skip_dict) = self._par_prep(filenames, index_cols,
                                                         use_cols,
                                                         fmts=mfile_fmt)
        par_data_cols = pyemu.pst_utils.pst_config["par_fieldnames"]
        if isinstance(par_name_base, str):
            par_name_base = [par_name_base]

        if len(par_name_base) == 1:
            pass
        elif use_cols is not None and len(par_name_base) == len(use_cols):
            pass
        else:
            self.logger.lraise("par_name_base should be a string, "
                               "single-element container, or container of "
                               "len use_cols, not '{0}'"
                               "".format(str(par_name_base)))

        if self.longnames:
            fmt = "_inst:{0}"
        else:
            fmt = "{0}"
        for i in range(len(par_name_base)):
            par_name_base[i] += fmt.format(self._next_count(par_name_base[i]))

        if index_cols is not None:
            # mult file name will take name from first par group in
            # passed par_name_base
            mlt_filename = "{0}_{1}.csv".format(
                par_name_base[0].replace(':', ''), par_type)
            tpl_filename = mlt_filename + ".tpl"

            if ult_lbound is None:
                ult_lbound = [None for _ in use_cols]
            if ult_ubound is None:
                ult_ubound = [None for _ in use_cols]
            if len(use_cols) == 1:
                if not isinstance(ult_lbound, list):
                    ult_lbound = [ult_lbound]
                if not isinstance(ult_ubound, list):
                    ult_ubound = [ult_ubound]
            if len(use_cols) != len(ult_lbound) != len(ult_ubound):
                self.logger.lraise("mismatch in number of columns to use {0} "
                                   "and number of ultimate lower {0} or upper "
                                   "{1} par bounds defined"
                                   "".format(len(use_cols), len(ult_lbound),
                                             len(ult_ubound)))

            self.logger.log(
                "writing list-based template file '{0}'".format(tpl_filename))
            df = write_list_tpl(
                file_dict.values(), par_name_base,
                tpl_filename=os.path.join(self.new_d, tpl_filename),
                par_type=par_type, suffix='', index_cols=index_cols,
                use_cols=use_cols, zone_array=zone_array,
                longnames=self.longnames, get_xy=self.get_xy,
                zero_based=self.zero_based,
                input_filename=os.path.join(self.mult_file_d, mlt_filename))

            self.logger.log("writing list-based template file "
                            "'{0}'".format(tpl_filename))
        else:
            mlt_filename = "{0}_{1}.csv".format(
                par_name_base[0].replace(':', ''), par_type)
            tpl_filename = mlt_filename + ".tpl"
            self.logger.log("writing array-based template file "
                            "'{0}'".format(tpl_filename))
            shape = file_dict[list(file_dict.keys())[0]].shape

            if par_type in {"constant", "zone", "grid"}:
                self.logger.log(
                    "writing template file "
                    "{0} for {1}".format(tpl_filename, par_name_base))
                df = write_array_tpl(
                    name=par_name_base[0],
                    tpl_filename=os.path.join(self.new_d, tpl_filename),
                    suffix='', par_type=par_type, zone_array=zone_array,
                    shape=shape, longnames=self.longnames, get_xy=self.get_xy,
                    fill_value=1.0,
                    input_filename=os.path.join(self.mult_file_d, mlt_filename))
                self.logger.log(
                    "writing template file"
                    " {0} for {1}".format(tpl_filename, par_name_base))

            elif par_type in {"pilotpoints", "pilot_points",
                              "pilotpoint", "pilot_point"}:
                # Stolen from helpers.PstFromFlopyModel()._pp_prep()
                if pp_space is None:
                    self.logger.warn("pp_space is None, using 10...\n")
                    pp_space = 10
                # TODO support independent geostructs for individual calls?
                if self.pp_geostruct is None:                      
                    self.logger.warn("pp_geostruct is None, using ExpVario "
                                     "with contribution=1 and "
                                     "a=(pp_space*max(delr,delc))")
                    pp_dist = pp_space * float(
                        max(self.m.dis.delr.array.max(),
                            self.m.dis.delc.array.max()))  # TODO set up new default? 
                    v = pyemu.geostats.ExpVario(contribution=1.0, a=pp_dist)
                    self.pp_geostruct = pyemu.geostats.GeoStruct(
                        variograms=v, name="pp_geostruct", transform="log")

                pp_df = mlt_df.loc[mlt_df.suffix == self.pp_suffix, :]
                layers = pp_df.layer.unique()
                layers.sort()
                pp_dict = {
                    l: list(pp_df.loc[pp_df.layer == l, "prefix"].unique()) for
                    l in layers}
                # big assumption here - if prefix is listed more than once, use the lowest layer index
                pp_dict_sort = {}
                for i, l in enumerate(layers):
                    p = set(pp_dict[l])
                    pl = list(p)
                    pl.sort()
                    pp_dict_sort[l] = pl
                    for ll in layers[i + 1:]:
                        pp = set(pp_dict[ll])
                        d = list(pp - p)
                        d.sort()
                        pp_dict_sort[ll] = d
                pp_dict = pp_dict_sort

                pp_array_file = {p: m for p, m in
                                 zip(pp_df.prefix, pp_df.mlt_file)}
                self.logger.statement("pp_dict: {0}".format(str(pp_dict)))

                self.log("calling setup_pilot_point_grid()")
                if self.use_pp_zones:
                    # check if k_zone_dict is a dictionary of dictionaries
                    if np.all([isinstance(v, dict) for v in
                               self.k_zone_dict.values()]):
                        ib = {p.split('.')[-1]: k_dict for p, k_dict in
                              self.k_zone_dict.items()}
                        for attr in pp_df.attr_name.unique():
                            if attr not in [p.split('.')[-1] for p in
                                            ib.keys()]:
                                if 'general_zn' not in ib.keys():
                                    warnings.warn(
                                        "Dictionary of dictionaries passed as zones, {0} not in keys: {1}. "
                                        "Will use ibound for zones".format(
                                            attr, ib.keys()), PyemuWarning)
                                else:
                                    self.logger.statement(
                                        "Dictionary of dictionaries passed as pp zones, "
                                        "using 'general_zn' for {0}".format(
                                            attr))
                            if 'general_zn' not in ib.keys():
                                ib['general_zn'] = {
                                    k: self.m.bas6.ibound[k].array for k in
                                    range(self.m.nlay)}
                    else:
                        ib = {'general_zn': self.k_zone_dict}
                else:
                    ib = {}
                    for k in range(self.m.nlay):
                        a = self.m.bas6.ibound[k].array.copy()
                        a[a > 0] = 1
                        ib[k] = a
                    for k, i in ib.items():
                        if np.any(i < 0):
                            u, c = np.unique(i[i > 0], return_counts=True)
                            counts = dict(zip(u, c))
                            mx = -1.0e+10
                            imx = None
                            for u, c in counts.items():
                                if c > mx:
                                    mx = c
                                    imx = u
                            self.logger.warn(
                                "resetting negative ibound values for PP zone" + \
                                "array in layer {0} : {1}".format(k + 1, u))
                            i[i < 0] = u
                    ib = {'general_zn': ib}
                pp_df = pyemu.pp_utils.setup_pilotpoints_grid(
                    self.m, ibound=ib, use_ibound_zones=self.use_pp_zones,
                    prefix_dict=pp_dict, every_n_cell=self.pp_space,
                    pp_dir=self.m.model_ws, tpl_dir=self.m.model_ws,
                    shapename=os.path.join(self.m.model_ws, "pp.shp"))
                self.logger.statement("{0} pilot point parameters created".
                                      format(pp_df.shape[0]))
                self.logger.statement("pilot point 'pargp':{0}".
                                      format(','.join(pp_df.pargp.unique())))
                self.log("calling setup_pilot_point_grid()")

                # calc factors for each layer
                pargp = pp_df.pargp.unique()
                pp_dfs_k = {}
                fac_files = {}
                pp_processed = set()
                pp_df.loc[:, "fac_file"] = np.NaN
                for pg in pargp:
                    ks = pp_df.loc[pp_df.pargp == pg, "k"].unique()
                    if len(ks) == 0:
                        self.logger.lraise(
                            "something is wrong in fac calcs for par group {0}".format(
                                pg))
                    if len(ks) == 1:
                        if np.all([isinstance(v, dict) for v in
                                   ib.values()]):  # check is dict of dicts
                            if np.any([pg.startswith(p) for p in ib.keys()]):
                                p = next(
                                    p for p in ib.keys() if pg.startswith(p))
                                # get dict relating to parameter prefix
                                ib_k = ib[p][ks[0]]
                            else:
                                p = 'general_zn'
                                ib_k = ib[p][ks[0]]
                        else:
                            ib_k = ib[ks[0]]
                    if len(ks) != 1:  # TODO
                        # self.logger.lraise("something is wrong in fac calcs for par group {0}".format(pg))
                        self.logger.warn(
                            "multiple k values for {0},forming composite zone array...".format(
                                pg))
                        ib_k = np.zeros((self.m.nrow, self.m.ncol))
                        for k in ks:
                            t = ib["general_zn"][k].copy()
                            t[t < 1] = 0
                            ib_k[t > 0] = t[t > 0]
                    k = int(ks[0])
                    kattr_id = "{}_{}".format(k, p)
                    kp_id = "{}_{}".format(k, pg)
                    if kp_id not in pp_dfs_k.keys():
                        self.log(
                            "calculating factors for p={0}, k={1}".format(pg,
                                                                          k))
                        fac_file = os.path.join(self.m.model_ws,
                                                "pp_k{0}.fac".format(kattr_id))
                        var_file = fac_file.replace("{0}.fac".format(kattr_id),
                                                    ".var.dat")
                        pp_df_k = pp_df.loc[pp_df.pargp == pg]
                        if kattr_id not in pp_processed:
                            self.logger.statement(
                                "saving krige variance file:{0}"
                                .format(var_file))
                            self.logger.statement(
                                "saving krige factors file:{0}"
                                .format(fac_file))
                            ok_pp = pyemu.geostats.OrdinaryKrige(
                                self.pp_geostruct, pp_df_k)
                            ok_pp.calc_factors_grid(self.m.sr,
                                                    var_filename=var_file,
                                                    zone_array=ib_k,
                                                    num_threads=10)
                            ok_pp.to_grid_factors_file(fac_file)
                            pp_processed.add(kattr_id)
                        fac_files[kp_id] = fac_file
                        self.log(
                            "calculating factors for p={0}, k={1}".format(pg,
                                                                          k))
                        pp_dfs_k[kp_id] = pp_df_k

                for kp_id, fac_file in fac_files.items():
                    k = int(kp_id.split('_')[0])
                    pp_prefix = kp_id.split('_', 1)[-1]
                    # pp_files = pp_df.pp_filename.unique()
                    fac_file = os.path.split(fac_file)[-1]
                    # pp_prefixes = pp_dict[k]
                    # for pp_prefix in pp_prefixes:
                    self.log("processing pp_prefix:{0}".format(pp_prefix))
                    if pp_prefix not in pp_array_file.keys():
                        self.logger.lraise(
                            "{0} not in self.pp_array_file.keys()".
                            format(pp_prefix, ','.
                                   join(pp_array_file.keys())))

                    out_file = os.path.join(self.arr_mlt, os.path.split(
                        pp_array_file[pp_prefix])[-1])

                    pp_files = pp_df.loc[pp_df.pp_filename.apply(
                        lambda x: "{0}pp".format(
                            pp_prefix) in x), "pp_filename"]
                    if pp_files.unique().shape[0] != 1:
                        self.logger.lraise(
                            "wrong number of pp_files found:{0}".format(
                                ','.join(pp_files)))
                    pp_file = os.path.split(pp_files.iloc[0])[-1]
                    pp_df.loc[pp_df.pargp == pp_prefix, "fac_file"] = fac_file
                    pp_df.loc[pp_df.pargp == pp_prefix, "pp_file"] = pp_file
                    pp_df.loc[pp_df.pargp == pp_prefix, "out_file"] = out_file

                pp_df.loc[:, "pargp"] = pp_df.pargp.apply(
                    lambda x: "pp_{0}".format(x))
                out_files = mlt_df.loc[mlt_df.mlt_file.
                                           apply(
                    lambda x: x.endswith(self.pp_suffix)), "mlt_file"]
                # mlt_df.loc[:,"fac_file"] = np.NaN
                # mlt_df.loc[:,"pp_file"] = np.NaN
                for out_file in out_files:
                    pp_df_pf = pp_df.loc[pp_df.out_file == out_file, :]
                    fac_files = pp_df_pf.fac_file
                    if fac_files.unique().shape[0] != 1:
                        self.logger.lraise(
                            "wrong number of fac files:{0}".format(
                                str(fac_files.unique())))
                    fac_file = fac_files.iloc[0]
                    pp_files = pp_df_pf.pp_file
                    if pp_files.unique().shape[0] != 1:
                        self.logger.lraise(
                            "wrong number of pp files:{0}".format(
                                str(pp_files.unique())))
                    pp_file = pp_files.iloc[0]
                    mlt_df.loc[
                        mlt_df.mlt_file == out_file, "fac_file"] = fac_file
                    mlt_df.loc[
                        mlt_df.mlt_file == out_file, "pp_file"] = pp_file
                self.par_dfs[self.pp_suffix] = pp_df

                mlt_df.loc[
                    mlt_df.suffix == self.pp_suffix, "tpl_file"] = np.NaN
                # TODO - other par types
                self.logger.lraise("array type 'pilotpoints' not implemented")
            elif par_type == "kl":
                self.logger.lraise("array type 'kl' not implemented")
            else:
                self.logger.lraise("unrecognized 'par_type': '{0}', "
                                   "should be in "
                                   "['constant','zone','grid','pilotpoints',"
                                   "'kl'")
            self.logger.log("writing array-based template file "
                            "'{0}'".format(tpl_filename))

        # accumulate information that relates mult_files (set-up here and
        # eventually filled by PEST) to actual model files so that actual
        # model input file can be generated
        relate_parfiles = []
        for mod_file in file_dict.keys():
            relate_parfiles.append(
                {"org_file": os.path.join(
                    *os.path.split(self.original_file_d)[1:],
                    mod_file),
                 "mlt_file": os.path.join(
                    *os.path.split(self.mult_file_d)[1:],
                    mlt_filename),
                 "model_file": mod_file,
                 "use_cols": use_cols,
                 "index_cols": index_cols,
                 "fmt": fmt_dict[mod_file],
                 "sep": sep_dict[mod_file],
                 "head_rows": skip_dict[mod_file],
                 "upper_bound": ult_ubound,
                 "lower_bound": ult_lbound})
        relate_pars_df = pd.DataFrame(relate_parfiles)
        self._parfile_relations.append(relate_pars_df)

        df.loc[:, "partype"] = par_type
        df.loc[:, "partrans"] = transform
        df.loc[:, "parubnd"] = upper_bound
        df.loc[:, "parlbnd"] = lower_bound
        #df.loc[:,"tpl_filename"] = tpl_filename

        self.tpl_filenames.append(tpl_filename)
        self.input_filenames.append(mlt_filename)
        for file_name in file_dict.keys():
            self.org_files.append(file_name)
            self.mult_files.append(mlt_filename)

        # add pars to par_data list BH: is this what we want?
        # - BH: think we can get away with dropping duplicates?
        missing = set(par_data_cols) - set(df.columns)
        for field in missing:
            df[field] = pyemu.pst_utils.pst_config['par_defaults'][field]
        df = df.loc[:, par_data_cols]
        self.par_dfs.append(df.drop_duplicates())
        self.logger.log("adding parameters for file(s) "
                        "{0}".format(str(filenames)))

        if rebuild_pst:
            if self.pst is not None:
                self.logger.log("Adding pars to control file "
                                "and rewriting pst")
                self.build_pst(filename=self.pst.filename, update='pars')
            else:
                self.build_pst(filename=self.pst.filename, update=False)
                self.logger.warn("pst object not available, "
                                 "new control file will be written")


def write_list_tpl(dfs, name, tpl_filename, suffix, index_cols, par_type,
                   use_cols=None, zone_array=None, longnames=False,
                   get_xy=None, zero_based=True, input_filename=None):
    """ Write template files for a list style input.

    Args:
        dfs:
        name:
        tpl_filename:
        suffix:
        index_cols:
        par_type:
        use_cols:
        zone_array:
        longnames:
        get_xy:
        zero_based:
        input_filename:

    Returns:

    """

    if not isinstance(dfs, list):
        dfs = list(dfs)
    # work out the union of indices across all dfs
    sidx = set()
    for df in dfs:
        didx = set(df.loc[:, index_cols].apply(lambda x: tuple(x), axis=1))
        sidx.update(didx)

    df_tpl = pd.DataFrame({"sidx": list(sidx)}, columns=["sidx"])
    # TO#DO using sets means that the rows of df and df_tpl are not necessarily
    #  aligned
    # - not a problem as this is just for the mult files the mapping to
    # model input files can be done by apply methods with the mapping
    # information provided by meta data within par names 
    # (or in the par_relations file).

    # get some index strings for naming
    if longnames:
        j = '_'
        fmt = "{0}:{1}"
        if isinstance(index_cols[0], str):
            inames = index_cols
        else:
            inames = ["idx{0}".format(i) for i in range(len(index_cols))]
    else:
        fmt = "{1:03d}"
        j = ''

    if not zero_based:
        # TODO: need to be careful here potential to have two
        #  conflicting/compounding `zero_based` actions
        #  by default we pass PestFrom zero_based object to this method
        #  so if not zero_based will subtract 1 from idx here...
        #  ----the get_xy method also -= 1 (checkout changes to get_xy())
        df_tpl.loc[:, "sidx"] = df_tpl.sidx.apply(
            lambda x: tuple(xx-1 for xx in x))
    df_tpl.loc[:, "idx_strs"] = df_tpl.sidx.apply(
        lambda x: j.join([fmt.format(iname, xx)
                          for xx, iname in zip(x, inames)]))

    # if zone type, find the zones for each index position
    if zone_array is not None and par_type in ["zone", "grid"]:
        if zone_array.ndim != len(index_cols):
            raise Exception("write_list_tpl() error: zone_array.ndim "
                            "({0}) != len(index_cols)({1})"
                            "".format(zone_array.ndim, len(index_cols)))
        df_tpl.loc[:, "zval"] = df_tpl.sidx.apply(lambda x: zone_array[x])

    # use all non-index columns if use_cols not passed
    if use_cols is None:
        use_cols = [c for c in df_tpl.columns if c not in index_cols]

    if get_xy is not None:
        df_tpl.loc[:, 'xy'] = df_tpl.sidx.apply(lambda x: get_xy(*x))
        df_tpl.loc[:, 'x'] = df_tpl.xy.apply(lambda x: x[0])
        df_tpl.loc[:, 'y'] = df_tpl.xy.apply(lambda x: x[1])

    for iuc, use_col in enumerate(use_cols):
        nname = name
        if not isinstance(name, str):
            nname = name[iuc]
        df_tpl.loc[:, "pargp{}".format(use_col)] = nname
        if par_type == "constant":
            if longnames:
                df_tpl.loc[:, use_col] = "{0}_use_col:{1}".format(
                    nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix

        elif par_type == "zone":

            if longnames:
                df_tpl.loc[:, use_col] = "{0}_use_col:{1}".format(nname,
                                                                  use_col)
                if zone_array is not None:
                    df_tpl.loc[:, use_col] += df_tpl.zval.apply(
                        lambda x: "_zone:{0}".format(x))
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix

        elif par_type == "grid":
            if longnames:
                df_tpl.loc[:, use_col] = "{0}_use_col:{1}".format(nname,
                                                                  use_col)
                if zone_array is not None:
                    df_tpl.loc[:, use_col] += df_tpl.zval.apply(
                        lambda x: "_zone:{0}".format(x))
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
                df_tpl.loc[:, use_col] += '_' + df_tpl.idx_strs

            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix
                df_tpl.loc[:, use_col] += df_tpl.idx_strs

        else:
            raise Exception("write_list_tpl() error: unrecognized 'par_type' "
                            "should be 'constant','zone', "
                            "or 'grid', not '{0}'".format(par_type))

    parnme = list(df_tpl.loc[:, use_cols].values.flatten())
    pargp = list(
        df_tpl.loc[:,
        ["pargp{0}".format(col) for col in use_cols]].values.flatten())
    df_par = pd.DataFrame({"parnme": parnme, "pargp": pargp}, index=parnme)
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


def write_array_tpl(name, tpl_filename, suffix, par_type, zone_array=None,
                    shape=None, longnames=False, fill_value=1.0, get_xy=None,
                    input_filename=None):
    """ write a template file for a 2D array.

        Parameters
        ----------
        name : str
            the base parameter name
        tpl_filename : str
            the template file to write - include path
        suffix:
        par_type:
        zone_array : numpy.ndarray
            an array used to skip inactive cells.  Values less than 1 are
            not parameterized and are assigned a value of fill_value.
            Default is None.
        shape:
        longnames:
        fill_value : float
            value to fill in values that are skipped.  Default is 1.0.
        get_xy:
        input_filename:

        Returns
        -------
        df : pandas.DataFrame
            a dataframe with parameter information

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
    df.loc[:, 'i'] = ii
    df.loc[:, 'j'] = jj
    if get_xy is not None:
        df.loc[:, 'x'] = xx
        df.loc[:, 'y'] = yy
    df.loc[:, "pargp"] = "{0}_{1}".format(
        name, suffix.replace('_', '')).rstrip('_')
    df.loc[:, "tpl_filename"] = tpl_filename
    df.loc[:, "input_filename"] = input_filename
    if input_filename is not None:
        arr = np.ones(shape)
        np.savetxt(input_filename, arr, fmt="%2.1f")

    return df


