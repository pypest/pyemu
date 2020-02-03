
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

    def __init__(self, original_d, new_d, longnames=True, remove_existing=False,
                 spatial_reference=None, zero_based=True):

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
        self.mod_command = 'forward_run.py'
        self.pre_py_cmds = []
        self.pre_sys_cmds = []
        self.mod_sys_cmds = []
        self.post_py_cmds = []
        self.post_sys_cmds = []

        self.tpl_filenames, self.input_filenames = [],[]
        self.ins_filenames, self.output_filenames = [],[]

        self.longnames=bool(longnames)
        self.logger = pyemu.Logger("PstFrom.log",echo=True)

        self.logger.statement("starting PstFrom process")

        self._prefix_count = {}

        self.get_xy = None

        self.initialize_spatial_reference()

        self._setup_dirs()
        # TODO: build an essentially empty pest control file object here?
        # something that the add_parameters() methods can hook into later?
        self.par_data = []
        self._parfile_relations = []
        self.par_dfs = {'list_pars': [], 'array_pars': []}
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
                    "seperator mismatch for {0}, seps passed {1}"
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

        elif len(args) == 2: #ij
            i, j = args[0], args[1]
        else:
            self.logger.lraise(("get_xy() error: wrong number of args, should be 3 (kij) or 2 (ij)"
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
        elif (hasattr(self._spatial_reference,"xcentergrid") and
              hasattr(self._spatial_reference,"ycentergrid")):
            self.get_xy = self._flopy_sr_get_xy
        elif (hasattr(self._spatial_reference, "xcellcenters") and
              hasattr(self._spatial_reference, "ycellcenters")):
            # support modelgrid style cell locs
            self.get_xy = self._flopy_mg_get_xy
        else:
            self.logger.lraise("initialize_spatial_reference() error: unsupported spatial_reference")


    def write_forward_run(self):
        pass


    def build_prior(self):
        pass


    def draw(self):
        pass

    def _init_pst(self, tpl_files=None, in_files=None,
                  ins_files=None, out_files=None):
        """Initialise a pest control file object through i/o files

        Args:
            tpl_files:
            in_files:
            ins_files:
            out_files:

        Returns:

        """
        #  TODO: amend so that pst can be built from PstFrom components
        tpl_files, in_files, ins_files, out_files = (
            [] if arg is None else arg for arg in
            [tpl_files, in_files, ins_files, out_files])
        # borrowing from PstFromFlopyModel() method:
        self.logger.statement("changing dir in to {0}".format(self.new_d))
        os.chdir(self.new_d)
        try:
            self.logger.log("instantiating control file from i/o files")
            self.logger.statement(
                "tpl files: {0}".format(",".join(tpl_files)))
            self.logger.statement(
                "ins files: {0}".format(",".join(ins_files)))
            pst = pyemu.Pst.from_io_files(tpl_files=tpl_files,
                                          in_files=in_files,
                                          ins_files=ins_files,
                                          out_files=out_files)
            self.logger.log("instantiating control file from i/o files")
        except Exception as e:
            os.chdir("..")
            self.logger.lraise("error build Pst:{0}".format(str(e)))
        os.chdir('..')
        self.pst = pst

    def build_pst(self, filename=None):
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

        Note:
            This builds a pest control file from scratch
            - overwriting anything already in self.pst object and
            anything already writen to `filename`
        """

        # parameter data from object
        par_data = pd.concat(self.par_data)
        # info relating parameter multiplier files to model input files
        parfile_relations = self.parfile_relations
        parfile_relations.to_csv(os.path.join(self.new_d,
                                              'mult2model_info.csv'))
        if filename is None:
            filename = os.path.join(self.new_d, self.original_d)
        elif os.path.dirname(filename) in ['', '.']:
            filename = os.path.join(self.new_d, filename)

        pst = pyemu.Pst(filename, load=False)
        pst.parameter_data = par_data


        # TODO: temporalily borowed from pst_utils.generic_pst()
        #  ----------------------------------------------------------------->
        obs_data = pyemu.pst_utils._populate_dataframe(
            [], pst.obs_fieldnames, pst.obs_defaults, pst.obs_dtype)
        obs_data.loc[:, "obsnme"] = []
        obs_data.index = []
        obs_data.sort_index(inplace=True)
        pst.observation_data = obs_data

        pst.template_files = self.tpl_filenames
        pst.input_files = [os.path.join(self.mult_file_d, f)
                           for f in self.ins_filenames]
        pst.instruction_files = self.ins_filenames
        pst.output_files = self.output_filenames
        pst.model_command = self.mod_command

        pst.prior_information = pst.null_prior
        self.pst = pst
        # TODO <--------------------------------------------------------------
        self.pst.write(filename)
        test = None
        # TODO: amend so that pst can be built from PstFrom components
        # self._init_pst(
        #     tpl_files=self.tpl_filenames, in_files=self.input_filenames,
        #     ins_files=self.ins_filenames, out_files=self.output_filenames)
        # 
        # if filename is None:
        #     filename = os.path.join(self.new_d, self.original_d)
        # self.logger.statement("writing pst {0}".format(filename))
        # 
        # self.pst.write(filename)

    def _setup_dirs(self):
        self.logger.log("setting up dirs")
        if not os.path.exists(self.original_d):
            self.logger.lraise("original_d '{0}' not found".format(self.original_d))
        if not os.path.isdir(self.original_d):
            self.logger.lraise("original_d '{0}' is not a directory".format(self.original_d))
        if os.path.exists(self.new_d):
            if self.remove_existing:
                self.logger.log("removing existing new_d '{0}'".format(self.new_d))
                shutil.rmtree(self.new_d)
                self.logger.log("removing existing new_d '{0}'".format(self.new_d))
            else:
                self.logger.lraise("new_d '{0}' already exists - use remove_existing=True".format(self.new_d))

        self.logger.log("copying original_d '{0}' to new_d '{1}'".format(self.original_d,self.new_d))
        shutil.copytree(self.original_d,self.new_d)
        self.logger.log("copying original_d '{0}' to new_d '{1}'".format(self.original_d, self.new_d))


        self.original_file_d = os.path.join(self.new_d, "org")
        if os.path.exists(self.original_file_d):
            self.logger.lraise("'org' subdir already exists in new_d '{0}'".format(self.new_d))
        os.makedirs(self.original_file_d)

        self.mult_file_d = os.path.join(self.new_d, "mult")
        if os.path.exists(self.mult_file_d):
            self.logger.lraise("'mult' subdir already exists in new_d '{0}'".format(self.new_d))
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
        if not isinstance(fmts, list):
            fmts = [fmts]
        if not isinstance(seps, list):
            seps = [seps]
        if not isinstance(skip_rows, list):
            skip_rows = [skip_rows]
        if fmts is None:
            fmts = ['free' for f in filenames]
        fmts = ['free' if fmt is None else fmt for fmt in fmts]
        if seps is None:
            seps = [None for f in filenames]
        if skip_rows is None:
            skip_rows = [None for f in filenames]
        skip_rows = [0 if s is None else s for s in skip_rows]
        if index_cols is None and use_cols is not None:
            self.logger.lraise("index_cols is None, but use_cols is not ({0})"
                               "".format(str(use_cols)))

        # load list type files
        if index_cols is not None:
            if not isinstance(index_cols,list):
                index_cols = [index_cols]
            if isinstance(index_cols[0],str):
                # index_cols can be from header str
                header=0
            elif isinstance(index_cols[0],int):
                # index_cols are column numbers in input file
                header=None
            else:
                self.logger.lraise("unrecognized type for index_cols, "
                                   "should be str or int, not {0}".
                                   format(str(type(index_cols[0]))))
            if use_cols is not None:
                if not isinstance(use_cols,list):
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
                                       format(str(itype),str(utype)))

                si = set(index_cols)
                su = set(use_cols)

                i = si.intersection(su)
                if len(i) > 0:
                    self.logger.lraise("use_cols also listed in "
                                       "index_cols: {0}".format(str(i)))

            for filename, sep, fmt, skip in zip(filenames, seps, fmts,
                                                skip_rows):
                # looping over model input filenames
                # TODO: fmt and sep checksout against previous calls with filename
                if fmt.lower() == 'free':
                    if sep is None:
                        delim_whitespace = True
                        sep = ' '
                        if filename.lower().endswith(".csv"):
                            delim_whitespace = False
                            sep = ','
                    else:
                        delim_whitespace = False
                    file_path = os.path.join(self.new_d, filename)
                    self.logger.log("loading list {0}".format(file_path))
                    if not os.path.exists(file_path):
                        self.logger.lraise("par filename '{0}' not found "
                                           "".format(file_path))
                    # read each input file
                    if skip > 0:
                        with open(file_path, 'r') as fp:
                            storehead = [next(fp) for _ in range(skip)]
                    else:
                        storehead=[]
                    df = pd.read_csv(file_path, header=header, skiprows=skip,
                                     delim_whitespace=delim_whitespace)
                else:
                    raise NotImplementedError("Only free format list "
                                              "par files currently supported")

                # ensure that column ids from index_col is in input file
                missing = []
                for index_col in index_cols:
                    if index_col not in df.columns:
                        missing.append(index_col)
                    df.loc[:,index_col] = df.loc[:,index_col].astype(np.int)
                if len(missing) > 0:
                    self.logger.lraise("the following index_cols were not found"
                                       " in file '{0}':{1}".
                                       format(file_path,str(missing)))
                # ensure requested use_cols are in input file
                for use_col in use_cols:
                    if use_col not in df.columns:
                        missing.append(use_col)
                if len(missing) > 0:
                    self.logger.lraise("the following use_cols were not found "
                                       "in file '{0}':{1}".
                                       format(file_path,str(missing)))
                hheader = header
                if hheader is None:
                    hheader = False
                self.logger.statement("loaded list '{0}' of shape {1}"
                                      "".format(file_path, df.shape))
                # TODO: do we need to be careful of the format of the model
                #  files? -- probs not necessary for the version in
                #  original_file_d - but for the eventual product model file,
                #  it might be format sensitive - yuck
                # Update: I think the `original files` saved can always
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

        # load array type files
        else:
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
                # read array type input file # TODO: check this handles both whitespace and comma delim
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
            #check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1, len(fnames)):
                    if file_dict[fnames[i]].shape != file_dict[fnames[j]].shape:
                        self.logger.lraise(
                            "shape mismatch for array types, '{0}' "
                            "shape {1} != '{2}' shape {3}"
                            "".format(fnames[i],file_dict[fnames[i]].shape,
                                      fnames[j],file_dict[fnames[j]].shape))

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

    def add_pars_from_template(self, tpl_filename, in_filename):
        # TODO: modify so that method adds to PstFrom object parameter data?
        #  Not pst
        """Method for adding parameters to Pest control file from pre-existing
        template, input file pairs

        Args:
            tpl_filename (`str` or `list`): filename(s) of template files
                with pars to add
            in_filename (`str` or `list`): filename(s) of (what pest views as)
                model input files

        Returns:

        Note:
            if arguments are lists they need to be in equivlent orders, so
            `tpl_filename[n]` relates to `in_filename[n]`

            if self.pst is None this method will initialise a pst object by
            calling pyemu.helpers.pst_from_io_files()
        """
        # quick argument checks
        if type(tpl_filename) != type(in_filename):
            raise TypeError("Both arguments need to be of the same type, "
                            "`tpl_filename` is type {0}, "
                            "`in_filename` is type {1}"
                            "".format(type(tpl_filename), type(in_filename)))
        if isinstance(tpl_filename, list):
            assert len(tpl_filename) == len(in_filename), \
                ("Lists provided for template and input filenames "
                 "are not the same length.")
        else:
            tpl_filename = [tpl_filename]
            in_filename = [in_filename]

        for tpl, infnme in zip(tpl_filename, in_filename):
            # need to set up a pst object container if it doesn't already exist
            if self.pst is None:
                self._init_pst(tpl_files=[tpl], in_files=[infnme])
            else:
                # add new pars to exisiting control file
                os.chdir(self.new_d)
                try:
                    self.pst.add_parameters(tpl, infnme)
                except Exception as e:
                    os.chdir("..")
                    self.logger.lraise(
                        "error adding parameters for tpl {0}:{1}"
                        "".format(tpl, str(e)))
                os.chdir("..")

    def _next_count(self,prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        else:
            self._prefix_count[prefix] += 1

        return self._prefix_count[prefix]

    def add_parameters(self, filenames, par_type, zone_array=None,
                       dist_type="gaussian", sigma_range=4.0,
                       upper_bound=1.0e10, lower_bound=1.0e-10,
                       transform="log", par_name_base="p", index_cols=None,
                       use_cols=None, pp_space=10, num_eig_kl=100,
                       spatial_reference=None, mfile_fmt='free',
                       ult_ubound=None, ult_lbound=None):
        """Add list or array style model input files to PstFrom object.
        This method

        Args:
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
                # TODO passing in the dictionary values worries me a little.
                #  can the order get screwed up? does it matter?
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

            if par_type in ["constant", "zone", "grid"]:
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
                # TODO
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
        self.par_data.append(df.drop_duplicates())
        # TODO workout where and how to store the mult -> model file info.
        # TODO workout how to get the mult to apply to the model files and
        #  mimic the model file formats!
        self.logger.log("adding parameters for file(s) "
                        "{0}".format(str(filenames)))


def write_list_tpl(dfs, name, tpl_filename, suffix, index_cols, par_type,
                   use_cols=None, zone_array=None, longnames=False, get_xy=None,
                   zero_based=True, input_filename=None):
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
    # TODO using sets means that the rows of df and df_tpl are not necessaril aligned
    # - not a problem as this is just for the mult files the mapping to model input files can be done
    # by apply methods with the mapping information provided by meta data within par names.

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

    for iuc,use_col in enumerate(use_cols):
        nname = name
        if not isinstance(name, str):
           nname = name[iuc]
        df_tpl.loc[:, "pargp{}".format(use_col)] = nname
        if par_type == "constant":
            if longnames:
                df_tpl.loc[:,use_col] = "{0}_use_col:{1}".format(nname,use_col)
                if suffix != '':
                    df_tpl.loc[:,use_col] += "_{0}".format(suffix)
            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix

        elif par_type == "zone":

            if longnames:
                df_tpl.loc[:, use_col] = "{0}_use_col:{1}".format(nname, use_col)
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
                df_tpl.loc[:, use_col] = "{0}_use_col:{1}".format(nname, use_col)
                if zone_array is not None:
                    df_tpl.loc[:, use_col] += df_tpl.zval.apply(
                        lambda x: "_zone:{0}".format(x))
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
                df_tpl.loc[:,use_col] += '_' + df_tpl.idx_strs

            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(nname, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix
                df_tpl.loc[:,use_col] += df_tpl.idx_strs

        else:
            raise Exception("write_list_tpl() error: unrecognized 'par_type' should be 'constant','zone',"+\
                            "or 'grid', not '{0}'".format(par_type))

    parnme = list(df_tpl.loc[:, use_cols].values.flatten())
    pargp = list(df_tpl.loc[:,
                 ["pargp{0}".format(col) for col in use_cols]].values.flatten())
    df_par = pd.DataFrame({"parnme": parnme, "pargp": pargp}, index=parnme)
    if not longnames:
        too_long = df_par.loc[df_par.parnme.apply(lambda x: len(x) > 12),"parnme"]
        if too_long.shape[0] > 0:
            raise Exception("write_list_tpl() error: the following parameter names are too long:{0}".
                            format(','.join(list(too_long))))
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
        raise Exception("write_array_tpl() error: must pass either zone_array or shape")
    elif shape is not None and zone_array is not None:
        if shape != zone_array.shape:
            raise Exception("write_array_tpl() error: passed shape != zone_array.shape")
    elif shape is None:
        shape = zone_array.shape
    if len(shape) != 2:
        raise Exception("write_array_tpl() error: shape '{0}' not 2D".format(str(shape)))

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


