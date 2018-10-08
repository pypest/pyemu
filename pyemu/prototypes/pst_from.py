
from __future__ import print_function, division
import os
from datetime import datetime
import shutil
import inspect
import warnings

import numpy as np
import pandas as pd
import pyemu
from ..pyemu_warnings import PyemuWarning
class PstFrom(object):

    def __init__(self,original_d,new_d,longnames=True,remove_existing=False,
                 spatial_reference=None):
        self.original_d = original_d
        self.new_d = new_d
        self.original_file_d = None
        self.mult_file_d = None
        self.remove_existing = bool(remove_existing)
        self.par_dfs = []
        self.obs_dfs = []
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
        self._spatial_reference = spatial_reference
        self.initialize_spatial_reference()

        self._setup_dirs()


    def _generic_get_xy(self,*args):
        if len(args) == 3: #kij
            return float(args[1]),float(args[2])
        elif len(args) == 2: #ij
            return float(args[0]),float(args[1])
        else:
            return 0.0,0.0

    def _flopy_structured_get_xy(self,*kwargs):
        if len(args) == 3: #kij
            return self._spatial_reference.xcentergrid[args[1],args[2]],\
                   self._spatial_reference.ycentergrid[args[1],args[2]]
        elif len(args) == 2: #ij
            return self._spatial_reference.xcentergrid[args[0], args[1]],\
                   self._spatial_reference.ycentergrid[args[0], args[1]]
        else:
            self.logger.lraise("_flopy_structured_get_xy() error: wrong number of args, should be 3 (kij) or 2 (ij)"+\
                               ", not '{0}'".format(str(args)))


    def initialize_spatial_reference(self):
        if self._spatial_reference is None:
            self.get_xy = self._generic_get_xy
        elif hasattr(self._spatial_reference,"xcentergrid") and\
            hasattr(self._spatial_reference,"ycentergrid"):
            self.get_xy = self._flopy_structured_get_xy
        else:
            self.logger.lraise("initialize_spatial_reference() error: unsupported spatial_reference")



    def write_forward_run(self):
        pass


    def build_prior(self):
        pass


    def draw(self):
        pass


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


    def _par_filename_prep(self,filenames,index_cols,use_cols):

        # todo: cast str column names, index_cols and use_cols to lower if str?
        # todo: copy files, load files, return file_dict
        # todo: check that all index_cols and use_cols are the same type
        # todo: check for shape consistency at least for array types
        file_dict = {}
        if not isinstance(filenames,list):
            filenames = [filenames]
        if index_cols is None and use_cols is not None:
            self.logger.lraise("index_cols is None, but use_cols is not ({0})".format(str(use_cols)))

        # load list type files
        if index_cols is not None:
            if not isinstance(index_cols,list):
                index_cols = [index_cols]
            if isinstance(index_cols[0],str):
                header=0
            elif isinstance(index_cols[0],int):
                header=None
            else:
                self.logger.lraise("unrecognized type for index_cols, should be str or int, not {0}".
                                   format(str(type(index_cols[0]))))
            if use_cols is not None:
                if not isinstance(use_cols,list):
                    use_cols = [use_cols]
                if isinstance(use_cols[0], str):
                    header = 0
                elif isinstance(use_cols[0], int):
                    header = None
                else:
                    self.logger.lraise("unrecognized type for use_cols, should be str or int, not {0}".
                                       format(str(type(use_cols[0]))))

                itype = type(index_cols[0])
                utype = type(use_cols[0])
                if itype != utype:
                    self.logger.lraise("index_cols type '{0} != use_cols type '{1}'".
                                       format(str(itype),str(utype)))

                si = set(index_cols)
                su = set(use_cols)

                i = si.intersection(su)
                if len(i) > 0:
                    self.logger.lraise("use_cols also listed in index_cols: {0}".format(str(i)))

            for filename in filenames:
                delim_whitespace = True
                sep = ' '
                if filename.lower().endswith(".csv"):
                    delim_whitespace = False
                    sep = ','
                file_path = os.path.join(self.new_d, filename)
                self.logger.log("loading array {0}".format(file_path))
                if not os.path.exists(file_path):
                    self.logger.lraise("par filename '{0}' not found ".format(file_path))
                df = pd.read_csv(file_path,header=header,delim_whitespace=delim_whitespace)
                missing = []
                for index_col in index_cols:
                    if index_col not in df.columns:
                        missing.append(index_col)
                if len(missing) > 0:
                    self.logger.lraise("the following index_cols were not found in file '{0}':{1}".
                                       format(file_path),str(missing))

                for use_col in use_cols:
                    if use_col not in df.columns:
                        missing.append(use_col)
                if len(missing) > 0:
                    self.logger.lraise("the following use_cols were not found in file '{0}':{1}".
                                       format(file_path),str(missing))
                if header is None:
                    header = False
                self.logger.statement("loaded list '{0}' of shape {1}".format(file_path,df.shape))
                df.to_csv(os.path.join(self.original_file_d,filename),index=False,sep=sep,header=header)
                file_dict[filename] = df
                self.logger.log("loading array {0}".format(file_path))


        # load array type files
        else:
            for filename in filenames:
                file_path = os.path.join(self.new_d, filename)
                self.logger.log("loading array {0}".format(file_path))
                if not os.path.exists(file_path):
                    self.logger.lraise("par filename '{0}' not found ".format(file_path))
                arr = np.loadtxt(os.path.join(self.new_d,filename))
                self.logger.log("loading array {0}".format(file_path))
                self.logger.statement("loaded array '{0}' of shape {1}".format(filename,arr.shape))
                np.savetxt(os.path.join(self.original_file_d,filename),arr)
                file_dict[filename] = arr
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1,len(fnames)):
                    if file_dict[fnames[i]].shape != file_dict[fnames[j]].shape:
                        self.logger.lraise("shape mismatch for array types, '{0}' shape {1} != '{2}' shape {3}".
                                           format(fnames[i],file_dict[fnames[i]].shape,
                                                  fnames[j],file_dict[fnames[j]].shape))

        return index_cols, use_cols, file_dict


    def add_pars_from_template(self,tpl_filename, in_filename):
        pass


    def _next_count(self,prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        count = self._prefix_count[prefix]
        self._prefix_count[prefix] += 1
        return count


    def add_parameters(self,filenames,par_type,zone_array=None,dist_type="gaussian",sigma_range=4.0,
                          upper_bound=1.0e10,lower_bound=1.0e-10,trans="log",
                          par_name_base="uni",index_cols=None,use_cols=None,
                          tpl_filename=None,pp_space=10,num_eig_kl=100,spatial_reference=None):
        
        self.logger.log("adding parameters for file(s) {0}".format(str(filenames)))
        index_cols, use_cols, file_dict = self._par_filename_prep(filenames,index_cols,use_cols)
        if tpl_filename is None:
            tpl_filename = "{0}_{1}.tpl".format(par_name_base,self._next_count(par_name_base))
        if tpl_filename in self.tpl_filenames:
            self.logger.lraise("tpl_filename '{0}' already listed".format(tpl_filename))

        if index_cols is not None:
            self.logger.log("writing list-based template file '{0}'".format(tpl_filename))
            df = write_list_tpl(file_dict.values(),par_name_base,os.path.join(self.new_d,tpl_filename),
                                par_type=par_type,suffix='',index_cols=index_cols,use_cols=use_cols,
                                zone_array=zone_array, longnames=self.longnames,
                                get_xy=self.get_xy)
            self.logger.log("writing list-based template file '{0}'".format(tpl_filename))
        else:
            self.logger.log("writing array-based template file '{0}'".format(tpl_filename))
            shape = file_dict[list(file_dict.keys())[0]].shape
            if par_type in ["constant","zone","grid"]:
                df = write_array_tpl(name=par_name_base,tpl_file=tpl_filename,suffix='',
                                     par_type=par_type,zone_array=zone_array,shape=shape,
                                     longnames=self.longnames,get_xy=self.get_xy,
                                     fill_value=1.0)
                
            elif par_type == "pilotpoints" or par_type == "pilot_points":
                self.logger.lraise("array type 'pilotpoints' not implemented")
            elif par_type == "kl":
                self.logger.lraise("array type 'kl' not implemented")
            else:
                self.logger.lraise("unrecognized 'par_type': '{0}', should be in "+\
                                   "['constant','zone','grid','pilotpoints','kl'")
            self.logger.log("writing array-based template file '{0}'".format(tpl_filename))
        df.loc[:,"partrans"] = trans
        df.loc[:,"parubnd"] = upper_bound
        df.loc[:,"parlbnd"] = lower_bound

        self.logger.log("adding parameters for file(s) {0}".format(str(filenames)))



def write_list_tpl(dfs, name, tpl_file, suffix, index_cols, par_type, use_cols=None,
                   zone_array=None,longnames=False,get_xy=None):

    if not isinstance(dfs,list):
        dfs = list(dfs)
    #work out the union of indices across all dfs
    sidx = set()
    for df in dfs:
        didx = set(df.loc[:,index_cols].apply(lambda x: tuple(x),axis=1))
        sidx.update(didx)
    df_tpl = pd.DataFrame({"sidx": sidx}, columns=["sidx"])



    # use all non-index columns if use_cols not passed
    if use_cols is None:
        use_cols = [c for c in df_tpl.columns if c not in index_cols]

    # get some index strings for naming
    if longnames:
        df_tpl.loc[:, "idx_strs"] = ["_".join(["{0}:{1}".format(idx,df.loc[i,idx])
                              for idx in index_cols]) for i in range(df_tpl.shape[0])]

    else:
        df_tpl.loc[:, "idx_strs"] = ["".join(["{1:03d}".format(idx, df.loc[i, idx])
                              for idx in index_cols]) for i in range(df_tpl.shape[0])]

    for use_col in use_cols:
        if par_type == "constant":
            if longnames:
                df_tpl.loc[:,use_col] = "{0}_{1}".format(name,use_col)
                if suffix != '':
                    df_tpl.loc[:,use_col] += "_{0}".format(suffix)
            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(name, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix

        elif par_type == "zone":
            if longnames:
                df_tpl.loc[:, use_col] = "{0}_{1}".format(name, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(name, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix

        elif par_type == "grid":
            if longnames:
                df_tpl.loc[:, use_col] = "{0}_{1}".format(name, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += "_{0}".format(suffix)
                df_tpl.loc[:,use_col] += '_' + df_tpl.idx_strs

            else:
                df_tpl.loc[:, use_col] = "{0}{1}".format(name, use_col)
                if suffix != '':
                    df_tpl.loc[:, use_col] += suffix
                df_tpl.loc[:,use_col] += df_tpl.idx_strs

        else:
            raise Exception("write_list_tpl() error: unrecognized 'par_type' should be 'constant','zone',"+\
                            "or 'grid', not '{0}'".format(par_type))

    parnme = list(df_tpl.loc[:,use_cols].values.flatten())
    df_par = pd.DataFrame({"parnme":parnme},index=parnme)

    for use_col in use_cols:
        df_tpl.loc[:,use_col] = df_tpl.loc[:,use_col].apply(lambda x: "~  {0}  ~".format(x))
    pyemu.helpers.write_df_tpl(filename=tpl_file,df=df_tpl,sep=',',tpl_marker='~')

    return df_par


def write_array_tpl(name, tpl_file, suffix, par_type, zone_array=None, shape=None,
                    longnames=False, fill_value=1.0,get_xy=None):
    """ write a template file for a 2D array.

        Parameters
        ----------
        name : str
            the base parameter name
        tpl_file : str
            the template file to write - include path
        zone_array : numpy.ndarray
            an array used to skip inactive cells.  Values less than 1 are
            not parameterized and are assigned a value of fill_value.
            Default is None.
        fill_value : float
            value to fill in values that are skipped.  Default is 1.0.

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

    def constant_namer(i,j):
        if longnames:
            pname =  "const_{0}".format(name)
            if suffix != '':
                pname += "_{0}".format(suffix)
        else:
            pname = "{0}{1}".format(name, suffix)
            if len(pname) > 12:
                raise ("constant par name too long:{0}". \
                       format(pname))
        return pname

    def zone_namer(i,j):
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
                raise ("zone par name too long:{0}". \
                       format(pname))
        return pname

    def grid_namer(i,j):
        if longnames:
            pname = "{0}_i:{0}_j:{1}".format(name, i, j)
            if get_xy is not None:
                pname += "_x:{0:10.2E}_y:{1:10.2E}".format(*get_xy(i,j))
            if suffix != '':
                pname += "_{0}".format(suffix)
        else:
            pname = "{0}{1:03d}{2:03d}".format(name, i, j)
            if len(pname) > 12:
                raise ("grid pname too long:{0}". \
                       format(pname))
        return pname

    if par_type == "constant":
        namer = constant_namer
    elif par_type == "zone":
        namer = zone_namer
    elif par_type == "grid":
        namer = grid_namer
    else:
        raise Exception("write_array_tpl() error: unsupported par_type"+
                        ", options are 'constant', 'zone', or 'grid', not"+\
                        "'{0}'".format(par_type))

    parnme = []
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zone_array is not None and zone_array[i, j] < 1:
                    pname = " {0} ".format(fill_value)
                else:
                    pname = namer(i,j)
                    parnme.append(pname)
                    pname = " ~   {0}    ~".format(pname)
                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    # df.loc[:,"pargp"] = "{0}{1}".format(self.cn_suffixname)
    df.loc[:, "pargp"] = "{0}_{1}".format(name, suffix.replace('_', ''))
    df.loc[:, "tpl"] = tpl_file
    return df


