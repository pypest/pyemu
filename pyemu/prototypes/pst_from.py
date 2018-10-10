
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
                 spatial_reference=None,zero_based=True):
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


    def _generic_get_xy(self,*args):
        if len(args) == 3: #kij
            return float(args[1]),float(args[2])
        elif len(args) == 2: #ij
            return float(args[0]),float(args[1])
        else:
            return 0.0,0.0

    def _flopy_structured_get_xy(self,*args):

        if len(args) == 3: #kij
            i,j = args[1],args[2]

        elif len(args) == 2: #ij
            i,j = args[0],args[1]
        else:
            self.logger.lraise("_flopy_structured_get_xy() error: wrong number of args, should be 3 (kij) or 2 (ij)"+\
                               ", not '{0}'".format(str(args)))
        if not self.zero_based:
            i -= 1
            j -= 1
        return self._spatial_reference.xcentergrid[i,j], \
               self._spatial_reference.ycentergrid[i,j]

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


    def _par_prep(self,filenames,index_cols,use_cols):

        # todo: cast str column names, index_cols and use_cols to lower if str?
        # todo: check that all index_cols and use_cols are the same type
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
                self.logger.log("loading list {0}".format(file_path))
                if not os.path.exists(file_path):
                    self.logger.lraise("par filename '{0}' not found ".format(file_path))
                df = pd.read_csv(file_path,header=header,delim_whitespace=delim_whitespace)
                missing = []
                for index_col in index_cols:
                    if index_col not in df.columns:
                        missing.append(index_col)
                    df.loc[:,index_col] = df.loc[:,index_col].astype(np.int)

                if len(missing) > 0:
                    self.logger.lraise("the following index_cols were not found in file '{0}':{1}".
                                       format(file_path,str(missing)))

                for use_col in use_cols:
                    if use_col not in df.columns:
                        missing.append(use_col)
                if len(missing) > 0:
                    self.logger.lraise("the following use_cols were not found in file '{0}':{1}".
                                       format(file_path,str(missing)))
                hheader = header
                if hheader is None:
                    hheader = False
                self.logger.statement("loaded list '{0}' of shape {1}".format(file_path,df.shape))
                df.to_csv(os.path.join(self.original_file_d,filename),index=False,sep=sep,header=hheader)
                file_dict[filename] = df
                self.logger.log("loading list {0}".format(file_path))

            # check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1,len(fnames)):
                    if file_dict[fnames[i]].shape[1] != file_dict[fnames[j]].shape[1]:
                        self.logger.lraise("shape mismatch for array types, '{0}' shape {1} != '{2}' shape {3}".
                                           format(fnames[i],file_dict[fnames[i]].shape[1],
                                                  fnames[j],file_dict[fnames[j]].shape[1]))


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

            #check for compatibility
            fnames = list(file_dict.keys())
            for i in range(len(fnames)):
                for j in range(i+1,len(fnames)):
                    if file_dict[fnames[i]].shape != file_dict[fnames[j]].shape:
                        self.logger.lraise("shape mismatch for array types, '{0}' shape {1} != '{2}' shape {3}".
                                           format(fnames[i],file_dict[fnames[i]].shape,
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

        return index_cols, use_cols, file_dict


    def add_pars_from_template(self,tpl_filename, in_filename):
        pass


    def _next_count(self,prefix):
        if prefix not in self._prefix_count:
            self._prefix_count[prefix] = 0
        else:
            self._prefix_count[prefix] += 1

        return self._prefix_count[prefix]


    def add_parameters(self,filenames,par_type,zone_array=None,dist_type="gaussian",sigma_range=4.0,
                          upper_bound=1.0e10,lower_bound=1.0e-10,transform="log",
                          par_name_base="p",index_cols=None,use_cols=None,
                          pp_space=10,num_eig_kl=100,spatial_reference=None):
        
        self.logger.log("adding parameters for file(s) {0}".format(str(filenames)))
        index_cols, use_cols, file_dict = self._par_prep(filenames,index_cols,use_cols)


        if isinstance(par_name_base,str):
            par_name_base = [par_name_base]

        if len(par_name_base) == 1:
            pass
        elif use_cols is not None and len(par_name_base) == len(use_cols):
            pass
        else:
            self.logger.lraise("par_name_base should be a string,single-element " + \
                               "container, or container of len use_cols " + \
                               "not '{0}'".format(str(par_name_base)))



        if self.longnames:
            fmt = "_inst:{0}"
        else:
            fmt = "{0}"
        for i in range(len(par_name_base)):
            par_name_base[i] += fmt.format(self._next_count(par_name_base[i]))


        if index_cols is not None:
            mlt_filename = "{0}_{1}.csv".format(par_name_base[0],par_type)
            tpl_filename = mlt_filename + ".tpl"

            self.logger.log("writing list-based template file '{0}'".format(tpl_filename))
            df = write_list_tpl(file_dict.values(),par_name_base,os.path.join(self.new_d,tpl_filename),
                                par_type=par_type,suffix='',index_cols=index_cols,use_cols=use_cols,
                                zone_array=zone_array, longnames=self.longnames,
                                get_xy=self.get_xy,zero_based=self.zero_based,
                                input_filename=os.path.join(self.mult_file_d,mlt_filename))

            self.logger.log("writing list-based template file '{0}'".format(tpl_filename))
        else:
            mlt_filename = "{0}_{1}.csv".format(par_name_base[0], par_type)
            tpl_filename = mlt_filename + ".tpl"
            self.logger.log("writing array-based template file '{0}'".format(tpl_filename))
            shape = file_dict[list(file_dict.keys())[0]].shape

            if par_type in ["constant","zone","grid"]:
                self.logger.log("writing template file {0} for {1}".\
                                format(tpl_filename,par_name_base))
                df = write_array_tpl(name=par_name_base[0],tpl_filename=os.path.join(self.new_d,tpl_filename),
                                     suffix='',par_type=par_type,zone_array=zone_array,shape=shape,
                                     longnames=self.longnames,get_xy=self.get_xy,fill_value=1.0,
                                     input_filename=os.path.join(self.mult_file_d,mlt_filename))
                self.logger.log("writing template file {0} for {1}". \
                                format(tpl_filename, par_name_base))

            elif par_type == "pilotpoints" or par_type == "pilot_points":
                self.logger.lraise("array type 'pilotpoints' not implemented")
            elif par_type == "kl":
                self.logger.lraise("array type 'kl' not implemented")
            else:
                self.logger.lraise("unrecognized 'par_type': '{0}', should be in "+\
                                   "['constant','zone','grid','pilotpoints','kl'")
            self.logger.log("writing array-based template file '{0}'".format(tpl_filename))
        df.loc[:,"partrans"] = transform
        df.loc[:,"parubnd"] = upper_bound
        df.loc[:,"parlbnd"] = lower_bound
        #df.loc[:,"tpl_filename"] = tpl_filename

        self.tpl_filenames.append(tpl_filename)
        self.input_filenames.append(mlt_filename)
        for file_name in file_dict.keys():
            self.org_files.append(file_name)
            self.mult_files.append(mlt_filename)

        self.logger.log("adding parameters for file(s) {0}".format(str(filenames)))



def write_list_tpl(dfs, name, tpl_filename, suffix, index_cols, par_type, use_cols=None,
                   zone_array=None,longnames=False,get_xy=None,zero_based=True,
                   input_filename=None):

    if not isinstance(dfs,list):
        dfs = list(dfs)
    #work out the union of indices across all dfs
    sidx = set()
    for df in dfs:
        didx = set(df.loc[:,index_cols].apply(lambda x: tuple(x),axis=1))
        sidx.update(didx)

    df_tpl = pd.DataFrame({"sidx": list(sidx)}, columns=["sidx"])


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
        df_tpl.loc[:,"sidx"] = df_tpl.sidx.apply(lambda x: tuple(xx-1 for xx in x))
        df_tpl.loc[:, "idx_strs"] = [j.join([fmt.format(iname, df.loc[i, idx]-1)
                    for iname,idx in zip(inames,index_cols)]) for i in range(df_tpl.shape[0])]
    else:
        df_tpl.loc[:, "idx_strs"] = [j.join([fmt.format(iname, df.loc[i, idx])
                    for iname, idx in zip(inames, index_cols)]) for i in range(df_tpl.shape[0])]


    # if zone type, find the zones for each index position
    if zone_array is not None and par_type in ["zone","grid"]:
        if zone_array.ndim != len(index_cols):
            raise Exception("write_list_tpl() error: zone_array.ndim "+\
                            "({0}) != len(index_cols)({1})".format(zone_array.ndim,len(index_cols)))
        df_tpl.loc[:,"zval"] = df_tpl.sidx.apply(lambda x: zone_array[x])


    # use all non-index columns if use_cols not passed
    if use_cols is None:
        use_cols = [c for c in df_tpl.columns if c not in index_cols]


    if get_xy is not None:
        df_tpl.loc[:,'xy'] = df_tpl.sidx.apply(lambda x: get_xy(*x))
        df_tpl.loc[:,'x'] = df_tpl.xy.apply(lambda x : x[0])
        df_tpl.loc[:, 'y'] = df_tpl.xy.apply(lambda x: x[1])


    for iuc,use_col in enumerate(use_cols):
        nname = name
        if not isinstance(name,str):
           nname = name[iuc]
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
                    df_tpl.loc[:, use_col] += df_tpl.zval.apply(lambda x: "_zone:{0}".format(x))
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
                    df_tpl.loc[:, use_col] += df_tpl.zval.apply(lambda x: "_zone:{0}".format(x))
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

    parnme = list(df_tpl.loc[:,use_cols].values.flatten())
    df_par = pd.DataFrame({"parnme":parnme},index=parnme)
    if not longnames:
        too_long = df_par.loc[df_par.parnme.apply(lambda x: len(x) > 12),"parnme"]
        if too_long.shape[0] > 0:
            raise Exception("write_list_tpl() error: the following parameter names are too long:{0}".
                            format(','.join(list(too_long))))
    for use_col in use_cols:
        df_tpl.loc[:,use_col] = df_tpl.loc[:,use_col].apply(lambda x: "~  {0}  ~".format(x))
    pyemu.helpers.write_df_tpl(filename=tpl_filename,df=df_tpl,sep=',',tpl_marker='~')

    if input_filename is not None:
        df_in = df_tpl.copy()
        df_in.loc[:,use_cols] = 1.0
        df_in.to_csv(input_filename)
    df_par.loc[:,"tpl_filename"] = tpl_filename
    df_par.loc[:,"input_filename"] = input_filename
    return df_par


def write_array_tpl(name, tpl_filename, suffix, par_type, zone_array=None, shape=None,
                    longnames=False, fill_value=1.0,get_xy=None,input_filename=None):
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
            pname = "{0}_i:{1}_j:{2}".format(name, i, j)
            if get_xy is not None:
                pname += "_x:{0:0.2f}_y:{1:0.2f}".format(*get_xy(i,j))
            if zone_array is not None:
                pname += "_zone:{0}".format(zone_array[i,j])
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
    xx,yy,ii,jj = [],[],[],[]
    with open(tpl_filename, 'w') as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zone_array is not None and zone_array[i, j] < 1:
                    pname = " {0} ".format(fill_value)
                else:
                    if get_xy is not None:
                        x,y = get_xy(i,j)
                        xx.append(x)
                        yy.append(y)
                    ii.append(i)
                    jj.append(j)

                    pname = namer(i,j)
                    parnme.append(pname)
                    pname = " ~   {0}    ~".format(pname)
                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    df.loc[:,'i'] = ii
    df.loc[:,'j'] = jj
    if get_xy is not None:
        df.loc[:,'x'] = xx
        df.loc[:,'y'] = yy
    df.loc[:, "pargp"] = "{0}_{1}".format(name, suffix.replace('_', ''))
    df.loc[:, "tpl_filename"] = tpl_filename
    df.loc[:,"input_filename"] = input_filename
    if input_filename is not None:
        arr = np.ones(shape)
        np.savetxt(input_filename,arr,fmt="%2.1f")

    return df


