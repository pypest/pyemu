
from __future__ import print_function, division
import os
from datetime import datetime
import shutil

import warnings

import numpy as np
import pandas as pd
import pyemu

class PstFrom(object):

    def __init__(self,original_d,new_d,longnames=True,remove_existing=False):
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

        self.longnames=bool(longnames)
        self.logger = pyemu.Logger("PstFrom.log",echo=True)

        self.logger.statement("starting PstFrom process")

        self._setup_dirs()


    def _add_pars(self,filenames,dist_type,sigma_range,
                        upper_bound,lower_bound):
        """private method to track new pars"""

        pass
        # todo: check for bound compat with log status


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
        pass
        # todo: cast str column names, index_cols and use_cols to lower if str?
        # todo: copy files, load files, return file_dict
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


    def add_pars_from_template(self,tpl_filename, in_filename):
        pass


    def add_uniform_pars(self,filenames,dist_type="gaussian",sigma_range=4.0,
                               upper_bound=1.0e10,lower_bound=1.0e-10,trans="log",
                               prefix="uni",index_cols=None,use_cols=None):
        self.logger.log("adding uniform pars for file(s) {0}".format(str(filenames)))
        file_dict = self._par_filename_prep(filenames,index_cols,use_cols)


        self.logger.log("adding uniform pars for file(s) {0}".format(str(filenames)))


    def add_zone_pars(self, filenames, zone_array, dist_type="gaussian",
                      sigma_range=4.0,upper_bound=1.0e10,
                      lower_bound=1.0e-10, trans="log",prefix="zon",
                      skip_zone_vals=None,index_cols=None,use_cols=None):

        #todo: check for zone array shape compatibility
        #todo: if list pars, what if zone array is not 3D?
        pass


    def add_grid_pars(self,filenames,zone_array=None,dist_type="gaussian",
                      sigma_range=4.0,upper_bound=1.0e10,
                      lower_bound=1.0e-10, trans="log",prefix="grd",
                      skip_zone_vals=None,geostruct=None,spatial_ref=None,
                      index_cols=None,use_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: if list pars, what if zone array is not 3D?
        pass


    def add_pilotpoint_pars(self,filenames,zone_array=None,dist_type="gaussian",
                            sigma_range=4.0,upper_bound=1.0e10,
                            lower_bound=1.0e-10, trans="log",prefix="ppt",
                            space=10,geostruct=None,spatial_ref=None,
                            index_cols=None,use_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: if list pars, what if zone array is not 3D?
        pass


    def add_kl_pars(self,filenames,zone_array=None,dist_type="gaussian",
                    sigma_range=4.0,upper_bound=1.0e10,
                    lower_bound=1.0e-10, trans="log",prefix="ppt",
                    num_comps=10,geostruct=None,spatial_ref=None,
                    index_cols=None,use_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: num_comps <= array entries
        # todo: if list pars, what if zone array is not 3D?
        pass


