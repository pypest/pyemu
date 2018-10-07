
from __future__ import print_function, division
import os
from datetime import datetime
import shutil

import warnings

import numpy as np
import pandas as pd
import pyemu

class PstFrom(object):

    def __init__(self,original_d,new_d,longnames=True):
        self.par_dfs = []
        self.obs_dfs = []
        self.pre_py_cmds = []
        self.pre_sys_cmds = []
        self.mod_sys_cmds = []
        self.post_py_cmds = []
        self.post_sys_cmds = []

        self.longnames=bool(longnames)
        self.logger = pyemu.Logger("PstFrom.log")
        #todo: setup dirs, add some intro log info

    def _add_pars(self,filenames,dist_type,sigma_range,
                        upper_bound,lower_bound):
        """private method to track new pars"""

        pass
        # todo: check for bound compat with log status


    def _setup_dirs(self):
        pass

    def _par_filename_prep(self,filenames,index_cols,par_cols):
        pass
        # todo: if par_cols, then index_cols
        # todo: parse index_cols based on type (int or str)
        # todo: if par_cols, then par_cols types must be same as index_cols types
        # todo: copy files, load files, return file_dict
    

    def add_pars_from_template(self,tpl_filename, in_filename):
        pass


    def add_uniform_pars(self,filenames,dist_type="gaussian",sigma_range=4.0,
                               upper_bound=1.0e10,lower_bound=1.0e-10,trans="log",
                               prefix="uni",index_cols=None,par_cols=None):
        pass


    def add_zone_pars(self, filenames, zone_array, dist_type="gaussian",
                      sigma_range=4.0,upper_bound=1.0e10,
                      lower_bound=1.0e-10, trans="log",prefix="zon",
                      skip_zone_vals=None,index_cols=None,par_cols=None):

        #todo: check for zone array shape compatibility
        #todo: if list pars, what if zone array is not 3D?
        pass


    def add_grid_pars(self,filenames,zone_array=None,dist_type="gaussian",
                      sigma_range=4.0,upper_bound=1.0e10,
                      lower_bound=1.0e-10, trans="log",prefix="grd",
                      skip_zone_vals=None,geostruct=None,spatial_ref=None,
                      index_cols=None,par_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: if list pars, what if zone array is not 3D?
        pass


    def add_pilotpoint_pars(self,filenames,zone_array=None,dist_type="gaussian",
                            sigma_range=4.0,upper_bound=1.0e10,
                            lower_bound=1.0e-10, trans="log",prefix="ppt",
                            space=10,geostruct=None,spatial_ref=None,
                            index_cols=None,par_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: if list pars, what if zone array is not 3D?
        pass


    def add_kl_pars(self,filenames,zone_array=None,dist_type="gaussian",
                    sigma_range=4.0,upper_bound=1.0e10,
                    lower_bound=1.0e-10, trans="log",prefix="ppt",
                    num_comps=10,geostruct=None,spatial_ref=None,
                    index_cols=None,par_cols=None):

        # todo: geostruct and spatial_ref must be used together
        # todo: num_comps <= array entries
        # todo: if list pars, what if zone array is not 3D?
        pass


