"""
A prototype for ensemble-based non-linear optimization under uncertainty.  
Employs the BFGS algorithm to solve the SQP problem.   
"""

from __future__ import print_function, division
import os
from datetime import datetime
import shutil
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pyemu
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov,Matrix

from pyemu.pst import Pst
from ..logger import Logger
from .ensemble_method import EnsembleMethod


class EnsembleSQP(EnsembleMethod):
	"""
	Description

	Parameters
	----------

	Example
	----------

	"""

	def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,submit_file=None,verbose=False,
                 port=4004,slave_dir="template",drop_bad_reals=None,save_mats=False):

	    super(EnsembleSQP,self).__init__(pst=pst,parcov=parcov,obscov=obscov,num_slaves=num_slaves,
                                         submit_file=submit_file,verbose=verbose,port=port,
                                         slave_dir=slave_dir)

        self.logger.warn("pyemu's EnsembleSQP is for prototyping only.  Use PESTPP-OPT for a production " +
                      "implementation of ensemble-based non-linear OUU")

    def initialize(self,num_reals=1,init_lambda=None,enforce_bounds="reset",
    				parensemble=None,obsensemble=None,restart_obsensemble=None,
                   regul_factor=0.0,use_approx_prior=True,build_empirical_prior=False):

    	"""
    	Description

    	Parameters
    	----------

    	Example
    	----------

    	"""
    	
    	# input echoing
    	# error checking
    	# etc


    def update():

    	# Include Wolfe tests
    	# I is default Hessian at k = 0; allow user to specify
    	# Pure python limited memory (L-BFGS) version - truncation of change vectors
    	# Use Oliver et al. (2008) scaled formulation
    	# bound handling and update length limiting like PEST


