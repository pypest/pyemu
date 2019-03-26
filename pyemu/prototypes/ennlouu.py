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
        pst : pyemu.Pst or str
            a control file instance or filename
        parcov : pyemu.Cov or str
            a prior parameter covariance matrix or filename. If None,
            parcov is constructed from parameter bounds (diagonal)
        obscov : pyemu.Cov or str
            a measurement noise covariance matrix or filename. If None,
            obscov is constructed from observation weights.
        num_slaves : int
            number of slaves to use in (local machine) parallel evaluation of the parmaeter
            ensemble.  If 0, serial evaluation is used.  Ignored if submit_file is not None
        submit_file : str
            the name of a HTCondor submit file.  If not None, HTCondor is used to
            evaluate the parameter ensemble in parallel by issuing condor_submit
            as a system command
        port : int
            the TCP port number to communicate on for parallel run management
        slave_dir : str
            path to a directory with a complete set of model files and PEST
            interface files
        drop_bad_reals : float
                drop realizations with phi greater than drop_bad_reals. If None, all
                realizations are kept. Default is None

    Example
    -------
    ``>>>import pyemu``

    ``>>>esqp = pyemu.EnsembleSQP(pst="pest.pst")``
    """

    def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,submit_file=None,verbose=False,
                 port=4004,slave_dir="template",drop_bad_reals=None,save_mats=False):

        super(EnsembleSQP,self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                         submit_file=submit_file, verbose=verbose, port=port,
                                         slave_dir=slave_dir)

        self.logger.warn("pyemu's EnsembleSQP is for prototyping only.")

    def initialize(self,num_reals=1,enforce_bounds="reset",
    			   parensemble=None,restart_obsensemble=None,draw_mult=0.1,dec_var_group="obj_fn"):

        """
    	Description

        Parameters
        ----------
            num_reals : int
                the number of realizations to draw.  Ignored if parensemble/obsensemble
                are not None
            enforce_bounds : str
                how to enfore parameter bound transgression.  options are
                reset, drop, or None
            parensemble : pyemu.ParameterEnsemble or str
                a parameter ensemble or filename to use as the initial
                parameter ensemble.  If not None, then obsenemble must not be
                None
            obsensemble : pyemu.ObservationEnsemble or str
                an observation ensemble or filename to use as the initial
                observation ensemble.  If not None, then parensemble must
                not be None
            restart_obsensemble : pyemu.ObservationEnsemble or str
                an observation ensemble or filename to use as an
                evaluated observation ensemble.  If not None, this will skip the initial
                parameter ensemble evaluation - user beware!
            draw_mult : float or int
                a multiplier for scaling (uniform) parensemble draw variance.  Used for drawing
                dec var en in tighter cluster around mean val (e.g., compared to ies and par priors).
                Dec var en stats here are ``computational'' rather than (pseudo-) physical.
            dec_var_group : str
                obsgnme containing a single obs serving as optimization objective function.
                Like ++opt_dec_var_groups(<group_names>) in PESTPP-OPT.
        
        # rename some of above vars in accordance with opt parlance
        # omitted args (from smoother.py): obsensemble=None, initial_lambda, regul_factor, 
        use_approx_prior, build_empirical_prior

        Example
        -------
        ``>>>import pyemu``
        ``>>>esqp = pyemu.EnsembleSQP(pst="pest.pst")``
        ``>>>esqp.initialize(num_reals=100)``

    	"""
    	
        self.enforce_bounds = enforce_bounds
        #self.drop_bad_reals = drop_bad_reals
        #self.save_mats = save_mats
        self.total_runs = 0

        self.draw_mult = draw_mult

        self.dec_var_group = dec_var_group#.lower()
        self.dec_var_obs = list(self.pst.observation_data.loc[self.pst.observation_data.obgnme == \
                                                              self.dec_var_group, :].obsnme)
        if len(self.dec_var_obs) != 1:
            raise Exception("number of obs serving as opt obj function " + \
                            "must equal 1, not {0} - see docstring".format(len(self.dec_var_obs)))

        # could use approx here to start with for especially high dim problems
        self.logger.statement("using full parcov.. forming inverse sqrt parcov matrix")
        self.parcov_inv_sqrt = self.parcov.inv.sqrt

        if parensemble is not None:
            self.logger.log("initializing with existing par ensembles")
            if isinstance(parensemble,str):
                self.logger.log("loading parensemble from file")
                if not os.path.exists(parensemble):
                    self.logger.lraise("can not find parensemble file: {0}".\
                                       format(parensemble))
                df = pd.read_csv(parensemble,index_col=0)
                #df.index = [str(i) for i in df.index]
                self.parensemble_0 = ParameterEnsemble.from_dataframe(df=df,pst=self.pst)
                self.logger.log("loading parensemble from file")

            elif isinstance(parensemble,ParameterEnsemble):
                self.parensemble_0 = parensemble.copy()

            else:
                raise Exception("unrecognized arg type for parensemble, " +\
                                "should be filename or ParameterEnsemble" +\
                                ", not {0}".format(type(parensemble)))

            self.parensemble = self.parensemble_0.copy()

            #assert self.parensemble_0.shape[0] == self.obsensemble_0.shape[0]
            num_reals = self.parensemble.shape[0] # defined here if par ensemble passed
            self.logger.log("initializing with existing par ensemble")

        else:
            self.logger.log("initializing by drawing {0} par realizations".format(num_reals))
            #self.parensemble_0 = ParameterEnsemble.from_gaussian_draw(self.pst,self.parcov,num_reals=num_reals)
            self.parensemble_0 = ParameterEnsemble.from_uniform_draw(self.pst,num_reals=num_reals)
            self.parensemble_0 = ParameterEnsemble.from_dataframe(df=self.parensemble_0 * self.draw_mult,pst=self.pst)
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename +\
                                      self.paren_prefix.format(0))
            self.logger.log("initializing by drawing {0} par realizations".format(num_reals))


        # repeated from above...
        # could use approx here to start with for especially high dim problems
        self.logger.statement("using full parcov.. forming inverse sqrt parcov matrix")
        self.parcov_inv_sqrt = self.parcov.inv.sqrt

        # self.obs0_matrix = self.obsensemble_0.nonzero.as_pyemu_matrix()
        # self.par0_matrix = self.parensemble_0.as_pyemu_matrix()
        
        # repeated
        self.enforce_bounds = enforce_bounds

        if restart_obsensemble is not None:
            # load prev obs ensemble
            self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))
            failed_runs,self.obsensemble = self._load_obs_ensemble(restart_obsensemble)
            #assert self.obsensemble.shape[0] == self.obsensemble_0.shape[0]
            #assert list(self.obsensemble.columns) == list(self.obsensemble_0.columns)
            self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))

        else:
            # run the initial parameter ensemble
            self.logger.log("evaluating initial ensembles")
            failed_runs, self.obsensemble = self._calc_obs(self.parensemble) # run
            self.obsensemble.to_csv(self.pst.filename + self.obsen_prefix.format(0))
            if self.raw_sweep_out is not None:
                self.raw_sweep_out.to_csv(self.pst.filename + "_sweepraw0.csv")
            self.logger.log("evaluating initial ensembles")

        if failed_runs is not None:
            self.logger.warn("dropping failed realizations")
            #failed_runs_str = [str(f) for f in failed_runs]
            #self.parensemble = self.parensemble.drop(failed_runs)
            #self.obsensemble = self.obsensemble.drop(failed_runs)
            self.parensemble.loc[failed_runs,:] = np.NaN
            self.parensemble = self.parensemble.dropna()
            self.obsensemble.loc[failed_runs,:] = np.NaN
            self.obsensemble = self.obsensemble.dropna()

        # check this
        if not self.parensemble.istransformed:
            self.parensemble._transform(inplace=True)
        if not self.parensemble_0.istransformed:
            self.parensemble_0._transform(inplace=True)

        #self.phi = Phi(self)


        '''if self.drop_bad_reals is not None:
            #drop_idx = np.argwhere(self.current_phi_vec > self.drop_bad_reals).flatten()
            #comp_phi = self.phi.comp_phi
            #drop_idx = np.argwhere(self.phi.comp_phi > self.drop_bad_reals).flatten()
            #meas_phi = self.phi.meas_phi
            drop_idx = np.argwhere(self.phi.meas_phi > self.drop_bad_reals).flatten()
            run_ids = self.obsensemble.index.values
            drop_idx = run_ids[drop_idx]
            if len(drop_idx) == self.obsensemble.shape[0]:
                raise Exception("dropped all realizations as 'bad'")
            if len(drop_idx) > 0:
                self.logger.warn("{0} realizations dropped as 'bad' (indices :{1})".\
                                 format(len(drop_idx),','.join([str(d) for d in drop_idx])))
                self.parensemble.loc[drop_idx,:] = np.NaN
                self.parensemble = self.parensemble.dropna()
                self.obsensemble.loc[drop_idx,:] = np.NaN
                self.obsensemble = self.obsensemble.dropna()

                self.phi.update()'''

        #self.phi.report(cur_lam=0.0)

        #self.last_best_mean = self.phi.comp_phi.mean()
        #self.last_best_std = self.phi.comp_phi.std()

        #self.logger.statement("initial phi (mean, std): {0:15.6G},{1:15.6G}".\
        #                      format(self.last_best_mean,self.last_best_std))
        #if init_lambda is not None:
         #   self.current_lambda = float(init_lambda)
        #else:
            #following chen and oliver
         #   x = self.last_best_mean / (2.0 * float(self.obsensemble.shape[1]))
          #  self.current_lambda = 10.0**(np.floor(np.log10(x)))
        #self.logger.statement("current lambda:{0:15.6g}".format(self.current_lambda))

        #self.delta_par_prior = self._calc_delta_par(self.parensemble_0)
        #u,s,v = self.delta_par_prior.pseudo_inv_components(eigthresh=self.pst.svd_data.eigthresh)
        #self.Am = u * s.inv
        #if self.save_mats:
         #   np.savetxt(self.pst.filename.replace(".pst",'.') + "0.prior_par_diff.dat", self.delta_par_prior.x, fmt="%15.6e")
         #   np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am_u.dat",u.x,fmt="%15.6e")
         #   np.savetxt(self.pst.filename.replace(".pst", '.') + "0.am_v.dat", v.x, fmt="%15.6e")
         #   np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am_s_inv.dat", s.inv.as_2d, fmt="%15.6e")
         #   np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am.dat", self.Am.x, fmt="%15.6e")

        self._initialized = True


    #def update():

    	# Include Wolfe tests
    	# I is default Hessian at k = 0; allow user to specify
    	# Pure python limited memory (L-BFGS) version - truncation of change vectors
        # Re-initialize ensemble at new mean upgrade locs - ala Oliver?
    	# Use Oliver et al. (2008) scaled formulation
    	# bound handling and update length limiting like PEST
        # `use_approx_prior`-like scaling of upgrade



