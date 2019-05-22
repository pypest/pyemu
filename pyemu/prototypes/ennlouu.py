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
    			   parensemble=None,restart_obsensemble=None,draw_mult=1.0,
                   hess=None,):#obj_fn_group="obj_fn"):

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
                (just for initial testing, especially on Rosenbrock) a multiplier for scaling
                parensemble draw variance.  Used for drawing dec var en in tighter cluster around mean val
                (e.g., compared to ies and par priors). Dec var en stats here are ``computational'' rather
                than (pseudo-)physical.  If bounds physically credible, as should be, no need to scale down.
            obj_fn_group : str
                obsgnme containing a single obs serving as optimization objective function.
                Like ++opt_dec_var_groups(<group_names>) in PESTPP-OPT.
            hess : pyemu.Matrix or str (optional)
            a matrix or filename to use as initial Hessian (for restarting)

        
        TODO: rename some of above vars in accordance with opt parlance
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

        #TODO: self-identify phi obs
        # new pestpp ++ arg?
        # #if self.pst.pestpp_options[""] is None:
         #   raise Exception("no pestpp_option['opt_obj_func'] entry passed")
        #else:
        #self.logger.warn("assuming pestpp_option[''] points " + \
         #                "to a (single) obs for now (could also be a pi eq or filename)")
        #self.obj_fn_group = obj_fn_group#.lower()
        #self.obj_fn_obs = list(self.pst.observation_data.loc[self.pst.observation_data.obgnme == \
         #                                                     self.obj_fn_group, :].obsnme)
        #if len(self.obj_fn_obs) != 1:
         #   raise Exception("number of obs serving as opt obj function " + \
          #                  "must equal 1, not {0} - see docstring".format(len(self.obj_fn_obs)))

        # could use approx here to start with for especially high dim problems
        self.logger.statement("using full parcov.. forming inverse sqrt parcov matrix")
        self.parcov_inv_sqrt = self.parcov.inv.sqrt

        # this matrix gets used a lot, so only calc once and store
        self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt

        # define dec var ensemble
        #TODO: add parcov load option here too
        if parensemble is not None:
            self.logger.log("initializing with existing par ensembles")
            if isinstance(parensemble,str):
                self.logger.log("loading parensemble from file")
                if not os.path.exists(parensemble):
                    self.logger.lraise("can not find parensemble file: {0}".format(parensemble))
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
            self.logger.log("initializing with existing par ensemble")

        else:
            self.logger.log("initializing by drawing {0} par realizations".format(num_reals))
            #self.parensemble_0 = ParameterEnsemble.from_uniform_draw(self.pst,num_reals=num_reals)
            self.parensemble_0 = ParameterEnsemble.from_gaussian_draw(self.pst,self.parcov,num_reals=num_reals)
            self.parensemble_0 = ParameterEnsemble.from_dataframe(df=self.parensemble_0 * self.draw_mult,pst=self.pst)
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename + self.paren_prefix.format(0))
            self.logger.log("initializing by drawing {0} par realizations".format(num_reals))

        self.num_reals = self.parensemble.shape[0]  # defined here if par ensemble passed
        # self.obs0_matrix = self.obsensemble_0.nonzero.as_pyemu_matrix()
        # self.par0_matrix = self.parensemble_0.as_pyemu_matrix()


        # define phi en by loading prev or computing
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

        # assert self.parensemble_0.shape[0] == self.obsensemble_0.shape[0]


        # Hessian
        if hess is not None:
            #TODO: add supporting for loading Hessian or assoc grad col vectors
            pass
            if not np.all(np.linalg.eigvals(self.hessian.as_2d) > 0):
                self.logger.lraise("Hessian matrix is not positive definite")
        else:
            pnames = self.pst.adj_par_names
            self.hessian = Matrix(x=np.eye(len(pnames),len(pnames)), row_names=pnames, col_names=pnames)
        self.hessian_0 = self.hessian.copy()
        self.inv_hessian = self.hessian.inv
        self.inv_hessian_0 = self.inv_hessian.copy()


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

        self._initialized = True


    def _calc_delta_par(self,parensemble):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        return self._calc_delta(parensemble, self.parcov_inv_sqrt)

    def _calc_delta_obs(self,obsensemble):
        '''
        calc the scaled observation ensemble differences from the mean
        '''
        return self._calc_delta(obsensemble.nonzero, self.obscov_inv_sqrt)


    def _calc_en_cov_decvar(self,parensemble):
        '''
        calc the dec var ensemble (approx) covariance vector (e.g., eq (8) of Dehdari and Oliver 2012 SPE)
        '''
        return self._calc_en_cov_crosscov(parensemble, parensemble)

    def _calc_en_crosscov_decvar_phi(self,parensemble,obsensemble):
        '''
        calc the dec var-phi ensemble (approx) cross-covariance vector (e.g., eq (9) of Dehdari and Oliver 2012 SPE)
        '''
        return self._calc_en_cov_crosscov(parensemble, obsensemble)

    def _calc_en_cov_crosscov(self,ensemble1,ensemble2):
        '''
        general func for calc of ensemble (approx) covariances and cross-covariances.
        '''
        mean1, mean2 = np.array(ensemble1.mean(axis=0)), np.array(ensemble2.mean(axis=0))
        delta1, delta2 = ensemble1.as_pyemu_matrix(), ensemble2.as_pyemu_matrix()
        for i in range(ensemble1.shape[0]):
            delta1.x[i, :] -= mean1
            delta2.x[i, :] -= mean2
        en_cov_crosscov = 1.0 / (ensemble1.shape[0] - 1.0) * ((delta1.x * delta2.x).sum(axis=0))
        if ensemble1.shape[1] == ensemble2.shape[1]:  # diag cov matrix
            en_cov_crosscov = np.diag(en_cov_crosscov)
            en_cov_crosscov = Matrix(x=en_cov_crosscov,
                                     row_names=self.pst.adj_par_names,col_names=self.pst.adj_par_names)
        else:  # cross-cov always a vector
            en_cov_crosscov = Matrix(x=(np.expand_dims(en_cov_crosscov, axis=0)),
                                     row_names=['cross-cov'],col_names=self.pst.adj_par_names)
        return en_cov_crosscov

    def _BFGS_hess_update(self,curr_inv_hess,curr_grad,new_grad,delta_par):
        '''
        see Oliver, Reynolds and Liu (2008) from pg. 180 for overview.
        '''
        # TODO: scaling and Nocedal's efficient implementation. Cross-ref below math.
        # TODO: okay to carry these with self?
        self.H = curr_inv_hess
        self.y = (curr_grad - new_grad)  # start with column vector
        self.s = delta_par.T  # start with column vector
        if (self.y.T * self.s).x <= 0:
            self.logger.lraise("curvature condition violated: yTs = {}; should be > 0".format((self.y.T * self.s).x) +
                               "  Hessian matrix will not be positive definite")

        #ys = self.y.T * self.s
        #Hy = self.H * self.y
        #yHy = self.y.T * Hy
        #self.H += ((ys + yHy) * (self.s.T * self.s)) / ys**2 # CHECK
        #self.H -= ((Hy,self.s) + (self.s,Hy)) / ys

        self.H += (self.y * self.y.T).x / (self.s.T * self.y).x
        self.H -= (self.H * self.s * self.s.T * self.H.T).x / (self.s.T * self.H * self.s).x

        return self.H # TODO: return others too for tests, e.g., grad diff?

    def _LBFGS_hess_update(self,curr_inv_hess,curr_grad,new_grad,step,idx,trunc_thresh):#scaling_method="ZhangReynolds")
        '''
        Use this for large problems
        '''
        # TODO


    def update(self,step_mult=[1.0],alg="BFGS"):#localizer=None,run_subset=None,
        """
        Perform one quasi-Newton update

        Parameters
        -------
        step_mult : list
            a list of step size (length) multipliers to test.  Each mult value will require
            evaluating the parameter ensemble (or a subset thereof).
        run_subset : int
            the number of realizations to test for each step_mult value.
        alg : string
            flag indicating which Hessian updating method to use. Options include "BFGS"
            (classic Broyden–Fletcher–Goldfarb–Shanno) (suited to small problems) or "LBFGS"
            (a limited-memory version of BFGS) (suited to large problems).

        Example
        -------
        ``>>>import pyemu``
        ``>>>esqp = pyemu.EnsembleSQP(pst="pest.pst")``
        ``>>>esqp.initialize(num_reals=100)``
        ``>>>for it in range(5):``
        ``>>>    esqp.update(step_mult=[1.0],run_subset=num_reals/len(step_mult))``

    	#TODO: calc par and obs delta wrt one another rather than mean?
    	#TODO: sub-setting
        """

        self.iter_num += 1
        self.logger.log("iteration {0}".format(self.iter_num))
        self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))

        # some checks first
        if self.obsensemble.shape[0] < 2:
            self.logger.lraise("at least active 2 realizations are needed to update")
        if not self._initialized:
            self.logger.lraise("must call initialize() before update()")

        # compute dec var covariance and dec var-phi cross covariance matrices - they are actually vectors
        self.logger.log("compute dec var en covariance vector")
        self.en_cov_decvar = self._calc_en_cov_decvar(self.parensemble)
        # and need mean for upgrades
        self.parensemble_mean = np.array(self.parensemble.mean(axis=0))
        self.parensemble_mean = Matrix(x=np.expand_dims(self.parensemble_mean, axis=0),\
                                       row_names=['mean'], col_names=self.pst.adj_par_names)
        self.logger.log("compute dec var en covariance vector")

        self.logger.log("compute dec var-phi en cross-covariance vector")
        self.en_crosscov_decvar_phi = self._calc_en_crosscov_decvar_phi(self.parensemble,self.obsensemble)
        self.logger.log("compute dec var-phi en cross-covariance vector")

        # compute gradient vector and undertake gradient-related checks
        # see e.g. eq (9) in Liu and Reynolds (2019 SPE)
        self.logger.log("calculate pseudo inv of ensemble dec var covariance vector")
        self.inv_en_cov_decvar = self.en_cov_decvar.pseudo_inv(eigthresh=self.pst.svd_data.eigthresh)
        self.logger.log("calculate pseudo inv of ensemble dec var covariance vector")

        # TODO: SVD on sparse form of dec var en cov matrix (do SVD on A where Cuu = AA^T - see Dehdari and Oliver)
        #self.logger.log("calculate pseudo inv comps")
        #u,s,v = self.en_cov_decvar.pseudo_inv_components(eigthresh=self.pst.svd_data.eigthresh)
        #self.logger.log("calculate pseudo inv comps")

        self.logger.log("calculate phi gradient vector")
        self.en_phi_grad = self.inv_en_cov_decvar * self.en_crosscov_decvar_phi.T
        self.logger.log("calculate phi gradient vector")

        # compute (quasi-)Newton search direction
        self.logger.log("calculate search direction")
        self.search_d = -1 * (self.inv_hessian * self.en_phi_grad)
        self.logger.log("calculate search direction")
        # TODO: prefer to have a function like `find_direction`? Y

        self.logger.log("phi gradient-related checks")
        if (self.search_d.T * self.en_phi_grad).x > 0:
            self.logger.lraise("search direction does not point down-hill! :facepalm:")
        if (self.search_d.T * self.en_phi_grad).x == 0:
            self.logger.warn("gradient is zero (unless Hessian is NOT pos definite, which is trapped elsewhere)")
        self.logger.log("phi gradient-related checks")

        # TODO: update mean dec var values and re-draw
        # TODO: handling of fixed, transformed etc. dec vars here
        # TODO: test multiple step sizes (multipliers?)
        step_lengths = []
        base_step = 1e-5  # wildass starting guess, just to check plumbing before implementing alpha testing
        # TODO: line search for getting "base" alpha, e.g., Powell (1978)
        for istep,step in enumerate(step_mult):
            step_size = base_step * step
            step_lengths.append(step_size)
            self.logger.log("undertaking calcs for step size (multiplier) : {0}...".format(step_size))

            self.logger.log("computing mean dec var upgrade".format(step_size))
            self.search_d.col_names = ['mean']  # TODO: temp hack
            self.parensemble_mean_1 = self.parensemble_mean + (step_size * self.search_d.T)
            np.savetxt(self.pst.filename + "_en_mean_step_{0}_it_{1}.dat".format(step_size,self.iter_num),\
                       self.parensemble_mean_1.x,fmt="%15.6e")
            # shift parval1
            self.pst.parameter_data.loc[:,"parval1"] += pd.Series(np.squeeze(self.parensemble_mean_1.x, axis=0)).values
            self.logger.log("computing mean dec var upgrade".format(step_size))

            self.logger.log("drawing {0} dec var realizations centred around new mean".format(self.num_reals))
            # self.parensemble_1 = ParameterEnsemble.from_uniform_draw(self.pst, num_reals=num_reals)
            self.parensemble_1 = ParameterEnsemble.from_gaussian_draw(self.pst, self.parcov, num_reals=self.num_reals)
            # TODO: update the parcov empirically based on success or otherwise of previous iteration in terms of phi
            # TODO: alternatively tighten/widen search region to reflect representativeness of gradient (mechanistic)
            # TODO: two sets of bounds: one hard on dec var and one for ensemble just to get grad
            self.parensemble_1 = ParameterEnsemble.from_dataframe(df=self.parensemble_1 * self.draw_mult, pst=self.pst)
            self.parensemble_1.enforce(enforce_bounds=self.enforce_bounds)
            self.parensemble = self.parensemble_1.copy()
            self.parensemble_1.to_csv(self.pst.filename + self.paren_prefix.format(0))
            self.logger.log("drawing {0} dec var realizations centred around new mean".format(self.num_reals))

            self.logger.log("undertaking calcs for step size (multiplier) : {0}...".format(step_size))

            # TODO: localization (with respect to gradient-dec var relationship only)

            # subset if needed
            # and combine lambda par ensembles into one par ensemble for evaluation

            # run the ensemble for diff step size lengths
            self.logger.log("evaluating ensembles for step size : {0}".\
                            format(','.join("{0:8.3E}".format(step_size))))
            failed_runs_1, self.obsensemble_1 = self._calc_obs(self.parensemble_1)  # run
            # TODO: unpack lambda obs ensembles from combined obs ensemble
            # TODO: failed run handling
            self.obsensemble_1.to_csv(self.pst.filename + ".{0}".format(self.iter_num) + self.obsen_prefix.format(0))
            self.logger.log("evaluating ensembles for step size : {0}".\
                            format(','.join("{0:8.3E}".format(step_size))))

        # TODO: undertake Wolfe and en tests. No - our need is superseded by parallel alpha tests
        # TODO: constraint and feasibility KKT checks here
        # TODO: select best upgrade
        # TODO: check for convergence in terms of dec var and phi changes

        # calc dec var changes (after picking best alpha etc)
        # this is needed for Hessian updating via BFGS but also needed for checks
        self.delta_parensemble_mean = self.parensemble_mean - self.parensemble_mean_1
        # TODO: dec var change related checks here

        # update Hessian via quasi-Newton methods - BFGS
        self.logger.log("updating Hessian using L-BFGS/BFGS")
        if self.iter_num == 1: # no pre-existing grad info
            curr_grad = Matrix(x=np.zeros((self.en_phi_grad.shape)),\
                               row_names=self.en_phi_grad.row_names,col_names=self.en_phi_grad.col_names)
        else: # prev grad info
            pass #curr_grad =

        if alg == "BFGS":
            self.inv_hessian = self._BFGS_hess_update(self.inv_hessian,\
                                                      curr_grad, self.en_phi_grad,
                                                      self.delta_parensemble_mean)
        else:  # LBFGS
            self.inv_hessian = self._LBFGS_hess_update(self.inv_hessian,\
                                                      curr_grad, self.en_phi_grad,
                                                      self.delta_parensemble_mean,L)
        self.logger.log("updating Hessian using L-BFGS/BFGS")
        # copy Hessian, write vectors


        # Hessian checks, e.g.,  positive-definite-ness
        if not np.all(np.linalg.eigvals(self.hessian.as_2d) > 0):
            self.logger.lraise("Hessian matrix is not positive definite")
        # TODO: check yT.s > 0 (curvature condition)

        # TODO: save Hessian vectors (as csv)

