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
                   hess=None,constraints=False):#obj_fn_group="obj_fn"):

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
            constraints :
                TODO: something derived from pestpp_options rather than bool

        
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
            self.parensemble_0 = ParameterEnsemble.from_gaussian_draw(self.pst, cov=self.parcov * self.draw_mult,
                                                                      num_reals=num_reals,)
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename + self.paren_prefix.format(0))
            #self.parensemble_0.to_csv(self.pst.filename + ".current" + self.paren_prefix.format(0))  # for `covert.py` for supply2 problem only...
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
            failed_runs, self.obsensemble = self._calc_obs(self.parensemble)  # run
            if "supply2" in self.pst.filename:  #TODO: temp only until pyemu can evaluate pi eqs
                self.obsensemble = self._append_pi_to_obs(self.obsensemble,"obj_func_en.csv","obj_func",
                                                          "prior_info_en.csv")
            self.obsensemble.to_csv(self.pst.filename + self.obsen_prefix.format(0))
            #TODO: pyemu method for eval prior information equations
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

        # nothing really needs to be done here for unconstrained problems..
        # need to start from feasible point in dec var space...
        if constraints:  # and constraints.shape[0] > 0:
            self.logger.log("checking here feasibility and initializing constraint filter")
            self._filter = pd.DataFrame()
            self._filter, _accept = self._filter_constraint_eval(self.obsensemble,self._filter)
            self._filter.to_csv("filter.{0}.csv".format(self.iter_num))
            self.logger.log("checking here feasibility and initializing constraint filter")

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

        self.curr_grad = None
        self.hess_progress, self.best_alpha_per_it = {},{}

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


    def _append_pi_to_obs(self,obsensemble,phi_en_fname="obj_func_en.csv",phi_obsnme="obj_func",
                          constraint_en_fname="prior_info_en.csv"):
        '''
        temp only until pyemu can evaluate pi equations - see convert.py
        '''

        # replace obj func obsensemble col as this is just for carrying purposes - we need ensemble grad info!
        df = pd.DataFrame.from_csv(phi_en_fname)
        obsensemble.loc[:,phi_obsnme] = df.loc[:,phi_obsnme]

        # and constraints
        df = pd.DataFrame.from_csv(constraint_en_fname)
        obsensemble.loc[:,[x for x in self.pst.prior_information.pilbl if "const" in x]] = df.loc[:]

        return obsensemble


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

        delta1 = self._calc_delta_(ensemble1)  # always dec var cov
        if type(ensemble1) != type(ensemble2):  # cross-cov
            delta2 = self._calc_delta_(ensemble2.loc[:,self.phi_obs])
        else:  # cov
            delta2 = self._calc_delta_(ensemble2)

        en_cov_crosscov = 1.0 / (ensemble1.shape[0] - 1.0) * ((delta1.x * delta2.x).sum(axis=0))
        if ensemble1.columns[0] == ensemble2.columns[0]:  # diag cov matrix
            en_cov_crosscov = np.diag(en_cov_crosscov)
            en_cov_crosscov = Matrix(x=en_cov_crosscov,
                                     row_names=self.pst.par_names,col_names=self.pst.par_names)  #TODO: here and elsewhere only adj dec vars
        else:  # cross-cov always a vector
            en_cov_crosscov = Matrix(x=(np.expand_dims(en_cov_crosscov, axis=0)),
                                     row_names=['cross-cov'],col_names=self.pst.par_names)

        return en_cov_crosscov

    def _calc_delta_(self,ensemble,use_dist_mean_for_delta=False):
        '''
        note: this is subtly different to self.calc_delta in EnsembleMethod

        Parameters
        -------
        use_dist_mean_for_delta : bool
            flag for method variation in which the mean is from ``prior'' dec var (i.e., from `parval1`).
            see Fonseca et al. 2013. default False (only used for cov mat adaptation components)
        '''
        if use_dist_mean_for_delta:
            mean = np.array(self.pst.parameter_data.parval1)  # TODO: Eq (10) Hansen 2006 - use mean from all iters?
        else:
            mean = np.array(ensemble.mean(axis=0))
        delta = ensemble.copy()
        delta = Matrix(x=delta.values, row_names=ensemble.index, col_names=ensemble.columns)
        for i in range(delta.shape[0]):
            delta.x[i, :] -= mean

        return delta


    def _BFGS_hess_update(self,curr_inv_hess,curr_grad,new_grad,delta_par,self_scale=True,damped=True,update=True):
        '''
        see, e.g., Oliver, Reynolds and Liu (2008) from pg. 180 for overview.

        Used to perform classic (rank-two quasi-Newton) Hessian update as well (or optionally only) Hessian scaling.
        This func does not implement Nocedal's ``efficient'' BFGS implementation - use L_BFGS for that implementation.

        Parameters
        -------
        self_scale : bool
            see EnsembleSQP.update args docstring
        update : bool
            see EnsembleSQP.update args docstring
        damped : bool
            see EnsembleSQP.update args docstring

        '''
        self.H = curr_inv_hess
        self.y = new_grad - curr_grad  # start with column vector
        self.s = delta_par.T  # start with column vector

        # curv condition related tests
        ys = self.y.T * self.s  # inner product

        if float(ys.x) <= 0:
            self.logger.warn("!! curvature condition violated: yTs = {}; should be > 0\n"
                             .format(float(ys.x)) +
                             "  If we update (or scale) Hessian matrix now it will not be positive definite !!\n" +
                             "  Either skipping scaling/updating (not recommended) or dampening...")
            if damped:  # damped (where required only--where curv cond violated)
                self.logger.log("using damped version of BFGS alg implementation..")
                damp_par = 0.2
                sHs = self.s.T * self.H * self.s  # a scalar
                dampening_cond = damp_par * float(sHs.x)
                if float(ys.x) < dampening_cond:
                    damp_factor = float(((1.0 - damp_par) * sHs).x / (sHs.x - ys.x))  # interpolate between curr and normal update
                else:
                    damp_factor = 1.0  # r = y
                r = (damp_factor * self.y.x) + ((1.0 - damp_factor) * (self.H * self.s).x)
                r = Matrix(x=r,row_names=self.y.row_names,col_names=self.y.col_names)
                rs = r.T * self.s
                #hess_scalar = float((r.T * r).x / rs.x)  # Dakota (Sandia)  #TODO: compare hess_scalars
                #hess_scalar = float(rs.x / (r.T * self.H * r).x)  # Nocedal and Wright, Oliver et al.
                hess_scalar = float(rs.x / (r.T * r).x)  # Nocedal and Wright, Oliver et al.
                self.logger.log("using damped version of BFGS alg implementation..")
                if hess_scalar < 0:  # abort
                    self.logger.warn("can't scale despite dampening...")
                    if self.iter_num == 1:
                        self.logger.warn("...this is expected given absence of grad info at iter_num 0...")  # TODO: skip--do not even attempt for it0?
                    self.hess_progress[self.iter_num] = "skip scaling despite using dampening"
                    return self.H, self.hess_progress
                else:
                    pass
            else:  # abort
                self.hess_progress[self.iter_num] = "yTs = {0:8.3E}".format(float(ys.x))
                if self.iter_num == 1:
                    self.logger.warn("skipping Hessian updates is expected given absence of grad info at iter_num 0...")
                return self.H, self.hess_progress

        # scale
        if self_scale:
            self.logger.log("scaling Hessian...")
            if not (float(ys.x) <= 0):  # not already scaled
                #hess_scalar = float((self.y.T * self.y.x) / ys.x)  # Dakota (Sandia)  #TODO: compare hess_scalars
                #hess_scalar = float(ys.x / (self.y.T * self.H * self.y).x)  # Nocedal and Wright, Oliver et al.
                hess_scalar = float(ys.x / (self.y.T * self.y).x)  # Nocedal and Wright, Oliver et al.
                if hess_scalar < 0:  # abort
                    self.logger.lraise("hessian scalar is not strictly positive!")
                    self.hess_progress[self.iter_num] = "skip scaling"
                    return self.H, self.hess_progress
            self.H *= hess_scalar
            self.H_cp = self.H.copy()  # in case the update step doesn't work below
            self.logger.log("scaling Hessian...")
            if update:
                if damped:
                    self.hess_progress[self.iter_num] = "scaled (using dampening) only: {0:8.3E}".format(hess_scalar)
                else:
                    self.hess_progress[self.iter_num] = "scaled only: {0:8.3E}".format(hess_scalar)
                return self.H, self.hess_progress

        # update
        if update:
            self.logger.log("trying to update Hessian...")
            yHy = self.y.T * self.H * self.y  # also a scalar
            ssT = self.s * self.s.T  # outer prod
            Hy = self.H * self.y

            if damped is True and float(ys.x) <= 0:
                # expanded form of Nocedal and Wright (18.16)
                self.logger.log("...using dampened update form...")
                Hr = self.H * r
                rHr = r.T * Hr
                self.H += (float(rs.x + rHr.x)) * ssT.x / float((rs ** 2).x)  # TODO: add scalar handling to mat_handler (Exception on line 473)
                self.H -= float((Hr.T * self.s).x + (self.s.T * Hr).x) / float(rs.x)
                self.logger.log("...using dampened update form...")
            else:
                # expanded form of Nocedal and Wright (6.17)
                self.logger.log("...using standard update form...")
                self.H += (float(ys.x + yHy.x)) * ssT.x / float((ys ** 2).x)  # TODO: add scalar handling to mat_handler (Exception on line 473)
                self.H -= float((Hy.T * self.s).x + (self.s.T * Hy).x) / float(ys.x)
                self.logger.log("...using standard update form...")
            self.logger.log("trying to update Hessian...")

            if not np.all(np.linalg.eigvals(self.H.as_2d) > 0):  #-1 * self.pst.svd_data.eigthresh):  #0): # can't soften!
                if float(ys.x) <= 0 and damped:
                    self.logger.warn("Hessian update causes pos-def status to be violated despite using dampening...")
                    if self_scale:
                        self.hess_progress[self.iter_num] = "scaled only (damped): {0:8.3E}".format(hess_scalar)
                        self.H = self.H_cp
                    else:
                        self.hess_progress[self.iter_num] = "not scaled or updated"
                        self.H = curr_inv_hess
                else:
                    self.logger.warn("Hessian update causes pos-def status to be violated...")
                    if self_scale:
                        self.hess_progress[self.iter_num] = "scaled only: {0:8.3E}".format(hess_scalar)
                        self.H = self.H_cp
                    else:
                        self.hess_progress[self.iter_num] = "not scaled or updated"
                        self.H = curr_inv_hess
            else:
                try:
                    hess_scalar
                    if self_scale and damped and float(ys.x) <= 0:
                        self.hess_progress[self.iter_num] = "scaled (damped) ({0:8.3E}) and updated".format(hess_scalar)
                    elif self_scale:
                        self.hess_progress[self.iter_num] = "scaled ({0:8.3E}) and updated".format(hess_scalar)
                    else:
                        self.hess_progress[self.iter_num] = "updated only"
                except NameError:
                    self.hess_progress[self.iter_num] = "updated only"

        return self.H, self.hess_progress

    def _LBFGS_hess_update(self,curr_inv_hess,curr_grad,new_grad,delta_par,idx,trunc_thresh=5):
        '''
        Use this for large problems.

        This involves calling _BFGS_hess_update() only with truncation of old gradient information vectors.
        '''
        # TODO

    def _filter_constraint_eval(self,obsensemble,filter,alpha=None,biobj_weight=1.0,biobj_transf=True,
                                opt_direction="min"):
        '''
        '''
        # TODO: description

        constraint_gps = [x for x in self.pst.obs_groups if x.startswith("g_") or x.startswith("greater_")
                           or x.startswith("l_") or x.startswith("less_")]
        if len(constraint_gps) == 0:
            self.logger.lraise("no constraint groups found")

        # compute constraint violation
        viol = 0
        constraints_violated = []
        for cg in constraint_gps:
            cs = list(self.pst.observation_data.loc[(self.pst.observation_data["obgnme"] == cg) & \
                                                    (self.pst.observation_data["weight"] > 0), "obsnme"])
            if cg.startswith("g_") or cg.startswith("greater_"):  #TODO: list of constraints to self at initialization/start of update
                if len(cs) > 0:  # TODO: improve effic here
                    for c in cs:
                        model_mean = obsensemble[c].mean()
                        constraint = self.pst.observation_data.loc[c,"obsval"]
                        viol_ = np.abs(min(model_mean - constraint, 0.0))
                        if viol_ > 0:
                            constraints_violated.append((c, viol_))
                        viol += viol_ ** 2
            elif cg.startswith("l_") or cg.startswith("less_"):
                if len(cs) > 0:
                    for c in cs:
                        model_mean = obsensemble[c].mean()
                        constraint = self.pst.observation_data.loc[c,"obsval"]
                        viol_ = np.abs(min(constraint - model_mean, 0.0))
                        if viol_ > 0:
                            constraints_violated.append((c, viol_))
                        viol += viol_ ** 2


        # mean phi
        phi_obs = [x for x in self.pst.obs_names if "obj_f" in x or "phi" in x]
        if len(phi_obs) != 1:
            self.logger.lraise("number of objective function (phi) obs not equal to one")
        mean_en_phi = obsensemble[phi_obs[0]].mean()

        # constraint filtering
        filter_thresh = 1e-4  #TODO: invest influence of filter_thresh
        #TODO: invest biobj params - this should also be related to transforms on dec vars and constraints
        if self.iter_num == 0:
            if viol > 0:
                self.logger.lraise("initial dec var violates constraints! we're toast!")
            else:  # just add to filter
                self._filter = pd.concat((self._filter, pd.DataFrame([[self.iter_num, 0, viol, mean_en_phi]],
                                                                     columns=['iter_num', 'alpha', 'beta', 'phi'])))
                acceptance = False
        else:
            acceptance = False  # until otherwise below

            # drop pairs that are dominated by new pair
            if opt_direction == "max":
                filter_drop_bool = (viol <= self._filter['beta']) & (mean_en_phi >= self._filter['phi'])
            else:
                filter_drop_bool = (viol <= self._filter['beta']) & (mean_en_phi <= self._filter['phi'])
            self._filter = self._filter.drop(self._filter[(filter_drop_bool)].index)

            # add new dominating pair
            if opt_direction == "max":
                filter_accept_bool = (viol < self._filter['beta']) | (mean_en_phi > self._filter['phi'])
            else:
                filter_accept_bool = (viol < self._filter['beta']) | (mean_en_phi < self._filter['phi'])
            if all(filter_accept_bool.values):
                # see slightly adjusted version in Liu and Reynolds (2019) SPE and accept if <=?
                self.logger.log("passes filter")
                self._filter = pd.concat((self._filter, pd.DataFrame([[self.iter_num, alpha, viol, mean_en_phi]],
                                                                     columns=['iter_num', 'alpha', 'beta', 'phi'])),
                                         ignore_index=True)

            # now we assess filter pairs following new (acceptable) pair for curr iter to choose best....
            curr_filter = self._filter.loc[self._filter['iter_num'] == self.iter_num, :]
            if curr_filter.shape[0] > 0 and curr_filter.loc[curr_filter['alpha'] == alpha,:].shape[0] > 0:
                # TODO: but.. what is the price of violating constraints? We don't want to at all...
                # TODO: either just pick min phi with constraint = 0, or weight constraint viol * 10, and transf?, e.g.
                if biobj_transf:
                    min_beta = curr_filter['beta'].min()
                    if opt_direction == "max":
                        minmax_phi = np.log10(curr_filter['phi'].max())
                    else:
                        minmax_phi = np.log10(curr_filter['phi'].min())
                    curr_filter.loc[:, "dist_from_min_origin"] = (((curr_filter['beta'] - min_beta) * biobj_weight) \
                                                                  ** 2 + (np.log10(curr_filter['phi']) - minmax_phi) \
                                                                  ** 2) ** 0.5
                    if curr_filter.loc[curr_filter['alpha'] == alpha, 'dist_from_min_origin'].values[0] == curr_filter[
                        'dist_from_min_origin'].min():
                        acceptance = True
                else:
                    min_beta = curr_filter['beta'].min()
                    if opt_direction == "max":
                        minmax_phi = curr_filter['phi'].max()
                    else:
                        minmax_phi = curr_filter['phi'].min()
                    curr_filter.loc[:, "dist_from_min_origin"] = (((curr_filter['beta'] - min_beta) * biobj_weight) \
                                                                  ** 2 + (curr_filter['phi'] - minmax_phi) ** 2) ** 0.5
                    if curr_filter.loc[curr_filter['alpha'] == alpha, 'dist_from_min_origin'].values[0] == \
                            curr_filter['dist_from_min_origin'].min():
                        acceptance = True


        return self._filter, acceptance


    def _cov_mat_adapt(self,en_cov,rank_mu=True,rank_one=False,learning_rate=0.1,mu_prop=0.25,
                       use_dist_mean_for_delta=True,mu_learning_prop=0.5):
        '''
        covariance matrix adaptation evolutionary strategy for dec var cov matrix
        see Fonseca et al. 2014 SPE

        rank_mu : bool
            perform rank-mu matrix update (i.e., use information within current iteration). default is True.
        rank_one : bool
            perform rank-one matrix update (i.e., use information between prev iterations). default is False.
        learning_rate : float
            important variable ranging between 0.0 and 1.0. low values are ``safe'' but are of less benefit;
            too large values may cause matrix degeneration.
        use_dist_mean_for_delta : bool
            flag to use mean based on ``prior'' dec var (i.e., from `parval1`).
            see Fonseca et al. 2013. default here is True
        mu_learning_prop : float
            proportion of learning (governed by learning_rate arg) via rank mu update (compared to rank one update).
            Hansen (2011) suggest mu learning more important with larger ensemble size, and vice versa.

        #TODO: try only adapting diag for increased robustness? as per Fonseca
        #TODO: finish implementing rank-one update
        #TODO: check use of ``distribution mean''

        '''

        if learning_rate < 0 or learning_rate > 1:
            self.logger.lraise("cov matrix adaptation learning rate not between 0.0 and 1.0")

        if mu_prop <= 0 or mu_prop > 1:
            self.logger.lraise("ruh roh")

        if rank_mu:
            par_en = self.parensemble.copy()
            mu = int(par_en.shape[0] * mu_prop)
            if self.opt_direction == "min":
                sorted_idx = self.obsensemble.sort_values(ascending=True, by=self.obsensemble.columns[0]).index
            else:
                sorted_idx = self.obsensemble.sort_values(ascending=False, by=self.obsensemble.columns[0]).index
            par_en.index = sorted_idx
            par_en = par_en[:mu]
            sub_delta = self._calc_delta_(par_en,
                                          use_dist_mean_for_delta=use_dist_mean_for_delta)
            if rank_one:  # only ever in addition to rank mu update
                if self.iter_num > 1: # TODO: just scalar?
                    en_cov = (1.0 - learning_rate) * en_cov + \
                             learning_rate * mu_learning_prop * 1.0 / mu * (sub_delta.T * sub_delta) + \
                             learning_rate * (1.0 - mu_learning_prop) * (p * p.T)
                    #self.logger.lraise("rank-one update not implemented... yet") #  p = (1.0 - r1_learning_rate) * p + (1.0 - (1.0 - r1_learning_rate) * mu)**0.5 * y check r1_learning_rate << (learning_rate * (1.0 - mu_learning_prop))
            else:  # TODO: just scalar?
                en_cov = (1.0 - learning_rate) * en_cov + \
                         learning_rate * 1.0 / mu * (sub_delta.T * sub_delta)
                if np.linalg.matrix_rank((sub_delta.T * sub_delta).x) > mu:
                    self.logger.lraise("matrix product should not be of rank greater than mu here")
        else:
            self.logger.lraise("rank_mu not True--skipping cov matrix adaptation")

        return en_cov


    def update(self,step_mult=[1.0],alg="BFGS",hess_self_scaling=True,damped=True,
               grad_calc_only=False,finite_diff_grad=False,hess_update=True,
               constraints=False,biobj_weight=1.0,biobj_transf=True,opt_direction="min",
               cma=False,
               rank_one=False, learning_rate=0.5, mu_prop=0.25,
               use_dist_mean_for_delta=False):#localizer=None,run_subset=None,
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
        hess_self_scaling : bool or int
            indicate whether/how current Hessian is to be scaled - i.e., multiplied by a scalar reflecting
            gradient and step information.  Highly recommended - particularly at early iterations.
            See Nocedal and Wright.
            False means do not scale at any iter; True means scale at all iters (at iter 1, grad is 0.0);
            int values mean scale at only that iter.
        damped : bool
            pg. 537 of Nocedal and Wright  # TODO: document
            # TODO: pass float (damp_param) and activated if not None..
        grad_calc_only : bool
            for testing ensemble based gradient approx (compared to finite differences).
        finite_diff_grad : bool
            flag indicating whether to use finite differences as means of computing gradients
            (rather than ensemble approx).  # TODO: could switch between these adaptively
        constraints :
            TODO: something derived from pestpp_options rather than bool
        opt_direction : str
            must be "min" or "max"
        cma : bool
            to perform or not to perform (dec var) covariance matrix adaptation (evolutionary strategy).
            rank_one, learning_rate, mu_prop and use_dist_mean_for_delta are all cma-related args #heuristic


        Example
        -------
        ``>>>import pyemu``
        ``>>>esqp = pyemu.EnsembleSQP(pst="pest.pst")``
        ``>>>esqp.initialize(num_reals=100)``
        ``>>>for it in range(5):``
        ``>>>    esqp.update(step_mult=[1.0],run_subset=num_reals/len(step_mult))``

    	# TODO: calc par and obs delta wrt one another rather than mean?
    	# TODO: sub-setting
    	# TODO: Langrangian formulation (and localization on constraint relationships)
        """

        if opt_direction is "min" or "max":
            self.opt_direction = opt_direction
        else:
            self.logger.lraise("need proper opt_direction entry")

        self.iter_num += 1
        self.logger.log("iteration {0}".format(self.iter_num))
        self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))

        if self.iter_num == 1:
            self.parensemble_mean = None

        # some checks first
        if self.obsensemble.shape[0] < 2:
            self.logger.lraise("at least active 2 realizations are needed to update")
        if not self._initialized:
            self.logger.lraise("must call initialize() before update()")

        # get phi component of obsensemble  # TODO: remove "obj_fn" option from below and instead x.startwith("phi_")..
        #TODO: move to initialization
        #TODO: similar for dec var par isolation
        phi_obs_gp = [x for x in self.pst.obs_groups if "obj" in x or "phi" in x]  #TODO: do this and below in initialization (in filter)
        if len(phi_obs_gp) != 1:
            self.logger.lraise("number of objective function (phi) obs group found != 1")
        self.phi_obs_gp = phi_obs_gp[0]
        self.phi_obs = list(self.pst.observation_data.loc[self.pst.observation_data["obgnme"] == self.phi_obs_gp,
                                                          "obsnme"])
        if len(self.phi_obs) != 1:
            self.logger.lraise("number of objective function (phi) obs found != 1")

        if finite_diff_grad:
            self.logger.log("compute phi grad using finite diffs")
            # TODO: implement and add test for this
            self.pst.control_data.noptmax = -2
            # self.pst.parameter_groups.derinc = 0.05
            self.pst.write(os.path.join("rosenbrock_2par_fds.pst"))
            pyemu.os_utils.run("pestpp rosenbrock_2par_fds.pst")
            jco = pyemu.Jco.from_binary("rosenbrock_2par_fds.jcb").to_dataframe()
            # TODO: get dims from npar_adj and pargp flagged as dec var
            # TODO: operate on phi vector of jco only
            #self.phi_grad = Matrix(x=jco.values,
             #                      row_names=self.pst.adj_par_names,col_names=['cross-cov'])
            #if grad_calc_only:
             #   return self.phi_grad
            self.logger.log("compute phi grad using finite diffs")
        else:
            self.logger.log("compute phi grad using ensemble approx")
            self.logger.log("compute dec var en covariance vector")
            # TODO: add check for parensemble var = 0 (all dec vars at (same) bounds). Or draw around mean on bound?
            self.en_cov_decvar = self._calc_en_cov_decvar(self.parensemble)
            # and need mean for upgrades
            if self.parensemble_mean is None:
                self.parensemble_mean = np.array(self.parensemble.mean(axis=0))
                self.parensemble_mean = Matrix(x=np.expand_dims(self.parensemble_mean, axis=0),
                                           row_names=['mean'], col_names=self.pst.par_names)
            self.logger.log("compute dec var en covariance vector")

            self.logger.log("compute dec var-phi en cross-covariance vector")
            self.en_crosscov_decvar_phi = self._calc_en_crosscov_decvar_phi(self.parensemble,self.obsensemble)
            self.logger.log("compute dec var-phi en cross-covariance vector")

            if cma:
                self.logger.log("undertaking dec var cov mat adaptation")
                self.en_cov_decvar = self._cov_mat_adapt(self.en_cov_decvar)
                self.logger.log("undertaking dec var cov mat adaptation")

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
            #self.phi_grad = self.inv_en_cov_decvar.T * self.en_crosscov_decvar_phi
            self.phi_grad = self.inv_en_cov_decvar * self.en_crosscov_decvar_phi.T
            self.logger.log("calculate phi gradient vector")
            if grad_calc_only:
                return self.phi_grad
            self.logger.log("compute phi grad using ensemble approx")

        # compute (quasi-)Newton search direction
        self.logger.log("calculate search direction")
        # TODO: for first itn can we make some assumption about step length from bounds? will reduce number of runs
        if self.opt_direction == "max":
            self.search_d = (self.inv_hessian * self.phi_grad)
        else:
            self.search_d = -1 * (self.inv_hessian * self.phi_grad)
        self.logger.log("calculate search direction")

        self.logger.log("phi gradient- and search direction-related checks")
        if (opt_direction == "min" and (self.search_d.T * self.phi_grad).x > 0) or \
                (opt_direction == "max" and (self.search_d.T * self.phi_grad).x < 0):
            self.logger.lraise("search direction does not point down-hill! :facepalm:")
            # TODO: rectify here rather than terminate
        if (self.search_d.T * self.phi_grad).x == 0:
            self.logger.warn("phi gradient is zero!")
        self.logger.log("phi gradient- and search direction-related checks")
        # TODO: using grad info only (with some expected step length), update Hessian from initial
        # TODO: handling of fixed, transformed etc. dec vars here

        self.parensemble_mean_next = None
        step_lengths, mean_en_phi_per_alpha, curv_per_alpha = [], pd.DataFrame(), pd.DataFrame()
        #base_step = 1.0  # start with 1.0 and progressively make smaller (will be 1.0 eventually if convex..)
        # TODO: check notion of adjusting alpha wrt Hessian?  similar to line searching...
        base_step = ((self.pst.parameter_data.parubnd.mean()-self.pst.parameter_data.parlbnd.mean()) * 0.1) \
                    / abs(self.search_d.x.mean())
        # TODO: handle log transforms here
        for istep, step in enumerate(step_mult):
            step_size = base_step * step
            step_lengths.append(step_size)
            self.logger.log("undertaking calcs for step size (multiplier) : {0}...".format(step_size))

            self.logger.log("computing mean dec var upgrade".format(step_size))
            self.search_d.col_names = ['mean']  # TODO: temp hack
            self.parensemble_mean_1 = self.parensemble_mean + (step_size * self.search_d.T)
            #np.savetxt(self.pst.filename + "_en_mean_step_{0}_it_{1}.dat".format(step_size,self.iter_num),
             #          self.parensemble_mean_1.x,fmt="%15.6e")
            # shift parval1  #TODO: change below line to adj dec var pars only..
            par = self.pst.parameter_data
            if "supply2" in self.pst.filename:  # TODO: temp hack: -1 here to account for par.scale
                par.loc[par['partrans'] == "none", "parval1"] = self.parensemble_mean_1.T.x * -1
            else:
                par.loc[par['partrans'] == "none", "parval1"] = self.parensemble_mean_1.T.x
            # and bound handling
            out_of_bounds = par.loc[(par.parubnd < par.parval1) | (par.parlbnd > par.parval1), :]
            if out_of_bounds.shape[0] > 0:
                self.logger.log("{0} mean dec vars for step {1} out-of-bounds: {2}..."
                                .format(out_of_bounds.shape[0],step_size,list(out_of_bounds.parnme)))
                # TODO: or some scaling/truncation strategy?
                # TODO: could try new alpha (between smaller and the bound violating one)?
                #continue
                par.loc[(par.parubnd < par.parval1), "parval1"] = par.parubnd
                par.loc[(par.parlbnd > par.parval1), "parval1"] = par.parlbnd
                # TODO: stop alpha testing from here...
                self.logger.log("{0} mean dec vars for step {1} out-of-bounds: {2}..."
                                .format(out_of_bounds.shape[0],step_size,list(out_of_bounds.parnme)))
            self.logger.log("computing mean dec var upgrade".format(step_size))

            self.logger.log("drawing {0} dec var realizations centred around new mean".format(self.num_reals))
            # self.parensemble_1 = ParameterEnsemble.from_uniform_draw(self.pst, num_reals=num_reals)
            self.parensemble_1 = ParameterEnsemble.from_gaussian_draw(self.pst, cov=self.parcov * self.draw_mult,
                                                                      num_reals=self.num_reals)
            # TODO: alternatively tighten/widen search region to reflect representativeness of gradient (mechanistic)
            # TODO: two sets of bounds: one hard on dec var and one (which can adapt during opt)
            #self.parensemble_1.enforce(enforce_bounds=self.enforce_bounds)  # suffic to check mean
            #self.parensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num,step_size)
             #                         + self.paren_prefix.format(0))
            #self.parensemble_1.to_csv(self.pst.filename + ".current" + self.paren_prefix.format(0))
            # for `covert.py` for supply2 problem only...
            self.logger.log("drawing {0} dec var realizations centred around new mean".format(self.num_reals))

            self.logger.log("undertaking calcs for step size (multiplier) : {0}...".format(step_size))

            self.logger.log("evaluating ensembles for step size : {0}".format(','.join("{0:8.3E}".format(step_size))))
            failed_runs_1, self.obsensemble_1 = self._calc_obs(self.parensemble_1)  # run
            if "supply2" in self.pst.filename:  #TODO: temp only until pyemu can evaluate pi eqs
                self.obsensemble = self._append_pi_to_obs(self.obsensemble,"obj_func_en.csv","obj_func",
                                                          "prior_info_en.csv")

            # TODO: constraints = from pst # contain constraint val (pcf) and constraint from obsen
            if constraints:  # and constraints.shape[0] > 0:
                # TODO: this is perhaps where Lagrangian should come in
                self.logger.log("adopting filtering method to handle constraints")
                self._filter, accept = self._filter_constraint_eval(self.obsensemble_1, self._filter, step_size,
                                                                    biobj_weight=biobj_weight,biobj_transf=biobj_transf,
                                                                    opt_direction=self.opt_direction)
                self.logger.log("adopting filtering method to handle constraints")
                if accept:
                    best_alpha = step_size
                    self.best_alpha_per_it[self.iter_num] = best_alpha
                    best_alpha_per_it_df = pd.DataFrame.from_dict([self.best_alpha_per_it])
                    best_alpha_per_it_df.to_csv("best_alpha.csv")

                    self.parensemble_mean_next = self.parensemble_mean_1.copy()
                    self.parensemble_next = self.parensemble_1.copy()
                    self.parensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, step_size) +
                                              self.paren_prefix.format(0))
                    [os.remove(x) for x in os.listdir() if (x.endswith(".obsensemble.0000.csv")
                                                            and x.split(".")[2] == str(self.iter_num))]
                    self.obsensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, step_size)
                                              + self.obsen_prefix.format(0))

            else:  # unconstrained opt
                mean_en_phi_per_alpha["{0}".format(step_size)] = self.obsensemble_1.mean()
                if float(mean_en_phi_per_alpha.idxmin(axis=1)) == step_size:
                    self.parensemble_mean_next = self.parensemble_mean_1.copy()
                    self.parensemble_next = self.parensemble_1.copy()
                    [os.remove(x) for x in os.listdir() if (x.endswith(".obsensemble.0000.csv")
                                                            and x.split(".")[2] == str(self.iter_num))]
                    self.obsensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, step_size)
                                              + self.obsen_prefix.format(0))

                    # TODO: test curv condition here too?
                    delta_parensemble_mean = self.parensemble_mean_next - self.parensemble_mean
                    curr_grad = Matrix(x=np.zeros((self.phi_grad.shape)),
                                       row_names=self.phi_grad.row_names, col_names=self.phi_grad.col_names)
                    y = self.phi_grad - curr_grad  # start with column vector
                    s = delta_parensemble_mean.T  # start with column vector
                    # curv condition related tests
                    ys = y.T * s  # inner product
                    curv_per_alpha.loc["{}".format(step_size), "curv_cond"] = float(ys.x)
                    curv_per_alpha.loc["{}".format(step_size), "mean_en_phi"] = self.obsensemble_1.mean()['obs']

            self.logger.log("evaluating ensembles for step size : {0}".format(','.join("{0:8.3E}".format(step_size))))

        if constraints:
            self._filter.to_csv("filter.{0}.csv".format(self.iter_num))
        else:
            best_alpha = float(mean_en_phi_per_alpha.idxmin(axis=1))
            self.best_alpha_per_it[self.iter_num] = best_alpha
            best_alpha_per_it_df = pd.DataFrame.from_dict([self.best_alpha_per_it])
            best_alpha_per_it_df.to_csv("best_alpha.csv")
            self.logger.log("best step length (alpha): {0}".format("{0:8.3E}".format(best_alpha)))
            self.parensemble_next.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, best_alpha) +
                                         self.paren_prefix.format(0))

        curv_per_alpha.to_csv("curv_and_phi_per_alpha_it{0}.csv".format(self.iter_num))
        mean_en_phi_per_alpha.to_csv("mean_phi_per_alpha_it{0}.csv".format(self.iter_num))

        # deal with unsuccessful iteration
        if self.parensemble_mean_next is None:
            self.logger.log("unsuccessful upgrade iteration.. using previous mean par en and increasing draw mult")
            self.parensemble_mean_next = self.parensemble_mean.copy()
            self.parensemble_next = self.parensemble.copy()
            # TODO: change draw mult if here

        # TODO: failed run handling
        # TODO: undertake Wolfe and en tests. No - our need is superseded by parallel alpha tests
        # TODO: constraint and feasibility KKT checks here
        # TODO: check for convergence in terms of dec var and phi changes


        # calc dec var changes (after picking best alpha etc)
        # this is needed for Hessian updating via BFGS but also needed for checks
        self.delta_parensemble_mean = self.parensemble_mean_next - self.parensemble_mean
        # TODO: dec var change related checks here - like PEST's RELPARMAX/FACPARMAX

        self.logger.log("scaling and/or updating Hessian via quasi-Newton")
        if self.iter_num == 1:  # no pre-existing grad or par delta info..
            hess_update = False  # never update at first iter
            if hess_self_scaling is True or hess_self_scaling == self.iter_num:
                self.curr_grad = Matrix(x=np.zeros((self.phi_grad.shape)),
                                        row_names=self.phi_grad.row_names,col_names=self.phi_grad.col_names)
                self_scale = True
            else:
                self_scale = False

        elif hess_self_scaling is True or hess_self_scaling == self.iter_num:
            self_scale = True
        else:
            self_scale = False

        if hess_update is True or self_scale is True:
            if alg == "BFGS":
                self.inv_hessian, self.hess_progress_d = self._BFGS_hess_update(self.inv_hessian, self.curr_grad,
                                                                                self.phi_grad,
                                                                                self.delta_parensemble_mean,
                                                                                self_scale=self_scale,
                                                                                update=hess_update,
                                                                                damped=damped)

            else:  # LBFGS
                self.logger.warn("LBFGS not implemented...yet...")
        else:
            self.hess_progress_d[self.iter_num] = "skipping scaling and updating"

        self.logger.log("scaling and/or updating Hessian via quasi-Newton")
        # copy Hessian, write vectors

        # track grad and dec vars for next iteration Hess scaling and updating
        self.curr_grad = self.phi_grad.copy()
        self.parensemble_mean = self.parensemble_mean_next.copy()
        self.parensemble = self.parensemble_next.copy()

        hess_progress_df = pd.DataFrame.from_dict([self.hess_progress_d])
        hess_progress_df.to_csv("hess_progress.csv")


        # TODO: save Hessian vectors (as csv)
        # TODO: phi mean and st dev report

