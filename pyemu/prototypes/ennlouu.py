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

    def initialize(self,num_reals=10,enforce_bounds="reset",finite_diff_grad=False,
    			   parensemble=None,restart_obsensemble=None,draw_mult=1.0,
                   hess=None,constraints=False,working_set=None):#obj_fn_group="obj_fn"):

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
            working_set : list
                list of observations in pst that are taken as initial bounding or ``active'' constraints.
                Suggest using simplex algorithm to get this working_set.

        
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

        self.phi_obs = [x for x in self.pst.obs_names if "obj_f" in x or "phi" in x or "obs" in x]
        if len(self.phi_obs) != 1:
            self.logger.lraise("number of objective function (phi) obs not equal to one")

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

        if finite_diff_grad is False:
            self.finite_diff_grad = False
            # could use approx here to start with for especially high dim problems
            self.logger.statement("using full parcov.. forming inverse sqrt parcov matrix")
            self.parcov_inv_sqrt = self.parcov.inv.sqrt

            # this matrix gets used a lot, so only calc once and store
            self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt

            # define dec var ensemble
            # TODO: add parcov load option here too
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
                # TODO: pyemu method for eval prior information equations
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

        else:  # finite diffs for grads
            self.finite_diff_grad = True
            self.logger.warn("using finite diffs as basis for phi grad vector rather than en approx")

            self.logger.log("running model forward with noptmax = 0")
            self.obsensemble = self._calc_obs_fd()
            # TODO: WRITE TO FILE FOR PHI_PROG PLOTTER TO GRAB
            self.obsensemble[self.phi_obs].to_csv(self.pst.filename + ".phi.{0}.initial.csv".format(self.iter_num))
            self.obsensemble_next = self.obsensemble.copy()  # for Wolfe tests only
            self.logger.log("running model forward with noptmax = 0")

            self.working_set, self.working_set_per_k = [], {}  #TODO: should do just w dict

        if constraints is True:
            constraint_gps = [x for x in self.pst.obs_groups if x.startswith("g_") or x.startswith("greater_")
                              or x.startswith("l_") or x.startswith("less_")]
            if len(constraint_gps) == 0:
                self.logger.lraise("no constraint groups found")

            all_constraints = []
            for cg in constraint_gps:
                cs = self.pst.observation_data.loc[(self.pst.observation_data["obgnme"] == cg) & \
                                                   (self.pst.observation_data["weight"] > 0), :]
                all_constraints.append(cs.obsnme.to_list())
            all_constraints = [i for subl in all_constraints for i in subl]
            if len(all_constraints) == 0:
                self.logger.lraise("no constraint obs found")
            self.constraint_set = self.pst.observation_data.loc[all_constraints, :]  # all constraints (that could become active)

            #self.biobj_per_k, self.working_set_per_k = {}, {}

            # assuming use of active-set method
            self.working_set = self.constraint_set.loc[[x for x in working_set], :]
            self.working_set_ineq = self.working_set.loc[self.working_set.obgnme.str.startswith("eq_") == False, :]
            self.not_in_working_set = self.constraint_set.drop(self.working_set.obsnme, axis=0)
            #TODO: add test here to ensure full-row-rank. Linear independence is a requirement. See active set func.

            self.logger.log("checking here feasibility and initializing constraint filter")
            self._filter = pd.DataFrame()
            self._filter, _accept, c_viol = self._filter_constraint_eval(self.obsensemble, self._filter)
            self._filter.to_csv("filter.{0}.csv".format(self.iter_num))
            self.logger.log("checking here feasibility and initializing constraint filter")

        # Hessian
        if hess is not None:
            # TODO: add supporting for loading Hessian or assoc grad col vectors
            pass
            if not np.all(np.linalg.eigvals(self.hessian.as_2d) > 0):
                self.logger.lraise("Hessian matrix is not positive definite")
        else:
            pnames = self.pst.adj_par_names
            self.hessian = Matrix(x=np.eye(len(pnames),len(pnames)), row_names=pnames, col_names=pnames)
        self.hessian_0 = self.hessian.copy()
        self.inv_hessian = self.hessian.inv
        self.inv_hessian_0 = self.inv_hessian.copy()

        self.search_d_per_k = {}

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
            mean = np.array(ensemble.mean(axis=0))
            # TODO: implement
            #mean = np.array(self.pst.parameter_data.parval1)  # TODO: Eq (10) Hansen 2006 - use mean from all iters?
        else:
            mean = np.array(ensemble.mean(axis=0))
        delta = ensemble.copy()
        delta = Matrix(x=delta.values, row_names=ensemble.index, col_names=ensemble.columns)
        for i in range(delta.shape[0]):
            delta.x[i, :] -= mean

        return delta

    def _calc_obs_fd(self):
        self.pst.control_data.noptmax = 0
        if self._initialized:  # testing alphas
            pst_fname = self.pst.filename.split(".pst")[0] + "_fds_1.pst"
        else:  # just running forward
            pst_fname = self.pst.filename
        self.pst.write(os.path.join(pst_fname))
        pyemu.os_utils.run("pestpp-glm {0}".format(pst_fname))
        rei = pyemu.pst_utils.read_resfile(os.path.join(pst_fname.replace(".pst", ".rei")))
        #phi = rei.loc[rei.group == "obj_fn", "modelled"]# - rei.loc[rei.group == "obj_fn", "measured"]
        #return phi
        obs = rei.loc[:, "modelled"]
        return obs

    def _calc_jco(self, derinc, suffix="_fds", hotstart_res=False):
        self.pst.control_data.noptmax = -2
        self.pst.parameter_groups.derinc = derinc
        if hotstart_res:
            self.pst.pestpp_options["hotstart_resfile"] = self.pst.filename.split(".pst")[0] + suffix + ".rei"
        pst_fname = self.pst.filename.split(".pst")[0] + suffix + ".pst"
        self.pst.write(os.path.join(pst_fname))
        pyemu.os_utils.run("pestpp-glm {0}".format(pst_fname))
        jco = pyemu.Jco.from_binary("{0}".format(pst_fname.replace(".pst",".jcb"))).to_dataframe()
        return jco


    def _BFGS_hess_update(self,curr_inv_hess,curr_grad,new_grad,delta_par,prev_constr_grad,new_constr_grad,
                          self_scale=True,damped=True,update=True,reduced=False):
        '''
        see, e.g., Oliver, Reynolds and Liu (2008) from pg. 180 for overview.

        Used to perform classic (rank-two quasi-Newton) Hessian update as well (or optionally only) Hessian scaling.

        #This func does not implement Nocedal's ``efficient'' BFGS implementation - use L_BFGS for that implementation.

        Parameters
        -------
        self_scale : bool
            see EnsembleSQP.update args docstring
        update : bool
            see EnsembleSQP.update args docstring
        damped : bool
            see EnsembleSQP.update args docstring
        reduced : bool
            see EnsembleSQP.update args docstring

        '''

        # ingredients: h, s, y
        if len(self.working_set_per_k[self.iter_num]) > 0 and reduced is True:  # A is non-empty and only reduced-hess update
            if self.z is not None:
                self.logger.log("implementing reduced-Hessian BFGS update")
                self.H = Matrix(x=self.zTgz, row_names=[x for x in curr_inv_hess.row_names[:self.z.shape[1]]],
                                col_names=[x for x in curr_inv_hess.row_names[:self.z.shape[1]]])
                #self.z = Matrix(x=self.z, row_names=curr_inv_hess.row_names,
                 #               col_names=[x for x in curr_inv_hess.row_names[:self.z.shape[1]]])
                #self.H = self.z.T * curr_inv_hess * self.z
            else:
                self.logger.lraise("ZTGZ is None as null space has no dim")  # TODO
        else:
            self.H = curr_inv_hess

        # s
        if len(self.working_set_per_k[self.iter_num]) > 0 and reduced is True:  # A is non-empty and only reduced-hess update
            # part of step in null space of constraint jco (pg. 538)
            self.s = self.best_alpha_per_it[self.iter_num] * \
                     Matrix(x=self.p_z, col_names=["mean"], row_names=self.H.row_names)  #["n{}".format(x + 1) for x in range(self.p_z.shape[1])])
        else:
            self.s = delta_par.T  # whole step

        # y
        if len(self.working_set_per_k[self.iter_num]) > 0:  # active set and A is non-empty
            new_grad.col_names, curr_grad.col_names = self.parensemble_mean.row_names, self.parensemble_mean.row_names

            # difference in A between k and k+1 - wrt working set at current iter_num (i.e., A^k (prev constraint grads) may contain elements that were not active at iteration k)
            # TODO: double-check math here - but I think this is the only way this can be done
            a_kp1 = new_constr_grad.df().drop(self.not_in_working_set.obsnme, axis=1)
            a_k = prev_constr_grad.df().drop(self.not_in_working_set.obsnme, axis=1)
            a_kp1 = Matrix(x=a_kp1, row_names=a_kp1.index, col_names=a_kp1.columns)  # note: A here is transposed compared to elsewhere
            a_k = Matrix(x=a_k, row_names=a_k.index, col_names=a_k.columns)  # note: A here is transposed compared to elsewhere
            assert a_kp1.shape[1] == len(self.working_set) == a_k.shape[1]

            self.y = (new_grad - curr_grad) - ((a_kp1 - a_k) * self.lagrang_mults)  # see eq (36) of Liu and Reynolds (2019) and Gill Ch. 3
            if reduced is True:
                self.y = self.z.T * self.y  #Matrix(x=self.z.T * self.y, col_names=["mean"], row_names=self.H.row_names)  #["n{}".format(x + 1) for x in range(self.p_z.shape[1])])
        else:
            self.y = new_grad - curr_grad

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
                #hess_scalar = float((self.s.T * self.s).x / rs.x)  # Zhang
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
                #hess_scalar = float((self.s.T * self.s).x / ys.x)  # Zhang
                if hess_scalar < 0:  # abort
                    self.logger.lraise("hessian scalar is not strictly positive!")
                    self.hess_progress[self.iter_num] = "skip scaling"
                    return self.H, self.hess_progress
            self.H *= hess_scalar
            self.H_cp = self.H.copy()  # in case the update step doesn't work below
            self.logger.log("scaling Hessian...")
            if update is False:
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
            Hy.col_names = self.s.col_names  # TODO: hack - rectify early on

            if damped is True and float(ys.x) <= 0:
                # expanded form of Nocedal and Wright (18.16)
                self.logger.log("...using dampened update form...")
                Hr = self.H * r
                Hr.col_names = self.s.col_names
                rHr = r.T * Hr
                self.H += (float(rs.x + rHr.x)) * ssT.x / float((rs ** 2).x)  # TODO: add scalar handling to mat_handler (Exception on line 473)
                self.H -= (((Hr * self.s.T).x + (self.s * Hr.T).x) / float(rs.x))
                self.logger.log("...using dampened update form...")
            else:
                # expanded form of Nocedal and Wright (6.17)
                self.logger.log("...using standard update form...")
                self.H += (float(ys.x + yHy.x)) * ssT.x / float((ys ** 2).x)  # TODO: add scalar handling to mat_handler (Exception on line 473)
                self.H -= (((Hy * self.s.T).x + (self.s * Hy.T).x) / float(ys.x))
                self.logger.log("...using standard update form...")
            self.logger.log("trying to update Hessian...")

            if not np.all(np.linalg.eigvals(self.H.as_2d) > 0):  #-1 * self.pst.svd_data.eigthresh):  #0): # can't soften otherwise point uphill!
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
                        self.hess_progress[self.iter_num] = "updated (damped) only"
                except NameError:
                    self.hess_progress[self.iter_num] = "updated only"

        # TODO: consider here some interpolation between prev and new H for update... espec approp for en approxs?

        return self.H, self.hess_progress  # TODO: add reduced hessian comments to hess_progress above

    def _LBFGS_hess_update(self,memory=5,self_scale=True,damped=False):
        '''
        Use this for large problems.

        This involves calling _BFGS_hess_update() only with truncation of old gradient information vectors.

        3 < M < 20 (Nocedal). H_0 can vary between iterations too...

        See pg. 178 of Nocedal and Wright and pg. 192 of Oliver et al.
        '''

        if self.reduced_hessian is True:
            self.logger.lraise("not implemented... yet")

        if self.iter_num == 1:
            self.hess_progress[self.iter_num] = "skip scaling and updating"
            self.s_d, self.y_d = {}, {}
            if self.opt_direction == "max":
                return self.inv_hessian_0 * self.phi_grad
            else:
                return -1 * self.inv_hessian_0 * self.phi_grad

        # storing necessary vectors etc
        # TODO: check self.iter_num - 1 as keys
        self.s_d[self.iter_num - 1] = self.delta_parensemble_mean.T
        self.y_d[self.iter_num - 1] = self.phi_grad - self.curr_grad

        # TODO: what about curv condition-based skipping and stability checks?
        #  Unsure how to skip update if curv cond violated when using efficient L-BFGS recursion. Cannot find soln.
        #  Instead, impose Wolfe (or strong Wolfe) conditions to avoid curv cond viol?
        #  Also, implement lim-mem form of standard BFGS (inefficient form) -- not for use but for testing
        # curv condition related tests (and for scaling Hessian below)
        ys = self.y_d[self.iter_num - 1].T * self.s_d[self.iter_num - 1]
        yy = self.y_d[self.iter_num - 1].T * self.y_d[self.iter_num - 1]
        if float(ys.x) <= 0:  # TODO: see alternative at https://arxiv.org/pdf/1802.05374.pdf
            self.logger.lraise("!! curvature condition violated: yTs = {}; should be > 0\n"
                               .format(float(ys.x)) +
                               "  If we update (or scale) Hessian matrix now it will not be positive definite !!\n" +
                               "  Either skipping scaling/updating (not recommended) or dampening...")
            if damped:  # from Powell
                self.logger.warn("TODO dampened form for limited memory BFGS")  # TODO? Maybes not - just strong Wolfe.
            else:  # abort
                self.hess_progress[self.iter_num] = "yTs = {0:8.3E}".format(float(ys.x))
                # effectively want search_d to be (unconstructed) H from prev it and new grad
                # so go through following (again)

        q = self.phi_grad

        if (self.iter_num - 1) <= memory:
            inc = 0
            ibd = self.iter_num
        else:
            inc = self.iter_num - memory - 1
            ibd = memory + 1

        al_d = {}
        for i in reversed(range(1, ibd - 1 + 1)):
            j = i + inc
            al_j = float(((1 / float((self.y_d[j].T * self.s_d[j]).x)) * self.s_d[j].T * q).x)
            al_d[j] = al_j  # TODO: check iterate?
            q -= (al_j * self.y_d[j])

        if self_scale is False:  # not recommended
            r = self.inv_hessian_0 * q
        else:  # scale at all iterations using most recent curv info - default
            hess_scalar = float((ys).x) / float((yy).x)  # Nocedal and Wright, Oliver et al.
            r = hess_scalar * self.inv_hessian_0 * q
        r.col_names = self.s_d[j].col_names  # TODO: hack

        for i in range(1, ibd - 1 + 1):
            j = i + inc
            be = float(((1 / float((self.y_d[j].T * self.s_d[j]).x)) * self.y_d[j].T * r).x)
            r += (self.s_d[j] * (al_d[j] - be))

        # TODO: discard vector pair from storage in k > M. Do when actually becomes memory intesive. Handy to have now.

        self.hess_progress[self.iter_num] = "updating"  # TODO
        if self.opt_direction == "max":
            return r
        else:
            return -1 * r

    def _impose_Wolfe_conds(self, alpha, strong=False, c1=10**-4, c2=0.9, first_cond_only=False,
                            skip_first_cond=False, finite_diff_grad=False):
        '''
        see pg. 172 of Oliver et al.

        First condition dictates sufficient phi decrease. Related to Armijo condition.

        Second condition is related to curvature and it ensures steps aren't too small.
        Strong Wolfe conditions involves a more stringent second condition.
        Therefore, don't impose this test in first iteration, as meaningless, when don't have grad change info.  #TODO: come back to this.

        Note non-Strong Wolfe conditions are sufficient for maintaining pos-def H.

        Require gradients for candidate alphas! Dayum!

        Parameters
        -------
        first_cond_only : bool
            To allow testing of first condition before committing to evaluating the gradient at proposed step
        '''

        # TODO: check not too small a step on first it - as no Hessian limiting here either

        # first condition (testing for sufficient decrease)
        if skip_first_cond is False:
            if self.opt_direction == "max":
                self.logger.lraise("TODO")
            else:
                if self.finite_diff_grad is False:
                    phi_red = self.obsensemble_1.mean() - (self.obsensemble_next.mean() +
                                                           float((c1 * alpha * (self.phi_grad.T * self.search_d)).x))
                else:
                    phi_red = self.obsensemble_1 - (self.obsensemble_next +
                                                    float((c1 * alpha * (self.phi_grad.T * self.search_d)).x))
                if float(phi_red) > 0:
                    self.logger.log("first Wolfe condition violated (with c1 = {0}): {1} !<= 0".format(c1, phi_red))
                    return False
            if first_cond_only:
                self.logger.log("skip second Wolfe condition test until have curv information based on candidate alpha")
                return True

        # second condition
        if strong:
            curv_fac = (np.abs(float((self.phi_grad_1.T * self.search_d).x))) - \
                       (c2 * np.abs(float((self.phi_grad.T * self.search_d).x)))
            if self.opt_direction == "max":
                self.logger.lraise("TODO")
            else:
                if curv_fac < 0:
                    self.logger.log("second (strong) Wolfe condition violated (with c2 = {0}): {1} !<= 0"
                                    .format(c2, curv_fac))
                    return False
        else:
            curv_fac = (float((self.phi_grad_1.T * self.search_d).x)) - \
                          (c2 * float((self.phi_grad.T * self.search_d).x))
            if self.opt_direction == "max":
                self.logger.lraise("TODO")
            else:
                if curv_fac < 0:
                    self.logger.log("second Wolfe condition violated (with c2 = {0}): {1} !<= 0"
                                    .format(c2, curv_fac))
                    return False

        return True


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
                        constraint = self.pst.observation_data.loc[c, "obsval"]
                        viol_ = np.abs(min(model_mean - constraint, 0.0))
                        if viol_ > 0:
                            constraints_violated.append((c, viol_))
                        viol += viol_ ** 2
            elif cg.startswith("l_") or cg.startswith("less_"):
                if len(cs) > 0:
                    for c in cs:
                        model_mean = obsensemble[c].mean()
                        constraint = self.pst.observation_data.loc[c, "obsval"]
                        viol_ = np.abs(min(constraint - model_mean, 0.0))
                        if viol_ > 0:
                            constraints_violated.append((c, viol_))
                        viol += viol_ ** 2


        # mean phi
        mean_en_phi = obsensemble[self.phi_obs[0]].mean()

        # constraint filtering
        filter_thresh = 1e-4
        if self.iter_num == 0:
            if viol > 0:
                self.logger.lraise("initial dec var violates constraints! no bueno! perhaps solve SLP then return.. ")
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
                self._filter = pd.concat((self._filter,
                                          pd.DataFrame([[self.iter_num, alpha, viol, mean_en_phi]],
                                                       columns=['iter_num', 'alpha', 'beta', 'phi'])),
                                         ignore_index=True)

            # now we assess filter pairs following new (acceptable) pair for curr iter to choose best....
            #curr_filter = self._filter.loc[self._filter['iter_num'] == self.iter_num, :]
            #if curr_filter.shape[0] > 0 and curr_filter.loc[curr_filter['alpha'] == alpha, :].shape[0] > 0:
            self._curr_filter = pd.concat((self._curr_filter,
                                           pd.DataFrame([[self.iter_num, alpha, viol, mean_en_phi]],
                                                        columns=['iter_num', 'alpha', 'beta', 'phi'])),
                                          ignore_index=True)  # dont need to drop dominated here as just getting min dist from origin
            curr_filter = self._curr_filter
            if biobj_transf:
                min_beta = curr_filter['beta'].min()  # should be 0.0
                if opt_direction == "max":
                    minmax_phi = np.log10(curr_filter['phi'].max())
                else:
                    minmax_phi = np.log10(curr_filter['phi'].min())
                curr_filter.loc[:, "dist_from_min_origin"] = (((curr_filter['beta'] - min_beta) * biobj_weight) \
                                                              ** 2 + (np.log10(curr_filter['phi']) - minmax_phi) \
                                                              ** 2) ** 0.5
                if curr_filter.loc[curr_filter['alpha'] == alpha, 'dist_from_min_origin'].values[0] == curr_filter[
                    'dist_from_min_origin'].min():
                    #self.biobj_per_k[self.iter_num] = \
                     #   curr_filter.loc[:, [x for x in curr_filter.columns if "phi" in x or "beta" in x]]
                    acceptance = True
            else:
                min_beta = curr_filter['beta'].min()  # should be 0.0
                if opt_direction == "max":
                    minmax_phi = curr_filter['phi'].max()
                else:
                    minmax_phi = curr_filter['phi'].min()
                curr_filter.loc[:, "dist_from_min_origin"] = (((curr_filter['beta'] - min_beta) * biobj_weight) \
                                                              ** 2 + (curr_filter['phi'] - minmax_phi) ** 2) ** 0.5
                if curr_filter.loc[curr_filter['alpha'] == alpha, 'dist_from_min_origin'].values[0] == \
                        curr_filter['dist_from_min_origin'].min():
                    #self.biobj_per_k[self.iter_num] = \
                     #   curr_filter.loc[:, [x for x in curr_filter.columns if "phi" in x or "beta" in x]]
                    acceptance = True


        return self._filter, acceptance, constraints_violated


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
        #TODO: or even just scalar - e.g., like draw mult variable?
        #TODO: finish implementing rank-one update
        #TODO: check use of ``distribution mean''
        #TODO: this needs to be scalable - i.e. when dealing with large en_covs

        '''

        if learning_rate < 0 or learning_rate > 1:
            self.logger.lraise("cov matrix adaptation learning rate not between 0.0 and 1.0")

        if mu_prop <= 0 or mu_prop > 1:
            self.logger.lraise("ruh roh")

        if rank_mu:
            par_en = self.parensemble.copy()
            mu = int(par_en.shape[0] * mu_prop)
            if mu < 2:
                mu = 2
            if self.opt_direction == "min":
                sorted_idx = self.obsensemble.sort_values(ascending=True, by=self.obsensemble.columns[0]).index
            else:
                sorted_idx = self.obsensemble.sort_values(ascending=False, by=self.obsensemble.columns[0]).index
            par_en.index = sorted_idx
            par_en = par_en[:mu]
            sub_delta = self._calc_delta_(par_en,
                                          use_dist_mean_for_delta=use_dist_mean_for_delta)
            if rank_one:  # only ever in addition to rank mu update
                if self.iter_num > 1:
                    en_cov = (1.0 - learning_rate) * en_cov + \
                             learning_rate * mu_learning_prop * 1.0 / mu * (sub_delta.T * sub_delta) + \
                             learning_rate * (1.0 - mu_learning_prop) * (p * p.T)
                    #self.logger.lraise("rank-one update not implemented... yet") #  p = (1.0 - r1_learning_rate) * p + (1.0 - (1.0 - r1_learning_rate) * mu)**0.5 * y check r1_learning_rate << (learning_rate * (1.0 - mu_learning_prop))
            else:
                en_cov = (1.0 - learning_rate) * en_cov + \
                         learning_rate * 1.0 / mu * (sub_delta.T * sub_delta)
                if np.linalg.matrix_rank((sub_delta.T * sub_delta).x) > mu:
                    self.logger.lraise("matrix product should not be of rank greater than mu here")
        else:
            self.logger.lraise("skipping cov matrix adaptation")

        return en_cov

    def _active_set_method(self,first_pass=True,add_to_working_set=None,drop_due_to_stall=False):
        '''
        see alg (16.3) in Nocedal and Wright (2006)

        this func involves the first phase of (16.3) - testing for optimality, dropping constraint from working set,
        and calculating alpha

        working_set is defined as current estimate of active constraints. the concept of the working set
        (and indeed the ``active set'') is specific to the active set method for QP with inequality constraints.

        ``drop due to stall'' arg relates relative stalling only - i.e., wrt the current estimated working set
        '''

        alpha, goto_next_it = 1.0, False

        if first_pass:  # stop-or-drop phase
            approx_converged = self._approx_converge_test()
            if approx_converged and (len(self.working_set) > 0) and \
                    self.working_set_per_k[self.iter_num].values == self.working_set_per_k[self.iter_num - 1].values:
                approx_converged = True
            else:
                approx_converged = False
            if self.qp_solve_method == "null_space":
                if self._detect_proj_off():
                    approx_converged = True
            if drop_due_to_stall and (len(self.working_set) > 0) and \
                    self.working_set_per_k[self.iter_num].values == self.working_set_per_k[self.iter_num - 1].values:
                drop_due_to_stall = True
            if np.all(np.isclose(self.search_d.x, 0.0, rtol=1e-3, atol=1e-3)) or drop_due_to_stall or approx_converged:  #drop_due_to_stall and np.all(np.isclose(self.search_d.x, 0.0, rtol=1, atol=1))):# or approx_converged:  # the former catches when two non-parallel equality constraints are present. TODO: occurs practically when filter stops updating? or search_d?
                # TODO: compute mults at new proposed pos with new A? (16.42)?
                lagrang_mults_ineq = self.lagrang_mults.df().loc[self.working_set_ineq.obsnme, :]  # multiplier sign only iterpret-able for ineq constraints (in working set)
                if np.all(lagrang_mults_ineq.values > 0):
                    self.logger.warn("reached optimal soln!")
                else:
                    to_drop = lagrang_mults_ineq.idxmin()[0]
                    self.working_set = self.working_set.drop(to_drop, axis=0)
                    self.working_set_ineq = self.working_set_ineq.drop(to_drop, axis=0)
                    self.not_in_working_set = self.constraint_set.drop(self.working_set.obsnme, axis=0)
                    goto_next_it = True  # to skips alpha-trial loop but sets x_k+1 for next it, etc.


            # block-and-add phase
            else:  # p != 0
                # compute alpha
                if len(self.not_in_working_set) == 0:
                    #self.logger.warn("all constraints are in active set")
                    alpha = 1.0    #self._compute_alpha_constrained()  # TODO: test this func
                else:
                    alpha = 1.0  #self._compute_alpha_constrained()  # TODO: test this func

        else:  # second pass
            # add constraint to working set where blocking constraints present
            self.working_set = pd.concat((self.working_set,
                                          self.constraint_set.loc[self.constraint_set.obsnme == add_to_working_set, :]))
            self.working_set_ineq = pd.concat((self.working_set_ineq,
                                               self.constraint_set.loc[self.constraint_set.obsnme == add_to_working_set,
                                               :]))
            self.not_in_working_set = self.constraint_set.drop(self.working_set.obsnme, axis=0)

            self.logger.log("check m <= n in A. If not, redundant information, use reduction strategy, SVD or QR")  # TODO!
            #lambdas, V = np.linalg.eig(matrix.T)
            #print(matrix[lambdas == 0, :]) # linearly dependent row vectors

        if first_pass is True and drop_due_to_stall is False:
            return alpha, goto_next_it

    def _kkt_direct(self, g, a, c, h, pyemu_matrix=True):

        if pyemu_matrix is True:
            g, a, c, h = g.x, a.x, c.x, h.x

        # TODO: check should not be -A^T
        coeff = np.concatenate((np.concatenate((g, a.T), axis=1),
                                np.concatenate((a, np.zeros((a.shape[0], a.shape[0]))), axis=1)))
        rhs = np.concatenate((-1.0 * c, h))
        x = np.linalg.solve(coeff, rhs)

        return x

    def _kkt_null_space(self, hessian, constraint_grad, constraint_diff, grad, constraints, cholesky=True, qr=True,
                        pyemu_matrix=True):
        '''
        see pg. 457 of Nocedal and Wright (2006)

        This approach ensures unique soln to kkt system (given that G may be indefinite).

        This approach is ``very effective'' when degrees of freedom (n - m) is small.

        hessian is G matrix in (16.5);
        constraint_grad is A matrix in (16.5)
        constraint_diff is h in (16.5) (h = Ax - b)  #TODO: check this - Ax - b in Ch. 16, but -b in Ch 18
        grad is g in (16.5) (g = c + Gx)  #TODO: check this - c + Gx in Ch. 16, but -c in Ch 18
        '''

        # Requires two conditions be satisfied
        # 0. A has full row rank (i.e., a_i vectors are linearly indep_ - this should have been caught before
        if np.linalg.matrix_rank(constraint_grad.x) != constraint_grad.shape[0]:
            self.logger.lraise("A does not have full row rank... This should have been caught previously... need to use SVD or QR factorization...")

        # 1. Z^TGZ is pos def
        # first, must compute ``null-space basis matrix'' Z (i.e., cols are null-space of A); see pgs. 430-432 and 457
        y, self.z, ay = self._compute_orthog_basis_matrices(a=constraint_grad, qr_mode=qr)  # TODO: re-use Z, Y if Wk same as prev it. Also QR factorization for efficiency here as Z will only change slightly from it to it...
        if self.alg is not "LBFGS":
            if self.reduced_hessian is False or self.iter_num < 3:  # TODO: TIDY
                if self.z is not None:
                    self.zTgz = np.dot(self.z.T, (hessian * self.z).x)
                    if not np.all(np.linalg.eigvals(self.zTgz) > 0):
                        self.logger.log("Z^TGZ not pos-def!")
                else:
                    self.zTgz = None  # TODO: check math here!!
            else:  # reduced hessian
                self.zTgz = hessian.x

        # first, solve for p_y (16.18) - regardless of quasi-Newton approach used
        if qr is False:
            ay = (constraint_grad * y).x
        #if self.reduced_hessian is False:
         #   self.p_y = np.linalg.solve(ay.x, -1.0 * constraint_diff.x)  # rhs should be [0]... if on constraint # TODO: check -1.0 * or not
        #else:
        self.p_y = np.linalg.solve(ay, -1.0 * constraint_diff.x)  # rhs should be [0]... if on constraint # TODO: check -1.0 * or not

        # now to solve linear system for p_z
        # best to do this via Cholesky factorization of reduced Hessian in (16.19) for speed-ups
        try:
            if self.alg is not "LBFGS":  # we have a hessian (potentially the reduced form, e.g., pg. 540)
                if self.z is not None:
                    if self.reduced_hessian is False:  # full hessian
                        zTgy = np.dot(self.z.T, (hessian * y).x)
                        rhs = (-1.0 * np.dot(zTgy, self.p_y)) - np.dot(self.z.T, grad.x)  # np.dot(self.z.T, self.phi_grad.x)  # TODO: +/- and drop second order?
                        if cholesky:
                            l = np.linalg.cholesky(self.zTgz)
                            rhs2 = np.linalg.solve(l, rhs)  # TODO: solve by forward substitution (triangular) for more speed-ups
                            self.p_z = np.linalg.solve(l.T, rhs2)
                        else:
                            self.p_z = np.linalg.solve(self.zTgz, rhs)
                    else:  # reduced hessian
                        # simplify by removing cross term (or ``partial hessian'') matrix (zTgy), which is approp when approximating hessian (zTgz) (as p_y goes to zero faster than p_z)
                        rhs = -1.0 * np.dot(self.z.T, self.phi_grad.x)  # note: grad vector here not multiplied by the product (hessian * x)!  # TODO: check
                        if cholesky:
                            self.logger.lraise("to do")  #TODO
                        else:
                            self.p_z = np.linalg.solve(self.zTgz, rhs)  # the reduced hess (zTgz) is much more likely to be pos-def
                else:
                    self.logger.log("null-space dim (wrt active constraints) is zero.. therefore no p_z component")

            else:  # we don't have the hessian (LBFGS)
                self.logger.lraise("not sure if this can be done...")  # TODO: see doi:10.3934/dcdss.2018071

        except LinAlgError:
            self.logger.lraise("Z^TGZ is not pos-def..")  # should have been caught above

        # total step
        if self.z is not None:
            p = np.dot(y, self.p_y) + np.dot(self.z, self.p_z)
        else:
            p = np.dot(y, self.p_y)

        # now to compute lagrangian multipliers
        if self.alg is not "LBFGS":
            if self.reduced_hessian is False:
                # pg. 457 and 538
                rhs = np.dot(y.T, grad.x + (hessian * p).x)  # self.phi_grad.x + (hessian * p).x  #  TODO: drop second order?
                lm = np.linalg.solve(ay.T, rhs)
            else:
                # pg. 539 of Nocedal and Wright (2006)
                # simplify by dropping dependency of lm on hess (considered appropr given p converges to zero whereas grad does not..
                lm = np.linalg.solve((constraint_grad * constraint_grad.T).x, (constraint_grad * grad).x)  #TODO: check this line  #((constraint_grad * constraint_grad.T) * (constraint_grad * self.phi_grad)).x  # note: grad vector here not multiplied by the product (hessian * x)!  # TODO: try * -1.0 here
        else:
            self.logger.lraise("not sure if this can be done...")

        return p, lm

    def _compute_orthog_basis_matrices(self, a, qr_mode=True):
        '''
        if A is sparse and large, QR may take a while, but robust..
        '''
        M, N = a.shape[0], a.shape[1]
        if M > N:
            self.logger.lraise("m > n! A cannot be this shape!")  # should have been caught before here

        if qr_mode is True:  # generalized form of (15.15) via QR decomp (see pg. 432)
            q, r = np.linalg.qr(a.T.x, 'complete')
            y, self.z, ay = q[:, :a.shape[0]], q[:, -(a.shape[1] - a.shape[0]):], r[:a.shape[0], :]  # q, q[:, -(a.shape[1] - a.shape[0]):]  # y cannot be full q... #TODO: break up here and add Y, Z shape tests below
            # TODO: revisit the partitioning here. for small case, same vector spanning Y and Z.. also, based on https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization-algorithms.html#brnox01, use full constraint grad matrix (not just active set).... which I don't understand...
        else:
            # null space basis
            self.z = self._null_space(a)

            # range space basis
            y = a.T.x  # "A^T is a valid choice for Y when A has full row rank" (pg. 539) of Nocedal and Wright (2006)
            # TODO: y via RREF or solve here alternatively? Only if we need to relax need for A to be full rank?

        # check in line with definitions
        if self.z is not None:
            if not np.all(np.isclose((a * self.z).x, 0.0, rtol=1e-2, atol=1e-3)):
                self.logger.lraise("null-space basis violates definition AZ = 0.. spewin..")

        if self.z is not None:
            yz = np.append(y, self.z, axis=1)
            if yz.shape != (a.shape[1], a.shape[1]):
                self.logger.lraise("Y|Z matrix is the wrong shape")
            if np.isclose(np.linalg.det(yz), 0.0, rtol=1e-5, atol=1e-6):
                self.logger.lraise("Y|Z not invertible.. spewin.. revisit orthogonal basis computation")

        if qr_mode is True:
            return y, self.z, ay
        else:
            return y, self.z, []

    def _kkt_schur(self,):
        self.logger.lraise("not implemented...")

    def _kkt_iterative_cg(self,):
        self.logger.lraise("not implemented...")

    def _null_space(self, a):
        tol = self.pst.svd_data.eigthresh
        if np.linalg.matrix_rank(a.x, tol=tol) == a.shape[1]:
            z = None  #np.zeros((a.shape[1], 1))
        else:
            u, s, v = np.linalg.svd(a.x, full_matrices=True)
            # rcond = np.finfo(s.dtype).eps * max(u.shape[0], v.shape[1])
            # tol = np.amax(s) * rcond
            num = np.sum(s > tol, dtype=int)
            z = v[num:, :].T.conj()

        return z

    def _solve_eqp(self,qp_solve_method="null_space"):
        '''
        Direct QP (KKT system) solve method herein---assuming convexity (pos-def-ness) and that A has full-row-rank.
        Direct method here is just for demonstrative purposes---will require the other methods offered here given that
        the KKT system will be indefinite (Nocedal and Wright, Ch 16)

        Refer to (16.5, 18.9) of Nocedal and Wright (2006)

        method : string
            indicates the method employed to solve the KKT system. default is direct (for demo purposes only).
            options include: null_space (pg. 457), schur (pg. 455) and the iterative (TODO) method.
        '''

        g = self.inv_hessian * 2.0  #Matrix(2.0 * np.eye((self.inv_hessian.shape[0])), row_names=self.inv_hessian.row_names, col_names=self.inv_hessian.col_names)
        # TODO: check hessian or inv_hessian

        a = self.constraint_jco.df().drop(self.not_in_working_set.obsnme, axis=1)  # pertains to active constraints only
        a = Matrix(x=a, row_names=a.index, col_names=a.columns).T  # note transpose
        assert a.shape[0] == len(self.working_set)

        # require A to have full row rank - i.e., a_i vectors are linearly indep
        if np.linalg.matrix_rank(a.x) != a.shape[0]:
            self.logger.warn("A does not have full row rank...")
            self.logger.warn("...linearly dependent constraints will be removed via SVD or QR factorization...")
            # TODO: remove linearly dependent constraints here

        x_ = self.parensemble_mean
        #x_.col_names = ['cross-cov']  # hack

        b = Matrix(x=np.expand_dims(self.pst.observation_data.loc[self.working_set.obsnme, "obsval"].values, axis=1),
                   row_names=self.pst.observation_data.loc[self.working_set.obsnme, "obsnme"].to_list(),
                   col_names=["mean"])
        cs = Matrix(x=np.expand_dims(self.obsensemble[[x for x in self.working_set.obsnme]].values, axis=1),
                    row_names=self.working_set.obsnme.to_list(), col_names=["mean"])

        h = (a * x_.T) - b  # TODO: check -1 * constraint grad
        if not np.all(np.isclose(h.x, 0.0, rtol=1e-2, atol=1e-3)):
            self.logger.warn("not sitting on constraint! Ax - b = {0}".format(h.x))  # TODO: enforce at filter rather than here as don't have direction info here.
            #self.logger.lraise("constraint violated! {0}".format(h.x))  # will have been encountered before this point

        grad_vect = self.phi_grad.copy()
        grad_vect.col_names = ['mean']  # hack
        c = grad_vect + np.dot(g, x_.T)  # small g  # TODO: determine whether should be c + Gx or c - Gx

        if self.qp_solve_method == "null_space":
            p, lm = self._kkt_null_space(hessian=g, constraint_grad=a, constraint_diff=h, grad=c, constraints=cs)
        elif self.qp_solve_method == "schur":
            self._kkt_schur()
        elif self.qp_solve_method == "iterative_cg":
            self._kkt_iterative_cg()
        else:  #method == "direct":
            x = self._kkt_direct(g=g, a=a, c=c, h=h)
            p, lm = x[:self.pst.npar_adj], x[self.pst.npar_adj:]  # TODO: do by parnme

        search_d = Matrix(x=p, row_names=x_.T.row_names, col_names=self.phi_grad.col_names)
        lagrang_mults = Matrix(x=lm, row_names=a.row_names, col_names=x_.T.col_names)
        search_d.to_ascii("search_d.{}.dat".format(self.iter_num))
        lagrang_mults.to_ascii("lagrang_mults.{}.dat".format(self.iter_num))
        return search_d, lagrang_mults

    def _compute_alpha_constrained(self,):
        # TODO: re-do A dim - for consistency with above only
        p = self.search_d
        a_ = self.constraint_jco.df().drop(self.working_set.obsnme, axis=1)  # pertains to inactive constraints only
        a_ = Matrix(x=a_, row_names=a_.index, col_names=a_.columns)
        assert a_.shape[1] == len(self.constraint_set) - len(self.working_set)
        assert a_.shape[1] == len(self.not_in_working_set)

        ap = a_.T * p  # transpose here is to do A-col-wise dot prod with p (i.e., np.dot(a.x.T, p.x)
        b = Matrix(x=np.expand_dims(self.pst.observation_data.loc[self.not_in_working_set.obsnme, "obsval"]
                                    .values, axis=0),
                   row_names=[self.pst.observation_data.loc[self.not_in_working_set.obsnme, "obsnme"][0]],
                   col_names=["mean"])
        # TODO: add lt or gt constraint conditional here or -1 * A
        bax = b - (a_.T * self.parensemble_mean.T)
        ap_bax = np.concatenate((ap, bax), axis=1)
        ap_bax_lt0 = ap_bax[ap_bax.x < 0.0]
        ap_bax_q = ap_bax_lt0[ap] / ap_bax_lt0[bax]

        min_q, min_idx = min(ap_bax_lt0), ap_bax_lt0.df().idxmin()
        alpha = min(1.0, min_q)
        if alpha < 1.0:
            self.logger.log("the blocking constraint is... {}".format(min_idx))

        return alpha

    def _detect_proj_off(self,):
        '''
        detects orthogonality of p wrt active constraint
        '''
        proj_off = False
        if self.p_y / self.p_z > 1.0:
            proj_off = False
        return proj_off

    def _approx_converge_test(self,):
        '''done on basis of filter (self.iter_num, self.working_set, distance_from_origin). Not done on basis of
        search_d because gradient direction, not only length, changes when you have a unique working set. Using
        distance from origin in phi, viol plot only general way to do this?'''

        is_converged = False
        if self.iter_num > 1:
            red_fac = np.sum(np.abs(self.search_d.x)) / np.sum(np.abs(self.search_d_per_k[self.iter_num - 1].values))
            if red_fac > 0.95 and (len(self.working_set) > 0) and self.working_set_per_k[self.iter_num].values == self.working_set_per_k[self.iter_num - 1].values:
                is_converged = True

        return is_converged


    def update(self,step_mult=[1.0],alg="BFGS",memory=5,hess_self_scaling=True,damped=True,
               grad_calc_only=False,finite_diff_grad=False,hess_update=True,strong_Wolfe=True,
               constraints=False,biobj_weight=1.0,biobj_transf=True,opt_direction="min",
               cma=False,derinc=0.01,
               rank_one=False, learning_rate=0.5, mu_prop=0.25,
               use_dist_mean_for_delta=False,mu_learning_prop=0.5,
               reduced_hessian=False,qp_solve_method="null_space"):#localizer=None,run_subset=None,
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
        hess_self_scaling : bool # or int
            indicate whether/how current Hessian is to be scaled - i.e., multiplied by a scalar reflecting
            gradient and step information.  Highly recommended - particularly at early iterations.
            See Nocedal and Wright and Oliver et al. for full treatment.
            False means do not scale at any iter;
            For BFGS, True means scale curr H at single iteration (at iter 2, i.e. once we have grad change info);
            For L-BFGS, True means scale H_0 by most recent grad info
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

        #hess_update = False

        self.alg = alg
        next_it = False
        self.reduced_hessian = reduced_hessian
        self.working_set.to_csv("working_set_it{}.csv".format(self.iter_num))
        self._curr_filter = pd.DataFrame()

        if opt_direction is "min" or "max":
            self.opt_direction = opt_direction
        else:
            self.logger.lraise("need proper opt_direction entry")

        if self.reduced_hessian is True:
            self.logger.warn("null-space QP soln scheme required for reduced hessian implementation.. switching...")
            qp_solve_method = "null_space"
        self.qp_solve_method = qp_solve_method

        self.iter_num += 1
        self.logger.log("iteration {0}".format(self.iter_num))
        if finite_diff_grad is False:
            self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))

        if self.iter_num == 1:
            self.parensemble_mean = None

        self.working_set_per_k[self.iter_num] = self.working_set.obsnme

        # some checks first
        if finite_diff_grad is False:
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

        if finite_diff_grad is True:
            self.logger.log("compute phi grad using finite diffs")
            if alg == "LBFGS" and self.iter_num > 2:
                self.logger.log("using jco from wolfe testing during previous upgrade evaluations")
                self.phi_grad = self.phi_grad_next.copy()
                self.logger.log("using jco from wolfe testing during previous upgrade evaluations")
            else:
                par = self.pst.parameter_data
                #if "supply2" in self.pst.filename:  # TODO: temp hack: -1 here to account for par.scale. And revisit for logs.
                 #   par.loc[par['partrans'] == "none", "parval1"] = self.parensemble_mean_1.T.x * -1
                #else:
                if self.iter_num > 1:
                    if self.parensemble_mean_next is not None:
                        par.loc[par['partrans'] == "none", "parval1"] = self.parensemble_mean_next.T.x
                jco = self._calc_jco(derinc=derinc)
                # TODO: get dims from npar_adj and pargp flagged as dec var, operate on phi vector of jco only
                self.phi_grad = Matrix(x=jco.loc[self.phi_obs,:].T.values,
                                       row_names=self.pst.adj_par_names, col_names=['cross-cov'])
                if constraints is True:
                    #if len(self.working_set.obsnme) > 0:
                    self.constraint_jco = Matrix(x=jco.loc[self.constraint_set.obsnme, :].T.values,
                                                 row_names=self.pst.adj_par_names, col_names=self.constraint_set.obsnme)
            if grad_calc_only:
                return self.phi_grad
            # and need mean for upgrades
            if self.parensemble_mean is None:
                self.parensemble_mean = np.array(self.pst.parameter_data.parval1)
                self.parensemble_mean = Matrix(x=np.expand_dims(self.parensemble_mean, axis=0),
                                               row_names=['mean'], col_names=self.pst.par_names)
            self.logger.log("compute phi grad using finite diffs")
        else:
            self.logger.log("compute phi grad using ensemble approx")
            if alg == "LBFGS" and self.iter_num > 2 and constraints is False:  # constraints as need phi_grad for Lagrangian
                self.logger.log("using jco from wolfe testing during previous upgrade evaluations")
                self.phi_grad = self.phi_grad_next.copy()
                self.logger.log("using jco from wolfe testing during previous upgrade evaluations")
            else:
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
                self.en_crosscov_decvar_phi = self._calc_en_crosscov_decvar_phi(self.parensemble,self.obsensemble)  #TODO: check self.obsenemble here is ``current''
                self.logger.log("compute dec var-phi en cross-covariance vector")

                if cma is True:
                    self.logger.log("undertaking dec var cov mat adaptation")
                    self.en_cov_decvar = self._cov_mat_adapt(self.en_cov_decvar,
                                                         rank_one=rank_one,learning_rate=learning_rate,mu_prop=mu_prop,
                                                         use_dist_mean_for_delta=use_dist_mean_for_delta,
                                                         mu_learning_prop=mu_learning_prop)
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

                self.logger.log("compute ensemble approx constraint jacobian")
                # TODO!
                self.logger.log("compute ensemble approx constraint jacobian")

                if grad_calc_only:
                    return self.phi_grad
                self.logger.log("compute phi grad using ensemble approx")

        # compute search direction
        self.logger.log("calculate search direction and perform tests")
        if constraints is True and len(self.working_set) > 0:  # active constraints present
                self.logger.log("calculate search direction and perform tests")
                self.logger.log("solve QP sub-problem (active set method)")
                #self.working_set_per_k[self.iter_num] = self.working_set.obsnme
                self.search_d, self.lagrang_mults = self._solve_eqp(qp_solve_method=self.qp_solve_method)
                self.logger.log("solve QP sub-problem (active set method)")
                self.logger.log("calculate search direction and perform tests")

        elif constraints is False or (constraints is True and len(self.working_set) == 0):  # unconstrained or no active constraints
            self.working_set_per_k[self.iter_num] = self.working_set.obsnme
            if alg == "LBFGS":
                self.logger.log("employing limited-memory BFGS quasi-Newton algorithm")
                self.search_d = self._LBFGS_hess_update(memory=memory,self_scale=hess_self_scaling)
                self.logger.log("employing limited-memory BFGS quasi-Newton algorithm")
            elif alg == "BFGS":
                if self.opt_direction == "max":
                    self.search_d = (self.inv_hessian * self.phi_grad)
                else:
                    self.search_d = -1 * (self.inv_hessian * self.phi_grad)
            else:
                self.logger.lraise("algorithm not recognized/supported")

            if constraints is True:
                alpha, next_it = self._active_set_method(first_pass=True)

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
            self.logger.log("calculate search direction and perform tests")

        self.search_d_per_k[self.iter_num] = self.search_d.df()  # for detecting convergence in _active_set_method()
        # implement active set method
        if constraints is True and len(self.working_set) > 0:  # active constraints present
            alpha, next_it = self._active_set_method(first_pass=True)

        if next_it is False:
            self.parensemble_mean_next = None
            step_lengths, mean_en_phi_per_alpha, curv_per_alpha = [], pd.DataFrame(), pd.DataFrame()
            #base_step = 1.0  # start with 1.0 and progressively make smaller (will be 1.0 eventually if convex..)
            # TODO: check notion of adjusting alpha wrt Hessian?  similar to line searching...
            if constraints is True:
                #if len(self.working_set.obsnme) > 0:
                 #   base_step = 1.0
                base_step = alpha
            else:
                base_step = ((self.pst.parameter_data.parubnd.mean()-self.pst.parameter_data.parlbnd.mean()) * 0.1) \
                        / abs(self.search_d.x.mean())  # TODO: check w JTW
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
                                    .format(out_of_bounds.shape[0], step_size, list(out_of_bounds.parnme)))
                    # TODO: or some scaling/truncation strategy?
                    # TODO: could try new alpha (between smaller and the bound violating one)?
                    #continue
                    par.loc[(par.parubnd < par.parval1), "parval1"] = par.parubnd
                    par.loc[(par.parlbnd > par.parval1), "parval1"] = par.parlbnd
                    # TODO: stop alpha testing from here...
                    self.logger.log("{0} mean dec vars for step {1} out-of-bounds: {2}..."
                                    .format(out_of_bounds.shape[0], step_size, list(out_of_bounds.parnme)))
                self.logger.log("computing mean dec var upgrade".format(step_size))

                if finite_diff_grad is False:
                    self.logger.log("drawing {0} dec var realizations centred around new mean".format(self.num_reals))
                    # self.parensemble_1 = ParameterEnsemble.from_uniform_draw(self.pst, num_reals=num_reals)
                    self.parensemble_1 = ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                              cov=self.parcov * self.draw_mult,
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

                if finite_diff_grad is False:
                    self.logger.log("evaluating ensembles for step size : {0}".format(','.join("{0:8.3E}".format(step_size))))
                    failed_runs_1, self.obsensemble_1 = self._calc_obs(self.parensemble_1)  # run
                    if "supply2" in self.pst.filename:  #TODO: temp only until pyemu can evaluate pi eqs
                        self.obsensemble = self._append_pi_to_obs(self.obsensemble,"obj_func_en.csv","obj_func",
                                                                  "prior_info_en.csv")

                    # TODO: constraints = from pst # contain constraint val (pcf) and constraint from obsen
                    if constraints:  # and constraints.shape[0] > 0:
                        self.logger.log("adopting filtering method to handle constraints")
                        self._filter, accept, c_viol = \
                            self._filter_constraint_eval(self.obsensemble_1, self._filter, step_size,
                                                         biobj_weight=biobj_weight, biobj_transf=biobj_transf,
                                                         opt_direction=self.opt_direction)
                        self.logger.log("adopting filtering method to handle constraints")
                        if accept:  # TODO: revise this as per the FD implementation---actually merge---this should be general and independent of nature of gradient computation
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

                        # Wolfe/strong Wolfe condition testing
                        if alg == "LBFGS":
                            if self.iter_num > 1:
                                if self._impose_Wolfe_conds(step_size, first_cond_only=True, strong=strong_Wolfe) is True:
                                    self.logger.log("first (sufficiency) Wolfe condition passed...")
                                else:
                                    self.logger.log(
                                        "first (sufficiency) Wolfe condition violated... abort alpha candidate...")
                                    continue  # next alpha # TODO: could potentially skip or go sparser with search from here on?

                                # eval of grad at candidate alpha
                                self.logger.log("compute phi grad using ensemble approx for candidate alpha")
                                # TODO: use rei so can save base run.
                                #  Also, copy and re-use gradients

                                # TODO: bundle below into func (as mostly same as grad eval and search direction calc above)

                                self.logger.log("compute dec var en covariance vector")
                                # TODO: add check for parensemble var = 0 (all dec vars at (same) bounds). Or draw around mean on bound?
                                self.en_cov_decvar_1 = self._calc_en_cov_decvar(self.parensemble_1)
                                # and need mean for upgrades
                                #if self.parensemble_mean is None:
                                 #   self.parensemble_mean = np.array(self.parensemble.mean(axis=0))
                                  #  self.parensemble_mean = Matrix(x=np.expand_dims(self.parensemble_mean, axis=0),
                                   #                                row_names=['mean'], col_names=self.pst.par_names)
                                self.logger.log("compute dec var en covariance vector")

                                self.logger.log("compute dec var-phi en cross-covariance vector")
                                self.en_crosscov_decvar_phi_1 = self._calc_en_crosscov_decvar_phi(self.parensemble_1,
                                                                                                  self.obsensemble_1)
                                self.logger.log("compute dec var-phi en cross-covariance vector")

                                # compute gradient vector and undertake gradient-related checks
                                # see e.g. eq (9) in Liu and Reynolds (2019 SPE)
                                self.logger.log("calculate pseudo inv of ensemble dec var covariance vector")
                                self.inv_en_cov_decvar_1 = self.en_cov_decvar_1.pseudo_inv(
                                    eigthresh=self.pst.svd_data.eigthresh)
                                self.logger.log("calculate pseudo inv of ensemble dec var covariance vector")

                                # TODO: SVD on sparse form of dec var en cov matrix (do SVD on A where Cuu = AA^T - see Dehdari and Oliver)
                                # self.logger.log("calculate pseudo inv comps")
                                # u,s,v = self.en_cov_decvar.pseudo_inv_components(eigthresh=self.pst.svd_data.eigthresh)
                                # self.logger.log("calculate pseudo inv comps")

                                self.logger.log("calculate phi gradient vector")
                                # self.phi_grad = self.inv_en_cov_decvar.T * self.en_crosscov_decvar_phi
                                self.phi_grad_1 = self.inv_en_cov_decvar_1 * self.en_crosscov_decvar_phi_1.T
                                self.logger.log("calculate phi gradient vector")

                                self.logger.log("compute phi grad using ensemble approx for candidate alpha")

                                # and again with Wolfe tests
                                if self._impose_Wolfe_conds(step_size, skip_first_cond=True, strong=strong_Wolfe) is True:
                                    self.logger.log("second (curvature) Wolfe condition passed...")
                                else:
                                    self.logger.log(
                                        "second (curvature) Wolfe condition violated... abort alpha candidate...")
                                    continue  # next alpha  # TODO: could potentially skip or go sparser with search from here on?

                        if float(mean_en_phi_per_alpha.idxmin(axis=1)) == step_size:
                            self.parensemble_mean_next = self.parensemble_mean_1.copy()
                            self.parensemble_next = self.parensemble_1.copy()
                            if alg == "LBFGS" and self.iter_num > 1:
                                self.phi_grad_next = self.phi_grad_1.copy()
                            self.obsensemble_next = self.obsensemble_1.copy()
                            [os.remove(x) for x in os.listdir() if (x.endswith(".obsensemble.0000.csv")
                                                                and x.split(".")[2] == str(self.iter_num))]
                            self.obsensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, step_size)
                                                      + self.obsen_prefix.format(0))

                            # TODO: test curv condition here too?
                            # TODO: calc_curv func (which BFGS func uses)
                            delta_parensemble_mean = self.parensemble_mean_next - self.parensemble_mean
                            curr_grad = Matrix(x=np.zeros((self.phi_grad.shape)),
                                           row_names=self.phi_grad.row_names, col_names=self.phi_grad.col_names)
                            y = self.phi_grad - curr_grad  # start with column vector
                            s = delta_parensemble_mean.T  # start with column vector
                            # curv condition related tests
                            ys = y.T * s  # inner product
                            curv_per_alpha.loc["{}".format(step_size), "curv_cond"] = float(ys.x)
                            curv_per_alpha.loc["{}".format(step_size), "mean_en_phi"] = self.obsensemble_1.mean()['obs']
                        #TODO: phi-curv trade-off here, not only for best alpha according to phi

                    self.logger.log("evaluating ensembles for step size : {0}".format(','.join("{0:8.3E}"
                                                                                               .format(step_size))))

                else:  # finite diffs for grads
                    self.logger.log("evaluating model for step size : {0}".format(','.join("{0:8.3E}"
                                                                                           .format(step_size))))
                    self.obsensemble_1 = self._calc_obs_fd()

                    if constraints is True:
                        self.logger.log("adopting filtering method to handle constraints")
                        self._filter, accept, c_viol = \
                            self._filter_constraint_eval(self.obsensemble_1, self._filter, step_size,
                                                         biobj_weight=biobj_weight, biobj_transf=biobj_transf,
                                                         opt_direction=self.opt_direction)
                        self.logger.log("adopting filtering method to handle constraints")
                        if accept:
                            best_alpha = step_size
                            self.best_alpha_per_it[self.iter_num] = best_alpha
                            best_alpha_per_it_df = pd.DataFrame.from_dict([self.best_alpha_per_it])
                            #best_alpha_per_it_df.to_csv("best_alpha.csv")

                            self.parensemble_mean_next = self.parensemble_mean_1.copy()
                            if finite_diff_grad is False:  #TODO: when merging FD and en into one for constraint
                                self.parensemble_next = self.parensemble_1.copy()
                                self.parensemble_1.to_csv(self.pst.filename + ".{0}.{1}"
                                                          .format(self.iter_num, step_size) +
                                                          self.paren_prefix.format(0))
                            [os.remove(x) for x in os.listdir() if (x.endswith(".obsensemble.0000.csv")
                                                                and x.split(".")[2] == str(self.iter_num))]
                            self.obsensemble_1.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, step_size)
                                                      + self.obsen_prefix.format(0))

                            # add blocking constraints
                            if len(c_viol) > 0:
                                # TODO: check search for min viol: constraints_viol['viol'].idxmin(axis=1)
                                add_to_working_set = c_viol[0][0]  # TODO: revisit - use df
                                if add_to_working_set not in self.working_set.obsnme:
                                    self._active_set_method(first_pass=False, add_to_working_set=add_to_working_set)
                                    break  # only allowed to add one per it; therefore break out of loop (could alteratively track changes to WS per it)

                    else:  # unconstrained opt
                        mean_en_phi_per_alpha["{0}".format(step_size)] = self.obsensemble_1[self.phi_obs]

                        # Wolfe/strong Wolfe condition testing
                        if alg == "LBFGS":  # note BFGS is implemented later
                            if self.iter_num > 1:
                                if self._impose_Wolfe_conds\
                                            (step_size, first_cond_only=True, strong=strong_Wolfe) is True:
                                    self.logger.log("first (sufficiency) Wolfe condition passed...")
                                else:
                                    self.logger.log("first (sufficiency) Wolfe condition violated... /"
                                                    "abort alpha candidate...")
                                    continue  # next alpha # TODO: could potentially skip or go sparser with search from here on?

                                # eval of grad at candidate alpha
                                self.logger.log("compute phi grad using finite diffs for candidate alpha")
                                # TODO: use rei so can save base run.
                                # TODO: MAKE SURE PAR SUBSTITUTED BEFORE THIS
                                jco = self._calc_jco(derinc=derinc, suffix="_fds_1_jco", hotstart_res=True)
                                # TODO: get dims from npar_adj and pargp flagged as dec var, operate on phi vector of jco only
                                self.phi_grad_1 = Matrix(x=jco.T.values, row_names=self.pst.adj_par_names,
                                                         col_names=['cross-cov'])
                                self.logger.log("compute phi grad using finite diffs for candidate alpha")

                                # and again with Wolfe tests
                                if self._impose_Wolfe_conds\
                                            (step_size, skip_first_cond=True, strong=strong_Wolfe) is True:
                                    self.logger.log("second (curvature) Wolfe condition passed...")
                                else:
                                    self.logger.log("second (curvature) Wolfe condition violated... /"
                                                    "abort alpha candidate...")
                                    continue  # next alpha  # TODO: could potentially skip or go sparser with search from here on?


                        # phi-curv trade-off per alpha
                        # TODO: eval_phi_curv_tradeoff()
                        if self.iter_num > 1:  # Hess never updated in first step so just take max phi red (no trade off)
                            delta_parensemble_mean = self.parensemble_mean_1 - self.parensemble_mean
                            y = self.phi_grad - self.curr_grad  # start with column vector
                            s = delta_parensemble_mean.T  # start with column vector
                            ys = y.T * s  # inner product
                            curv_per_alpha.loc["{}".format(step_size), "curv_cond"] = float(ys.x)
                            curv_per_alpha.loc["{}".format(step_size), "mean_en_phi"] = self.obsensemble_1['obs']

                        if float(mean_en_phi_per_alpha.idxmin(axis=1)) == step_size:
                            self.parensemble_mean_next = self.parensemble_mean_1.copy()
                            if alg == "LBFGS" and self.iter_num > 1:
                                self.phi_grad_next = self.phi_grad_1.copy()
                            self.obsensemble_next = self.obsensemble_1.copy()
                            [os.remove(x) for x in os.listdir() if
                             (x.startswith("{0}.phi.{1}".format(self.pst.filename,self.iter_num)))
                             and (x.endswith(".csv"))]
                            self.obsensemble_1.to_csv(self.pst.filename + ".phi.{0}.{1}.csv"
                                                      .format(self.iter_num, step_size))

                    self.logger.log("evaluating model for step size : {0}".format(','.join("{0:8.3E}"
                                                                                           .format(step_size))))

        if next_it is False:
            if constraints:
                self._filter.to_csv("filter.{0}.csv".format(self.iter_num))
                if self.parensemble_mean_next is not None:
                    self.parensemble_mean_next.df().to_csv(self.pst.filename + ".{0}.{1}.csv"
                                                           .format(self.iter_num, best_alpha))  #TODO: or forgive here for unsuccessful iter
            else:
                best_alpha = float(mean_en_phi_per_alpha.idxmin(axis=1))
                self.best_alpha_per_it[self.iter_num] = best_alpha
                best_alpha_per_it_df = pd.DataFrame.from_dict([self.best_alpha_per_it])
                best_alpha_per_it_df.to_csv("best_alpha.csv")
                self.logger.log("best step length (alpha): {0}".format("{0:8.3E}".format(best_alpha)))
                if finite_diff_grad is False:
                    self.parensemble_next.to_csv(self.pst.filename + ".{0}.{1}".format(self.iter_num, best_alpha) +
                                             self.paren_prefix.format(0))
                self.parensemble_mean_next.df().to_csv(self.pst.filename + ".{0}.{1}.csv"
                                                       .format(self.iter_num, best_alpha))

            curv_per_alpha.to_csv("curv_and_phi_per_alpha_it{0}.csv".format(self.iter_num))
            mean_en_phi_per_alpha.to_csv("mean_phi_per_alpha_it{0}.csv".format(self.iter_num))

        # deal with unsuccessful iteration
        if next_it is False and self.parensemble_mean_next is None:  # TODO: change to using best from curr k
            self.logger.warn("unsuccessful upgrade iteration.. using previous mean par en")
            self.parensemble_mean_next = self.parensemble_mean.copy()

            if finite_diff_grad is False:  # TODO: change draw mult?
                self.parensemble_next = self.parensemble.copy()

            if constraints and len(self.working_set) > 0:
                self._active_set_method(first_pass=True, drop_due_to_stall=True)  # TODO: stall status could be less naive - e.g., filter stalling over successive iterations...

            hess_update, self_scale = False, False

        # calc dec var changes (after picking best alpha etc)
        # this is needed for Hessian updating via BFGS but also needed for checks
        self.delta_parensemble_mean = self.parensemble_mean_next - self.parensemble_mean
        # TODO: dec var change related checks here - like PEST's RELPARMAX/FACPARMAX

        self.logger.log("scaling and/or updating Hessian via quasi-Newton")
        if self.iter_num == 1 or next_it is True:  # no pre-existing grad or par delta info..
            hess_update = False  # never update at first iter
            #if hess_self_scaling is True or hess_self_scaling == self.iter_num:
             #   self.curr_grad = Matrix(x=np.zeros((self.phi_grad.shape)),
              #                          row_names=self.phi_grad.row_names,col_names=self.phi_grad.col_names)
               # self_scale = True
            #else:
             #   self_scale = False
            self_scale = False
        elif hess_self_scaling is True and self.iter_num == 2:  # or hess_self_scaling == self.iter_num:
            self_scale = True
        else:
            self_scale = False

        if hess_update is True or self_scale is True:
            if alg == "BFGS":
                self.inv_hessian, self.hess_progress = self._BFGS_hess_update(curr_inv_hess=self.inv_hessian,
                                                                              curr_grad=self.curr_grad,
                                                                              new_grad=self.phi_grad,
                                                                              delta_par=self.delta_parensemble_mean,
                                                                              self_scale=self_scale,
                                                                              update=hess_update,
                                                                              damped=damped,
                                                                              prev_constr_grad=self.prev_constr_grad,
                                                                              new_constr_grad=self.constraint_jco,
                                                                              reduced=self.reduced_hessian)

            elif alg == "LBFGS":
                self.logger.log("LBFGS implemented above")
            else:
                self.logger.lraise("alg not recognized/supported")
        else:
            self.hess_progress[self.iter_num] = "skip scaling and updating"

        self.logger.log("scaling and/or updating Hessian via quasi-Newton")
        # copy Hessian, write vectors

        # track grad and dec vars for next iteration Hess scaling and updating
        self.curr_grad = self.phi_grad.copy()
        self.parensemble_mean = self.parensemble_mean_next.copy()
        if finite_diff_grad is False:
            self.parensemble = self.parensemble_next.copy()

        if constraints:
            self.prev_constr_grad = self.constraint_jco.copy()

        pd.DataFrame.from_dict([self.hess_progress]).to_csv("hess_progress.csv")
        # TODO: phi mean and st dev report

        if self.pst.npar_adj < 10:
            #self.H.df()
            self.inv_hessian.to_ascii("hess_it{}.dat".format(self.iter_num))
            if cma is True:
                self.en_cov_decvar.to_ascii("en_decvar_cov_it{}.dat".format(self.iter_num))

        # TODO: implement termination criteria here based on both dec var and phi (abs and rel) convergence