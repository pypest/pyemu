import os
import numpy as np
import pandas as pd

import pyemu
from pyemu.smoother import EnsembleMethod


class ParetoObjFunc(object):

    def __init__(self, pst, obj_function_dict, logger):

        self.logger = logger
        self.pst = pst
        self.max_distance = 1.0e+30
        obs = pst.observation_data
        pi = pst.prior_information
        self.obs_dict, self.pi_dict = {}, {}
        for name,direction in obj_function_dict.items():

            if name in obs.obsnme:
                if direction.lower().startswith("max"):
                    self.obs_dict[name] = "max"
                elif direction.lower().startswith("min"):
                    self.obs_dict[name] = "min"
                else:
                    self.logger.lraise("unrecognized direction for obs obj func {0}:'{1}'".\
                                       format(name,direction))
            elif  name in pi.pilbl:
                if direction.lower().startswith("max"):
                    self.pi_dict[name] = "max"
                elif direction.lower().startswith("min"):
                    self.pi_dict[name] = "min"
                else:
                    self.logger.lraise("unrecognized direction for pi obj func {0}:'{1}'".\
                                       format(name,direction))
            else:
                self.logger.lraise("objective function not found:{0}".format(name))

        if len(self.pi_dict) > 0:
            self.logger.lraise("pi obj function not yet supported")

        self.logger.statement("{0} obs objective functions registered".\
                              format(len(self.obs_dict)))
        for name,direction in self.obs_dict.items():
            self.logger.statement("obs obj function: {0}, direction: {1}".\
                                  format(name,direction))

        self.logger.statement("{0} pi objective functions registered". \
                              format(len(self.pi_dict)))
        for name, direction in self.pi_dict.items():
            self.logger.statement("pi obj function: {0}, direction: {1}". \
                                  format(name, direction))

    def is_feasible(self, obs_df, risk=0.5):
        """identify which candidate solutions in obs_df (rows)
        are feasible with respect obs constraints (obs_df)

        Parameters
        ----------
        obs_df : pandas.DataFrame
            a dataframe with columns of obs names and rows of realizations
        risk : float
            risk value. If != 0.5, then risk shifting is used.  Otherwise, the
            obsval in Pst is used.  Default is 0.5.


        Returns
        -------
        is_feasible : pandas.Series
            series with obs_df.index and bool values

        """
        # todo deal with pi eqs

        is_feasible = pd.Series(data=True, index=obs_df.index)
        for lt_obs in self.pst.less_than_obs_constraints:
            if risk != 0.5:
                val = self.get_risk_shifted_value(risk,obs_df.loc[lt_obs])
            else:
                val = self.pst.observation_data.loc[lt_obs,"obsval"]
            is_feasible.loc[obs_df.loc[:,lt_obs]>=val] = False
        for gt_obs in self.pst.greater_than_obs_constraints:
            if risk != 0.5:
                val = self.get_risk_shifted_value(risk,obs_df.loc[gt_obs])
            else:
                val = self.pst.observation_data.loc[gt_obs,"obsval"]
            is_feasible.loc[obs_df.loc[:,gt_obs] <= val] = False
        return is_feasible



    def is_nondominated_pathetic(self, obs_df):
        """identify which candidate solutions are pareto non-dominated -
        super patheically slow...

        Parameters
        ----------
        obs_df : pandas.DataFrame
            dataframe with columns of observation names and rows of realizations

        Returns
        -------
        is_dominated : pandas.Series
            series with index of obs_df and bool series
        """
        signs = []
        obj_names = list(self.obs_dict.keys())
        for obj in obj_names:
            if self.obs_dict[obj] == "max":
                signs.append(1.0)
            else:
                signs.append(-1.0)
        signs = np.array(signs)

        obj_df = obs_df.loc[:,obj_names]

        def dominates(idx1,idx2):
            r1 = obj_df.loc[idx1,:]
            r2 = obj_df.loc[idx2,:]
            d = signs * (obj_df.loc[idx1,:] -  obj_df.loc[idx2,:])
            if np.all(d >= 0.0) and np.any(d > 0.0):
                return True
            return False

        is_nondom = []
        for i,iidx in enumerate(obj_df.index):
            ind = True
            for jidx in obj_df.index:
                if iidx == jidx:
                    continue
                if dominates(jidx,iidx):
                    ind = False
                    break
            is_nondom.append(ind)
        is_nondom = pd.Series(data=is_nondom,index=obs_df.index,dtype=bool)
        return is_nondom

    def is_nondominated_continuous(self, obs_df):
        """identify which candidate solutions are pareto non-dominated continuously updated,
        but still slow

        Parameters
        ----------
        obs_df : pandas.DataFrame
            dataframe with columns of observation names and rows of realizations

        Returns
        -------
        is_dominated : pandas.Series
            series with index of obs_df and bool series
        """
        signs = []
        obj_names = list(self.obs_dict.keys())
        for obj in obj_names:
            if self.obs_dict[obj] == "max":
                signs.append(1.0)
            else:
                signs.append(-1.0)
        signs = np.array(signs)

        obj_df = obs_df.loc[:,obj_names]

        def dominates(idx1,idx2):
            r1 = obj_df.loc[idx1,:]
            r2 = obj_df.loc[idx2,:]
            d = signs * (obj_df.loc[idx1,:] -  obj_df.loc[idx2,:])
            if np.all(d >= 0.0) and np.any(d > 0.0):
                return True
            return False

        P = list(obj_df.index)
        PP = set()
        PP.add(P[0])

        iidx = 1
        while iidx < len(P):
            jidx = 0
            drop = []
            keep = True
            for jidx in PP:
                if dominates(iidx,jidx):
                    drop.append(jidx)
                elif dominates(jidx,iidx):
                    keep = False
                    break
            for d in drop:
                PP.remove(d)
            if keep:
                PP.add(iidx)
            iidx += 1




        is_nondom = pd.Series(data=False,index=obs_df.index,dtype=bool)
        is_nondom.loc[PP] = True
        return is_nondom


    def is_nondominated(self, obs_df):
        """identify which candidate solutions are pareto non-dominated using Kungs algorithm

        Parameters
        ----------
        obs_df : pandas.DataFrame
            dataframe with columns of observation names and rows of realizations

        Returns
        -------
        is_dominated : pandas.Series
            series with index of obs_df and bool series
        """
        signs = []
        obj_names = list(self.obs_dict.keys())
        for obj in obj_names:
            if self.obs_dict[obj] == "max":
                signs.append(1.0)
            else:
                signs.append(-1.0)
        signs = np.array(signs)

        obj_df = obs_df.loc[:,obj_names]

        def dominates(idx1,idx2):
            r1 = obj_df.loc[idx1,:]
            r2 = obj_df.loc[idx2,:]
            d = signs * (obj_df.loc[idx1,:] -  obj_df.loc[idx2,:])
            if np.all(d >= 0.0) and np.any(d > 0.0):
                return True
            return False
        ascending = False
        if self.obs_dict[obj_names[0]] == "min":
            ascending = True

        obj_df.sort_values(by=obj_names[0],ascending=ascending,inplace=True)
        P = list(obj_df.index)

        def front(p):
            if len(p) == 1:
                return p
            p = list(obj_df.loc[p,:].sort_values(by=obj_names[0],ascending=ascending).index)
            half = int(len(p) / 2)
            T = front(p[:half])
            B = front(p[half:])
            M = []
            i = 0
            while i < len(B):
                j = 0
                while j < len(T):
                    if dominates(T[j],B[i]):
                        break
                    j += 1
                if (j == len(T)):
                    M.append(B[i])
                i += 1
            T.extend(M)
            return T


        PP = front(P)
        is_nondom = pd.Series(data=False,index=obs_df.index,dtype=bool)
        is_nondom.loc[PP] = True
        return is_nondom


    def crowd_distance(self,obs_df):
        """determine the crowding distance for each candidate solution

        Parameters
        ----------
        obs_df : pandas.DataFrame
            dataframe with columns of observation names and rows of realizations

        Returns
        -------
        crowd_distance : pandas.Series
            series with index of obs_df and values of crowd distance
        """

        # initialize the distance container
        crowd_distance = pd.Series(data=0.0,index=obs_df.index)

        for name,direction in self.obs_dict.items():
            # make a copy - wasteful, but easier
            obj_df = obs_df.loc[:,name].copy()

            # sort so that largest values are first
            obj_df.sort_values(ascending=False,inplace=True)

            # set the ends so they are always retained
            crowd_distance.loc[obj_df.index[0]] += self.max_distance
            crowd_distance.loc[obj_df.index[-1]] += self.max_distance

            # process the vector
            i = 1
            for idx in obj_df.index[1:-1]:
                crowd_distance.loc[idx] += obj_df.iloc[i-1] - obj_df.iloc[i+1]
                i += 1

        return crowd_distance


    def get_risk_shifted_value(self,risk,series):
        n = series.name

        if n in self.obs_dict.keys():
            d = self.obs_dict[n]
            t = "obj"
        elif n in self.pst.less_than_obs_constraints:
            d = "min"
            t = "lt_obs"
        elif n in self.pst.greater_than_obs_constraints:
            d = "max"
            t = "gt_obs"
        else:
            self.logger.lraise("series is not an obs obj func or obs inequality contraint:{0}".\
                               format(n))

        ascending = True
        if d == "min":
            ascending = False
        s = series.shape[0]
        shift = int(s * risk)
        if shift >= s:
            shift = s - 1

        cdf = series.sort_values(ascending=ascending).apply(np.cumsum)
        val = float(cdf.iloc[shift])
        #print(cdf)
        #print(shift,cdf.iloc[shift])
        self.logger.statement("risk-shift for {0}->type:{1}dir:{2},shift:{3},val:{4}".format(n,t,d,shift,val))
        return val

    def reduce_stack_with_risk_shift(self,oe,num_reals,risk):

        stochastic_cols = list(self.obs_dict.keys())
        stochastic_cols.extend(self.pst.less_than_obs_constraints)
        stochastic_cols.extend(self.pst.greater_than_obs_constraints)
        stochastic_cols = set(stochastic_cols)

        vvals = []
        for i in range(0,oe.shape[0],num_reals):
            oes = oe.iloc[i:i+num_reals]
            vals = []
            for col in oes.columns:
                if col in stochastic_cols:
                    val = self.get_risk_shifted_value(risk=risk,series=oes.loc[:,col])
                # otherwise, just fill with the mean value
                else:
                    val = oes.loc[:,col].mean()
                vals.append(val)
            vvals.append(vals)
        df = pd.DataFrame(data=vvals,columns=oe.columns)
        return df



class EvolAlg(EnsembleMethod):
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template"):
        super(EvolAlg, self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                      submit_file=submit_file, verbose=verbose, port=port,
                                      slave_dir=slave_dir)


    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None,par_ensemble=None,risk=0.5,
                   dv_names=None,par_names=None):
        # todo : setup a run results store for all candidate solutions?  or maybe
        # just nondom, feasible solutions?


        if risk != 0.5:
            if risk > 1.0 or risk < 0.0:
                self.logger.lraise("risk not in 0.0:1.0 range")
        self.risk = risk
        self.obj_func = ParetoObjFunc(self.pst,obj_func_dict, self.logger)

        self.par_ensemble_base = None

        # all adjustable pars are dec vars
        if dv_ensemble is None and par_ensemble is None:

            if dv_names is not None:
                aset = set(self.pst.adj_par_names)
                dvset = set(dv_names)
                diff = dvset - aset
                if len(diff) > 0:
                    self.logger.lraise("the following dv_names were not " + \
                                       "found in the adjustable parameters: {0}". \
                                       format(",".join(diff)))
                how = {p: "uniform" for p in dv_names}
            else:
                if risk != 0.5:
                    self.logger.lraise("risk != 0.5 but all adjustable pars are dec vars")
                how = {p: "uniform" for p in self.pst.adj_par_names}

            self.dv_ensemble_base = pyemu.ParameterEnsemble.from_mixed_draws(self.pst, how_dict=how,
                                                                    num_reals=num_dv_reals, cov=self.parcov)
            if risk != 0.5:
                aset = set(self.pst.adj_par_names)
                dvset = set(self.dv_ensemble_base.columns)
                diff = aset - dvset
                if len(diff) > 0:
                    self.logger.lraise("risk!=0.5 but all adjustable parameters are dec vars")
                self.par_ensemble_base = pyemu.ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                                    num_reals=num_par_reals,
                                                                                    cov=self.parcov)
            else:
                self.par_ensemble_base = None

        # dv_ensemble supplied, but not pars, so check if any adjustable pars are not
        # in dv_ensemble, and if so, draw reals for them
        elif dv_ensemble is not None:
            aset = set(self.pst.adj_par_names)
            dvset = set(dv_ensemble.columns)
            diff = dvset - aset
            if len(diff) > 0:
                self.logger.lraise("the following dv_ensemble names were not " + \
                                   "found in the adjustable parameters: {0}". \
                                   format(",".join(diff)))
            self.dv_ensemble_base = dv_ensemble
            if risk != 0.5:
                if par_names is not None:
                    pset = set(par_names)
                    diff = pset - aset
                    if len(diff) > 0:
                        self.logger.lraise("the following par_names were not " + \
                                           "found in the adjustable parameters: {0}". \
                                           format(",".join(diff)))
                        how = {p: "gaussian" for p in par_names}
                else:
                    adj_pars = aset - dvset
                    if len(adj_pars) == 0:
                        self.logger.lraise("risk!=0.5 but all adjustable pars are dec vars")
                    how = {p:"gaussian" for p in adj_pars}
                self.par_ensemble_base = pyemu.ParameterEnsemble.from_mixed_draws(self.pst,how_dict=how,
                                         num_reals=num_par_reals,cov=self.parcov)


        # par ensemble supplied but not dv_ensmeble, so check for any adjustable pars
        # that are not in par_ensemble and draw reals.  Must be at least one...
        elif par_ensemble is not None:
            aset = set(self.pst.par_names)
            pset = set(par_ensemble.columns)
            diff = aset - pset
            if len(diff) > 0:
                self.logger.lraise("the following par_ensemble names were not " + \
                                   "found in the pst par names: {0}". \
                                   format(",".join(diff)))
            self.par_ensemble_base = par_ensemble

            # aset = set(self.pst.adj_par_names)
            # adj_pars = aset - pset
            # if len(adj_pars) == 0:
            #     self.logger.lraise("all adjustable pars listed in par_ensemble, no dec vars available")
            if dv_names is None:
                self.logger.lraise("dv_names must be passed if dv_ensemble is None and par_ensmeble is not None")

            dvset = set(dv_names)
            diff = dvset - aset
            if len(diff) > 0:
                self.logger.lraise("the following dv_names were not " + \
                                   "found in the adjustable parameters: {0}". \
                                   format(",".join(diff)))
            how = {p: "uniform" for p in dv_names}
            self.dv_ensemble_base = pyemu.ParameterEnsemble.from_mixed_draws(self.pst, how_dict=how,
                                                                             num_reals=num_dv_reals,
                                                                             cov=self.parcov,partial=True)

        # both par_ensemble and dv_ensemble were passed, so check
        # for compatibility
        else:
            aset = set(self.pst.adj_par_names)
            ppset = set(self.pst.par_names)
            dvset = set(dv_ensemble.columns)
            pset = set(par_ensemble.columns)
            diff = ppset - aset
            if len(diff) > 0:
                self.logger.lraise("the following par_ensemble names were not " + \
                                   "found in the pst par names: {0}". \
                                   format(",".join(diff)))
            if len(diff) > 0:
                self.logger.lraise("the following dv_ensemble names were not " + \
                                   "found in the adjustable parameters: {0}". \
                                   format(",".join(diff)))

            self.par_ensemble_base = par_ensemble
            self.dv_ensemble_base = dv_ensemble


        self.obs_ensemble_base = self._calc_obs(self.dv_ensemble_base)
        self.obs_ensemble = self.obs_ensemble_base.copy()
        self.dv_ensemble = self.dv_ensemble_base.copy()

        self._initialized = True


    def _calc_obs(self,dv_ensemble):

        # todo - deal with failed runs

        if self.par_ensemble_base is None:
            failed_runs, oe = super(EvolAlg,self)._calc_obs(dv_ensemble)
        else:
            # make a copy of the org par ensemble but as a df instance
            df_base = pd.DataFrame(self.par_ensemble_base.loc[:,:])
            # stack up the par ensembles for each solution
            dfs = []
            for i in range(dv_ensemble.shape[0]):
                solution = dv_ensemble.loc[i,:]
                df = df_base.copy()
                df.loc[:,solution.index] = solution.values
                dfs.append(df)
            df = pd.concat(dfs)
            # reset with a range index
            org_index = df.index.copy()
            df.index = np.arange(df.shape[0])
            failed_runs, oe = super(EvolAlg,self)._calc_obs(df)
            if oe.shape[0] != dv_ensemble.shape[0] * self.par_ensemble_base.shape[0]:
                self.logger.lraise("wrong number of runs back from stack eval")


            pe_reals = self.par_ensemble_base.shape[0]
            df = self.obj_func.reduce_stack_with_risk_shift(oe,pe_reals,risk=self.risk)
            # big assumption the run results are in the same order
            df.index = dv_ensemble.index
            oe = pyemu.ObservationEnsemble.from_dataframe(df=df,pst=self.pst)



        return oe


    def update(self):
        if not self._initialized:
            self.logger.lraise("not initialized")











