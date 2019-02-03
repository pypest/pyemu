"""a prototype multiobjective opt under uncertainty algorithm
    Otis to contribute here
"""
import os
import numpy as np
import pandas as pd
import time

import pyemu
from .ensemble_method import EnsembleMethod


class ParetoObjFunc(object):
    """multiobjective function calculator.
    """

    def __init__(self, pst, obj_function_dict, logger):

        self.logger = logger
        self.pst = pst
        self.max_distance = np.inf
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
            elif name in pi.pilbl:
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

        self.is_nondominated = self.is_nondominated_continuous
        self.obs_obj_names = list(self.obs_dict.keys())
        self.cdf_dfs = None
        self.cdf_loc = None

    def set_cdf_df(self, observation_ensemble, num_realisations):
        self.cdf_dfs = dict()
        if observation_ensemble.shape[0] % num_realisations != 0:
            self.logger.lraise('incorrect number of realisations supplied')
        stochastic_cols = list(self.obs_dict.keys())
        stochastic_cols.extend(self.pst.less_than_obs_constraints)
        stochastic_cols.extend(self.pst.greater_than_obs_constraints)
        stochastic_cols = set(stochastic_cols)
        objectives = observation_ensemble.loc[:, stochastic_cols]
        cdf_names = ['cdf_{}'.format(i + 1) for i in range(objectives.shape[0] // num_realisations)]
        self.cdf_loc = pd.DataFrame(data=np.NaN, index=cdf_names, columns=self.obs_obj_names)
        for i in range(objectives.shape[0] // num_realisations):
            cdf = objectives.iloc[i * num_realisations:(i + 1) * num_realisations, :]
            self.cdf_dfs['cdf_{}'.format(i + 1)] = cdf - cdf.mean(axis=0)  # centering
            # store location data for each cdf
            self.cdf_loc.loc['cdf_{}'.format(i + 1), :] = cdf.loc[:, self.obs_obj_names].mean(axis=0).values

    def single_point_cdf(self, observation_ensemble):
        stochastic_cols = list(self.obs_dict.keys())
        stochastic_cols.extend(self.pst.less_than_obs_constraints)
        stochastic_cols.extend(self.pst.greater_than_obs_constraints)
        stochastic_cols = set(stochastic_cols)
        cdf = observation_ensemble.loc[:, stochastic_cols]
        self.cdf_dfs = cdf - cdf.mean(axis=0)
        print(self.cdf_dfs)

    def get_approximation_points(self, obs_ensemble):
        """

        :param obs_ensemble:
        :return: list of indexes in obs_ensemble which should be used to calculate a cdf
        """
        is_nondominated = self.is_nondominated_kung(obs_ensemble)
        front_loc = np.where(is_nondominated.values == True)[0]
        time.sleep(0.1)
        front = obs_ensemble.loc[front_loc, self.obs_obj_names]
        min = front.idxmin(axis=0)
        max = front.idxmax(axis=0)
        nadir = []  # nadir objective vector for front (not accounting for obs obj signs)
        for obj, idx in zip(min.index, min.values):
            nadir.append(front.loc[idx, obj])
        ideal = []  # ideal objective vector for front (not accounting for obs obj signs)
        for obj, idx in zip(max.index, max.values):
            ideal.append(front.loc[idx, obj])
        nadir = np.array(nadir)
        ideal = np.array(ideal)
        mean = (nadir + ideal) / 2  # point located in centre of all possible pareto fronts
        # find index of point on pareto front closest to the mean vector
        distances = np.linalg.norm(front - mean, 2, axis=1)
        mid_pareto_index = front.index[np.argmin(distances)]
        approximation_points = set(min) | set(max)
        approximation_points.add(mid_pareto_index)
        print(approximation_points)
        return list(approximation_points)

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
            is_feasible.loc[obs_df.loc[:,lt_obs] >= val] = False
        for gt_obs in self.pst.greater_than_obs_constraints:
            if risk != 0.5:
                val = self.get_risk_shifted_value(risk, obs_df.loc[gt_obs])
            else:
                val = self.pst.observation_data.loc[gt_obs,"obsval"]
            is_feasible.loc[obs_df.loc[:,gt_obs] <= val] = False
        return is_feasible

    def constraint_violation_vector(self, obs_df, risk=0.5):
        """calculate the constraint violation for constraints in obs_df.

        Parameters
        ----------
        obs_df : pandas.DataFrame
            a dataframe with columns of obs names and rows of realizations
        risk : float
            risk value. If != 0.5, then risk shifting is used.  Otherwise, the
            obsval in Pst is used.  Default is 0.5.


        Returns
        -------
        Series of constraint observation names (index) and constraint violation values

        """
        constraint_values = pd.Series(data=0, index=obs_df.index)
        for lt_obs in self.pst.less_than_obs_constraints:
            if risk != 0.5:
                val = self.get_risk_shifted_value(risk, obs_df.loc[lt_obs])
            else:
                val = self.pst.observation_data.loc[lt_obs, 'obsval']
            constraint_values.loc[obs_df.loc[:, lt_obs] >= val] += obs_df.loc[:, lt_obs] - val
        for gt_obs in self.pst.greater_than_obs_constraints:
            if risk != 0.5:
                val = self.get_risk_shifted_value(risk, obs_df.loc[gt_obs])
            else:
                val = self.pst.observation_data.loc[gt_obs, 'obsval']
            constraint_values.loc[obs_df.loc[:, gt_obs] <= val] += val - obs_df.loc[:, gt_obs]
        return constraint_values

    def objective_vector(self, obs_df):
        return obs_df.loc[:, self.obs_obj_names] * self.obs_obj_signs

    @property
    def obs_obj_signs(self):
        signs = []
        for obj in self.obs_obj_names:
            if self.obs_dict[obj] == "max":
                signs.append(1.0)
            else:
                signs.append(-1.0)
        signs = np.array(signs)
        return signs

    def dominates(self,sol1,sol2, constraint_violation1=None, constraint_violation2=None):
        def _dominates(sol1, sol2):
            d = self.obs_obj_signs * (sol1 - sol2)
            if np.all(d >= 0.0) and np.any(d > 0.0):
                return True
            return False
        if constraint_violation1 is None and constraint_violation2 is None:
            return _dominates(sol1, sol2)
        elif constraint_violation1 is None or constraint_violation2 is None:
            self.logger.lraise('constraint violation for one individual given as None while other is not None')
        else:
            if constraint_violation1 < constraint_violation2: # soln1 has smaller constraint violation
                return True
            elif constraint_violation2 < constraint_violation1:
                return False
            else:  # the constraint violations are both zero or equal
                return _dominates(sol1, sol2)

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
        obj_df = obs_df.loc[:,self.obs_obj_names]
        is_nondom = []
        for i,iidx in enumerate(obj_df.index):
            ind = True
            for jidx in obj_df.index:
                if iidx == jidx:
                    continue
                # if dominates(jidx,iidx):
                #     ind = False
                #     break
                if self.dominates(obj_df.loc[jidx,:], obj_df.loc[iidx,:]):
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

        obj_df = obs_df.loc[:,self.obs_obj_names]
        P = list(obj_df.index)
        PP = set()
        PP.add(P[0])

        #iidx = 1
        #while iidx < len(P):
        for iidx in P:
            jidx = 0
            drop = []
            keep = True
            for jidx in PP:
                # if dominates(iidx,jidx):
                #     drop.append(jidx)
                # elif dominates(jidx,iidx):
                #     keep = False
                #     break
                if jidx == iidx:
                    continue
                if self.dominates(obj_df.loc[iidx, :], obj_df.loc[jidx, :]):
                    drop.append(jidx)
                elif self.dominates(obj_df.loc[jidx, :], obj_df.loc[iidx, :]):
                    keep = False
                    break
            for d in drop:
                PP.remove(d)
            if keep:
                PP.add(iidx)
            #iidx += 1




        is_nondom = pd.Series(data=False,index=obs_df.index,dtype=bool)
        is_nondom.loc[PP] = True
        return is_nondom

    def is_nondominated_kung(self, obs_df):
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

        if np.any(obs_df.index.duplicated()):
            self.logger.lraise("is_nondominated_kung() requires a non-duplicated index for obs_df")
        obj_df = obs_df.loc[:,self.obs_obj_names]
        obj_names = self.obs_obj_names
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
                    #if dominates(T[j],B[i]):
                    if self.dominates(obj_df.loc[T[j],:], obj_df.loc[B[i],:]):
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

        for name, direction in self.obs_dict.items():
            # make a copy - wasteful, but easier
            obj_df = obs_df.loc[:,name].copy()

            # sort so that largest values are first
            obj_df.sort_values(ascending=False,inplace=True)

            # set the ends so they are always retained
            crowd_distance.loc[obj_df.index[0]] += self.max_distance
            crowd_distance.loc[obj_df.index[-1]] += self.max_distance

            # calculate the maximum separation for this objective
            max_separation = obj_df.iloc[0] - obj_df.iloc[-1]

            # process the vector
            i = 1
            for idx in obj_df.index[1:-1]:
                crowd_distance.loc[idx] += (obj_df.iloc[i-1] - obj_df.iloc[i+1]) / max_separation
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

        ascending = False
        if d == "min":
            ascending = True
        s = series.shape[0]
        shift = int(s * risk)
        if shift >= s:
            shift = s - 1

        cdf = series.sort_values(ascending=ascending).apply(np.cumsum)
        val = float(cdf.iloc[shift])
        #print(cdf)
        #print(shift,cdf.iloc[shift])
        #self.logger.statement("risk-shift for {0}->type:{1}dir:{2},shift:{3},val:{4}".format(n,t,d,shift,val))
        return val

    def full_recalculation_risk_shift(self, oe, num_reals, risk):

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

    def partial_recalculation_risk_shift(self, observation_ensemble, risk):
        """

        :param observation_ensemble: ensemble of observations with parameter means - each index represents a dv
        :param risk: risk level for risk shifting
        :return: observation_ensemble with constraints and objectives risk shifted
        """
        stochastic_cols = list(self.obs_dict.keys())
        stochastic_cols.extend(self.pst.less_than_obs_constraints)
        stochastic_cols.extend(self.pst.greater_than_obs_constraints)
        stochastic_cols = set(stochastic_cols)
        distances = pd.DataFrame(data=np.NaN, index=observation_ensemble.index, columns=self.cdf_loc.index)
        risk_shifted_observation_ensemble = observation_ensemble.copy()
        for idx in observation_ensemble.index:
            distances.loc[idx, :] = np.linalg.norm(observation_ensemble.loc[idx, self.obs_obj_names] -
                                                   self.cdf_loc, 2, axis=1)
        cdf_names = distances.idxmin(axis=1)
        for idx, cdf_name in zip(observation_ensemble.index, cdf_names):
            cdf = self.cdf_dfs[cdf_name]
            for col in stochastic_cols:
                series = cdf.loc[:, col] + observation_ensemble.loc[idx, col]
                val = self.get_risk_shifted_value(risk=risk, series=series)
                risk_shifted_observation_ensemble.loc[idx, col] = val
        return risk_shifted_observation_ensemble

    def no_recalculation_risk_shift(self, observation_ensemble, risk):
        risk_shifted_obs_ensemble = observation_ensemble.copy()
        stochastic_cols = list(self.obs_dict.keys())
        stochastic_cols.extend(self.pst.less_than_obs_constraints)
        stochastic_cols.extend(self.pst.greater_than_obs_constraints)
        stochastic_cols = set(stochastic_cols)
        for idx in observation_ensemble.index:
            for col in stochastic_cols:
                series = self.cdf_dfs.loc[:, col] + risk_shifted_obs_ensemble.loc[idx, col]
                risk_shifted_obs_ensemble.loc[idx, col] = self.get_risk_shifted_value(risk=risk, series=series)
        return risk_shifted_obs_ensemble

    def nsga2_non_dominated_sort(self, obs_df, risk):
        """

        :param obs_df: dataframe of observations
        :return: pd series of indexes and
        """
        rank = pd.Series(data=-1, index=obs_df.index, dtype=np.int16)
        dominated_sets = {idx: [] for idx in obs_df.index}
        domination_counts = pd.Series(data=0, index=obs_df.index)
        constraint_vector = self.constraint_violation_vector(obs_df, risk=risk)
        obj_df = obs_df.loc[:, self.obs_obj_names]
        current_front = set()
        for idx1 in obj_df.index: # p
            individual1 = obj_df.loc[idx1, :]
            violation1 = constraint_vector.loc[idx1]
            for idx2 in obj_df.index[idx1 + 1:]: # q
                individual2 = obj_df.loc[idx2, :]
                violation2 = constraint_vector.loc[idx2]
                if self.dominates(individual1, individual2, violation1, violation2):  # self.dominates(p, q): p dominates q
                    dominated_sets[idx1].append(idx2)
                    domination_counts.loc[idx2] += 1
                elif self.dominates(individual2, individual1, violation2, violation1):  # self.dominates(q, p):
                    dominated_sets[idx2].append(idx1)
                    domination_counts.loc[idx1] += 1
            if domination_counts[idx1] == 0:
                rank.loc[idx1] = 1
                current_front.add(idx1)
        prev_front = current_front
        i = 0
        while prev_front:
            current_front = set()
            for idx1 in prev_front:
                for idx2 in dominated_sets[idx1]:
                    domination_counts.loc[idx2] -= 1
                    if domination_counts.loc[idx2] == 0:
                        rank.loc[idx2] = i + 2
                        current_front.add(idx2)
            prev_front = current_front
            i += 1
        return rank


class EvolAlg(EnsembleMethod):
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template"):
        super(EvolAlg, self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                      submit_file=submit_file, verbose=verbose, port=port,
                                      slave_dir=slave_dir)


    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None,par_ensemble=None,risk=0.5,
                   dv_names=None,par_names=None, when_calculate=0):
        """

        :param obj_func_dict:
        :param num_par_reals:
        :param num_dv_reals:
        :param dv_ensemble:
        :param par_ensemble:
        :param risk:
        :param dv_names:
        :param par_names:
        :param when_calculate: flag to indicate when to recalculate the cdf. If -1, calculate the cdf once then
        propagate for all solutions. If 0, use full recalculation for each solution. for 1, 2, ... recalculate at 3
        points along the pareto front (edges and closest point to nadir ideal solution). default is full recalculation
        :return:
        """
        # todo : setup a run results store for all candidate solutions?  or maybe
        # just nondom, feasible solutions?

        # todo : check that the dv ensemble index is not duplicated

        self.dv_ensemble_archive = None
        self.obs_ensemble_archive = None

        if risk != 0.5:
            if risk > 1.0 or risk < 0.0:
                self.logger.lraise("risk not in 0.0:1.0 range")
        self.risk = risk
        self.obj_func = ParetoObjFunc(self.pst,obj_func_dict, self.logger)
        if not isinstance(when_calculate, int):
            self.logger.lraise('when_calculate must be an integer')
        if when_calculate < -1:
            self.logger.lraise('when calculate must be an integer in range [-1, ...]')
        self.when_calculate = when_calculate
        self.par_ensemble = None

        # all adjustable pars are dec vars
        if dv_ensemble is None and par_ensemble is None:
            self.num_dv_reals = num_dv_reals
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
                self.dv_names = dv_names

            self.dv_ensemble = pyemu.ParameterEnsemble.from_mixed_draws(self.pst, how_dict=how,
                                                                        num_reals=num_dv_reals, cov=self.parcov,
                                                                        partial=True) # TODO: add partial=True?
            if risk != 0.5:
                aset = set(self.pst.adj_par_names)
                dvset = set(self.dv_ensemble.columns)
                diff = aset - dvset
                if len(diff) == 0:
                    self.logger.lraise("risk!=0.5 but all adjustable parameters are dec vars")
                self.par_ensemble = pyemu.ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                                    num_reals=num_par_reals,
                                                                                    cov=self.parcov)
            else:
                self.par_ensemble = None


        # both par ensemble and dv ensemble were passed
        elif par_ensemble is not None and dv_ensemble is not None:
            self.num_dv_reals = dv_ensemble.shape[0]
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

            self.par_ensemble = par_ensemble
            self.dv_ensemble = dv_ensemble


        # dv_ensemble supplied, but not pars, so check if any adjustable pars are not
        # in dv_ensemble, and if so, draw reals for them
        elif dv_ensemble is not None and par_ensemble is None:
            self.num_dv_reals = dv_ensemble.shape[0]
            aset = set(self.pst.adj_par_names)
            dvset = set(dv_ensemble.columns)
            diff = dvset - aset
            if len(diff) > 0:
                self.logger.lraise("the following dv_ensemble names were not " + \
                                   "found in the adjustable parameters: {0}". \
                                   format(",".join(diff)))
            self.dv_ensemble = dv_ensemble
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
                self.par_ensemble = pyemu.ParameterEnsemble.from_mixed_draws(self.pst,how_dict=how,
                                         num_reals=num_par_reals,cov=self.parcov)
            else:
                diff = aset - dvset
                if len(diff) > 0:
                    self.logger.warn("adj pars {0} missing from dv_ensemble".\
                                       format(','.join(diff)))
                    df = pd.DataFrame(self.pst.parameter_data.loc[:,"parval1"]).T

                    self.par_ensemble = pyemu.ParameterEnsemble.from_dataframe(df=df,pst=self.pst)

        # par ensemble supplied but not dv_ensmeble, so check for any adjustable pars
        # that are not in par_ensemble and draw reals.  Must be at least one...
        elif par_ensemble is not None and dv_ensemble is None:
            self.num_dv_reals = num_dv_reals
            aset = set(self.pst.par_names)
            pset = set(par_ensemble.columns)
            diff = aset - pset
            if len(diff) > 0:
                self.logger.lraise("the following par_ensemble names were not " + \
                                   "found in the pst par names: {0}". \
                                   format(",".join(diff)))
            self.par_ensemble = par_ensemble

            if dv_names is None:
                self.logger.lraise("dv_names must be passed if dv_ensemble is None and par_ensmeble is not None")

            dvset = set(dv_names)
            diff = dvset - aset
            if len(diff) > 0:
                self.logger.lraise("the following dv_names were not " + \
                                   "found in the adjustable parameters: {0}". \
                                   format(",".join(diff)))
            how = {p: "uniform" for p in dv_names}
            self.dv_ensemble = pyemu.ParameterEnsemble.from_mixed_draws(self.pst, how_dict=how,
                                                                             num_reals=num_dv_reals,
                                                                             cov=self.parcov,partial=True)


        self.dv_names = self.dv_ensemble.columns
        self.last_stack = None
        self.logger.log("evaluate initial dv ensemble of size {0}".format(self.dv_ensemble.shape[0]))
        self.obs_ensemble = self._calc_obs(self.dv_ensemble) # TODO maybe change to copy()?
        self.logger.log("evaluate initial dv ensemble of size {0}".format(self.dv_ensemble.shape[0]))
        self._initialized = True


        # isfeas = self.obj_func.is_feasible(self.obs_ensemble,risk=self.risk)
        # isnondom = self.obj_func.is_nondominated(self.obs_ensemble)
        #
        # vc = isfeas.value_counts()
        # if True not in vc:
        #     self.logger.lraise("no feasible solutions in initial population")
        # self.logger.statement("{0} feasible individuals in initial population".format(vc[True]))
        # self.dv_ensemble = self.dv_ensemble.loc[isfeas,:]
        # self.obs_ensemble = self.obs_ensemble.loc[isfeas,:]
        # vc = isnondom.value_counts()
        # if True in vc:
        #     self.logger.statement("{0} nondominated solutions in initial population".format(vc[True]))
        # else:
        #     self.logger.statement("no nondominated solutions in initial population")
        # self.dv_ensemble = self.dv_ensemble.loc[isnondom,:]  # TODO: check this - before was isfeas?
        # self.obs_ensemble = self.obs_ensemble.loc[isnondom,:]  # seemed to be doing the same thing twice
        # self.pst.add_transform_columns()
        # self._initialized = True

    @staticmethod
    def _drop_failed(failed_runs, dv_ensemble, obs_ensemble):
        if failed_runs is None:
            return
        print(dv_ensemble)
        dv_ensemble.loc[failed_runs,:] = np.NaN
        print(dv_ensemble)
        print(obs_ensemble)
        dv_ensemble = dv_ensemble.dropna(axis=0)
        obs_ensemble.loc[failed_runs,:] = np.NaN
        obs_ensemble = obs_ensemble.dropna(axis=0)
        print(obs_ensemble)

    def _archive(self,dv_ensemble,obs_ensemble):
        self.logger.log("archiving {0} solutions".format(dv_ensemble.shape[0]))
        if dv_ensemble.shape[0] != obs_ensemble.shape[0]:
            self.logger.lraise("EvolAlg._archive() error: shape mismatch: {0} : {1}".\
                               format(dv_ensemble.shape[0],obs_ensemble.shape[0]))

        obs_ensemble = obs_ensemble.copy()
        dv_ensemble = dv_ensemble.copy()
        isfeas = self.obj_func.is_feasible(obs_ensemble)
        isnondom = self.obj_func.is_nondominated(obs_ensemble)
        cd = self.obj_func.crowd_distance(obs_ensemble)
        obs_ensemble.loc[isfeas.index,"feasible"] = isfeas
        obs_ensemble.loc[isnondom.index,"nondominated"] = isnondom
        dv_ensemble.loc[isfeas.index,"feasible"] = isfeas
        dv_ensemble.loc[isnondom.index,"nondominated"] = isnondom
        obs_ensemble.loc[:,"iteration"] = self.iter_num
        dv_ensemble.loc[:,"iteration"] = self.iter_num
        obs_ensemble.loc[cd.index,"crowd_distance"] = cd
        dv_ensemble.loc[cd.index,"crowd_distance"] = cd
        if self.obs_ensemble_archive is None:
            self.obs_ensemble_archive = pd.DataFrame(obs_ensemble.loc[:,:])
            self.dv_ensemble_archive = pd.DataFrame(dv_ensemble.loc[:,:])
        else:
            self.obs_ensemble_archive = self.obs_ensemble_archive.append(pd.DataFrame(obs_ensemble.loc[:,:]))
            self.dv_ensemble_archive = self.dv_ensemble_archive.append(pd.DataFrame(dv_ensemble.loc[:,:]))

    def _calc_obs(self, dv_ensemble):
        """
        Note: assumes that pars in dv ensemble have already been transformed. also assumes pst has transform columns
        added.
        :param dv_ensemble: dv to calculate obs_ensemble for
        :return obs_ensemble: ensemble of (risk shifted) observations
        """
        oe = None
        dv = dv_ensemble.copy()
        if self.par_ensemble is None:  # MOO setting is being used
            eval_ensemble = self._add_missing_pars(dv)
            failed_runs, oe = super(EvolAlg,self)._calc_obs(eval_ensemble)  # TODO: set up some fail run dv removal
        else:  # Do MOOUU
            if self.when_calculate == 0:  # calculate ensemble for every point - no reduction in evals
                df = self._evaluation_ensemble(dv, par_ensemble=self.par_ensemble)
                failed_runs, oe = super(EvolAlg, self)._calc_obs(df)
                if oe.shape[0] != dv_ensemble.shape[0] * self.par_ensemble.shape[0]:
                    self.logger.lraise("wrong number of runs back from stack eval")
                EvolAlg._drop_failed(failed_runs, df, oe)  # TODO: check what happens if all runs for one dv fail
                try:
                    num_failed = len(failed_runs)
                except TypeError:
                    num_failed = 0
                self.logger.statement("dropped {0} failed runs, {1} remaining". \
                                      format(num_failed, dv_ensemble.shape[0]))
                self.last_stack = oe.copy()

                self.logger.log("reducing initial stack evaluation")
                df = self.obj_func.full_recalculation_risk_shift(oe, self.par_ensemble.shape[0],
                                                                 self.risk)
                self.logger.log("reducing initial stack evaluation")
                # big assumption the run results are in the same order
                df.index = dv_ensemble.index
                oe = pyemu.ObservationEnsemble.from_dataframe(df=df, pst=self.pst)
            elif self.when_calculate == -1:  # calculate ensemble only for initial point then propagate
                if self._initialized is False:
                    midpoint = self._get_bounds().mean(axis=0) # calculate
                    print(midpoint)
                    time.sleep(0.1)
                    midpoint = pd.DataFrame(data=midpoint, index=self.dv_names).T  # dv dataframe using mean of dvs
                    eval_ensemble = self._evaluation_ensemble(midpoint, self.par_ensemble)
                    failed_runs, stack = super()._calc_obs(eval_ensemble)
                    self._drop_failed(failed_runs, eval_ensemble, stack)
                    self.obj_func.single_point_cdf(stack)
                eval_ensemble = self._add_missing_pars(dv)
                failed_runs, obs_ensemble = super()._calc_obs(eval_ensemble)
                self._drop_failed(failed_runs, dv_ensemble, obs_ensemble)
                oe = self.obj_func.no_recalculation_risk_shift(obs_ensemble, risk=self.risk)
                oe = pyemu.ObservationEnsemble.from_dataframe(df=oe, pst=self.pst)
            elif self.when_calculate > 0 and self.iter_num % self.when_calculate == 0:
                self.logger.statement('updating stack using decision variables')
                eval = self._add_missing_pars(dv) # calculate at mean of pars first
                self.logger.log('running model with mean parameters')
                # maybe would be good to include prev generations - i.e points to calculate stack at chosen from
                # the whole dv population (incl. those that have already been evaluated)
                failed_runs, obs_ensemble = super()._calc_obs(eval)
                self.logger.log('running model with mean parameters')
                EvolAlg._drop_failed(failed_runs, eval, obs_ensemble)
                self.logger.log('finding best points to run stack evaluation')
                points = self.obj_func.get_approximation_points(obs_ensemble)
                self.logger.log('finding best points to run stack evaluation')
                # calculate ensembles for best points
                eval_ensemble = self._evaluation_ensemble(dv.loc[points, :], self.par_ensemble)
                self.logger.log('running stack evaluation at {} points'.format(len(points)))
                failed_runs, stack = super()._calc_obs(eval_ensemble)
                self.logger.log('running stack evaluation at {} points'.format(len(points)))
                EvolAlg._drop_failed(failed_runs, eval_ensemble, stack)
                self.obj_func.set_cdf_df(stack, self.par_ensemble.shape[0])
                oe = self.obj_func.partial_recalculation_risk_shift(obs_ensemble, risk=self.risk)
                oe = pyemu.ObservationEnsemble.from_dataframe(df=oe, pst=self.pst)
            elif self.when_calculate > 0:
                eval_ensemble = self._add_missing_pars(dv)
                self.logger.log('running model with mean parameters')
                failed_runs, obs_ensemble = super()._calc_obs(eval_ensemble)
                self.logger.log('running model with mean parameters')
                EvolAlg._drop_failed(failed_runs, eval_ensemble, obs_ensemble)
                oe = self.obj_func.partial_recalculation_risk_shift(obs_ensemble, risk=self.risk)
                oe = pyemu.ObservationEnsemble.from_dataframe(df=oe, pst=self.pst)
            else:
                self.logger.lraise('did not calculate observations - check logic flow')
        self._archive(dv_ensemble, oe)
        return oe

    def _evaluation_ensemble(self, dv_ensemble, par_ensemble):
        dfs = []
        df_base = par_ensemble.reindex(columns=self.pst.par_names)
        for idx in dv_ensemble.index:
            df_base.loc[:, dv_ensemble.columns] = dv_ensemble.loc[idx, :].values
            dfs.append(df_base.copy())
        df = pd.concat(dfs)
        df.index = np.arange(df.shape[0])
        return df

    def update(self,*args,**kwargs):
        self.logger.lraise("EvolAlg.update() must be implemented by derived types")

    def _get_bounds(self):
        """
        get the bounds given for each of the dv parameters. assumes problem has been initialised so that transform
        columns have been added
        :return: transformed bounds
        """
        bounds = self.pst.parameter_data.loc[self.dv_names, ['parlbnd_trans', 'parubnd_trans']].values
        return bounds.T


class EliteDiffEvol(EvolAlg):
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template", ):
        super(EliteDiffEvol, self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                      submit_file=submit_file, verbose=verbose, port=port,
                                      slave_dir=slave_dir)
    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None,par_ensemble=None,risk=0.5,
                   dv_names=None,par_names=None, when_calculate=0):
        super(EliteDiffEvol, self).initialize(obj_func_dict=obj_func_dict, num_par_reals=num_par_reals, num_dv_reals=num_dv_reals,
                           dv_ensemble=dv_ensemble, par_ensemble=par_ensemble, risk=risk, dv_names=dv_names,
                           par_names=par_names, when_calculate=when_calculate)
        isfeas = self.obj_func.is_feasible(self.obs_ensemble,risk=self.risk)
        isnondom = self.obj_func.is_nondominated(self.obs_ensemble)

        vc = isfeas.value_counts()
        if True not in vc:
            self.logger.lraise("no feasible solutions in initial population")
        self.logger.statement("{0} feasible individuals in initial population".format(vc[True]))
        self.dv_ensemble = self.dv_ensemble.loc[isfeas,:]
        self.obs_ensemble = self.obs_ensemble.loc[isfeas,:]
        vc = isnondom.value_counts()
        if True in vc:
            self.logger.statement("{0} nondominated solutions in initial population".format(vc[True]))
        else:
            self.logger.statement("no nondominated solutions in initial population")
        self.dv_ensemble = self.dv_ensemble.loc[isnondom,:]  # TODO: check this - before was isfeas?
        self.obs_ensemble = self.obs_ensemble.loc[isnondom,:]  # seemed to be doing the same thing twice
        self.iter_num = 1

    def update(self,mut_base = 0.8,cross_over_base=0.7,num_dv_reals=None):
        if not self._initialized:
            self.logger.lraise("not initialized")
        if num_dv_reals is None:
            num_dv_reals = self.num_dv_reals
        if self.dv_ensemble.shape[0] < 4:
            self.logger.lraise("not enough individuals in population to continue")

        # function to get unique index names
        self._child_count = 0

        def next_name():
            while True:
                sol_name = "c_i{0}_{1}".format(self.iter_num, self._child_count)
                if sol_name not in self.dv_ensemble.index.values:
                    break
                self._child_count += 1
            return sol_name

        # generate self.num_dv_reals offspring using diff evol rules
        dv_offspring = []
        child2parent = {}
        offspring_idx = []
        tol = 1.0
        num_dv = self.dv_ensemble.shape[1]
        dv_names = self.dv_ensemble.columns
        dv_log = self.pst.parameter_data.loc[dv_names, "partrans"] == "log"
        lb = self.pst.parameter_data.loc[dv_names,"parlbnd"].copy()
        ub = self.pst.parameter_data.loc[dv_names,"parubnd"].copy()
        lb.loc[dv_log] = lb.loc[dv_log].apply(np.log10)
        ub.loc[dv_log] = ub.loc[dv_log].apply(np.log10)
        dv_ensemble_trans = self.dv_ensemble.copy()

        for idx in dv_ensemble_trans.index:
            dv_ensemble_trans.loc[idx,dv_log] = dv_ensemble_trans.loc[idx,dv_log].apply(lambda x: np.log10(x))

        for i in range(num_dv_reals):
            # every archive gets an offspring
            if i < self.dv_ensemble.shape[0]:
                parent_idx = i
                mut = mut_base
                cross_over = cross_over_base
            else:
                #otherwise, some parents get more than one offspring
                # could do something better here - like pick a good archive
                # make a wild population
                parent_idx = np.random.randint(0,dv_ensemble_trans.shape[0])
                mut = 0.9
                cross_over = 0.9

            parent = dv_ensemble_trans.iloc[parent_idx,:]


            # select the three other members in the population
            abc_idxs = np.random.choice(dv_ensemble_trans.index,3,replace=False)

            abc = dv_ensemble_trans.loc[abc_idxs,:].copy()

            mutant= abc.iloc[0] + (mut * (abc.iloc[1] - abc.iloc[2]))

            # select cross over genes (dec var values)
            cross_points = np.random.rand(num_dv) < cross_over
            if not np.any(cross_points):
                cross_points[np.random.randint(0,num_dv)] = True

            #create an offspring
            offspring = parent.copy()
            offspring.loc[cross_points] = mutant.loc[cross_points]

            #enforce bounds
            out = offspring > ub
            offspring.loc[out] = ub.loc[out]
            out = offspring < lb
            offspring.loc[out] = lb.loc[out]

            # back transform
            offspring.loc[dv_log] = 10.0**offspring.loc[dv_log]
            offspring = offspring.loc[self.dv_ensemble.columns]


            sol_name = "c_{0}".format(i)
            dv_offspring.append(offspring)
            offspring_idx.append(sol_name)
            child2parent[sol_name] = dv_ensemble_trans.index[parent_idx]

        dv_offspring = pd.DataFrame(dv_offspring,columns=self.dv_ensemble.columns,index=offspring_idx)

        # run the model with offspring candidates
        self.logger.log("running {0} canditiate solutions for iteration {1}".\
                        format(dv_offspring.shape[0],self.iter_num))
        obs_offspring = self._calc_obs(dv_offspring)


        # evaluate offspring fitness WRT feasibility and nondomination (elitist) -
        # if offspring dominates archive, replace in
        # self.dv_ensemble and self.obs_ensemble.  if not, drop candidate.
        # If tied, keep both
        isfeas = self.obj_func.is_feasible(obs_offspring)
        isnondom = self.obj_func.is_nondominated(obs_offspring)

        for child_idx in obs_offspring.index:
            if not isfeas[child_idx]:
                self.logger.statement("population {0} is not feasible".format(child_idx))
                continue

            child_sol = obs_offspring.loc[child_idx,:]
            parent_idx = child2parent[child_idx]
            if parent_idx is None:
                # the archive was already removed by another population, so if this population is
                # feasible and nondominated, keep it
                if isnondom(child_idx):
                    self.logger.statement("orphaned population {0} retained".format(child_idx))
                    sol_name = next_name()
                    self.dv_ensemble.loc[sol_name, child_sol.index] = child_sol
                    self.obs_ensemble.loc[sol_name, obs_offspring.columns] = obs_offspring.loc[child_idx, :]

            else:
                parent_sol = self.obs_ensemble.loc[parent_idx,:]
                if self.obj_func.dominates(parent_sol.loc[self.obj_func.obs_obj_names],\
                                           child_sol.loc[self.obj_func.obs_obj_names]):
                    self.logger.statement("population {0} dominated by archive {1}".format(child_idx,parent_idx))
                    # your dead to me!
                    pass
                elif self.obj_func.dominates(child_sol.loc[self.obj_func.obs_obj_names],\
                                             parent_sol.loc[self.obj_func.obs_obj_names]):
                    # hey dad, what do you think about your son now!
                    self.logger.statement("population {0} dominates archive {1}".format(child_idx,parent_idx))
                    self.dv_ensemble.loc[parent_idx,dv_offspring.columns] = dv_offspring.loc[child_idx,:]
                    self.obs_ensemble.loc[parent_idx,obs_offspring.columns] = obs_offspring.loc[child_idx,:]
                    child2parent[idx] = None
                else:
                    self.logger.statement("population {0} and archive {1} kept".format(child_idx,parent_idx))
                    sol_name = next_name()
                    self.dv_ensemble.loc[sol_name,dv_offspring.columns] = dv_offspring.loc[child_idx,:]
                    self.obs_ensemble.loc[sol_name,obs_offspring.columns] = obs_offspring.loc[child_idx,:]


        #if there are too many individuals in self.dv_ensemble,
        # first drop dominated,then reduce by using crowding distance.

        # self.logger.statement("number of solutions:{0}".format(self.dv_ensemble.shape[0]))
        isnondom = self.obj_func.is_nondominated(self.obs_ensemble)
        dom_idx = isnondom.loc[isnondom == False].index
        nondom_idx = isnondom.loc[isnondom==True].index
        self.logger.statement("number of dominated solutions:{0}".format(dom_idx.shape[0]))
        # self.logger.statement("nondominated solutions: {0}".format(','.join(nondom_idx)))
        self.logger.statement("dominated solutions: {0}".format(','.join(str(dom_idx))))
        ndrop = self.dv_ensemble.shape[0] - num_dv_reals
        if ndrop > 0:
            isnondom = self.obj_func.is_nondominated(self.obs_ensemble)
            vc = isnondom.value_counts()
            # if there a dominated solutions, drop those first, using
            # crowding distance as the order
            if False in vc.index:
                # get dfs for the dominated solutions
                dv_dom = self.dv_ensemble.loc[dom_idx,:].copy()
                obs_dom = self.obs_ensemble.loc[dom_idx,:].copy()

                self.dv_ensemble.drop(dom_idx,inplace=True)
                self.obs_ensemble.drop(dom_idx,inplace=True)
                self.logger.statement("dropping {0} dominated individuals based on crowd distance".\
                                      format(min(ndrop,dv_dom.shape[0])))

                self._drop_by_crowd(dv_dom,obs_dom,min(ndrop,dv_dom.shape[0]))
                #add any remaining dominated solutions back
                self.dv_ensemble = self.dv_ensemble.append(dv_dom)
                self.obs_ensemble = self.obs_ensemble.append(obs_dom)


        # drop remaining nondom solutions as needed
        if self.dv_ensemble.shape[0] > num_dv_reals:
            self._drop_by_crowd(self.dv_ensemble,self.obs_ensemble,self.dv_ensemble.shape[0] - num_dv_reals)

        self.iter_report()
        self.iter_num += 1

        return

    def iter_report(self):

        oe = self.obs_ensemble.copy()
        dv = self.dv_ensemble.copy()
        isfeas = self.obj_func.is_feasible(oe)
        isnondom = self.obj_func.is_nondominated(oe)
        cd = self.obj_func.crowd_distance(oe)
        for df in [oe,dv]:
            df.loc[isfeas.index,"feasible"] = isfeas
            df.loc[isnondom.index, "nondominated"] = isnondom
            df.loc[cd.index,"crowd_distance"] = cd

        dv.to_csv("dv_ensemble.{0}.csv".format(self.iter_num + 1))
        oe.to_csv("obs_ensemble.{0}.csv".format(self.iter_num + 1))
        self.logger.statement("*** iteration {0} report".format(self.iter_num+1))
        self.logger.statement("{0} current solutions".format(dv.shape[0]))
        self.logger.statement("{0} infeasible".format(isfeas[isfeas==False].shape[0]))
        self.logger.statement("{0} nondomiated".format(isnondom[isnondom==True].shape[0]))

    def _drop_by_crowd(self,dv_ensemble, obs_ensemble, ndrop,min_dist=0.1):
        if ndrop > dv_ensemble.shape[0]:
            self.logger.lraise("EliteDiffEvol.drop_by_crowd() error: ndrop"+
                               "{0} > dv_ensemble.shape[0] {1}".\
                               format(ndrop,dv_ensemble.shape[0]))
        self.logger.statement("dropping {0} of {1} individuals based on crowd distance".\
                              format(ndrop,dv_ensemble.shape[0]))
        # if min_dist is not None:
        #     while True:
        #         cd = self.obj_func.crowd_distance(obs_ensemble)
        #         if cd.min() >= min_dist or ndrop == 0:
        #             break
        #         cd.sort_values(inplace=True, ascending=False)
        #
        #         drop_idx = cd.index[-1]
        #         self.logger.statement("dropping solution {0} - less then 'min_dist' apart{1}".\
        #                               format(drop_idx,cd.loc[drop_idx]))
        #
        #         dv_ensemble.drop(drop_idx,inplace=True)
        #         obs_ensemble.drop(drop_idx,inplace=True)
        #         ndrop -= 1%


        for idrop in range(ndrop):
            cd = self.obj_func.crowd_distance(obs_ensemble)
            cd.sort_values(inplace=True,ascending=False)
            #drop the first element in cd from both dv_ensemble and obs_ensemble
            drop_idx = cd.index[-1]
            self.logger.statement("solution {0} removed based on crowding distance {1}".\
                                  format(drop_idx,cd[drop_idx]))
            dv_ensemble.drop(drop_idx,inplace=True)
            obs_ensemble.drop(drop_idx,inplace=True)







