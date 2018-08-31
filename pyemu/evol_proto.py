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

    def is_feasible(self, obs_df):
        """identify which candidate solutions in obs_df (rows)
        are feasible with respect obs constraints (obs_df)

        Parameters
        ----------
        obs_df : pandas.DataFrame
            a dataframe with columns of obs names and rows of realizations


        Returns
        -------
        is_feasible : pandas.Series
            series with obs_df.index and bool values

        """
        # todo deal with pi eqs

        is_feasible = pd.Series(data=True, index=obs_df.index)
        for lt_obs in self.pst.less_than_obs_constraints:
            val = self.pst.observation_data.loc[lt_obs,"obsval"]
            is_feasible.loc[obs_df.loc[:,lt_obs]>=val] = False
        for gt_obs in self.pst.greater_than_obs_constraints:
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


class EvolAlg(EnsembleMethod):
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template"):
        super(EvolAlg, self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                      submit_file=submit_file, verbose=verbose, port=port,
                                      slave_dir=slave_dir)


    def initialize(self,obj_function_names):

        self.obj_func = ParetoObjFunc(self.pst,obj_function_names, self.logger)
        pass

    def update(self):
        pass











