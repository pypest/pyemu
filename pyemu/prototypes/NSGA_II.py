"""
NSGA-II algorithim for multi objective optimisation - supports MOO and MOOUU
version proposed by Deb[2002] IEEE
GNS.cri
Otis Rea
IMPORTANT!!!!: Old class of NSGA_II (if checks needed/issues arise - revert to this one using git checkout)
                Find this version in commits BEFORE 30/01/2019
better documentation coming soon...
"""
import numpy as np
from .Abstract_Moo import *
from .GeneticOperators import *


class NSGA_II(EvolAlg):

    def __init__(self, pst, parcov=None, obscov=None, num_slaves=0, use_approx_prior=True,
                 submit_file=None, verbose=False, port=4004, slave_dir="template",
                 cross_prob=0.9, cross_dist=15, mut_prob=0.01, mut_dist=20):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions to be optimised
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           ------------------------------------- Optional parameters -------------------------------------------------
           archive_size: size of archive population. Full population size is 2 * archive due to population population
           cross_prob: probability of crossover occurring for any population
           cross_dist: distribution parameter of the crossover operation
           mut_prob: probability of mutation occurring for any population
           mut_dist: distribution parameter of the mutation operation
           iterations: number of iterations of the algorithm
        """
        super().__init__(pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves, use_approx_prior=use_approx_prior,
                 submit_file=submit_file, verbose=verbose, port=port, slave_dir=slave_dir)
        # --------------------------------sets of population members------------------------------------------
        self.archive_obs = None
        self.population_obs = None
        self.archive_dv = None
        self.population_dv = None
        self.joint_obs = None
        self.joint_dv = None
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        initialising_variables = 'pst {}\nparcov {}\nobscov {}\nnum_slaves {}\nuse_approx_prior {}\n ' \
                                 'submit_file {}\nverbose {}\nport {}\nslave_dir {}\n' \
                                 'cross_prob {}\ncross_dist {}\nmut_prob {}\n' \
                                 'mut_dist {}'.format(self.pst.filename, parcov, obscov, num_slaves, use_approx_prior,
                                                      submit_file, verbose, port, slave_dir, cross_prob, cross_dist,
                                                      mut_prob, mut_dist)
        self.logger.statement('using NSGA-II as evolutionary algorithm.\nParameters:\n{}'.format(initialising_variables))

    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None,par_ensemble=None,risk=0.5,
                   dv_names=None,par_names=None, when_calculate=0):
        super().initialize(obj_func_dict=obj_func_dict, num_par_reals=num_par_reals, num_dv_reals=num_dv_reals,
                           dv_ensemble=dv_ensemble, par_ensemble=par_ensemble, risk=risk, dv_names=dv_names,
                           par_names=par_names, when_calculate=when_calculate)
        self.logger.log("initialising NSGA-II")
        if when_calculate < 0:
            use = 'single (mean of bounds) point uncertainty calculation'
        elif when_calculate == 0:
            use = 'complete recalculation of uncertainty for every point in decision space'
        else:
            use = 'will recalculate uncertainty at 3 points every {} iterations'.format(when_calculate)
        self.logger.statement('when_calculate set as {}\n'.format(when_calculate, use))
        self.joint_dv = pyemu.ParameterEnsemble(pst=self.pst, data=np.NaN, index=np.arange(2 * self.num_dv_reals),
                                                columns=self.dv_names)
        self.joint_obs = pyemu.ObservationEnsemble(pst=self.pst, data=np.NaN, index=np.arange(2 * self.num_dv_reals),
                                                   columns=self.obs_ensemble.columns)
        self.archive_dv = self.dv_ensemble
        self.archive_obs = self.obs_ensemble
        self.population_dv = pyemu.ParameterEnsemble(pst=self.pst, data=np.NaN,
                                                     index=np.arange(self.num_dv_reals, 2 * self.num_dv_reals),
                                                     columns=self.dv_names)
        self.population_obs = pyemu.ParameterEnsemble(pst=self.pst, data=np.NaN,
                                                      index=np.arange(self.num_dv_reals, 2 * self.num_dv_reals),
                                                      columns=self.obs_ensemble.columns)
        rank = self.obj_func.nsga2_non_dominated_sort(obs_df=self.archive_obs, risk=self.risk)
        self.archive_dv.loc[:, 'rank'] = rank.values
        fronts = self.get_fronts(rank)
        for front in fronts:
            crowding_distance = self.obj_func.crowd_distance(self.archive_obs.loc[front, :])
            self.archive_dv.loc[front, 'crowding_distance'] = crowding_distance
        new_population_index = self.tournament_selection(self.archive_dv, self.num_dv_reals)
        self.population_dv.loc[:, :] = self.archive_dv.loc[new_population_index, self.dv_names].values
        self.population_obs.loc[:, :] = self.archive_obs.loc[new_population_index, :].values
        to_update = Crossover.sbx(self.population_dv, self._get_bounds(), self.cross_prob, self.mut_dist)
        to_update | Mutation.polynomial(self.population_dv, self._get_bounds(), self.mut_prob, self.mut_dist)
        to_update = list(to_update)
        self.population_obs.loc[to_update, :] = self._calc_obs(self.population_dv.loc[to_update, self.dv_names]).values
        self.iter_num = 1
        self.logger.log("initialising NSGA-II")

    def update(self):
        self.logger.log('iteration number {}'.format(self.iter_num))
        # create joint population from previous archive and population
        self.joint_dv.loc[self.archive_dv.index, :] = self.archive_dv.loc[:, self.dv_names].values
        self.joint_dv.loc[self.population_dv.index, :] = self.population_dv.values
        self.joint_obs.loc[self.archive_obs.index, :] = self.archive_obs.values
        self.joint_obs.loc[self.population_obs.index, :] = self.population_obs.values
        # set old archive values as NaN
        self.archive_obs.loc[:, :] = np.NaN
        self.archive_dv.loc[:, :] = np.NaN
        # sort joint population into non dominated fronts and rank it
        rank = self.obj_func.nsga2_non_dominated_sort(self.joint_obs, risk=self.risk)
        fronts = self.get_fronts(rank)
        j = 0
        num_filled = 0
        # put all nondominated fronts that fit into the archive, into the archive
        while num_filled + len(fronts[j]) < self.num_dv_reals:
            index = np.arange(num_filled, num_filled + len(fronts[j]))
            self.archive_dv.loc[index, self.dv_names] = self.joint_dv.loc[fronts[j], :].values
            self.obs_ensemble.loc[index, :] = self.joint_obs.loc[fronts[j], :].values
            self.archive_dv.loc[index, 'rank'] = rank[fronts[j]].values
            cd = self.obj_func.crowd_distance(self.archive_obs.loc[index, :])
            self.archive_dv.loc[index, 'crowding_distance'] = cd.values
            num_filled += len(fronts[j])
            j += 1
        # fill up the archive using the remaining front, choosing new values based on crowding distance
        joint_dvj = self.joint_dv.loc[fronts[j]]
        joint_dvj.loc[:, 'rank'] = rank.loc[fronts[j]]
        joint_dvj.loc[:, 'crowding_distance'] = self.obj_func.crowd_distance(self.joint_obs.loc[fronts[j]])
        joint_dvj.sort_values(by=['rank', 'crowding_distance'], ascending=[True, False], inplace=True)
        index = np.arange(num_filled, self.num_dv_reals)
        self.archive_dv.loc[index, :] = joint_dvj.loc[joint_dvj.index[:len(index)], :].values
        self.archive_obs.loc[index, :] = self.joint_obs.loc[joint_dvj.index[:len(index)], :].values
        # use tournament selection, and the genetic operators to create new population
        new_population_index = self.tournament_selection(self.archive_dv, self.num_dv_reals)
        self.population_dv.loc[:, :] = self.archive_dv.loc[new_population_index, self.dv_names].values
        self.population_obs.loc[:, :] = self.archive_obs.loc[new_population_index, :].values
        to_update = Crossover.sbx(self.population_dv, self._get_bounds(), self.cross_prob, self.cross_dist)
        to_update | Mutation.polynomial(self.population_dv, self._get_bounds(), self.mut_prob, self.mut_dist)
        to_update = list(to_update)
        # calculate observations for updated individuals in populations
        self.population_obs.loc[to_update, :] = self._calc_obs(self.population_dv.loc[to_update, self.dv_names]).values
        self.logger.log('iteration number {}'.format(self.iter_num))
        self.iter_num += 1
        return self.joint_dv.loc[fronts[0], :], self.joint_obs.loc[fronts[0], :]

    def get_fronts(self, ranks):
        rank_copy = ranks.sort_values(ascending=True, inplace=False)
        start = 0
        finish = 1
        fronts = []
        while finish < len(rank_copy.index):
            if rank_copy.loc[rank_copy.index[start]] != rank_copy.loc[rank_copy.index[finish]]:
                fronts.append(rank_copy.index[start: finish])
                start = finish
            finish += 1
        fronts.append(rank_copy.index[start:])
        return fronts

    def tournament_selection(self, dv_ensemble, num_to_select):
        def _is_better(idx1, idx2):
            return bool(dv_ensemble.loc[idx1, 'rank'] < dv_ensemble.loc[idx2, 'rank'] or
                        (dv_ensemble.loc[idx1, 'rank'] == dv_ensemble.loc[idx2, 'rank'] and
                         dv_ensemble.loc[idx1, 'crowding_distance'] > dv_ensemble.loc[idx2, 'crowding_distance'])
                        )
        first = self.dv_ensemble.index[-1] + 1
        last = first + num_to_select
        child_population_index = []
        even = np.arange(0, (num_to_select // 2) * 2, 2)
        odd = np.arange(1, (num_to_select // 2) * 2, 2)
        index = np.array(dv_ensemble.index)
        i = 0
        for _ in range(2):
            np.random.shuffle(index)
            for idx1, idx2 in zip(index[even], index[odd]):
                if _is_better(idx1, idx2):
                    child_population_index.append(idx1)
                else:
                    child_population_index.append(idx2)
                i += 1
        if num_to_select % 2 == 1:  # i.e is odd
            child_population_index.append(index[-1])
        return child_population_index


# old class of NSGA_II for safety purposes find before commit on 30/01/19
