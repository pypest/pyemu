"""SPEA_2 algorithm"""

from .GeneticOperators import *
from .moouu import *
import os


class SPEA_2(EvolAlg):
    """pyemu version of SPEA2"""

    def __init__(self, pst, parcov=None, obscov=None, num_slaves=0, use_approx_prior=True,
                 submit_file=None, verbose=False, port=4004, slave_dir='template',
                 crossover_probability=0.9, crossover_distribution=15,
                 mutation_probability=0.1, mutation_distribution=20):
        """
        initialise the SPEA2 algorithm for multi objective optimisation under uncertainty

        :param pst: Pst object instatiated with a pest control file
        :param parcov: parameter covariance matrix (optional)
        :param obscov: observation covariance matrix (optional)
        :param num_slaves: number of pipes/slaves to use if running in parallel
        :param use_approx_prior: Does nothing currently (I think...)
        :param submit_file: the name of a HTCondor submit file.  If not None, HTCondor is used to
        evaluate the parameter ensemble in parallel by issuing condor_submit as a system command
        :param verbose: if False, less output to the command line. If True, more output. If a string giving a
        file instance is passed, logger output will be written to that file
        :param port: the TCP port number to communicate on for parallel run management
        :param slave_dir: directory to use as a template for parallel run management (should contain pest and model
        executables).


        ----------------------------don't change these unless you know what you are doing-----------------------
        :param crossover_probability: probability of crossover operation occurring
        :param crossover_distribution: distribution parameter for SBX crossover
        :param mutation_probability: probability of mutation
        :param mutation_distribution: distribution parameter for mutation operator
        """
        super().__init__(pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves, use_approx_prior=use_approx_prior,
                         submit_file=submit_file, verbose=verbose, port=port, slave_dir=slave_dir)
        self.archive_obs = None
        self.population_obs = None
        self.archive_dv = None
        self.population_dv = None
        self.joint_dv = None
        self.joint_obs = None
        self.cross_prob = crossover_probability
        self.cross_dist = crossover_distribution
        self.mut_prob = mutation_probability
        self.mut_dist = mutation_distribution
        initialising_variables = 'pst {}\nparcov {}\nobscov {}\nnum_slaves {}\nuse_approx_prior {}\n ' \
                                 'submit_file {}\nverbose {}\nport {}\nslave_dir {}\n' \
                                 'cross_prob {}\ncross_dist {}\nmut_prob {}\n' \
                                 'mut_dist {}'.format(pst.filename, parcov, obscov, num_slaves, use_approx_prior,
                                                      submit_file, verbose, port, slave_dir, self.cross_prob,
                                                      self.cross_dist, self.mut_prob, self.mut_dist)
        self.logger.statement('using SPEA-2 evolutionary algorithm.\nParameters:\n{}'.format(initialising_variables))

    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None,par_ensemble=None,risk=0.5,
                   dv_names=None,par_names=None, when_calculate=0):
        super().initialize(obj_func_dict=obj_func_dict, num_par_reals=num_par_reals, num_dv_reals=num_dv_reals,
                           dv_ensemble=dv_ensemble, par_ensemble=par_ensemble, risk=risk, dv_names=dv_names,
                           par_names=par_names, when_calculate=when_calculate)
        self.logger.log('Initialising SPEA-2')
        self.population_dv = self.dv_ensemble
        self.population_obs = self.obs_ensemble
        self.joint_dv = pyemu.ParameterEnsemble(pst=self.pst, data=np.NaN, index=np.arange(2 * self.num_dv_reals),
                                                columns=self.dv_names)
        self.joint_obs = pyemu.ObservationEnsemble(pst=self.pst, data=np.NaN, index=np.arange(2 * self.num_dv_reals),
                                                 columns=self.pst.obs_names)
        self.archive_dv = pyemu.ParameterEnsemble(pst=self.pst, data=np.NaN,
                                                  index=np.arange(self.num_dv_reals, 2 * self.num_dv_reals),
                                                  columns=self.dv_names)
        self.archive_obs = pyemu.ObservationEnsemble(pst=self.pst, data=np.NaN,
                                                     index=np.arange(self.num_dv_reals, 2 * self.num_dv_reals),
                                                     columns=self.pst.obs_names)
        self.logger.log('Initialising SPEA-2')
        self.iter_num = 0

    def update(self):
        self.iter_num += 1
        self.logger.log('iteration number {}'.format(self.iter_num))
        self.joint_dv.loc[self.population_dv.index, :] = self.population_dv.values
        self.joint_dv.loc[self.archive_dv.index, :] = self.archive_dv.values
        self.joint_obs.loc[self.population_obs.index, :] = self.population_obs.values
        self.joint_obs.loc[self.archive_obs.index, :] = self.archive_obs.values
        # Fitness assignment
        self.logger.log('Fitness assignment')
        fitness_df, distance_df = self.obj_func.spea2_fitness_assignment(self.joint_obs, risk=self.risk,
                                                                         pop_size=self.num_dv_reals)
        self.logger.log('Fitness assignment')
        # enviromental selection (SPEA-2 sorting)
        self.logger.log('Selecting individuals for archive')
        archive_positions = Selection.spea2_enviromental_selection(fitness_df, distance_df, self.num_dv_reals)
        self.archive_dv.loc[:, :] = self.joint_dv.loc[archive_positions, :].values
        self.archive_obs.loc[:, :] = self.joint_obs.loc[archive_positions, :].values
        archive_fitness = pd.Series(data=fitness_df.loc[archive_positions].values, index=self.archive_dv.index)
        self.logger.log('Selecting individuals for archive')
        # Population selection (tournament selection)
        self.logger.log('Using tournament selection to create population')
        population_positions = self.tournament_selection(archive_fitness)
        self.population_dv.loc[:, :] = self.archive_dv.loc[population_positions, :].values
        self.population_obs.loc[:, :] = self.archive_obs.loc[population_positions, :].values
        self.logger.log('Using tournament selection to create population')
        # Crossover
        self.logger.log('Using Crossover and Mutation to introduce variation into population')
        to_update = Crossover.sbx(self.population_dv, self._get_bounds(), self.cross_prob, self.cross_dist)
        # Mutation
        to_update | Mutation.polynomial(self.population_dv, self._get_bounds(), self.mut_prob, self.mut_dist)
        to_update = list(to_update)
        self.logger.log('Using Crossover and Mutation to introduce variation into population')
        # Run model for updated individuals
        self.population_obs.loc[to_update, :] = self._calc_obs(self.population_dv.loc[to_update, self.dv_names]).values
        self.iter_report(self.dv_ensemble, self.obs_ensemble)
        self.logger.log('iteration number {}'.format(self.iter_num))
        return self.archive_dv, self.archive_obs

    def tournament_selection(self, fitness):
        def fitness_compare(idx1, idx2):
            return fitness.loc[idx1] < fitness.loc[idx2]
        return Selection.tournament_selection(fitness.index, self.num_dv_reals, comparison_key=fitness_compare)

    def iter_report(self, dv_ensemble, obs_ensemble):
        dv = dv_ensemble.copy()
        oe = obs_ensemble.copy()
        self.logger.log('removing previous csv files of dv and oe')
        dv_file_name = 'dv_ensemble_curr.csv'
        oe_file_name = 'obs_ensemble_curr.csv'
        try:
            os.remove(dv_file_name)
            os.remove(oe_file_name)
        except FileNotFoundError:
            self.logger.warn('Cannot find previous obs_ensemble or dv_ensemble csv')
        self.logger.log('removing previous csv files of dv and oe')
        self.logger.log('Writing csv files of dv_ensemble and obs_ensemble')
        dv.to_csv(dv_file_name)
        oe.to_csv(oe_file_name)
        self.logger.log('Writing csv files of dv_ensemble and obs_ensemble')

