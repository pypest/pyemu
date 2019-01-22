"""
NSGA-II algorithim for multi objective optimisation
benchmark testing for SAM mooUU using Differential evolution
version proposed by Deb[2002] IEEE
GNS.cri
Otis Rea
"""
import numpy as np
from .Abstract_Moo import *


class NSGA_II(AbstractMOEA):
    """implements the NSGA_II algorithm for multi objective optimisation
       Methods to be used:
       __init__: initialise the algorithm.
            Objectives: a list of functions to be optimised, e.g [f1, f2...]
            bounds: limits on the decision variables in the problem, to be specified as
            ([x1_lower, x1_upper], [x2_lower, x2_upper], ...)
            Other d_vars are just numbers specifying parameters of the algorithm
        run: runs the algorithm and returns an approximation to the pareto front
    """

    # ------------------------External methods--------------------------------
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template",
                 archive_size=100, cross_prob=0.9, cross_dist=15, mut_prob=0.01, mut_dist=20):
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
                 submit_file=submit_file, verbose=verbose, port=port, slave_dir=slave_dir,
                 cross_prob=cross_prob, cross_dist=cross_dist, mut_prob=mut_prob, mut_dist=mut_dist)
        # --------------------------------sets of population members------------------------------------------
        self.archive = None
        self.population = None

    def __str__(self):
        return "NSGA-II"

    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None, par_ensemble=None, risk=0.5, dv_names=None,
                   par_names=None):
        super().initialize(obj_func_dict=obj_func_dict, num_par_reals=num_par_reals, num_dv_reals=num_dv_reals,
                           dv_ensemble=dv_ensemble, par_ensemble=par_ensemble, risk=risk, dv_names=dv_names,
                           par_names=par_names)
        self.archive = Population.from_pyemu_ensemble(dv_ensemble=self.dv_ensemble, constrained=self.is_constrained,
                                                      individual_class=PopIndividual)
        objectives = self.obj_func.objective_vector(self.obs_ensemble).values
        constraints = self.obj_func.constraint_violation_vector(self.obs_ensemble).values
        self.archive.update_objectives(objectives=objectives, constraints=constraints)
        self.archive.non_dominated_sort()
        self.population = self.archive.tournament_selection(num_to_select=self.num_dv_reals, is_better=lambda x, y: x < y)
        self.population.crossover(bounds=self.bounds, crossover_probability=self.cross_prob,
                                  crossover_distribution=self.cross_dist)
        self.population.mutation(bounds=self.bounds, mutation_probability=self.mut_prob,
                                 mutation_distribution=self.mut_dist)
        self.run_model(population=self.population)

    def update(self):
        t0 = time.perf_counter()
        joint = self.archive + self.population
        fronts = joint.non_dominated_sort()
        self.archive = Population([], constrained=self.is_constrained, dv_names=self.dv_ensemble.columns)
        j = 0
        while len(self.archive) + len(fronts[j]) < self.num_dv_reals:
            self.crowding_distance_assignment(fronts[j])
            self.archive += fronts[j]
            j += 1
        self.crowding_distance_assignment(fronts[j])
        fronts[j].sort()
        self.archive += fronts[j][:(self.num_dv_reals - len(self.archive))]
        self.population = self.archive.tournament_selection(self.num_dv_reals, is_better=lambda x, y: x < y)
        self.population.crossover(bounds=self.bounds, crossover_probability=self.cross_prob,
                                  crossover_distribution=self.cross_dist)
        self.population.mutation(bounds=self.bounds, mutation_probability=self.mut_prob,
                                 mutation_distribution=self.mut_dist)
        self.archive.reset_population()
        self.run_model(self.population)
        iteration_time = time.perf_counter() - t0
        self.iter_report(iteration_time)
        self.archive.reset_population()
        return fronts[0]

    def crowding_distance_assignment(self, front):
        """Internal method. Calculates and assigns the crowding distance for each individual in the population"""
        i = 0
        while i < self.number_objectives:
            front.sort(key=lambda x: x.objective_values[i])
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            max_objective = front[-1].objective_values[i]
            min_objective = front[0].objective_values[i]
            if not np.isclose(min_objective, max_objective):
                for j in range(1, len(front) - 1):
                    front[j].crowding_distance += (front[j + 1].objective_values[i] - front[j - 1].objective_values[i])\
                                                  / (max_objective - min_objective)
                    j += 1
            i += 1
            # TODO: Sorting by objective d_vars: will not work in constrained problem

    def iter_report(self, iteration_time):
        non_dominated = len([individual.rank == 1 for individual in self.archive])
        self.logger.statement('iteration took: {} s'.format(iteration_time))
        self.logger.statement("{0} non-dominated individuals in population".format(non_dominated))


class Population(AbstractPopulation):

    def __init__(self, population, constrained, dv_names):
        super().__init__(population, constrained=constrained, dv_names=dv_names)

    def non_dominated_sort(self):
        fronts = []
        Q = set()
        for i, individual1 in enumerate(self.population): # p
            for individual2 in self.population[i + 1:]: # q
                if individual1.dominates(individual2):  # self.dominates(p, q): p dominates q
                    individual1.dominated_set.append(individual2) # append q to dominated set of p
                    individual2.domination_count += 1 # increase q domination count
                elif individual2.dominates(individual1):  # self.dominates(q, p):
                    individual2.dominated_set.append(individual1)
                    individual1.domination_count += 1
            if individual1.domination_count == 0:
                individual1.rank = 1
                Q.add(individual1)
        fronts.append(Population(population=list(Q), constrained=self.constrained, dv_names=self.dv_names))
        i = 0
        while fronts[i]:
            Q = set()
            for individual1 in fronts[i]:
                for individual2 in individual1.dominated_set:
                    individual2.domination_count -= 1
                    if individual2.domination_count == 0:
                        individual2.rank = i + 2
                        Q.add(individual2)
            i += 1
            fronts.append(Population(population=list(Q), constrained=self.constrained, dv_names=self.dv_names))
        fronts.pop()
        return fronts


class PopIndividual(AbstractPopIndividual):
    """represents an individual in a population for NSGA-II"""

    def __init__(self, d_vars, is_constrained=False, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        super().__init__(d_vars, is_constrained=is_constrained, objective_values=objective_values,
                         total_constraint_violation=total_constraint_violation)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def __str__(self):
        """string representation of the individual"""
        s = "solution at {}, rank {}".format(self.d_vars, self.rank)
        return s

    def __repr__(self):
        """representation of the individual"""
        return "[{:.3f}, {:.3f}]".format(self.d_vars[0], self.d_vars[1])

    def __lt__(self, other):
        return bool((self.rank < other.rank) or (
                (self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def clear(self):
        self.domination_count = 0
        self.dominated_set = []
        self.crowding_distance = 0
