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
    def __init__(self, objectives, bounds, number_objectives, pst=None, parcov=None, obscov=None,
                 constraints=None, parent_pop_size=100, cross_prob=0.9, cross_dist=15,mut_prob=0.01, mut_dist=20,
                 iterations=20, submit_file=None, num_slaves=0, verbose=False):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions to be optimised
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           ------------------------------------- Optional parameters -------------------------------------------------
           parent_pop_size: size of parent population. Full population size is 2 * parent_pop due to child population
           cross_prob: probability of crossover occurring for any child
           cross_dist: distribution parameter of the crossover operation
           mut_prob: probability of mutation occurring for any child
           mut_dist: distribution parameter of the mutation operation
           iterations: number of iterations of the algorithm
        """
        super().__init__(objectives=objectives, bounds=bounds, number_objectives=number_objectives, pst=pst,
                         parcov=parcov, obscov=obscov, num_slaves=num_slaves, submit_file=submit_file,
                         port = 4004, slave_dir = "template", constraints=None, cross_prob=0.8, cross_dist=20,
                         mut_prob=0.01, mut_dist=20, iterations=20, verbose=verbose)
        # --------------------------------sets of population members------------------------------------------
        self.parent_pop_size = parent_pop_size
        self.parent_pop = []
        self.child_pop = []
        self.fronts = []
        self.population = []

    def __str__(self):
        return "NSGA-II"

    def run(self):
        """Run the NSGA-II algorithm. This will return an approximation to the pareto front"""
        self.parent_pop = self.initialise_population(self.parent_pop_size, PopIndividual)
        self.run_model(self.parent_pop)
        self.non_dominated_sort(init=True)
        self.child_pop = []
        self.child_pop = self.tournament_selection(self.parent_pop, self.parent_pop_size, is_better=lambda x, y: x < y)
        self.crossover_step_SBX(self.child_pop)
        self.mutation_step_polynomial(self.child_pop)
        self.run_model(self.child_pop)
        for i in range(self.iterations):
            self.population = self.child_pop + self.parent_pop
            self.non_dominated_sort()
            self.parent_pop = []
            j = 0
            while len(self.parent_pop) + len(self.fronts[j]) < self.parent_pop_size:
                self.crowding_distance_assignment(self.fronts[j])
                self.parent_pop += self.fronts[j]
                j += 1

            self.crowding_distance_assignment(self.fronts[j])
            self.fronts[j].sort()
            self.parent_pop += self.fronts[j][:(self.parent_pop_size - len(self.parent_pop))]
            self.child_pop = []
            self.child_pop = self.tournament_selection(self.parent_pop, self.parent_pop_size,
                                                       is_better=lambda x, y: x < y)
            self.crossover_step_SBX(self.child_pop)
            self.mutation_step_polynomial(self.child_pop)
            self.reset_population(self.parent_pop)
            self.run_model(self.child_pop)
        return self.fronts[0]

    def run_IO(self):
        parent = Population.draw_uniform(constrained=self.is_constrained, bounds=self.bounds,
                                         population_size=self.parent_pop_size, individual_class=PopIndividual)
        self.run_model_IO(population=parent, decision_variable_template='individual_{}.inp',
                       objective_template='individual_{}.out')
        fronts = parent.non_dominated_sort()
        child = parent.tournament_selection(self.parent_pop_size, is_better=lambda x, y: x < y)
        child.crossover(bounds=self.bounds, crossover_probability=self.cross_prob,
                        crossover_distribution=self.cross_dist)
        child.mutation(bounds=self.bounds, mutation_probability=self.mut_prob, mutation_distribution=self.mut_dist)
        self.run_model_IO(population=child, decision_variable_template='individual_{}.inp',
                           objective_template='individual_{}.out')
        for i in range(self.iterations):
            population = parent + child
            fronts = population.non_dominated_sort()
            parent = Population([], constrained=self.is_constrained)
            j = 0
            while len(parent) + len(fronts[j]) < self.parent_pop_size:
                self.crowding_distance_assignment(fronts[j])
                parent += fronts[j]
                j += 1

            self.crowding_distance_assignment(fronts[j])
            fronts[j].sort()
            parent += fronts[j][:(self.parent_pop_size - len(parent))]
            child = parent.tournament_selection(self.parent_pop_size, is_better=lambda x, y: x < y)
            child.crossover(bounds=self.bounds, crossover_probability=self.cross_prob,
                            crossover_distribution=self.cross_dist)
            child.mutation(bounds=self.bounds, mutation_probability=self.mut_prob, mutation_distribution=self.mut_dist)
            parent.reset_population()
            self.run_model_IO(population=child, decision_variable_template='individual_{}.inp',
                               objective_template='individual_{}.out')
        return fronts[0]

    def non_dominated_sort(self, init=False):
        """Internal method. Sorts the population into a set of non-dominated pareto fronts F1, F2 ..."""
        if init:
            self.population = self.parent_pop
        self.fronts = []
        Q = set()
        for i in range(len(self.population)):
            p = self.population[i]
            for j in range(i + 1, len(self.population)):
                q = self.population[j]
                if p.dominates(q): #self.dominates(p, q):
                    p.dominated_set.append(q)
                    q.domination_count += 1
                elif q.dominates(p): #self.dominates(q, p):
                    q.dominated_set.append(p)
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                Q.add(p)
        self.fronts.append(list(Q))
        i = 0
        while self.fronts[i]:
            Q = set()
            for p in self.fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        Q.add(q)
            i += 1
            self.fronts.append(list(Q))
        self.fronts.pop()

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


class Population(AbstractPopulation):

    def __init__(self, population, constrained):
        super().__init__(population, constrained)

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
        fronts.append(Population(population=list(Q), constrained=self.constrained))
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
            fronts.append(Population(population=list(Q), constrained=self.constrained))
        fronts.pop()
        return fronts


class PopIndividual(AbstractPopIndividual):
    """represents an individual in a population for NSGA-II"""

    def __init__(self, d_vars, constraints=None, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        super().__init__(d_vars, constraints, objective_values, total_constraint_violation)
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
