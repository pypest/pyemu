
"""abstract class for MOEA and PopIndividual classes"""

import numpy as np
import subprocess
import os
import time
import pandas as pd
from .moouu import ParetoObjFunc, EvolAlg
import pyemu


class AbstractMOEA(EvolAlg):
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
                 cross_prob=0.8, cross_dist=20, mut_prob=0.01, mut_dist=20):
        """initialise the algorithm. Parameters:
           objectives: callable function that returns an ndarray of objective values
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           ------------------------------------- Optional parameters -------------------------------------------------
           archive_size: size of archive population. Full population size is 2 * archive due to population population
           cross_prob: probability of crossover occurring for any population
           cross_dist: distribution parameter of the crossover operation
           mut_prob: probability of mutation occurring for any population
           mut_dist: distribution parameter of the mutation operation
           iterations: number of iterations of the algorithm
        """
        super().__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                         submit_file=submit_file, verbose=verbose, port=port, slave_dir=slave_dir)
        # self.model_path = pst
        # self.model = model
        # self.constraints = constraints
        # self.number_objectives = number_objectives
        # self.is_constrained = constraints is not None
        # self.objectives = objectives
        # self.bounds = bounds

        # self.pst = pst done by super()
        self.bounds = None
        self.is_constrained = None
        self.number_objectives = None
        self.dv_names = None
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        self.animator_points = []

    def run(self):
        pass

    def initialize(self,obj_func_dict,num_par_reals=100,num_dv_reals=100,
                   dv_ensemble=None, par_ensemble=None, risk=0.5, dv_names=None,
                   par_names=None, when_calculate=0):
        super().initialize(obj_func_dict=obj_func_dict, num_par_reals=num_par_reals, num_dv_reals=num_dv_reals,
                           dv_ensemble=dv_ensemble, par_ensemble=par_ensemble, risk=risk, dv_names=dv_names,
                           par_names=par_names, when_calculate=when_calculate)
        self.bounds = self._get_bounds()
        self.is_constrained = not(self.pst.greater_than_obs_constraints.empty
                                  and self.pst.less_than_obs_constraints.empty)
        self.number_objectives = len(obj_func_dict)
        if dv_names is None:
            self.dv_names = dv_ensemble.columns
        else:
            self.dv_names = dv_names

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def run_model(self, population):
        """
        to be called after all changes to the population have been made. individuals whose objectives
        need calculating are calculated.
        """
        df = population.to_pyemu_ensemble(pst=self.pst)
        observation_ensemble = super()._calc_obs(df)
        objectives = self.obj_func.objective_vector(observation_ensemble).values
        constraints = self.obj_func.constraint_violation_vector(self.obs_ensemble)
        population.update_objectives(objectives, constraints)


class AbstractPopulation:
    """represents a population of individuals"""

    def __init__(self, population, dv_names, constrained=False):
        """create either from uniform draw or from old population"""
        if type(population) != np.ndarray:
            self.population = np.array(population)
        else:
            self.population = population
        self.constrained = constrained
        self.dv_names = dv_names

    def __len__(self):
        return len(self.population)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ensemble = self.to_decision_variable_array()
        index = ['individual {}'.format(i) for i in range(1, len(self) + 1)]
        columns = ['Decision variable {}'.format(i) for i in range(1, ensemble.shape[1] + 1)]
        df = pd.DataFrame(ensemble, index=index, columns=columns)
        return str(df)

    def __add__(self, other):
        population = np.concatenate((self.population, other.population))
        if self.constrained is not other.constrained:
            raise Exception("population {} and population {} are inconsistent with constraints".format(self, other))
        return self.__class__(population=population, constrained=self.constrained, dv_names=self.dv_names)

    def __iter__(self):
        return iter(self.population)

    def __next__(self):
        return next(self.population)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(self.population.__getitem__(item), constrained=self.constrained,
                                  dv_names=self.dv_names)
        else:
            return self.population.__getitem__(item)

    def __setitem__(self, key, value):
        self.population.__setitem__(key, value)

    @classmethod
    def from_pyemu_ensemble(cls, dv_ensemble, individual_class, constrained):
        population = []
        for i, row in dv_ensemble.iterrows():
            population.append(individual_class(d_vars=row.values, is_constrained=constrained))
        return cls(population=population, dv_names=dv_ensemble.columns, constrained=constrained)

    def tournament_selection(self, num_to_select, is_better=lambda x, y: x.fitness < y.fitness):
        next_population = []
        even = np.arange(0, (num_to_select // 2) * 2, 2)
        odd = np.arange(1, (num_to_select // 2) * 2, 2)
        for _ in range(2):
            np.random.shuffle(self.population)
            for individual1, individual2 in zip(self.population[even], self.population[odd]):
                if is_better(individual1, individual2):
                    next_population.append(individual1.clone())
                else:
                    next_population.append(individual2.clone())
        if num_to_select % 2 == 1:  # i.e is odd
            next_population.append(self.population[-1].clone())
        return self.__class__(next_population, constrained=self.constrained, dv_names=self.dv_names)

    def crossover(self, bounds, crossover_probability, crossover_distribution):
        for individual1 in self.population:
            if np.random.random() <= crossover_probability:
                individual2 = np.random.choice(self.population)
                individual1.crossover_SBX(individual2, bounds, crossover_distribution)

    def mutation(self, bounds, mutation_probability, mutation_distribution):
        k = 0
        i = 0
        while i < len(self):
            self.population[i].mutate_polynomial(k, bounds, mutation_distribution)
            length = int(np.ceil(- 1 / mutation_probability * np.log(1 - np.random.random())))
            i += (k + length) // len(bounds)
            k = (k + length) % len(bounds)

    def to_decision_variable_array(self, model_update=False):
        if model_update is False:
            positions = np.arange(0, len(self))
        else:
            positions = np.where([individual.run_model for individual in self.population])[0]
        return np.atleast_2d(np.array([individual.d_vars for individual in self.population[positions]]))

    def to_pyemu_ensemble(self, pst):
        """for now just use this simple output method. In future use pyemu and pest to write model input file"""
        ensemble = self.to_decision_variable_array(model_update=True)
        columns = [dv_name for dv_name in self.dv_names]
        return pyemu.ParameterEnsemble(pst=pst, data=ensemble, columns=columns)

    def update_objectives(self, objectives, constraints=None):
        positions = np.where([individual.run_model for individual in self.population])[0]
        to_update = self.population[positions]
        if not constraints.empty:
            for objective, constraint, individual in zip(objectives, constraints.values, to_update):
                individual.objective_values = objective
                individual.total_constraint_violation = np.sum(constraint)
                individual.run_model = False
        else:
            for objective, individual in zip(objectives, to_update):
                individual.objective_values = objective
                individual.run_model = False
        time.sleep(0.1)

    def reset_population(self):
        for individual in self.population:
            individual.clear()

    def sort(self, key=None):
        if key is None:
            self.population.sort()
        else:
            key_values = np.array([key(individual) for individual in self.population])
            args = key_values.argsort()
            self.population = self.population[args]


class AbstractPopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, d_vars, is_constrained=False, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        self.d_vars = np.array(d_vars, dtype=float)
        # self.objectives = objectives
        self.fitness = 0
        self.is_constrained = is_constrained
        self.run_model = False
        if objective_values is None:
            self.run_model = True
            self.objective_values = None
        else:
            self.objective_values = objective_values
        if self.is_constrained:
            if total_constraint_violation is None:
                self.violates = None
            else:
                self.total_constraint_violation = total_constraint_violation
                self.violates = bool(self.total_constraint_violation > 0)

    def update(self):
        self.objective_values = None
        self.violates = None
        self.run_model = True
        if self.is_constrained:
            self.total_constraint_violation = None

    def __str__(self):
        try:
            return "d_vars: {}, objectives: {}".format(self.d_vars, self.objective_values)
        except AttributeError:
            return "d_vars: {}, objectives not calculated".format(self.d_vars)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return bool(self.fitness < other.fitness)

    def __le__(self, other):
        return bool(self.fitness <= other.fitness)

    def __gt__(self, other):
        return bool(self.fitness > other.fitness)

    def __ge__(self, other):
        return bool(self.fitness >= other.fitness)

    def fitness_equals(self, other):
        return bool(self.fitness == other.fitness)

    def calculate_objective_values(self):
        print("calculate_objective_values should not be called. called by {}".format(str(self)))

    def dominates(self, other):
        def _unconstrained_dominate(a, b):
            weak_dom = True
            strong_condition = False
            for i in range(len(a.objective_values)):
                if a.objective_values[i] < b.objective_values[i]:
                    weak_dom = False
                    i = len(a.objective_values)
                elif a.objective_values[i] > b.objective_values[i]:
                    strong_condition = True
                i += 1
            return weak_dom and strong_condition
        if self.is_constrained:
            if self.violates and other.violates:
                result = bool(self.total_constraint_violation < other.total_constraint_violation)
            elif self.violates:
                result = False
            elif other.violates:
                result = True
            else:
                result = _unconstrained_dominate(self, other)
        else:
            result = _unconstrained_dominate(self, other)
        return result

    def covers(self, other):
        def _unconstrained_covers(a, b):
            condition = True
            for i in range(len(a.objective_values)):
                if a.objective_values[i] > b.objective_values[i]:
                    condition = False
                    i = len(a.objective_values)
                i += 1
            return condition
        if self.is_constrained:
            if self.violates and other.violates:
                result = self.total_constraint_violation < other.total_constraint_violation
            elif self.violates:
                result = False
            elif other.violates:
                result = True
            else:
                result = _unconstrained_covers(self, other)
        else:
            result = _unconstrained_covers(self, other)
        return result

    def clone(self):
        cls = self.__class__
        if self.is_constrained and self.violates:
            return cls(self.d_vars, is_constrained=self.is_constrained,
                       total_constraint_violation=self.total_constraint_violation)
        elif self.is_constrained:
            return cls(self.d_vars, is_constrained=self.is_constrained, objective_values=self.objective_values,
                       total_constraint_violation=self.total_constraint_violation)
        else:
            return cls(self.d_vars, objective_values=self.objective_values)

    def crossover_SBX(self, other, bounds, distribution_parameter):
        """uses simulated binary crossover"""
        for i in range(len(self.d_vars)): #TODO change bound handling method
            if np.random.random() >= 0.5:
                p1 = self.d_vars[i]
                p2 = other.d_vars[i]
                if np.isclose(p1, p2, rtol=0, atol=1e-15):
                    beta_1, beta_2 = self.get_beta(np.NaN, np.NaN, distribution_parameter, values_are_close=True)
                else:
                    lower_transformation = (p1 + p2 - 2 * bounds[0][i]) / (abs(p2 - p1))
                    upper_transformation = (2 * bounds[1][i] - p1 - p2) / (abs(p2 - p1))
                    beta_1, beta_2 = self.get_beta(lower_transformation, upper_transformation, distribution_parameter)
                self.d_vars[i] = 0.5 * ((p1 + p2) - beta_1 * abs(p2 - p1))
                other.d_vars[i] = 0.5 * ((p1 + p2) + beta_2 * abs(p2 - p1))
        self.update()
        other.update()

    @staticmethod
    def get_beta(transformation1, transformation2, distribution_parameter, values_are_close=False):
        rand = np.random.random()
        beta_values = []
        for transformation in [transformation1, transformation2]:
            if values_are_close:
                p = 1
            else:
                p = 1 - 1/(2 * transformation ** (distribution_parameter + 1))
            u = rand * p
            if u <= 0.5:
                beta_values.append((2 * u) ** (1 / (distribution_parameter + 1)))
            else:
                beta_values.append((1 / (2 - 2 * u)) ** (1 / (distribution_parameter + 1)))
        return beta_values

    def mutate_polynomial(self, variable, bounds, distribution_parameter):
        u = np.random.random() # TODO bounds change
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.d_vars[variable] = self.d_vars[variable] + delta * (self.d_vars[variable] - bounds[0][variable])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.d_vars[variable] = self.d_vars[variable] + delta * (bounds[1][variable] - self.d_vars[variable])
        self.update()

    def clear(self):
        self.fitness = 0
