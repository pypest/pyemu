
"""abstract class for MOEA and PopIndividual classes"""

import numpy as np
import subprocess
import os
from .moouu import EvolAlg, ParetoObjFunc
import time
import pandas as pd




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
    def __init__(self, objectives, bounds, number_objectives, pst, parcov = None, obscov = None, num_slaves = 0,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template",
                 constraints=None, cross_prob=0.8, cross_dist=20, mut_prob=0.01, mut_dist=20,
                 iterations=20):
        """initialise the algorithm. Parameters:
           objectives: callable function that returns an ndarray of objective values
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           ------------------------------------- Optional parameters -------------------------------------------------
           parent_pop_size: size of parent population. Full population size is 2 * parent_pop due to child population
           cross_prob: probability of crossover occurring for any child
           cross_dist: distribution parameter of the crossover operation
           mut_prob: probability of mutation occurring for any child
           mut_dist: distribution parameter of the mutation operation
           iterations: number of iterations of the algorithm
        """
        super().__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                         submit_file=submit_file, verbose=verbose, port=port,slave_dir=slave_dir)
        self.constraints = constraints
        self.number_objectives = number_objectives
        self.is_constrained = constraints is not None
        self.objectives = objectives
        self.bounds = bounds
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        self.iterations = iterations
        self.animator_points = []

    def run(self):
        pass

    def run_model(self, population):
        """
        to be called after all changes to the population have been made. individuals whose objectives
        need calculating are calculated.
        """
        if type(population) != "numpy.ndarray":
            population = np.array(population)
        positions = np.where([individual.calculate_objectives for individual in population])[0]
        ensemble = np.atleast_2d(np.array([individual.d_vars for individual in population[positions]]))
        ensemble = ensemble.T
        if len(ensemble) > 0:
            data = np.atleast_2d(self.objectives(ensemble))
            data = data.T
            for i, individual in enumerate(population[positions]):
                individual.objective_values = data[i]

    def run_pyemu(self):
        raise Exception('run_pyemu must be implemented by child class')

    def run_model_IO(self, population, decision_variable_template='individual_{}.inp',
                  objective_template='individual_{}.out'):
        input_files = population.write_decision_variables(decision_variable_template)
        output_files = [objective_template.format(i + 1) for i in range(len(input_files))]
        for I, O in zip(input_files, output_files):  # TODO: substitute this for parallel run use pyemu.helpers.setup_slaves (or something)
            cmd = self.model_path + [self.model, '--input_file', I, '--output_file', O]
            subprocess.run(cmd, shell=True)
        population.read_objectives(objective_template)

    @staticmethod
    def tournament_selection(old_population, new_max_size, is_better=lambda x, y: x.fitness < y.fitness):
        new_population = []
        for _ in range(2):
            np.random.shuffle(old_population)
            i = 0
            while i < (new_max_size // 2):
                if is_better(old_population[2 * i], old_population[2 * i + 1]):  # Change this to a method -> __lt__ no more
                    new_population.append(old_population[2 * i].clone())
                else:
                    new_population.append(old_population[2 * i + 1].clone())
                i += 1
        if new_max_size % 2 == 1:  # i.e is odd
            new_population.append(old_population[-1].clone())
        return new_population

    def initialise_population(self, population_size, population_class):
        new_population = []
        for i in range(population_size):
            d_vars = []
            for j in range(len(self.bounds)):
                d_vars.append(self.bounds[j][0] + np.random.random() * (self.bounds[j][1] - self.bounds[j][0]))
            new_population.append(population_class(d_vars, self.constraints))
        return new_population

    def crossover_step_SBX(self, population):
        for i in range(len(population)):
            if np.random.random() > self.cross_prob:
                rand_index = np.random.randint(0, len(population))
                population[i].crossover_SBX(population[rand_index], self.bounds, self.cross_dist)

    def mutation_step_polynomial(self, population):
        k = 0
        i = 0
        while i < len(population):
            population[i].mutate_polynomial(k, self.bounds, self.mut_dist)
            length = int(np.ceil(- 1 / self.mut_prob * np.log(1 - np.random.random())))
            i += (k + length) // len(self.bounds)
            k = (k + length) % len(self.bounds)

    def reset_population(self, population):
        gen = []
        for _ in range(self.number_objectives):
            gen.append(np.zeros(len(population)))
        count = 0
        for individual in population:
            for i, obj in enumerate(individual.objective_values):
                gen[i][count] = obj
            individual.clear()
            count += 1
        self.animator_points.append(gen)

    def get_animation_points(self):
        return self.animator_points


class AbstractPopulation:
    """represents a population of individuals"""

    def __init__(self, population, constrained=False):
        """create either from uniform draw or from old population"""
        if type(population) != np.ndarray:
            self.population = np.array(population)
        else:
            self.population = population
        self.constrained = constrained

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
        return self.__class__(population, self.constrained)

    def __iter__(self):
        return iter(self.population)

    def __next__(self):
        return next(self.population)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(self.population.__getitem__(item), constrained=self.constrained)
        else:
            return self.population.__getitem__(item)

    def __setitem__(self, key, value):
        self.population.__setitem__(key, value)

    @classmethod
    def draw_uniform(cls, constrained, bounds, population_size, individual_class):
        population = []
        for i in range(population_size):
            d_vars = []
            for bound in bounds:
                d_vars.append(min(bound) + np.random.random() * (max(bound) - min(bound)))
            population.append(individual_class(d_vars))
        return cls(population, constrained)

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
        return self.__class__(next_population, self.constrained)

    def crossover(self, bounds, crossover_probability, crossover_distribution):
        for individual1 in self.population:
            if np.random.random() > crossover_probability:
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
            positions = np.where([individual.calculate_objectives for individual in self.population])[0]
        return np.atleast_2d(np.array([individual.d_vars for individual in self.population[positions]]))

    def write_decision_variables(self, filename_template):
        """for now just use this simple output method. In future use pyemu and pest to write model input file"""
        ensemble = self.to_decision_variable_array(model_update=True)
        file_names = [filename_template.format(i) for i in range(1, ensemble.shape[0] + 1)]
        for file in file_names:
            if os.path.exists(file):
                os.remove(file)
        index = ['d_var{}'.format(i) for i in range(1, ensemble.shape[1] + 1)]
        for decision_variables, filename in zip(ensemble, file_names):
            df = pd.Series(decision_variables, index=index, name='MODEL INPUT FILE')
            df.to_csv(filename, header=True, encoding='ascii')
        return file_names

    def read_objectives(self, filename_template):
        positions = np.where([individual.calculate_objectives for individual in self.population])[0]
        file_names = [filename_template.format(i) for i in range(1, len(positions) + 1)]
        for filename, individual in zip(file_names, self.population[positions]):
            df = pd.read_csv(filename, header=0, index_col=0, encoding='ascii').T
            objectives = np.array(df)[0]
            individual.objective_values = objectives

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

    def __init__(self, d_vars, constraints=None, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        self.d_vars = np.array(d_vars, dtype=float)
        # self.objectives = objectives
        self.fitness = 0
        self.is_constrained = not (constraints is None)
        self.calculate_objectives = False
        self.calculate_constraints = False
        if objective_values is None:
            self.calculate_objectives = True
            self.objective_values = None
        else:
            self.objective_values = objective_values
        if self.is_constrained:
            self.constraints = constraints
            if total_constraint_violation is None:
                self.total_constraint_violation = self.calculate_constrained_values(self.constraints, self.d_vars)
                self.calculate_constraints = True
            else:
                self.total_constraint_violation = total_constraint_violation
                self.calculate_constraints = False
            self.violates = bool(self.total_constraint_violation > 0)

    def update(self):
        if not self.is_constrained:
            self.calculate_objectives = True
        else:  # TODO: check this whole section. Strategy for constrained problems needs to be updated
            self.total_constraint_violation = self.calculate_constrained_values(self.constraints, self.d_vars)
            self.violates = bool(self.total_constraint_violation > 0)
            self.calculate_objectives = True
            self.calculate_constraints = True

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
        """tests for dominance between self and other. returns true if self dominates other, false otherwise"""
        # weak_dom = True
        # strong_condition = False
        # for i in range(len(self.objective_values)):
        #     if self.objective_values[i] > other.objective_values[i]:
        #         weak_dom = False
        #         i = len(self.objective_values)
        #     elif self.objective_values[i] < other.objective_values[i]:
        #         strong_condition = True
        #     i += 1
        # return weak_dom and strong_condition
        def _unconstrained_dominate(a, b):
            weak_dom = True
            strong_condition = False
            for i in range(len(a.objective_values)):
                if a.objective_values[i] > b.objective_values[i]:
                    weak_dom = False
                    i = len(a.objective_values)
                elif a.objective_values[i] < b.objective_values[i]:
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
            return cls(self.d_vars, self.constraints,
                       total_constraint_violation=self.total_constraint_violation)
        elif self.is_constrained:
            return cls(self.d_vars, self.constraints, self.objective_values,
                       self.total_constraint_violation)
        else:
            return cls(self.d_vars, objective_values=self.objective_values)

    def crossover_SBX(self, other, bounds, distribution_parameter):
        """uses simulated binary crossover"""
        for i in range(len(self.d_vars)):
            if np.random.random() >= 0.5:
                p1 = self.d_vars[i]
                p2 = other.d_vars[i]
                if np.isclose(p1, p2, rtol=0, atol=1e-15):
                    beta_1, beta_2 = self.get_beta(np.NaN, np.NaN, distribution_parameter, values_are_close=True)
                else:
                    lower_transformation = (p1 + p2 - 2 * bounds[i][0]) / (abs(p2 - p1))
                    upper_transformation = (2 * bounds[i][1] - p1 - p2) / (abs(p2 - p1))
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
        u = np.random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.d_vars[variable] = self.d_vars[variable] + delta * (self.d_vars[variable] - bounds[variable][0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.d_vars[variable] = self.d_vars[variable] + delta * (bounds[variable][1] - self.d_vars[variable])
        self.update()

    @staticmethod
    def calculate_constrained_values(constraints, d_vars):
        """return the total constraint violation. Assumes all constraints are of the form g(X) >= 0"""
        evaluated_constraints = constraints(d_vars)
        where_violates = np.where(evaluated_constraints < 0)
        return np.sum(np.abs(evaluated_constraints[where_violates]))
        # total_constraint_violation = 0
        # evaluated_constraints = constraints(d_vars)
        # for g in evaluated_constraints:
        #     if g < 0:
        #         total_constraint_violation += abs(g)
        # return total_constraint_violation

    def clear(self):
        self.fitness = 0
