
"""
Benchmark test problems script. Will parse command line arguments in format:
"""

import argparse
import numpy as np
import scipy.stats as stat
import os
import pandas as pd
import re
import subprocess
import time


class BenchmarkTestProblem:
    """
    General class for stochastic benchmark test problems
    """
    def __init__(self):
        raise Exception("Class should not be instantiated")

    @staticmethod
    def calculate_objectives(d_vars, pars):
        raise Exception("Child class should define this function")

    @staticmethod
    def number_parameters():
        raise Exception("Child class should define this function")

    @staticmethod
    def number_decision_variables():
        raise Exception("Child class should define this function")

    @staticmethod
    def number_objectives():
        raise Exception("Child class should define this function")

    @staticmethod
    def parameter_means():
        raise Exception("Child class should define this function")

    @staticmethod
    def parameter_covariance():
        raise Exception("Child class should define this function")

    @staticmethod
    def bounds():
        raise Exception("Child class should define this function")

    @staticmethod
    def constrained():
        return False

    @staticmethod
    def _quadratic_positive_root(a, b, c):
        return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


class StochasticParaboloid(BenchmarkTestProblem):

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(d_vars, pars):
        return (d_vars[0] - 2) ** 2 + d_vars[1] ** 2 + 0.1 * (d_vars[0] + pars[0]) ** 3

    @staticmethod
    def f2(d_vars, pars):
        return (d_vars[0] + 2) ** 2 + d_vars[1] ** 2 - 0.1 * (d_vars[0] + pars[0]) ** 3

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([StochasticParaboloid.f1(d_vars, pars), StochasticParaboloid.f2(d_vars, pars)])

    @staticmethod
    def number_parameters():
        return 1

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def number_objectives():
        return 2

    @staticmethod
    def parameter_means():
        return np.zeros(StochasticParaboloid.number_parameters())

    @staticmethod
    def bounds():
        return [(-5, 5), (-5, 5)]

    @staticmethod
    def parameter_covariance():
        return np.diag([1])

    @staticmethod
    def pareto_front(risk):
        """:returns pareto front given a certain level of risk"""
        beta_up = stat.norm.ppf(risk)
        beta_down = stat.norm.ppf(1 - risk)
        a, b, c = (3 * 0.1, 2 + 6 * 0.1 * beta_up, 3 * 0.1 * beta_up ** 2 - 4)
        argminf1 = BenchmarkTestProblem._quadratic_positive_root(a, b, c)
        a, b, c = (-3 * 0.1, 2 - 6 * 0.1 * beta_down, 4 - 3 * 0.1 * beta_down ** 2)
        argminf2 = BenchmarkTestProblem._quadratic_positive_root(a, b, c)
        pareto_set = np.vstack((np.linspace(argminf2, argminf1), np.zeros(50)))
        f1 = StochasticParaboloid.f1(pareto_set, [beta_up])
        f2 = StochasticParaboloid.f2(pareto_set, [beta_down])
        return np.vstack((f1, f2))


class StochasticParaboloid2(StochasticParaboloid):

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(d_vars, pars):
        return (d_vars[0] - 2) ** 2 + d_vars[1] ** 2 + 0.05 * (d_vars[1] + pars[0]) ** 3

    @staticmethod
    def f2(d_vars, pars):
        return (d_vars[0] + 2) ** 2 + d_vars[1] ** 2 + 0.05 * (d_vars[1] + pars[0]) ** 3

    @staticmethod
    def pareto_front(risk):
        beta_up = stat.norm.ppf(risk)
        arg_min_f1 = BenchmarkTestProblem._quadratic_positive_root(0.15, 2 - 0.3 * beta_up, 0.15 * beta_up ** 2)
        pareto_set = np.vstack((np.linspace(-2, 2), np.full(50, arg_min_f1)))
        f1 = StochasticParaboloid2.f1(pareto_set, [beta_up])
        f2 = StochasticParaboloid2.f2(pareto_set, [beta_up])
        return np.vstack((f1, f2))


class DeterministicBenchmark(BenchmarkTestProblem):

    def __init__(self):
        super().__init__()

    @staticmethod
    def number_parameters():
        return 0

    @staticmethod
    def parameter_means():
        raise Exception("Deterministic problem has no parameters")

    @staticmethod
    def parameter_covariance():
        raise Exception("Deterministic problem has no parameters")


class Simple(DeterministicBenchmark):

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x):
        return 1/x[0] + x[1]

    @staticmethod
    def bounds():
        return [(0.1, 2), (0.1, 2)]

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([Simple.f1(d_vars), Simple.f2(d_vars)])

    @staticmethod
    def pareto_front():
        x = np.array([np.linspace(Simple.bounds()[0][0], Simple.bounds()[0][1], 500)])
        return np.array([Simple.f1(x), Simple.f2(x)])


class ZDT1(DeterministicBenchmark):
    """represents ZDT problem 1"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ZDT1"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(ZDT1.f1(x) / g))

    @staticmethod
    def number_decision_vars():
        return 30

    @staticmethod
    def bounds():
        return [(0, 1) for _ in range(ZDT1.number_decision_vars())]

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([ZDT1.f1(d_vars), ZDT1.f2(d_vars)])

    @staticmethod
    def pareto_front():
        x = np.array([np.linspace(ZDT1.bounds()[0][0], ZDT1.bounds()[0][1], 500)])
        return np.array([Simple.f1(x), ZDT1.f2(x, front=True)])


class ZDT2(DeterministicBenchmark):
    """represents ZDT problem 2"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ZDT2"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.power(ZDT2.f1(x) / g, 2))

    @staticmethod
    def number_decision_vars():
        return 30

    @staticmethod
    def bounds():
        return [(0, 1) for _ in range(ZDT2.number_decision_vars())]

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([ZDT1.f1(d_vars), ZDT1.f2(d_vars)])

    @staticmethod
    def pareto_front():
        x = np.array([np.linspace(ZDT1.bounds()[0][0], ZDT1.bounds()[0][1], 500)])
        return np.array([Simple.f1(x), ZDT1.f2(x, front=True)])


class ZDT3(DeterministicBenchmark):
    """ZDT problem 3"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 30
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

    def __str__(self):
        return "ZDT3"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(ZDT3.f1(x) / g) - (ZDT3.f1(x) / g) * np.sin(10 * np.pi * ZDT3.f1(x)))

    @staticmethod
    def number_decision_vars():
        return 30


class ZDT4(DeterministicBenchmark):
    """ZDT problem 4"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 10
        self.bounds = [(0, 1) if i == 0 else (-5, 5) for i in range(self.number_of_decision_vars)]

    def __str__(self):
        return "ZDT4"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 10 * (len(x) - 1) + np.sum(np.power(x[1:], 2) - 10 * np.cos(4 * np.pi * x[1:]))
        return g * (1 - np.sqrt(ZDT4.f1(x) / g))

    @staticmethod
    def number_decision_vars():
        return 10


class ZDT6(DeterministicBenchmark):
    """ZDT problem 6"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 10
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

    def __str__(self):
        return "ZDT6"

    @staticmethod
    def f1(x):
        return 1 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.power(np.sum(x[1:]) / (len(x) - 1), 0.25)
        return g * (1 - np.power(ZDT6.f1(x) / g, 2))

    @staticmethod
    def number_decision_vars():
        return 10


class CONSTR(DeterministicBenchmark):
    """CONSTR problem for testing constraint handling"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CONSTR"

    @staticmethod
    def constrained():
        return True

    @staticmethod
    def bounds():
        return [(0.1, 1), (0, 5)]

    @staticmethod
    def constraint1(x):
        return x[1] + 9 * x[0]  # greater than or equal to 6

    @staticmethod
    def constraint2(x):
        return -x[1] + 9 * x[0]  # greater than or equal to 1

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            raise Exception("This problem does not support the front flag")
        return (1 + x[1]) / x[0]

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([CONSTR.f1(d_vars), CONSTR.f2(d_vars)])

    @staticmethod
    def calculate_constraints(d_vars, pars):
        return np.array([CONSTR.constraint1(d_vars), CONSTR.constraint2(d_vars)])

    def get_pareto_front(self):
        x1_1 = np.linspace(7 / 18, 2 / 3)
        x1_2 = np.linspace(2 / 3, 1)
        f1 = np.concatenate((x1_1, x1_2))
        f2 = np.concatenate(((7 - 9 * x1_1) / x1_1, 1 / x1_2))
        return f1, f2

    def feasible_reigon(self):
        pass


class SRN(DeterministicBenchmark):
    """CONSTR problem for testing constraint handling"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SRN"

    @staticmethod
    def constrained():
        return True

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def bounds():
        return [(-20, 20), (-20, 20)]

    @staticmethod
    def constraint1(x):
        return np.power(x[0], 2) + np.power(x[1], 2)  # lest than or equal to 225

    @staticmethod
    def constraint2(x):
        return 3 * x[1] - x[0]  # greater than or equal to 10

    @staticmethod
    def f1(x):
        return np.power(x[0] - 2, 2) + np.power(x[1] - 1, 2) + 2

    @staticmethod
    def f2(x, front=False):
        if front:
            raise Exception("This problem does not support the front flag")
        return 9 * x[0] - np.power(x[1] - 1, 2)

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([SRN.f1(d_vars), SRN.f2(d_vars)])

    @staticmethod
    def calculate_constraints(d_vars, pars):
        return np.array([SRN.constraint1(d_vars), SRN.constraint2(d_vars)])

    def get_pareto_front(self):
        maximiser1 = -2.5
        upper_bound = 1.1
        lowerx1 = np.full(100, maximiser1)
        lowerx2 = np.linspace((maximiser1 + 10) / 3, np.sqrt(225 - maximiser1 ** 2), 100)
        upperx1 = np.linspace(maximiser1, upper_bound, 100)
        upperx2 = (10 + upperx1) / 3
        x1 = np.concatenate((lowerx1, upperx1))
        x2 = np.concatenate((lowerx2, upperx2))
        return SRN.f1(np.array([x1, x2])), SRN.f2(np.array([x1, x2]))

    def feasible_reigon(self):
        """plot randomly generated selection of points to approximate feasible reigon"""
        n_points = 20000
        constr_bounds = (-15, -1 + 3 * np.sqrt(86) / 2)
        x1 = np.zeros(n_points)
        x2 = np.zeros(n_points)
        for i in range(n_points):
            a = constr_bounds[0] + np.random.random() * (constr_bounds[1] - constr_bounds[0])
            if a < -1 - 3 * np.sqrt(86)/2:
                x2_bounds = (-np.sqrt(225 - a ** 2), np.sqrt(225 - a ** 2))
            else:
                x2_bounds = ((10 + a)/3, np.sqrt(225 - a ** 2))
            b = x2_bounds[0] + np.random.random() * (x2_bounds[1] - x2_bounds[0])
            x1[i] = a
            x2[i] = b
        return SRN.f1([x1, x2]), SRN.f2([x1, x2])


class Test1(DeterministicBenchmark):

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x):
        return x[1]

    @staticmethod
    def bounds():
        return [(0, 1), (0, 1)]

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([Test1.f1(d_vars), Test1.f2(d_vars)])


class Test2(DeterministicBenchmark):

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def bounds():
        return [(0, 1)]

    @staticmethod
    def number_decision_variables():
        return 1

    @staticmethod
    def calculate_objectives(d_vars, pars):
        return np.array([Test2.f1(d_vars)])


test_functions = {"stochasticparaboloid": StochasticParaboloid, "stochasticparaboloid2": StochasticParaboloid2,
                  'simple': Simple, 'zdt1': ZDT1, 'zdt2': ZDT2, 'test1': Test1, 'test2': Test2, 'constr': CONSTR,
                  'srn': SRN}


class IOWrapper:

    def __init__(self):
        args = self.parse()
        model = test_functions[args.benchmark_function.lower()]
        d_vars, pars = self.read_input_file(args.input_file, model)
        objectives = model.calculate_objectives(d_vars, pars)
        if model.constrained():
            constraints = model.calculate_constraints(d_vars, pars)
        else:
            constraints = []
        self.write_output_file(objectives=objectives, constraints=constraints, output_file=args.output_file)

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("benchmark_function", default='stochasticparaboloid')
        parser.add_argument("--input_file", dest='input_file', default='input.dat')
        parser.add_argument("--output_file", dest='output_file', default='output.dat')
        args = parser.parse_args()
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        if args.benchmark_function.lower() not in test_functions.keys():
            raise Exception("benchmark_function {} not found in known functions".format(args.benchmark_function))
        if not os.path.exists(args.input_file):
            raise Exception("input file not found")
        return args

    @staticmethod
    def read_input_file(input_file, model):
        df = pd.read_csv(input_file, encoding='ascii', squeeze=True, index_col=0)
        par_template = re.compile('par[0-9]+')
        d_var_template = re.compile('d_var[0-9]+')
        number_d_vars = 0
        number_pars = 0
        for i, index in enumerate(df.index):
            if d_var_template.fullmatch(index):
                number_d_vars += 1
            if par_template.fullmatch(index):
                number_pars += 1
        if number_d_vars != model.number_decision_variables():
            raise Exception("Incorrect number of decision variables for benchmark function")
        if number_pars != model.number_parameters():
            raise Exception("Incorrect number of parameters for benchmark function")
        data = df.values
        d_vars = data[:number_d_vars]
        pars = data[number_d_vars: number_d_vars + number_pars]
        return d_vars, pars

    @staticmethod
    def write_output_file(objectives, output_file, constraints=None):
        num_objectives = len(objectives)
        num_constraints = len(constraints)
        data = np.concatenate((objectives, constraints))
        index = ['objective{}'.format(i + 1) for i in range(num_objectives)]
        index += ['constraint{}'.format(i + 1) for i in range(num_constraints)]
        df = pd.Series(data, index)
        f = open(output_file, 'w')
        f.write('MODEL OUTPUT FILE\n')
        f.close()
        df.to_csv(output_file, encoding='ascii', mode='a')


if __name__ == "__main__":
    IOWrapper()




