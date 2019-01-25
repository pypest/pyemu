
import numpy as np
import pandas as pd
import pyemu


class Crossover:
    """
    class containing crossover operators for generating new individuals
    """

    @staticmethod
    def sbx(dv_ensemble, bounds, crossover_probability, crossover_distribution_parameter):
        to_update = set()
        for index in dv_ensemble.index:
            if np.random.random() <= crossover_probability:
                index_other = np.random.choice(dv_ensemble.index)
                solution1, solution2 = Crossover._sbx_helper(dv_ensemble.loc[index, :].values,
                                                             dv_ensemble.loc[index_other, :].values, bounds,
                                                             crossover_distribution_parameter)
                dv_ensemble.loc[index, :] = solution1
                dv_ensemble.loc[index_other, :] = solution2
                to_update.add(index)
                to_update.add(index_other)
        return to_update

    @staticmethod
    def _sbx_helper(values1, values2, bounds, cross_dist):
        for i in range(len(values1)):
            if np.random.random() <= 0.5:
                p1 = values1[i]
                p2 = values2[i]
                if np.isclose(p1, p2, rtol=0, atol=1e-15):
                    beta_1, beta_2 = Crossover._get_beta(np.NaN, np.NaN, cross_dist, values_are_close=True)
                else:
                    lower_transformation = (p1 + p2 - 2 * bounds[0][i]) / (abs(p2 - p1))
                    upper_transformation = (2 * bounds[1][i] - p1 - p2) / (abs(p2 - p1))
                    beta_1, beta_2 = Crossover._get_beta(lower_transformation, upper_transformation, cross_dist)
                values1[i] = 0.5 * ((p1 + p2) - beta_1 * abs(p2 - p1))
                values2[i] = 0.5 * ((p1 + p2) + beta_2 * abs(p2 - p1))
        return values1, values2

    @staticmethod
    def _get_beta(transformation1, transformation2, distribution_parameter, values_are_close=False):
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


class Mutation:
    """
    class containing mutation operators
    """

    @staticmethod
    def polynomial(dv_ensemble, bounds, mutation_probability, mutation_distribution_parameter):
        to_update = set()
        k = 0
        i = 0
        while i < len(dv_ensemble.index):
            dv = dv_ensemble.loc[dv_ensemble.index[i], dv_ensemble.columns[k]]
            dv = Mutation._polynomial_helper(dv, bounds[:, k], mutation_distribution_parameter)
            dv_ensemble.loc[dv_ensemble.index[i], dv_ensemble.columns[k]] = dv
            to_update.add(dv_ensemble.index[i])
            length = int(np.ceil(- 1 / mutation_probability * np.log(1 - np.random.random())))
            i += (k + length) // len(bounds)
            k = (k + length) % len(bounds)
        return to_update

    @staticmethod
    def _polynomial_helper(dv, bound, mut_dist):
        u = np.random.random()  # TODO bounds change
        if u <= 0.5:
            delta = (2 * u) ** (1 / (mut_dist + 1)) - 1
            dv = dv + delta * (dv - bound[0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + mut_dist))
            dv = dv + delta * (bound[1] - dv)
        return dv

