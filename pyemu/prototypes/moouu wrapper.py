import os
import numpy as np
import pandas as pd
from .GeneticAlgorithms import *
import pyemu
from .moouu import *

ga = {'nsga2': NSGA_II, 'spea': SPEA, 'spea2': SPEA_2}


class MOOUU:
    """
    Class for multi objective optimisation under uncertainty. initializer accepts a pest object
    """

    def __init__(self, pst, obj_dict,
                 risk=0.5, uncertainty_calculator='fosm', genetic_algorithm='nsga_ii', jco=None, number_realisations=30):
        """
        :param pst:
                pst handler object
        :param risk: float
                level of risk, default 0.5. must be in [0, 1]
        :param uncertainty_calculator: str
                method of calcualting risk shifted values, can be fosm (First Order Second Moment propogation of
                uncertainty) or mc (Monte Carlo) methods
        :param genetic_algorithm:
                genetic algorithm to use for optimisation. choose nsga2 for a low number of objective
                functions, and spea2 for when there are many objectives to optimise
        :param jco:
                Jacobian file name, if not set as pest_file_name.jcb
        """
        objectives = ObjectiveFuncCalculator(pst=pst, uncertainty_calculator=uncertainty_calculator, risk=risk, jco=jco)



class ObjectiveFuncCalculator(ParetoObjFunc):
    """
    objective_function calculator - should be passed as 'objectives' into genetic algorithm
    model should be responsible for running model in parallel with
    """
    def __init__(self, pst, uncertainty_calculator, obj_dict, risk=0.5, number_realisations=30, jco=None,
                 parcov=None, obscov=None, num_slaves=0, submit_file=None, verbose=False, port=4004,
                 slave_dir="template"):
        super().__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                         submit_file=submit_file, verbose=verbose, port=port, slave_dir=slave_dir)
        self.risk = risk
        if uncertainty_calculator.lower() == 'fosm':
            self.sc = pyemu.Schur(jco=jco, pst=self.pst, forcast=obj_dict.keys())
            self.shift_method = self.risk_shifted_fosm
        elif uncertainty_calculator.lower() == 'monte carlo':
            self.shift_method = self.risk_shifted_monte_carlo
            self.parameter_ensemble = pyemu.ParameterEnsemble.from_gaussian_draw(pst=self.pst, cov=self.parcov,
                                                                                 num_reals=number_realisations)
        else:
            self.logger.lraise('No uncertainty propagation method {} available'.format(uncertainty_calculator.lower()))

    def risk_shifted_monte_carlo(self, d_vars):
        """

        :param d_vars:
        :return:
        """
        risk_shifted_obs = -1
        return risk_shifted_obs

    def risk_shifted_fosm(self, d_vars):
        standard_deviation_df = self.sc.get_forecast_summary(include_map=True)['post_stdev']
        # I hope this is how to get model output standard deviation? ^^^ TODO check this
        standard_deviation = np.array(standard_deviation_df)
        risk_shifted_obs = -1
        return risk_shifted_obs

    def calculate_objectives(self, d_vars):
        """

        :param d_vars: array of decision variables to evaluate in parallel
        :return: risk shifted objectives
        """
        self.shift_method(d_vars=d_vars)

    def _calc_obs(self, dv_ensemble):
        """

        :param dv_ensemble: dv to be evaluated
        :return: observation ensemble found by calculating
        """
        # the parameter and decision variable ensembles need to be added together.
        pass