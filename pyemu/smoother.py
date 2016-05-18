from __future__ import print_function, division
import os
import copy
import math
import numpy as np
import pandas as pd
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov
from pyemu.pst import Pst


class LM_enRML(object):

    def __init__(self,pst,parcov=None,obscov=None):
        assert isinstance(pst,Pst)
        self.pst = pst
        if parcov is not None:
            assert isinstance(parcov,Cov)
        else:
            parcov = Cov.from_parameter_data(self.pst)
        if obscov is not None:
            assert isinstance(obscov,Cov)
        else:
            obscov = Cov.from_observation_data(pst)

        self.parcov = parcov
        self.obscov = obscov

        self.__initialized = False
        self.num_reals = 0
        self.half_parcov_diag = None
        self.half_obscov_diag = None
        self.delta_par_prior = None


    def initialize(self,num_reals):
        '''
        (re)initialize the process
        '''
        self.num_reals = int(num_reals)
        self.parensemble = ParameterEnsemble(self.pst)
        self.parensemble.draw(cov=self.parcov,num_reals=num_reals)

        self.obsensemble_0 = ObservationEnsemble(self.pst)
        self.obsensemble_0.draw(cov=self.obscov,num_reals=num_reals)
        self.obsensemble = self.obsensemble_0.copy()

        if self.parcov.isdiagonal:
            self.half_parcov_diag = self.parcov.sqrt
        else:
            self.half_parcov_diag = Cov(x=np.diag(self.parcov.x),
                                       names=self.parcov.col_names,
                                       isdiagonal=True)
        if self.obscov.isdiagonal:
            self.half_obscov_diag = self.obscov.sqrt
        else:
            self.half_obscov_diag = Cov(x=np.diag(self.obscov.x),
                                        names=self.obscov.col_names,
                                        isdiagonal=True)

        self.delta_par_prior = self._calc_delta_par()

        self.__initialized = True

    def _calc_delta_par(self):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        mean = np.array(self.parensemble.mean(axis=0))
        delta = self.parensemble.as_matrix()
        for i in range(self.num_reals):
            delta[i,:] -= mean
        delta = self.half_parcov_diag * delta.transpose()
        return delta * (1.0 / np.sqrt(float(self.num_reals - 1.0)))

    def _calc_delta_obs(self):
        '''
        calc the scaled observation ensemble differences from the mean
        '''

        raise NotImplementedError()


    def _calc_obs(self):
        '''
        propagate the ensemble forward...
        '''
        self.parensemble.to_csv("LM_enRML.pars.csv")
        #todo: modifiy sweep to be interactive...


    def update(self):
        if not self.__initialized:
            self.initialize()
        raise NotImplementedError()
