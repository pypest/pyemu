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

    def initialize(self,num_reals):
        self.parensemble = ParameterEnsemble(self.pst)
        self.parensemble.draw(cov=self.parcov,num_reals=num_reals)

        self.obsensemble = ObservationEnsemble(self.pst)
        self.obsensemble.draw(cov=self.obscov,num_reals=num_reals)

        
    def update(self):
        raise NotImplementedError()
