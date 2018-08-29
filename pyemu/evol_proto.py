import os
import numpy as np
import pandas as pd

import pyemu
from pyemu.smoother import EnsembleMethod


def EvolAlg(EnsembleMethod):
    def __init__(self, pst, parcov = None, obscov = None, num_slaves = 0, use_approx_prior = True,
                 submit_file = None, verbose = False, port = 4004, slave_dir = "template"):
        super(EvolAlg, self).__init__(pst=pst, parcov=parcov, obscov=obscov, num_slaves=num_slaves,
                                      submit_file=submit_file, verbose=verbose, port=port,
                                      slave_dir=slave_dir)

    def initialize(self):
        pass

    def update(self):
        pass

    









