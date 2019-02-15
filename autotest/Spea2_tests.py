

from pyemu.prototypes.SPEA_2 import *
import pyemu
import os
import matplotlib.pyplot as plt
import numpy as np
import random
os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
srn = pyemu.Pst('SRN.pst')
simple = pyemu.Pst('Simple.pst')
srn_objectives = {'obj1': 'min', 'obj2': 'min'}
simple_objectives = srn_objectives
sp = pyemu.Pst('StochasticParaboloid.pst')
logger = pyemu.Logger(True)


def update_test():
    np.random.seed(12929)
    random.seed(18291)
    evolAlg = SPEA_2(simple, verbose=True, num_slaves=4, slave_dir='template')
    evolAlg.initialize(simple_objectives, dv_names=simple.par_names, num_dv_reals=5, when_calculate=0)
    for i in range(0):
        evolAlg.update()
    archive_dv, archive_obs = evolAlg.update()
    f1, f2 = simple_objectives.keys()
    # plt.plot(archive_obs.loc[:, f1], archive_obs.loc[:, f2], 'o')
    # x = np.linspace(0.1, 2)
    # y = 1/x
    # plt.plot(x, y)
    # plt.show()
    #   ---------------- turned off for travis---------------------------


if __name__ == '__main__':
    update_test()


