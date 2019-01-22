from pyemu.prototypes.NSGA_II import NSGA_II
import pyemu
import os
import matplotlib.pyplot as plt
import numpy as np
import random


def test():
    np.random.seed(12929)
    random.seed(18291)
    os.chdir(os.path.join('moouu', 'StochasticProblemSuite', 'template'))
    pst = pyemu.Pst('StochasticParaboloid.pst')
    dv_names = pst.par_names[:2]
    evolAlg = NSGA_II(pst, verbose=True, slave_dir='template')
    obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    evolAlg.initialize(obj_func_dict=obj_func_dict, num_par_reals=5, num_dv_reals=5, dv_names=dv_names)
    evolAlg.update()
    evolAlg.update()
    evolAlg.update()
    front = evolAlg.update()
    print(front)
    f1, f2 = np.array([individual.objective_values for individual in front]).T
    f1 *= -1
    f2 *= -1
    plt.plot(f1, f2, 'o')
    plt.show()


def test_simple():
    pst = pyemu.Pst('Simple.pst')
    #data = [[range(1, 2, 0.5)] for i in range()
    dv_ensemble = pyemu.ParameterEnsemble(pst, )

if __name__ == "__main__":
    test()
    test_simple()