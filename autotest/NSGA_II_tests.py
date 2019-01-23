from pyemu.prototypes.NSGA_II import NSGA_II
from pyemu.prototypes.moouu import ParetoObjFunc
import pyemu
import os
import matplotlib.pyplot as plt
import numpy as np
import random
os.chdir(os.path.join('moouu', 'StochasticProblemSuite', 'template'))





def test():
    np.random.seed(12929)
    random.seed(18291)
    pst = pyemu.Pst('StochasticParaboloid.pst')
    dv_names = pst.par_names[:2]
    print(dv_names)
    evolAlg = NSGA_II(pst, verbose=True, slave_dir='template')
    obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    evolAlg.initialize(obj_func_dict=obj_func_dict, num_dv_reals=30, dv_names=dv_names, risk=0.5)
    for i in range(10):
        evolAlg.update()
    front = evolAlg.update()
    f1, f2 = np.array([individual.objective_values for individual in front]).T
    f1 *= -1
    f2 *= -1
    plt.plot(f1, f2, 'o')
    plt.show()


def test_simple():
    pst = pyemu.Pst('Simple.pst')
    evolAlg = NSGA_II(pst=pst, verbose=True, slave_dir='template')
    obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    evolAlg.initialize(obj_func_dict=obj_func_dict, dv_names=pst.par_names, num_dv_reals=10)
    for i in range(10):
        evolAlg.update()
    front = evolAlg.update()
    f1, f2 = np.array([individual.objective_values for individual in front]).T
    f1 *= -1
    f2 *= -1
    plt.plot(f1, f2, 'o')
    x = np.linspace(0.1, 2)
    y = 1/x
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    test()
    #test_simple()