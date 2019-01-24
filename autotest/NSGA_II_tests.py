from pyemu.prototypes.NSGA_II import *
from pyemu.prototypes.moouu import ParetoObjFunc
import pyemu
import os
import matplotlib.pyplot as plt
import numpy as np
import random
os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
srn = pyemu.Pst('SRN.pst')
simple = pyemu.Pst('Simple.pst')
srn_objectives = {'obj1': 'min', 'obj2': 'min'}
simple_objecives = srn_objectives
logger = pyemu.Logger(True)


def test_population_pyemu_add():
    dv_en = pyemu.ParameterEnsemble(pst=simple, data=1, columns=simple.par_names, index=np.arange(0, 5))
    obs_en = pyemu.ObservationEnsemble(pst=simple, data=1, columns=simple.obs_names, index=np.arange(0, 5))
    p = Population_pyemu(dv_en, obs_en, logger)
    dv_en = pyemu.ParameterEnsemble(pst=simple, data=0, columns=simple.par_names, index=np.arange(0, 2))
    obs_en = pyemu.ObservationEnsemble(pst=simple, data=0, columns=simple.par_names, index=np.arange(0, 2))
    q = Population_pyemu(dv_en, obs_en, logger)
    z = p + q


def test_fronts_from_rank():
    obj_func = ParetoObjFunc(pst=srn, obj_function_dict=srn_objectives, logger=logger)
    x = np.arange(1, 5)
    y = np.concatenate((1 / x, 1 / x + 1))
    obj_data = np.stack((np.concatenate((x, x)), y))
    constr_data = np.array([[224, 11], [223, 12], [222, 15], [226, 9], [222, 12], [221, 20], [226, 13], [227, 9]])
    data = np.column_stack((obj_data.T, constr_data))
    obj_df = pyemu.ParameterEnsemble(pst=srn, data=data, columns=srn.obs_names)
    rank = obj_func.nsga2_non_dominated_sort(obj_df, risk=0.5)
    p = Population_pyemu(dv_ensemble=obj_df, obs_ensemble=obj_df, logger=logger)
    fronts = p.fronts_from_rank(rank)
    assert np.all(fronts[0].obs_ensemble.loc[range(3), :].values == obj_df.loc[range(3), :].values)
    assert np.all(fronts[1].obs_ensemble.loc[range(2), :].values == obj_df.loc[range(4, 6), :].values)
    assert np.all(fronts[2].obs_ensemble.loc[0, :].values == obj_df.loc[6, :].values)
    assert np.all(fronts[3].obs_ensemble.loc[0, :].values == obj_df.loc[3, :].values)
    assert np.all(fronts[4].obs_ensemble.loc[0, :].values == obj_df.loc[7, :].values)


def test_tournament_selection():
    np.random.seed(12645678)
    pyemu.ParameterEnsemble()
    a = AbstractPopIndividual([1])
    b = AbstractPopIndividual([2])
    c = AbstractPopIndividual([3])
    d = AbstractPopIndividual([4])
    e = AbstractPopIndividual([5])
    a.fitness = 1
    b.fitness = 2
    c.fitness = 3
    d.fitness = 6
    e.fitness = 5
    population = [a, b, c, d]
    new_population = moo.tournament_selection(population, 4)
    expected = [c, a, a, c]
    for i, individual in enumerate(new_population):
        assert individual.d_vars == expected[i].d_vars
    np.random.seed(12345678)
    moo = AbstractMOEA(objectives, bounds, 1)
    population = [a, b, c, d, e]
    new_population = moo.tournament_selection(population, 5)
    expected = [b, a, a, b, e]
    for i, individual in enumerate(new_population):
        assert individual.d_vars == expected[i].d_vars


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
    evolAlg = NSGA_II_pyemu(pst=simple, verbose=True, slave_dir='template')
    evolAlg.initialize(obj_func_dict=simple_objecives, dv_names=simple.par_names, num_dv_reals=2)
    for i in range(1):
        evolAlg.update()
    _, objective_df = evolAlg.update()
    f1, f2 = simple_objecives.keys()
    plt.plot(objective_df.loc[:, f1], objective_df.loc[f2], 'o')
    x = np.linspace(0.1, 2)
    y = 1/x
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    #test()
    test_simple()
    # test_population_pyemu_add()
    # test_fronts_from_rank()