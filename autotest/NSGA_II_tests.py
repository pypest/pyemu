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
simple_objectives = srn_objectives
logger = pyemu.Logger(True)


def test_set_cdf():
    obj_func = ParetoObjFunc(pst=simple, obj_function_dict=simple_objectives, logger=logger)
    obs_ensemble = pd.DataFrame(data=np.arange(12).reshape((4, 3)), columns=['obj1', 'random_obs', 'obj2'])
    obj_func.set_cdf_df(obs_ensemble, 4)
    assert np.all(np.isclose(obj_func.cdf_dfs['cdf_1'].values,
                             (obs_ensemble.loc[:, simple_objectives.keys()] -
                              obs_ensemble.loc[:, simple_objectives.keys()].mean(axis=0)).values))
    assert np.all(np.isclose(obj_func.cdf_loc.values, obs_ensemble.loc[:, simple_objectives.keys()].mean(axis=0).values))
    try:
        obj_func.set_cdf_df(obs_ensemble, 3)
    except Exception as e:
        assert str(e) == 'incorrect number of realisations supplied'
    obs_ensemble = pd.DataFrame(data=np.arange(24).reshape((8, 3)), columns=['obj1', 'random_obs', 'obj2'])
    obj_func.set_cdf_df(obs_ensemble, 4)
    assert np.all(np.isclose(obj_func.cdf_dfs['cdf_1'].values,
                             (obs_ensemble.loc[:3, simple_objectives.keys()] -
                              obs_ensemble.loc[:3, simple_objectives.keys()].mean(axis=0)).values))
    assert np.all(np.isclose(obj_func.cdf_loc.loc['cdf_1', :].values,
                             obs_ensemble.loc[:3, simple_objectives.keys()].mean(axis=0).values))


def test_partial_recalculation_risk_shift():
    obj_func = ParetoObjFunc(pst=simple, obj_function_dict=simple_objectives, logger=logger)
    obs_ensemble = pd.DataFrame(data=np.arange(24).reshape((8, 3)), columns=['obj1', 'random_obs', 'obj2'])
    obj_func.set_cdf_df(obs_ensemble, 4)
    obs = pd.DataFrame(data=[[0, 12, 0], [1, 11, 1], [2, 10, 3]], columns=['obj1', 'random_obs', 'obj2'])
    risk_shifted = obj_func.partial_recalculation_risk_shift(obs, risk=0)
    print(risk_shifted)


def test():
    np.random.seed(12929)
    random.seed(18291)
    pst = pyemu.Pst('zdt1.pst')
    dv_names = pst.par_names[:30]
    evolAlg = NSGA_II(pst, verbose=True, slave_dir='template')
    obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    evolAlg.initialize(obj_func_dict=obj_func_dict, num_dv_reals=20, num_par_reals=30, dv_names=dv_names, risk=0.9,
                       when_calculate=-1)
    for i in range(2):
        evolAlg.update()
    front = evolAlg.update()
    f1, f2 = np.array([individual.objective_values for individual in front]).T
    f1 *= -1
    f2 *= -1
    plt.plot(f1, f2, 'o')
    plt.show()


def test_simple():
    np.random.seed(21212)
    data = np.random.random(size=(5, 2))
    dv_ensemble = pyemu.ParameterEnsemble(pst=simple, data=data)
    evolAlg = NSGA_II(pst=simple, verbose=False, slave_dir='template')
    evolAlg.initialize(obj_func_dict=simple_objectives, dv_ensemble=dv_ensemble, num_dv_reals=5)
    for i in range(10):
        evolAlg.update()
    _, objective_df = evolAlg.update()
    f1, f2 = simple_objectives.keys()
    plt.plot(objective_df.loc[:, f1], objective_df.loc[:, f2], 'o')
    x = np.linspace(0.1, 2)
    y = 1/x
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    #test()
    test_simple()
    #test_set_cdf()
    #test_partial_recalculation_risk_shift()
