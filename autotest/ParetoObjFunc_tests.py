
import pyemu
import os
from pyemu.prototypes.moouu import *
import itertools as it


# setup
os.chdir('moouu')
xsec = pyemu.Pst(os.path.join('10par_xsec', '10par_xsec.pst'))
srn = pyemu.Pst(os.path.join('StochasticProblemSuite', 'SRN.pst'))
xsec_objectives = {'h01_04': 'max', 'h01_06': 'min'}
srn_objectives = {'obj1': 'min', 'obj2': 'min'}
logger = pyemu.Logger(True)


def test_init():
    obj_func = ParetoObjFunc(xsec, obj_function_dict=xsec_objectives, logger=logger)
    assert set(obj_func.obs_dict.keys()) == set(xsec_objectives.keys())


def test_is_feasible():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    data=np.array([[0, 0, 224, 11], [0, 0, 226, 11], [0, 0, 224, 9], [0, 0, 224.99, 10.0001], [10, 20, 224.99, 10.0001]])
    objective_df = pyemu.ParameterEnsemble(pst=srn, data=data, columns=srn.observation_data.index)
    feasible = obj_func.is_feasible(objective_df, risk=0.5)
    assert np.all(feasible.values == [True, False, False, True, True])


def test_constraint_violation_vector():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    data = np.array([[0, 0, 224, 11], [0, 0, 226, 11], [0, 0, 224, 9], [0, 0, 224.99, 10.0001], [10, 20, 224.99, 10.0001]])
    objective_df = pyemu.ParameterEnsemble(pst=srn, data=data, columns=srn.observation_data.index)
    vector = obj_func.constraint_violation_vector(objective_df, risk=0.5)
    truth_data = np.array([0, 1, 1, 0, 0])
    assert np.all(np.isclose(truth_data, vector.values))


def test_objective_vector():
    obj_func = ParetoObjFunc(xsec, obj_function_dict=xsec_objectives, logger=logger)
    data = np.random.random(size=(10, len(xsec.obs_names))) * 100
    objective_df = pyemu.ParameterEnsemble(pst=xsec, data=data, columns=xsec.obs_names)
    obj_vector = objective_df.loc[:, xsec_objectives.keys()]
    assert np.all(obj_func.objective_vector(objective_df).values == obj_vector.values * [1, -1])


def test_obs_obj_signs():
    obj_func = ParetoObjFunc(xsec, obj_function_dict=xsec_objectives, logger=logger)
    assert np.all(obj_func.obs_obj_signs == [1, -1])


def test_dominates():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    soln_df = pd.DataFrame([[2, 1], [1, 2], [3, 1], [2, 2]], columns=srn_objectives.keys())
    assert obj_func.dominates(soln_df.loc[0, :], soln_df.loc[1, :]) is False
    assert obj_func.dominates(soln_df.loc[0, :], soln_df.loc[2, :])
    assert obj_func.dominates(soln_df.loc[0, :], soln_df.loc[3, :])
    assert obj_func.dominates(soln_df.loc[1, :], soln_df.loc[0, :]) is False
    assert obj_func.dominates(soln_df.loc[3, :], soln_df.loc[1, :]) is False


def test_is_nondominated_kung():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    soln_df = pd.DataFrame([[2, 1], [1, 2], [3, 1], [2, 2]], columns=srn_objectives.keys())
    is_nondominated = obj_func.is_nondominated_kung(soln_df)
    assert np.all(is_nondominated.values == [True, True, False, False])


def test_crowding_distance():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    d_vars = np.array([[i for i in range(1, 5)], [1 for i in range(1, 5)]])
    objectives = np.array([d_vars[0], 1 / d_vars[0] + d_vars[1]])
    obj_df = pyemu.ParameterEnsemble(pst=srn, data=objectives.T, columns=srn_objectives.keys())
    cd = obj_func.crowd_distance(obj_df)
    assert cd.loc[0] == np.inf
    assert np.isclose(cd.loc[1], 14/9)
    assert np.isclose(cd.loc[2], 1)
    assert cd.loc[3] == np.inf


if __name__ == '__main__':
    test_init()
    test_is_feasible()
    test_constraint_violation_vector()
    test_objective_vector()
    test_dominates()
    test_is_nondominated_kung()
    test_crowding_distance()
