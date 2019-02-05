
import pyemu
import os
from pyemu.prototypes.moouu import *
import itertools as it


# setup
os.chdir('moouu')
xsec = pyemu.Pst(os.path.join('10par_xsec', '10par_xsec.pst'))
srn = pyemu.Pst(os.path.join('StochasticProblemSuite', 'SRN.pst'))
simple = pyemu.Pst(os.path.join('StochasticProblemSuite', 'Simple.pst'))
xsec_objectives = {'h01_04': 'max', 'h01_06': 'min'}
srn_objectives = {'obj1': 'min', 'obj2': 'min'}
simple_objecives = srn_objectives
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
    # test constrained dominates
    data = np.array([[2, 1, 224, 11], [0, 0, 226, 11], [0, 0, 224, 9], [3, 1, 224.99, 10.0001],
                     [3, 2, 224.99, 10.0001]])
    obj_names = srn_objectives.keys()
    objective_df = pyemu.ObservationEnsemble(pst=srn, data=data, columns=srn.observation_data.index)
    constraint_df = obj_func.constraint_violation_vector(objective_df, risk=0.5)
    assert obj_func.dominates(objective_df.loc[0, obj_names], objective_df.loc[1, obj_names], constraint_df.loc[0],
                              constraint_df.loc[1]) is True
    assert obj_func.dominates(objective_df.loc[0, obj_names], objective_df.loc[2, obj_names], constraint_df.loc[0],
                              constraint_df.loc[2]) is True
    assert obj_func.dominates(objective_df.loc[1, obj_names], objective_df.loc[2, obj_names], constraint_df.loc[1],
                              constraint_df.loc[0]) is False
    assert obj_func.dominates(objective_df.loc[3, obj_names], objective_df.loc[2, obj_names], constraint_df.loc[3],
                              constraint_df.loc[2]) is True
    assert obj_func.dominates(objective_df.loc[2, obj_names], objective_df.loc[3, obj_names], constraint_df.loc[2],
                              constraint_df.loc[3]) is False


def test_is_nondominated_kung():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    soln_df = pd.DataFrame([[2, 1], [1, 2], [3, 1], [2, 2]], columns=srn_objectives.keys())
    is_nondominated = obj_func.is_nondominated_kung(soln_df)
    assert np.all(is_nondominated.values == [True, True, False, False])


def test_crowding_distance():
    obj_func = ParetoObjFunc(srn, obj_function_dict=srn_objectives, logger=logger)
    d_vars = np.array([[i for i in range(1, 5)], [1 for j in range(1, 5)]])
    objectives = np.array([d_vars[0], 1 / d_vars[0] + d_vars[1]])
    obj_df = pyemu.ParameterEnsemble(pst=srn, data=objectives.T, columns=srn_objectives.keys())
    cd = obj_func.crowd_distance(obj_df)
    assert cd.loc[0] == np.inf
    assert np.isclose(cd.loc[1], 14/9)
    assert np.isclose(cd.loc[2], 1)
    assert cd.loc[3] == np.inf


def test_nsga2_non_dominated_sort():
    obj_func_dict = {'obj1': 'min'}
    obj_func = ParetoObjFunc(simple, obj_function_dict=obj_func_dict, logger=logger)
    d_vars = np.array([[i for i in range(5)]])
    objectives = np.array([d_vars[0]])
    obj_df = pyemu.ParameterEnsemble(pst=simple, data=objectives.T, columns=['obj1'])
    rank = obj_func.nsga2_non_dominated_sort(obj_df, risk=0.5)
    for i, idx in enumerate(obj_df.index):
        assert rank[idx] == i + 1
    obj_func = ParetoObjFunc(simple, obj_function_dict=simple_objecives, logger=logger)
    d_vars = []
    for i in range(1, 5):
        for j in range(1, 5):
            d_vars.append([i, j])
    d_vars = np.array(d_vars).T
    objectives = np.array([d_vars[0], 1 / d_vars[0] + d_vars[1]])
    obj_df = pyemu.ParameterEnsemble(pst=simple, data=objectives.T, columns=simple_objecives.keys())
    rank = obj_func.nsga2_non_dominated_sort(obj_df, risk=0.5)
    for idx in rank.index:
        assert rank.loc[idx] == d_vars[1, idx]
    # test sorting with constraints
    obj_func = ParetoObjFunc(pst=srn, obj_function_dict=srn_objectives, logger=logger)
    x = np.arange(1, 5)
    y = np.concatenate((1 / x, 1 / x + 1))
    obj_data = np.stack((np.concatenate((x, x)), y))
    constr_data = np.array([[224, 11], [223, 12], [222, 15], [226, 9], [222, 12], [221, 20], [226, 13], [227, 9]])
    data = np.column_stack((obj_data.T, constr_data))
    obj_df = pyemu.ParameterEnsemble(pst=srn, data=data, columns=srn.obs_names)
    rank = obj_func.nsga2_non_dominated_sort(obj_df, risk=0.5)
    for i in range(3):
        assert rank.loc[i] == 1
    assert rank.loc[3] == 4
    for i in range(4, 6):
        assert rank.loc[i] == 2
    assert rank.loc[6] == 3
    assert rank.loc[7] == 5


def test_spea2_fitness_assignment():
    obj_func = ParetoObjFunc(simple, obj_function_dict=simple_objecives, logger=logger)
    obs_values = []
    for i in range(1, 4):
        for j in range(1, 4):
            obs_values.append([j, 1/j + i])
    obs_df = pyemu.ObservationEnsemble(pst=simple, data=obs_values)
    index = obs_df.index
    fitness, distance = obj_func.spea2_fitness_assignment(obs_df, risk=0.5, pop_size=4.5)
    assert np.isclose(fitness.loc[index[0]], 0.320715)
    assert np.isclose(fitness.loc[index[1]], 0.320715)
    assert np.isclose(fitness.loc[index[2]], 0.282758)
    assert np.isclose(fitness.loc[index[3]], 6.320715)
    assert np.isclose(fitness.loc[index[4]], 10.33181)
    assert np.isclose(fitness.loc[index[5]], 12.33181)
    assert np.isclose(fitness.loc[index[6]], 9.262966)
    assert np.isclose(fitness.loc[index[7]], 15.32071)
    assert np.isclose(fitness.loc[index[8]], 18.30287)




if __name__ == '__main__':
    # test_init()
    # test_is_feasible()
    # test_constraint_violation_vector()
    # test_objective_vector()
    # test_dominates()
    # test_is_nondominated_kung()
    # test_crowding_distance()
    # test_nsga2_non_dominated_sort()
    test_spea2_fitness_assignment()
