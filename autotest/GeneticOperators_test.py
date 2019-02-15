

from pyemu.prototypes.GeneticOperators import *
import numpy as np


def test_crossover_sbx():
    pass  # testing crossover module


def test_mutation_polynomial():
    pass


def test_spea2_enviromental_selection():
    fitness = pd.Series(data=[0.1, 0.99, 1, 3], index=np.arange(0, 4))
    distance_df = pd.DataFrame(data=[[np.NaN, 1, 3, 5], [1, np.NaN, 2, 4], [3, 2, np.NaN, 1], [5, 4, 1, np.NaN]],
                               index=np.arange(0, 4), columns=np.arange(0, 4))
    num_dv_reals = 4
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    results = archive == pd.Series(data=[True, True, True, True])
    assert results.all()
    num_dv_reals = 3
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    results = archive == pd.Series(data=[True, True, True, False])
    assert results.all()
    num_dv_reals = 2
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    results = archive == pd.Series(data=[True, True, False, False])
    assert results.all()
    num_dv_reals = 1
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    results = archive == pd.Series(data=[True, False, False, False])
    assert results.all()
    # test with larger set of fitness and distance for archive truncation
    fitness = pd.Series(data=[0.1, 0.13, 0.5, 0.9], index=np.arange(0, 4))
    num_dv_reals = 4
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    result = archive == pd.Series(data=[True, True, True, True])
    assert result.all()
    num_dv_reals = 3
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    result = archive == pd.Series(data=[True, True, False, True])
    assert result.all()
    num_dv_reals = 2
    archive = Selection.spea2_enviromental_selection(fitness, distance_df, num_dv_reals)
    result = archive == pd.Series(data=[True, False, False, True])
    assert result.all()


def test_tournament_selection():
    index = np.arange(0, 10)
    num_to_select = 5
    # use simple fitness values to check method
    fitness = pd.Series(data=[1, 4, 5, 6, 7, 9, 2, 3, 8, 10], index=index)

    def comparison(idx1, idx2):
        return fitness.loc[idx1] < fitness.loc[idx2]
    selected = Selection.tournament_selection(index, num_to_select, comparison)
    selected = set(selected)
    assert 0 in selected


if __name__ == '__main__':
    test_crossover_sbx()
    test_mutation_polynomial()
    test_spea2_enviromental_selection()
    test_tournament_selection()

