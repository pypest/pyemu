import os
from pyemu.prototypes.Abstract_Moo import *
import pyemu


class AbstractPopIndividualTests:
    """Test the methods and entire AbstractPopIndividual class"""

    @staticmethod
    def test__init__():
        a = AbstractPopIndividual([0.2, 1])
        assert a.fitness == 0
        assert a.is_constrained is False
        assert a.run_model is True
        assert a.objective_values is None
        b = AbstractPopIndividual([0.2, 1], objective_values=np.array([0.2, 10], dtype=float))
        assert b.is_constrained is False
        assert b.run_model is False
        assert np.all(np.isclose(b.objective_values, [0.2, 10]))
        c = AbstractPopIndividual([0.2, 1], is_constrained=True)
        assert c.is_constrained
        assert c.run_model is True
        assert c.violates is None
        d = AbstractPopIndividual([0, 0], is_constrained=True, total_constraint_violation=2)
        assert d.violates is True
        assert d.total_constraint_violation == 2
        assert d.run_model is True
        e = AbstractPopIndividual([0, 0], is_constrained=True, total_constraint_violation=0)
        assert e.violates is False
        assert e.total_constraint_violation == 0
        assert e.run_model is True


    @staticmethod
    def test_update():
        a = AbstractPopIndividual([0.3, 1], is_constrained=True)
        assert a.run_model is True
        a.run_model = False
        a.d_vars = np.array([0.2, 1], dtype=float)
        a.update()
        assert a.run_model is True
        assert a.objective_values is None
        assert a.violates is None
        assert a.total_constraint_violation is None

    @staticmethod
    def test_dominates():
        # test very simple objective function
        def objectives(x):
            return np.array([x[0]])
        population = []
        for i in range(5):
            population.append(AbstractPopIndividual([i], objectives))
        moo = AbstractMOEA(objectives, (0, 5), 2)
        moo.run_model(population)
        a = AbstractPopIndividual([0])
        a.objective_values = np.array([0])
        b = AbstractPopIndividual([1])
        b.objective_values = np.array([1])
        c = AbstractPopIndividual([2])
        c.objective_values = np.array([2])
        d = AbstractPopIndividual([3])
        d.objective_values = np.array([3])
        d = AbstractPopIndividual([4])
        d.objective_values = np.array([4])
        assert population[0].dominates(population[1]) is True
        assert population[0].dominates(population[3]) is True
        assert population[3].dominates(population[0]) is False
        assert population[0].dominates(population[0]) is False

        def objectives(x):
            return np.array([x[0], (1 + x[1]) / x[0]])

        def constraints(x):
            return np.array([x[1] + 9 * x[0] - 6, -x[1] + 9 * x[0] - 1])

        a = AbstractPopIndividual([0.6, 0.7], is_constrained=True)
        b = AbstractPopIndividual([0.8, 2], is_constrained=True)
        c = AbstractPopIndividual([0.2, 1], is_constrained=True)
        d = AbstractPopIndividual([0.1, 3], is_constrained=True)
        population = [a, b, c, d]
        moo = AbstractMOEA(objectives, (0, 5), 2, constraints=constraints)
        moo.run_model(population)
        assert a.dominates(b)
        assert a.dominates(c)
        assert a.dominates(d)
        assert b.dominates(c)
        assert b.dominates(d)
        assert c.dominates(d)
        assert b.dominates(a) is False
        assert c.dominates(a) is False
        assert d.dominates(a) is False
        assert c.dominates(b) is False
        assert d.dominates(b) is False
        assert d.dominates(c) is False

    @staticmethod
    def test_SBX():
        def objectives(x):
            return np.array([x[0]])

        np.random.seed(1002402)
        bounds = [(-np.inf, np.inf)]
        dist_param = 3
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.022211826787321, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.977788173212678, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.0, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 10.0, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.0, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 10.0, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.998068382956265, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.001931617043736, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.746398124977722, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.25360187502228, atol=1e-6)
        np.random.seed(100)
        bounds = [(0, 5)]
        p1 = AbstractPopIndividual([3], objectives)
        p2 = AbstractPopIndividual([4], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 3.068122818593165)
        assert np.isclose(p2.d_vars[0], 3.9312316083035745)

    @staticmethod
    def test_mutate_polynomial():
        def objectives(x):
            return np.array([x[0]])

        np.random.seed(12645678)
        bounds = [(-5, 5)]
        distribution_param = 2
        d_var = 0
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.07299526594720773)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.027816685123562834)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.9019673485855295)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 0.49704076190606683)
        np.random.seed(12645678)
        d_var = 4.9
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.755469373424529)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.844922963455346)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 3.1141046498006517)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.909940815238122)


class AbstractMOEATests:
    """test methods of abstract MOEA class"""

    @staticmethod
    def test_run_model():
        def objective(x):
            return np.array([x[0], x[1]])
        population = []
        for i in range(5):
            individual = AbstractPopIndividual([i, -i], objective)
            population.append(individual)
        moo = AbstractMOEA(objective, (0, 5), 2)
        moo.run_model(population)
        for i in range(5):
            assert np.all(population[i].objective_values == np.array([i, -i]))

    @staticmethod
    def test_run_model_new():
        model = 'test1'
        model_path = os.path.join(os.getcwd(), 'TestProblems', 'StochasticProblemSuite.py')
        pst = ['python', model_path]
        bounds = [(0, 1), (0, 1)]
        population = AbstractPopulation.draw_uniform(False, bounds, 5, AbstractPopIndividual)
        moo = AbstractMOEA(objectives=None, bounds=bounds, number_objectives=2, model=model, pst=pst)
        moo.run_model_IO(population)
        for individual in population:
            assert np.all(np.isclose(individual.d_vars, individual.objective_values))


    @staticmethod
    def test_tournament_selection():
        def objectives(x):
            return np.array(x[0])
        np.random.seed(12645678)
        bounds = [(-10, 10)]
        moo = AbstractMOEA(objectives, bounds, 1)
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


class AbstractPopulationTests:

    @staticmethod
    def test__init__():
        population = [1, 2, 3, 4]
        parent = AbstractPopulation(population, True)
        assert np.all(parent.population == np.array(population))
        assert len(parent) == 4
        assert parent.constrained is True
        population2 = np.array(population)
        parent = AbstractPopulation(population2)
        assert np.all(parent.population == population2)
        assert len(parent) == 4

    @staticmethod
    def test__add__():
        population1 = [1, 2, 3]
        population2 = [2, 6, 10]
        a = AbstractPopulation(population1)
        b = AbstractPopulation(population2)
        c = a + b
        assert np.all(c.population == np.array(population1 + population2))
        assert len(c) == 6
        assert c.constrained is False

    @staticmethod
    def test_draw_uniform():
        population = AbstractPopulation.draw_uniform(True, [(0, 1)], 4, AbstractPopIndividual)
        assert len(population) == 4
        for individual in population:
            assert len([(0, 1)]) == len(individual.d_vars)
            assert np.all(individual.d_vars >= 0) and np.all(individual.d_vars <= 1)
        assert population.constrained is True
        bounds = [(0, 1), (1, 3), (2, 4)]
        population = AbstractPopulation.draw_uniform(True, bounds, 4, AbstractPopIndividual)
        for individual in population:
            assert len(individual.d_vars) == len(bounds)
            for d_var, bound in zip(individual.d_vars, bounds):
                assert min(bound) <= d_var <= max(bound)

    @staticmethod
    def test_tournament_selection():
        def objectives(x):
            return np.array(x[0])
        np.random.seed(12645678)
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
        population = AbstractPopulation([a, b, c, d])
        new_population = population.tournament_selection(4)
        expected = [c, a, a, c]
        for i, individual in enumerate(new_population):
            assert individual.d_vars == expected[i].d_vars
        np.random.seed(12345678)
        population = AbstractPopulation([a, b, c, d, e])
        new_population = population.tournament_selection(5)
        expected = [b, a, a, b, e]
        for i, individual in enumerate(new_population):
            assert individual.d_vars == expected[i].d_vars

    @staticmethod
    def test_to_decision_variable_array():
        a = [AbstractPopIndividual([i]) for i in range(5)]
        b = AbstractPopulation(a)
        decision_variable_array = b.to_decision_variable_array()
        assert np.all(decision_variable_array == np.array([[0], [1], [2], [3], [4]]))
        a = [AbstractPopIndividual([i, -i]) for i in range(5)]
        b = AbstractPopulation(a)
        decision_variable_array = b.to_decision_variable_array()
        assert np.all(decision_variable_array == np.array([[0, 0], [1, -1], [2, -2], [3, -3], [4, -4]]))

    @staticmethod
    def test_write_decision_variables():
        a = [AbstractPopIndividual([i, -i]) for i in range(5)]
        b = AbstractPopulation(a)
        filename_template = 'individual_{}.dvar'
        b.write_decision_variables(filename_template)
        filenames = [filename_template.format(i + 1) for i in range(5)]
        for i, filename in enumerate(filenames):
            df = pd.read_csv(filename, header=0, index_col=0)
            d_vars = np.array(df).T[0]
            assert np.all(d_vars == np.array([i, -i]))

    @staticmethod
    def test_read_objectives():
        objectives = np.array([[1, 3, 2], [2, 0, 10], [82, 32, 2]])
        index = ['objective {}'.format(i) for i in range(1, 4)]
        filename_template = 'individual_{}.objs'
        i = 1
        for objective in objectives:
            df = pd.Series(objective, index=index, name='MODEL OUTPUT FILE')
            filename = filename_template.format(i)
            df.to_csv(filename, header=True)
            i += 1
        a = [AbstractPopIndividual([i, -i]) for i in range(3)]
        b = AbstractPopulation(a)
        b.read_objectives(filename_template)
        for individual, objective in zip(b, objectives):
            assert np.all(individual.objective_values == objective)





if __name__ == "__main__":
    AbstractPopIndividualTests.test__init__()


