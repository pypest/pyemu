
"""
run benchmark (ZDT) scripts with multi objective optimisation under uncertainty
"""
import os
from pest_file_creator import *
from pyemu.prototypes.NSGA_II import *
import pyemu


def run_benchmarks(par_interaction, number_iterations, when_calculate):
    benchmarks = ['zdt1']
    os.chdir(os.path.join('moouu', 'verbose_files'))
    for f in os.listdir():
        for benchmark in benchmarks:
            if f.startswith(benchmark):
                os.remove(f)
    verbose_file_base = os.getcwd()
    os.chdir('..')
    os.chdir('..')
    for name in benchmarks:
        pst = setup_files(name, parameter_bounds=(-1, 1), par_interaction=par_interaction)
        # get objectives
        objectives = {obj: 'min' for obj in pst.obs_names}
        # initialise decision variables
        how = {p: 'uniform' for p in pst.par_names if p.startswith('dvar')}  # dictionary for draw method (only dvs)
        d_vars = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict=how, num_reals=20, partial=True)
        verbose_name = '{}_single_point.rec'.format(name)
        verbose = os.path.join(verbose_file_base, verbose_name)
        os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
        evolAlg = NSGA_II(pst=pst, num_slaves=4, verbose=verbose, slave_dir='template')
        # running with single point calculation
        evolAlg.initialize(obj_func_dict=objectives, num_par_reals=1, dv_ensemble=d_vars, risk=0.9, when_calculate=1)
        for i in range(number_iterations):
            evolAlg.update()
        dv_opt, pareto_front = evolAlg.update()
        os.chdir('..')
        os.chdir('verbose_files')
        dv_opt.to_csv(path_or_buf='{}_dvar_opt_sp.rec'.format(name))
        pareto_front.to_csv(path_or_buf='{}_pareto_front_sp.rec'.format(name))
        os.chdir('..')
        os.chdir('..')


if __name__ == "__main__":
    run_benchmarks('additive', 2, 1)





