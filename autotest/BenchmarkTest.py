
"""
run benchmark (ZDT) scripts with multi objective optimisation under uncertainty
Note: needs to start in .../pyemu folder in the run settings
"""
import os
from autotest.pest_file_creator import *
from pyemu.prototypes.NSGA_II import *
import pyemu
import shutil


def run_benchmarks(par_interaction, number_iterations, when_calculate, benchmarks=('zdt1', 'zdt3')):
    # get abbreviation based on value of when calculate
    sd = os.getcwd()
    if when_calculate == -1:
        # full reuse
        abbreviation = 'fr'
        interaction_name = 'full_reuse'
    elif when_calculate == 0:
        # no reuse (full ensemble evaluation)
        abbreviation = 'nr'
        interaction_name = 'no_reuse'
    elif when_calculate == 1:
        # partial resue (evaluate at 3 points every generation)
        abbreviation = 'pr'
        interaction_name = 'partial_reuse'
    # names of record files and such
    base = os.path.join(os.getcwd(), 'moouu', 'verbose_files', par_interaction)
    record_base = os.path.join(base, '{name}_{interaction}.rec')
    dvar_base = os.path.join(base, '{name}_gen{generation}_dvar_opt_{abbreviation}.rec')
    pareto_front_base = os.path.join(base, '{name}_gen{generation}_pareto_front_{abbreviation}.rec')
    # create file names for running
    verbose_file_base = os.path.join(os.getcwd(), 'moouu', 'verbose_files', par_interaction)
    for name in benchmarks:
        pst = setup_files(name, parameter_bounds=(-1, 1), par_interaction=par_interaction)
        # get objectives
        objectives = {obj: 'min' for obj in pst.obs_names}
        # initialise decision variables
        how = {p: 'uniform' for p in pst.par_names if p.startswith('dvar')}  # dictionary for draw method (only dvs)
        d_vars = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict=how, num_reals=100, partial=True) # 100
        print(d_vars)
        # get verbose file name
        verbose_name = record_base.format(name=name, interaction=interaction_name)
        verbose = os.path.join(verbose_file_base, verbose_name)
        os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
        evolAlg = NSGA_II(pst=pst, num_slaves=0, verbose=verbose, slave_dir='template') # 16
        # running algorithm
        evolAlg.initialize(obj_func_dict=objectives, num_par_reals=50, dv_ensemble=d_vars, risk=0.9, when_calculate=when_calculate) # 50
#        print(os.getcwd())
#        evolAlg.par_ensemble.to_csv('parameter_ensemble.csv')
        for i in range(number_iterations - 1):
            evolAlg.update()
        dv_opt, pareto_front = evolAlg.update()
        # have to manually close the logger otherwise we get permission errors in the file removal part...
        evolAlg.logger.f.close()
        os.chdir('..')
        os.chdir('verbose_files')
        dv_opt.to_csv(path_or_buf=dvar_base.format(name=name, generation=250, abbreviation=abbreviation))
        pareto_front.to_csv(path_or_buf=pareto_front_base.format(name=name, generation=250, abbreviation=abbreviation))
        os.chdir('..')
        os.chdir('..')
    os.chdir(sd)

def clear_verbose_folder(par_interaction):
    pwd = os.getcwd()
    os.chdir(os.path.join('moouu', 'verbose_files'))
    # clear and remove old directory
    if os.path.exists(par_interaction):
        shutil.rmtree(par_interaction)
    os.mkdir(par_interaction)
    os.chdir(pwd)

#def clear_verbose_folder(par_interaction, benchmarks=('zdt1', 'zdt3')):
#    pwd = os.getcwd()
#    print(pwd)
#    os.chdir(os.path.join('autotest', 'moouu', 'verbose_files'))
#    if not os.path.exists(par_interaction)
#    os.mkdir(par_interaction)
#    os.chdir(par_interaction)
#    interaction_names = ['full_reuse', 'partial_reuse', 'no_reuse']
#    abbreviations = ['fr', 'nr', 'pr']
#    print(os.listdir())
#    for abbreviation, interaction_name in zip(abbreviations, interaction_names):
#        # create file names for running
#        record_base = '{name}_{interaction}.rec'
#        dvar_base = '{name}_gen{generation}_dvar_opt_{abbreviation}'
#        pareto_front = '{name}_gen{generation}_pareto_front_{abbreviation}'
#        file_names = []
#        for benchmark in benchmarks:
#            file_names.append(record_base.format(name=benchmark, interaction=interaction_name))
#            file_names.append(dvar_base.format(name=benchmark, generation=200, abbreviation=abbreviation))
#            file_names.append(dvar_base.format(name=benchmark, generation=250, abbreviation=abbreviation))
#            file_names.append(pareto_front.format(name=benchmark, generation=200, abbreviation=abbreviation))
#            file_names.append(pareto_front.format(name=benchmark, generation=250, abbreviation=abbreviation))
#        print(file_names)
#        for f in os.listdir():
#            if f in file_names:
#                os.remove(f)
#    os.chdir(pwd)
#    print(os.getcwd())

def modified_run(name, par_interaction):
    abbreviation = 'nr'
    base = os.path.join(os.getcwd(), 'moouu', 'verbose_files', par_interaction)
    record_base = os.path.join(base, '{name}_{interaction}.rec')
    dvar_base = os.path.join(base, '{name}_gen{generation}_dvar_opt_{abbreviation}.rec')
    pareto_front_base = os.path.join(base, '{name}_gen{generation}_pareto_front_{abbreviation}.rec')
    cur_dir = os.getcwd()
    os.chdir('..')
    os.chdir('..')
    os.chdir('PycharmProjects')
    os.chdir('GNSMOO')
    pars = pd.read_csv('parameter_ensemble.csv')
    cols = []
    for col in pars.columns:
        if col.startswith('par') or col.startswith('dvar'):
            cols.append(col)
    pars = pars.loc[:, cols]
    dvars = pd.read_csv('zdt3_dvars.csv')
    cols = []
    for col in dvars.columns:
        if col.startswith('dvar'):
            cols.append(col)
    dvars = dvars.loc[:, cols]
    os.chdir(cur_dir)
    pst = setup_files(name, parameter_bounds=(-1, 1), par_interaction=par_interaction)
    dvars = pd.DataFrame(data=dvars.values, columns=pst.par_names[:30])
    objectives = {obj: 'min' for obj in pst.obs_names}
    os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
    evolAlg = NSGA_II(pst=pst)  # 16
    # running algorithm
    evolAlg.initialize(obj_func_dict=objectives, dv_ensemble=dvars, risk=0.9,
                      when_calculate=0, par_ensemble=pars)
    dv_opt, pareto_front = evolAlg.update()
    os.chdir('..')
    os.chdir('verbose_files')
    dv_opt.to_csv(path_or_buf=dvar_base.format(name=name, generation=250, abbreviation=abbreviation))
    pareto_front.to_csv(path_or_buf=pareto_front_base.format(name=name, generation=250, abbreviation=abbreviation))
    os.chdir('..')
    os.chdir('..')



if __name__ == "__main__":
    clear_verbose_folder('nonlinear')
    modified_run('zdt3', 'nonlinear')
    # generations = 0
    # clear_verbose_folder('nonlinear')
    # run_benchmarks('nonlinear', generations, 0) # partial reuse
#    clear_verbose_folder('multiplicative')
#    run_benchmarks('multiplicative', generations, 1) # partial reuse
#    run_benchmarks('multiplicative', generations, -1) # no reuse
#    clear_verbose_folder('nonlinear')
#    run_benchmarks('nonlinear', generations, 1) # partial reuse
#    run_benchmarks('nonlinear', generations, -1) # no reuse







