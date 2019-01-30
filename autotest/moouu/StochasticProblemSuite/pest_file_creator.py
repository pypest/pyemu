

from autotest.moouu.StochasticProblemSuite.StochasticProblemSuite import *
import pyemu
import os
import sys

head = 'pcf\n'
control_data = "* control data\n" \
               "restart estimation\n" \
               "{npar}  {nobs}  {npargp}  0  {nobsgp}\n" \
               "1  1  single  point  1  0  0\n" \
               "1e-1  -4.0   0.3  0.03  10  999\n" \
               "5.0   5.0   1.0e-3  absparmax(5)=1\n" \
               "0.1\n" \
               "0   .005  4   4  .005   4\n" \
               "1    1    1\n"
singular_values = "* singular value decomposition\n" \
                  "1\n" \
                  "11  1.0000000E-06\n" \
                  "0\n"
par_groups = "* parameter groups\n" \
             "var	absolute	1.0000E-10	0.000	switch	2.000	parabolic\n"
par_data_template = "{name}  none  absolute(5)  {parval1}  {parlbnd}  {parubnd}  var  1.0  0.0  1\n"
obs_groups = "* observation groups\n" \
             "objective\n"
obs_data_template = "{obsnme}  1  10.0  objective\n"
command_line = "* model command line\n" \
               "{exec} StochasticProblemSuite.py {problem} --stochastic {par_interaction}\n"
model_IO = "* model input/output\n" \
           "{template} input.dat\n" \
           "{instruction} output.dat\n"
pestpp = "++# PEST++ optional input\n" \
         "++sweep_parameter_csv_file(sweep_in.csv)\n"


def create_template(problem, file_base, num_pars):
    filename = str(file_base) + '.tpl'
    f = open(filename, mode='w')
    f.write('ptf #\n')
    f.write('MODEL INPUT FILE\n')
    for i in range(problem.number_decision_variables()):
        f.write('d_var{0},#dvar{0}          #\n'.format(i + 1))
    for i in range(num_pars):
        f.write('par{0},#par{0}          #\n'.format(i + 1))
    f.close()


def create_instruction_file(problem, file_base):
    filename = str(file_base) + '.ins'
    f = open(filename, mode='w')
    f.write('pif #\n')
    for i in range(problem.number_objectives()):
        f.write('#objective{0},# !obj{0}!\n'.format(i + 1))
    f.close()


def create_pest_file(problem, file_base, num_pars, parbnd, stochastic_flag):
    pst_name = str(file_base) + '.pst'
    template = str(file_base) + '.tpl'
    instruction = str(file_base) + '.ins'
    f = open(pst_name, mode='w')
    f.write(head)
    npar = num_pars + problem.number_decision_variables()
    cd = control_data.format(npar=npar, nobs=str(problem.number_objectives()), npargp=str(1),
                             nobsgp=str(1))
    f.write(cd)
    f.write(singular_values)
    f.write(par_groups)
    f.write("* parameter data\n")
    for i, bound in zip(range(problem.number_decision_variables()), problem.bounds()):
        parval1 = np.mean(bound)
        lbnd = min(bound)
        ubnd = max(bound)
        f.write(par_data_template.format(name='dvar{}'.format(i + 1), parval1=parval1, parlbnd=lbnd, parubnd=ubnd))
    parval1 = np.mean(parbnd)
    lbnd = min(parbnd)
    ubnd = max(parbnd)
    for i in range(num_pars):
        name = 'par{}'.format(i + 1)
        f.write(par_data_template.format(name=name, parval1=parval1, parlbnd=lbnd, parubnd=ubnd))
    f.write(obs_groups)
    f.write("* observation data\n")
    for i in range(problem.number_objectives()):
        name = 'obj{}'.format(i + 1)
        f.write(obs_data_template.format(obsnme=name))
    exec = sys.executable
    f.write(command_line.format(exec=exec, problem=problem.name(), par_interaction=stochastic_flag))
    f.write(model_IO.format(template=template, instruction=instruction))
    f.write(pestpp)
    f.close()


if __name__ == '__main__':
    name = 'zdt2'
    num_pars = 30
    parameter_bounds = (-1, 1)
    parameter_interaction = 'additive'
    prob = test_functions[name]
    create_template(prob, name, num_pars)
    create_instruction_file(prob, name)
    create_pest_file(prob, name, num_pars=num_pars, parbnd=parameter_bounds, stochastic_flag=parameter_interaction)


