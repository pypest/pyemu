

from autotest.moouu.StochasticProblemSuite.StochasticProblemSuite import *
import pyemu


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


def create_pest_file(problem, file_base):
    tpl_filenames = [str(file_base) + '.tpl']
    ins_filename = [str(file_base) + '.ins']
    pst_name = str(file_base) + '.pst'
    pst = pyemu.Pst.from_io_files(tpl_files=tpl_filenames, ins_files=ins_filename, in_files=['input.dat'],
                            out_files=['output.dat'], pst_filename=pst_name)


if __name__ == '__main__':
    name = 'zdt1'
    problem = test_functions[name]
    create_template(problem, name, 30)
    create_instruction_file(problem, name)
    create_pest_file(problem, name)


