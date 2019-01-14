import os, sys
import subprocess as sp
import pyemu
import shutil
from pyemu.utils import os_utils
import pandas as pd
"""

"""


class Restart(object):
    def __init__(self):
        pass



    def update_input_files(self, pars = None, in_files = None, tpl_files = None,
                           output_dir = None, exe_swp = None,  options = None,):
        """
        Update input files according to a template file. The execuatbale sweep is used to imitate
        to make the changes bu generating a pest file with a dummy model run.

        :param pars: a dictionary of keys with parameter name, and values are parameter values
        :param in_files: a list of input files to be changed
        :param tpl_files: a list of template files
        :param output_dir:
        :param options:
        :return:
        """

        if pars is None:
            raise ValueError("No parameters to change ....")
        elif not (isinstance(pars, dict)) :
            raise ValueError("Parameters to be changed must be passed as dictionary")


        if not isinstance(tpl_files, list):
            raise ValueError("Template file names must be passed as a list")

        # check if template file exists
        for tpfile in tpl_files:
            if not (os.path.isfile(tpfile)):
                raise ValueError("{} does not exist...".format(tpfile))

        # temporary folder where the inputfiles will initially written
        temp_dir = "temp_dir_sweep__"

        # create a dummy pst file
        pst = pyemu.Pst.from_par_obs_names(par_names=pars.keys(),  obs_names=['obs1'])

        # Generate a file (e.g. sweep_in.csv file) that sweep.exe  needs to make a run
        if output_dir is None:
            if os.path.isabs(tpl_files[0]): #
                ws = os.path.dirname(tpl_files[0])
            else:
                ws = os.getcwd()
        else:
            ws = os.path.abspath(output_dir)

        # Create the temporary directory ...
        temp_dir = os.path.join(ws, temp_dir)
        os.mkdir(temp_dir)

        # make a copy of tpl files in the temporary folder
        for tpfile in tpl_files:
            src = tpfile
            dst = os.path.join(temp_dir , os.path.basename(tpfile))
            shutil.copy(src, dst)

        # sweep in and out files
        sweep_in_file = os.path.basename(in_files[0]) + "_sweep_in_temp.csv"
        sweep_out_file = os.path.basename(in_files[0]) + "_sweep_out_temp.csv"
        sweep_in_file = os.path.join(temp_dir, sweep_in_file)
        sweep_out_file = os.path.join(temp_dir, sweep_out_file)

        # Generate sweep_in.csv file
        columns = pars.keys()
        fidw  = open(sweep_in_file, 'w')
        lin1 = ",".join(columns)
        lin1 = ","+lin1
        fidw.write(lin1)
        lin2 = "\n0"
        for pnam in columns:
            lin2 = lin2 + "," + str(pars[pnam])
        fidw.write(lin2)
        fidw.close()

        # create a dummy model, just to allow pestpp-swp to make run without failure
        # the model generates an dummy output file with single value
        dummy_outfile = 'outfile__.dat'
        fidw = open(os.path.join(temp_dir, "temp_model__.py"), 'w')
        script = "fidw = open('{}', 'w')".format(dummy_outfile)
        script = script + "\n"
        script = script + "fidw.write(str(1))"
        script = script + "\n"
        script = script + "fidw.close()"
        fidw.write(script)
        fidw.close()
        pst.model_command = sys.executable + " temp_model__.py"

        # wrtite ins file
        dummy_ins = 'outfile__.ins'
        fidw = open(os.path.join(temp_dir, dummy_ins), 'w')
        fidw.write("pif ~\n")
        fidw.write("l1 !obs1!")
        fidw.close()

        # add inputfiles
        pst.input_files = [os.path.basename(file) for file in in_files]
        pst.template_files = [os.path.basename(file) for file in tpl_files]
        pst.output_files = [dummy_outfile]
        pst.instruction_files = [dummy_ins]

        # add sweep files to pst files
        pst.pestpp_options["sweep_parameter_csv_file"] = os.path.basename(sweep_in_file)
        pst.pestpp_options["sweep_output_csv_file"]= os.path.basename(sweep_out_file)
        pst.control_data.noptmax = 0
        pst_file = os.path.join(temp_dir, 'ppest_swp.pst')
        pst.write(new_filename=pst_file)

        # run ppest_swp
        if exe_f is None:
            if shutil.which(exe_f) is None:
                raise ValueError("Cannot find executable pestpp-swp.exe...")
            exe =  "pestpp-swp.exe" # exists in th environment path
        else:
            exe = exe_f

        # run sweep

        with open(os.devnull, 'w') as f:
            args = [exe, 'ppest_swp.pst']
            p = sp.Popen(args, cwd= temp_dir)
        p.wait()

        # move the modefied inputfiles to output directory
        for ifile in in_files:
            src = os.path.join(temp_dir, os.path.basename(ifile))
            dst = os.path.join(ws, os.path.basename(ifile))
            shutil.move(src, dst)
        try:
            shutil.rmtree(temp_dir, onerror=os_utils.remove_readonly)  # , onerror=del_rw)
        except Exception as e:
            raise Exception("unable to remove existing master dir:" + \
                            "{0}\n{1}".format(temp_dir, str(e)))



    def extract_output(self, out_files = None, ins_files = None,
                       output_dir = None, exe_swp = None):
        """
        This function extracts specified output values from a list of output files as guided by a list
        of instruction files

        :param out_files: a list of output files
        :param ins_files: a list of instruction files
        :param output_dir: directory where output will be saved at
        :param exe_swp: executable ppest_swp
        :return:
        """


        if not isinstance(ins_files, list):
            raise ValueError("Template file names must be passed as a list")

        # check if ins file exists
        for insfile in ins_files:
            if not (os.path.isfile(insfile)):
                raise ValueError("{} does not exist...".format(insfile))

        # check if output files exist
        for outfile in out_files:
            if not (os.path.isfile(outfile)):
                raise ValueError("{} does not exist...".format(outfile))

        # temporary folder where the extraction will happen
        temp_dir = "temp_dir_sweep__"

        # get
        obs = None
        for outf in ins_files:
            if obs is None:
                obs = pyemu.pst.pst_utils.parse_ins_file(outf)
            else:
                obs.append(pyemu.pst.pst_utils.parse_ins_file(outf))
        # create a dummy pst file
        pst = pyemu.Pst.from_par_obs_names(par_names=['par1'], obs_names=obs)

        # Generate a file (e.g. sweep_in.csv file) that sweep.exe  needs to make a run
        if output_dir is None:
            if os.path.isabs(out_files[0]):  #
                ws = os.path.dirname(out_files[0])
            else:
                ws = os.getcwd()
        else:
            ws = os.path.dirname(os.path.abspath(output_dir))

        # Create the temporary directory ...
        temp_dir = os.path.join(ws, temp_dir)
        os.mkdir(temp_dir)
        pass

        # make a copy of ins files in the temporary folder
        for insfile in ins_files:
            src = insfile
            dst = os.path.join(temp_dir, os.path.basename(insfile))
            shutil.copy(src, dst)

        # sweep in and out files
        sweep_in_file = os.path.basename(out_files[0]) + "_sweep_in_temp.csv"
        sweep_out_file = os.path.basename(out_files[0]) + "_sweep_out_temp.csv"
        sweep_in_file = os.path.join(temp_dir, sweep_in_file)
        sweep_out_file = os.path.join(temp_dir, sweep_out_file)

        # Generate sweep_in.csv file
        columns = ['par1']
        fidw = open(sweep_in_file, 'w')
        lin1 = ",".join(columns)
        lin1 = "," + lin1
        fidw.write(lin1)
        lin2 = "\n0"
        for pnam in columns:
            lin2 = lin2 + "," + str(1.0)
        fidw.write(lin2)
        fidw.close()

        # create a dummy model, just to allow pestpp-swp to make run without failure
        # the model copy output files from original folder to the temporary folder to convince ppest++ that
        # model output is generated

        # write a txt file with output files
        outf_list = os.path.join(temp_dir, 'output_files_list.txt')
        fid = open(outf_list, 'w')
        for fn in out_files:
            fid.write(fn)
            fid.write("\n")
        fid.close()

        #******  Write script that will act as a model
        script = ["import shutil\n"]
        script.append("import os\n")
        script.append("fid = open(r'{}', 'r')\n".format(outf_list))
        script.append("out_files = fid.readlines()\n")
        script.append("for out_file in out_files:\n")
        script.append("\tsrc = out_file.strip()\n")
        script.append("\tdst = os.path.join(r'{}', os.path.basename(out_file))\n".format(temp_dir))
        script.append("\tshutil.copy(src, dst.strip())")

        #write
        fidw = open(os.path.join(temp_dir, 'temp_model__.py'), 'w')
        for sline in script:
            fidw.write(sline)
        fidw.close()
        pst.model_command = sys.executable + " temp_model__.py"

        # write dummy tpl
        dummy_tpl = 'infile__.tpl'
        fidw = open(os.path.join(temp_dir, dummy_tpl), 'w')
        fidw.write("ptf ~\n")
        fidw.write("~    par1     ~\n")
        fidw.close()

        # add file names
        pst.input_files = ['infile.dat']
        pst.template_files = [dummy_tpl]
        pst.output_files = [os.path.basename(f) for f in out_files]
        pst.instruction_files = [os.path.basename(f) for f in ins_files]

        # add sweep files to pst files
        pst.pestpp_options["sweep_parameter_csv_file"] = os.path.basename(sweep_in_file)
        pst.pestpp_options["sweep_output_csv_file"] = os.path.basename(sweep_out_file)
        pst.control_data.noptmax = 0
        pst_file = os.path.join(temp_dir, 'ppest_swp.pst')
        pst.write(new_filename=pst_file)

        # run ppest_swp
        if exe_f is None:
            if shutil.which(exe_f) is None:
                raise ValueError("Cannot find executable pestpp-swp.exe...")
            exe = "pestpp-swp.exe"  # exists in th environment path
        else:
            exe = exe_f

        # run sweep
        with open(os.devnull, 'w') as f:
            args = [exe, 'ppest_swp.pst']
            p = sp.Popen(args, cwd=temp_dir)
        p.wait()

        # move the resulting sweep out to output directory
        src = sweep_out_file
        dst = os.path.join(ws, os.path.basename(sweep_out_file))
        shutil.move(src, dst)
        try:
            shutil.rmtree(temp_dir, onerror=os_utils.remove_readonly)  # , onerror=del_rw)
        except Exception as e:
            raise Exception("unable to remove existing master dir:" + \
                            "{0}\n{1}".format(temp_dir, str(e)))

        return dst



    def update_using_output(self, inlist = None, outlist = None,  output_dir = None, exe_swp = None ):
        """
        Update model input from model output files, for example update initial states for model restart.
        It is also a general utility to update a file by extracting information from another file.
        Description of : Given a pair of file lists (inlist, outlist), extract information from outfile using an
        instruction file and update infile according to template file.

        :param inlist: a list of file pairs, where first file is infile and and the second is template file
        :param outlist: a list of file pairs, where first file is output file and the seconf is instruction file
        :param output_dir: location of outputs
        :param exe_swp: executable ppest_swp

        :return:

        :Example:
        inlist = [[infile1, infile1.tpl], [infile2, infile2.tpl]]
        outlist = [[outfile1, outfile1.ins], [outfile2, outfile2.ins]]
        """

        drop_list = ['run_id', 'input_run_id', 'failed_flag' , 'phi' , 'meas_phi', 'regul_phi', 'OBGNME']

        for out_pair, in_pair in outlist, inlist:
            if not (isinstance(out_pair, list) and len(out_pair) == 2):
                raise ValueError("outlist must be a list of lists, where the nested list has only two files output "
                                 "file and *.ins file")

            if not (isinstance(in_pair, list) and len(in_pair) == 2):
                raise ValueError("inlist must be a list of lists, where the nested list has only two files input file "
                                 "and *.tpl file")
            outfile = out_pair[0] #
            insfile = out_pair[1]
            infile = in_pair[0]
            tplfile = in_pair[1]

            #
            if output_dir is None:
                if os.path.isabs(outfile):  #
                    ws = os.path.dirname(outfile)
                else:
                    ws = os.getcwd()
            else:
                ws = os.path.dirname(os.path.abspath(output_dir))

            # Output is extracted and saved at sweep_outfile
            sweep_out = self.extract_output(out_files=[outfile], ins_files = [insfile], output_dir = ws, exe_swp = exe_swp)

            #Now, use extracted output to update input file
            df_par = pd.read_csv(sweep_out)
            df_par.drop(drop_list, axis=1, inplace=True)
            self.update_input_files(pars=df_par.to_dict(), in_files=[infile], tpl_files=[tplfile],
                               output_dir=ws, exe_swp=exe_swp)




        pass




##
if __name__ == "__main__":

    if False: # test change parameter
        parnam = ['rch_1', 'rch_2']
        parval = [7.0, 0.825466911]
        exe_f = r"D:\Workspace\projects\mississippi\pyemu\bin\win\pestpp-swp.exe"
        par_dict = {}
        for i, par in enumerate(parnam):
            par_dict[par] = parval[i]

        infile = r"D:\Workspace\trash\template\freyberg.rch"
        tpl_file = r"D:\Workspace\trash\template\freyberg.rch.tpl"
        Restart().update_input_files(pars = par_dict, in_files = [infile], tpl_files = [tpl_file],
                                     output_dir = os.path.abspath(infile), exe_swp = exe_f)
    if False: # test "extract outputs"
        exe_f = r"D:\Workspace\projects\mississippi\pyemu\bin\win\pestpp-swp.exe"
        outfs = [r"D:\Workspace\trash\template\freyberg.truth.hds"]
        ins_files = [r"D:\Workspace\trash\template\freyberg.hds.ins"]
        Restart().extract_output(out_files=outfs, ins_files= ins_files,
                                 output_dir=os.path.abspath(outfs[0]), exe_swp = exe_f)

    # general tests
    exe_f = r"D:\Workspace\projects\mississippi\pyemu\bin\win\pestpp-swp.exe"
    ws = r"D:\Workspace\trash\testRestart"
    if os.path.isdir(ws):
        shutil.rmtree(ws)
    os.mkdir(ws)
    pars = {'par1': 1.0, 'par2': 2.0}
    pyemu.utils.simple_tpl_from_pars(parnames=pars.keys(), tplfilename=os.path.join(ws, 'infile.tpl'))
    Restart().update_input_files(pars=pars, in_files = [os.path.join(ws, 'infile.dat')],
                               tpl_files=[os.path.join(ws, 'infile.tpl')], output_dir=ws, exe_swp=exe_f)
    pyemu.utils.simple_ins_from_obs(obsnames=pars.copy(), insfilename=os.path.join(ws, 'outfile.ins'))

    Restart().update_using_output()

    pass



