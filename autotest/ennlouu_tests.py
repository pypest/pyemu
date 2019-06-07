import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def rosenbrock_2par_setup():
    import pyemu
    os.chdir(os.path.join("ennlouu","rosenbrock_2par"))
    in_file = os.path.join("par.dat")
    tpl_file = in_file+".tpl"
    out_file = os.path.join("obs.dat")
    ins_file = out_file+".ins"
    pst = pyemu.helpers.pst_from_io_files(tpl_file,in_file,ins_file,out_file)
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    par.loc[:,"parval1"] = 1.0
    par.loc[:,"parubnd"] = 7.0
    par.loc[:,"parlbnd"] = -3.0
    obs = pst.observation_data
    obs.loc[:,"obsval"] = 0.0
    obs.loc[:,"weight"] = 1.0
    obs.loc[:,"obgnme"] = "obj_fn"
    #pst.pestpp_options["opt_obj_func"] = "obj_fn"
    pst.model_command = ["python rosenbrock_2par.py"]
    pst.control_data.noptmax = 0
    pst.write(os.path.join("rosenbrock_2par.pst"))

    os.chdir(os.path.join("..",".."))

def rosenbrock_2par_initialize():
    import pyemu
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    esqp = pyemu.EnsembleSQP(pst="rosenbrock_2par.pst")
    esqp.initialize(num_reals=5,draw_mult=0.01)
    os.chdir(os.path.join("..", ".."))

def rosenbrock_2par_initialize_diff_args_test():
    import numpy as np
    import pyemu
    import shutil
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    for i,c in enumerate([(None,None),("sweep_in.cp.csv",None),("sweep_in.cp.csv","sweep_out.cp.csv")]):
        esqp = pyemu.EnsembleSQP(pst="rosenbrock_2par.pst")
        esqp.initialize(num_reals=1,parensemble=c[0],restart_obsensemble=c[1],draw_mult=0.05)
        if i == 0:
            shutil.copy("sweep_in.csv", "sweep_in.cp.csv")  # default pestpp filename
            shutil.copy("sweep_out.csv", "sweep_out.cp.csv")  # default pestpp filename
        shutil.copy("sweep_out.csv","sweep_out.{}.csv".format(i))
        oe = pyemu.ObservationEnsemble.from_csv("sweep_out.{}.csv".format(i))
        if i > 0:
            oe_last = pyemu.ObservationEnsemble.from_csv("sweep_out.{}.csv".format(i-1))
            print(oe.OBS, oe_last.OBS)
            if np.isclose(oe.OBS, oe_last.OBS):
                pass
            else:
                raise Exception("rosenbrock initialization example gives different answers with different args..")
    os.chdir(os.path.join("..", ".."))

def rosenbrock_2par_single_update():
    import pyemu
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    esqp = pyemu.EnsembleSQP(pst="rosenbrock_2par.pst", )
    esqp.initialize(num_reals=2,draw_mult=0.001)
    esqp.update(step_mult=[0.1,0.01,0.001,0.0001],hess_self_scaling=True)  #run_subset=num_reals/len(step_mult)
    os.chdir(os.path.join("..", ".."))

def rosenbrock_2par_grad_approx_invest():
    import pyemu
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    pst = pyemu.Pst("rosenbrock_2par.pst")
    pst.parameter_data.parval1 *= 2
    esqp = pyemu.EnsembleSQP(pst=pst)
    # for dm in [0.05,0.005,0.0005]
    # for en_size in [30,50]
    esqp.initialize(num_reals=30,draw_mult=0.01)
    en_phi_grad = esqp.update(grad_calc_only=True).T.to_dataframe()
    en_phi_grad_rel = en_phi_grad.par1 / en_phi_grad.par2
    pst.control_data.noptmax = -2
    #pst.parameter_groups.derinc = 0.01
    pst.write(os.path.join("rosenbrock_2par_fds.pst"))
    pyemu.os_utils.run("pestpp rosenbrock_2par_fds.pst")
    jco = pyemu.Jco.from_binary("rosenbrock_2par_fds.jcb").to_dataframe()
    jco_rel = jco.par1 / jco.par2
    compare = en_phi_grad_rel[0] / jco_rel[0]
    if compare >= 1.5 or compare <= 0.5:
        raise Exception("ensemble grad approx is unacceptable.. en grad is {} times that from finite diffs..".
                        format(compare))
    os.chdir(os.path.join("..", ".."))


def rosenbrock_2par_multiple_update(nit=5):
    import pyemu
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    esqp = pyemu.EnsembleSQP(pst="rosenbrock_2par.pst")
    esqp.initialize(num_reals=3,draw_mult=0.1)
    for it in range(nit):
        esqp.update()
    os.chdir(os.path.join("..", ".."))

#def rosenbrock_2par_opt_and_draw_setting_invest():
    # function for identifying appropr default values (for simple problem)

#TODO: copy test dirs and make changes in there...

if __name__ == "__main__":
    #rosenbrock_2par_setup()
    #rosenbrock_2par_initialize()
    #rosenbrock_2par_initialize_diff_args_test()
    #rosenbrock_2par_single_update()
    rosenbrock_2par_grad_approx_invest()
