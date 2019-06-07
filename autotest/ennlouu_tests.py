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
    par.loc[:,"parval1"] = 2.0
    par.loc[:,"parubnd"] = 8.0
    par.loc[:,"parlbnd"] = -4.0
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
    import pandas as pd
    import matplotlib.pyplot as plt
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    pst = pyemu.Pst("rosenbrock_2par.pst")

    # finite diffs
    pst.control_data.noptmax = -2
    # pst.parameter_groups.derinc = 0.05
    pst.write(os.path.join("rosenbrock_2par_fds.pst"))
    pyemu.os_utils.run("pestpp rosenbrock_2par_fds.pst")
    jco = pyemu.Jco.from_binary("rosenbrock_2par_fds.jcb").to_dataframe()
    jco_rel = jco.par1 / jco.par2

    # en approx
    esqp = pyemu.EnsembleSQP(pst=pst)
    df = pd.DataFrame([(en_size, draw_m) for en_size in [10,20,30, 40, 50, 70, 100]
                       for draw_m in [0.0003,0.003,0.03]],columns=["en_size", "draw_m"])
    # middle draw_m stdev approx corresp with derinc

    en_grad_par1, grad_diff_par1, en_grad_par2, grad_diff_par2 = [],[],[],[]
    for i,v in df.iterrows():
        esqp.initialize(num_reals=int(v[0]),draw_mult=v[1])
        en_phi_grad = esqp.update(grad_calc_only=True).T.to_dataframe()

        en_grad_par1.append(en_phi_grad.par1[0])
        en_grad_par2.append(en_phi_grad.par2[0])
        grad_diff_par1.append(en_phi_grad.par1[0] / jco.par1[0])
        grad_diff_par2.append(en_phi_grad.par2[0] / jco.par2[0])
        # TODO: add below exceptions
        #if (en_phi_grad.par1[0] / jco.par1[0] >= 10) or (en_phi_grad.par1[0] / jco.par1[0] <= 0.1):
         #   raise Exception("ensemble grad approx for par1 is unacceptable.. " +
          #                  "en grad is {} times that from finite diffs..".format(en_phi_grad.par1[0] / jco.par1[0]))
        #if (en_phi_grad.par2[0] / jco.par2[0] >= 10) or (en_phi_grad.par2[0] / jco.par2[0] <= 0.1):
         #   raise Exception("ensemble grad approx for par2 is unacceptable.. " +
          #                  "en grad is {} times that from finite diffs..".format(en_phi_grad.par2[0] / jco.par2[0]))

    df_par1 = pd.concat((df, pd.DataFrame(data=grad_diff_par1, columns=["grad_diff_par1"])), axis=1)
    df_par2 = pd.concat((df, pd.DataFrame(data=grad_diff_par2, columns=["grad_diff_par2"])), axis=1)
    dfs = [df_par1,df_par2]

    # some plots
    fontsize = 12
    fig,axs = plt.subplots(1,2,sharey=True)
    for i,ax in enumerate(axs):
        for dm in df_par1.draw_m.unique():
            df_, col = dfs[i], dfs[i].columns[-1]
            ax.plot(df_.loc[df_.draw_m == dm, "en_size"], df_.loc[df_.draw_m == dm, col],
                 label="draw mult = {}".format(dm),marker='o')
        ax.axhline(y=1,xmin=0,xmax=100,linestyle='--',color='k')
        if i == 0:
            ax.set_ylabel(
                "$\\frac{(\\frac{\Delta \Phi}{\Delta \\mathtt{dec\ var}})_{en\ approx}}"
                "{(\\frac{\Delta \Phi}{\Delta \\mathtt{dec\ var}})_{finite\ diffs}}$",
                fontsize=fontsize)
        ax.set_title("rosenbrock dec var {0}".format(i+1),fontsize=fontsize)
        ax.set_xlabel("ensemble size $N_e$",fontsize=fontsize)
        ax.legend()
    plt.show()
    os.chdir(os.path.join("..", ".."))


def rosenbrock_2par_multiple_update(nit=2):
    import pyemu
    os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    esqp = pyemu.EnsembleSQP(pst="rosenbrock_2par.pst")
    esqp.initialize(num_reals=3,draw_mult=0.01)
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
