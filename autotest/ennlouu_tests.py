import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def rosenbrock_setup(version,initial_decvars=1.6,constraints=False):
    import pyemu
    if version == "2par":
        if constraints:
            os.chdir(os.path.join("ennlouu", "rosenbrock_2par_constrained"))
        else:
            os.chdir(os.path.join("ennlouu","rosenbrock_2par"))
    elif version == "high_dim":
        if constraints:
            raise Exception
        else:
            os.chdir(os.path.join("ennlouu","rosenbrock_high_dim"))
    in_file = os.path.join("par.dat")
    tpl_file = in_file+".tpl"
    out_file = [os.path.join("obs.dat")]
    ins_file = [out_file[0]+".ins"]
    if constraints:
        out_file.append(os.path.join("constraint.dat"))
        ins_file.append(out_file[1]+".ins")
    pst = pyemu.helpers.pst_from_io_files(tpl_file,in_file,ins_file,out_file)
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    if version == "2par" and type(initial_decvars) is not float:
        par.loc[par.parnme[0], "parval1"] = initial_decvars[0]
        par.loc[par.parnme[1], "parval1"] = initial_decvars[1]
    else:
        par.loc[:,"parval1"] = initial_decvars
    par.loc[:,"parubnd"] = 2.2
    par.loc[:,"parlbnd"] = -2.2
    # TODO: repeat with log transform
    par.loc[:, "parchglim"] = "relative"
    obs = pst.observation_data
    obs.loc["obs","obsval"] = 0.0
    obs.loc["obs","obgnme"] = "obj_fn"  #pst.pestpp_options["opt_obj_func"] = "obj_fn"
    if constraints:
        obs.loc["constraint", "obgnme"] = "g_constraint"  # inherit from pestpp_options
        obs.loc["constraint", "obsval"] = 10.0  # inherit from pestpp_options
    obs.loc[:, "weight"] = 1.0
    pst.control_data.noptmax = 0
    if version == "2par":
        if constraints:
            pst.model_command = ["python rosenbrock_2par_constrained.py"]
            pst.write(os.path.join("rosenbrock_2par_constrained.pst"))
        else:
            pst.model_command = ["python rosenbrock_2par.py"]
            pst.write(os.path.join("rosenbrock_2par.pst"))
    elif version == "high_dim":
        if constraints:
            raise Exception
        else:
            pst.model_command = ["python rosenbrock_high_dim.py"]
            pst.write(os.path.join("rosenbrock_high_dim.pst"))

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
    pst.parameter_groups.derinc = 0.001
    pst.write(os.path.join("rosenbrock_2par_fds.pst"))
    pyemu.os_utils.run("pestpp-glm rosenbrock_2par_fds.pst")
    jco = pyemu.Jco.from_binary("rosenbrock_2par_fds.jcb").to_dataframe()

    # en approx
    esqp = pyemu.EnsembleSQP(pst=pst)
    df = pd.DataFrame([(en_size, draw_m) for en_size in [5,10,30]
                       for draw_m in [3e-5,3e-4,3e-3]],columns=["en_size", "draw_mult"])
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
    dfs = [df_par1, df_par2]

    # some plots
    fontsize = 12
    fig, axs = plt.subplots(1, 2, sharey=True)
    for i, ax in enumerate(axs):
        for dm in df_par1.draw_mult.unique():
            df_, col = dfs[i], dfs[i].columns[-1]
            ax.plot(df_.loc[df_.draw_mult == dm, "en_size"], df_.loc[df_.draw_mult == dm, col],
                 label="draw mult = {}".format(dm),marker='o')
        ax.axhline(y=1, xmin=0, xmax=100, linestyle='--', color='k')
        if i == 0:
            ax.set_ylabel(
                "$\\frac{(\\frac{\Delta \Phi}{\Delta \\mathtt{dec\ var}})_{en\ approx}}"
                "{(\\frac{\Delta \Phi}{\Delta \\mathtt{dec\ var}})_{finite\ diffs}}$",
                fontsize=fontsize)
        ax.set_title("rosenbrock dec var {0}".format(i+1),fontsize=fontsize)
        ax.set_xlabel("ensemble size $N_e$",fontsize=fontsize)
        ax.legend()
    #plt.show()
    plt.savefig("fd_v_en.pdf")
    os.chdir(os.path.join("..", ".."))


def rosenbrock_multiple_update(version,nit=10,draw_mult=3e-5,en_size=20,finite_diff_grad=False,
                               constraints=False,biobj_weight=1.0,biobj_transf=True,
                               hess_self_scaling=True,hess_update=True,damped=True,
                               cma=False,derinc=0.01,alg="BFGS",memory=5,strong_Wolfe=False,
                               rank_one=False,learning_rate=0.5,
                               mu_prop=0.25,use_dist_mean_for_delta=False,mu_learning_prop=0.5): #filter_thresh=1e-2
    import pyemu
    import numpy as np
    if version == "2par":
        if constraints:
            os.chdir(os.path.join("ennlouu","rosenbrock_2par_constrained"))
        else:
            os.chdir(os.path.join("ennlouu","rosenbrock_2par"))
    elif version == "high_dim":
        if constraints:
            raise Exception
        else:
            os.chdir(os.path.join("ennlouu","rosenbrock_high_dim"))

    if constraints:
        ext = version + "_constrained"
    else:
        ext = version

    [os.remove(x) for x in os.listdir() if x.endswith("obsensemble.0000.csv")]
    [os.remove(x) for x in os.listdir() if x.endswith("parensemble.0000.csv")]
    [os.remove(x) for x in os.listdir() if (x.startswith("rosenbrock_{}.pst.".format(ext))) and (x.endswith(".csv"))]
    [os.remove(x) for x in os.listdir() if (x.startswith("filter.") and "csv" in x)]
    [os.remove(x) for x in os.listdir() if ("_per_" in x) and ("_alpha_" in x)]
    [os.remove(x) for x in os.listdir() if x == "hess_progress.csv"]

    step_mults = list(np.logspace(-5, 0, 12))
    #step_mults = [0.1]  # small step to check updating etc.
    #step_mults = np.linspace(0.1,1.0,10)

    esqp = pyemu.EnsembleSQP(pst="rosenbrock_{}.pst".format(ext))#,num_slaves=10)
    esqp.initialize(num_reals=en_size,draw_mult=draw_mult,constraints=constraints,finite_diff_grad=finite_diff_grad)
    for it in range(nit):
        esqp.update(step_mult=step_mults,constraints=constraints,biobj_weight=biobj_weight,
                    hess_self_scaling=hess_self_scaling,hess_update=hess_update,damped=damped,
                    finite_diff_grad=finite_diff_grad,derinc=derinc,alg=alg,memory=memory,strong_Wolfe=strong_Wolfe,
                    cma=cma,rank_one=rank_one,learning_rate=learning_rate,mu_prop=mu_prop,
                    use_dist_mean_for_delta=use_dist_mean_for_delta,mu_learning_prop=mu_learning_prop)
    os.chdir(os.path.join("..", ".."))

   #  TODO: critical that draw_mult is refined as we go?
   #  TODO: want alpha to increase from it to it; getting nan paren vals when diff starting vals and when step mult
   # is egt 0.01 with H = I -- large alpha/Hess forces all at bounds therefore no cov.
   # feedback something about at bounds so don't waste runs.


def rosenbrock_phi_progress(version,label="phi_progress.pdf",finite_diff_grad=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    if version == "2par":
        os.chdir(os.path.join("ennlouu","rosenbrock_2par"))
    elif version == "high_dim":
        os.chdir(os.path.join("ennlouu","rosenbrock_high_dim"))

    if finite_diff_grad is False:
        obs_ens = ["rosenbrock_{}.pst.obsensemble.0000.csv".format(version)]  #  from initialization sweep
        obs_ens += [x for x in os.listdir() if (x.endswith(".obsensemble.0000.csv"))
                and not (x.startswith("rosenbrock_{}.pst.obs".format(version)))
                and not (x.startswith("rosenbrock_{}.pst_1".format(version)))]
    else:
        obs_ens = [x for x in os.listdir() if x.startswith("rosenbrock_2par.pst.phi.")]

    oes = pd.DataFrame()
    for obs_en in obs_ens:
        if obs_en != "rosenbrock_{}.pst.obsensemble.0000.csv".format(version):
            if finite_diff_grad is False:
                it = int(obs_en.split(".")[2])
            else:
                it = int(obs_en.split(".")[3])
        else:
            it = 0
        if finite_diff_grad is False:
            oe = pd.read_csv(obs_en,index_col=0)
        else:
            oe = pd.read_csv(obs_en, index_col=0, header=None)
        oe.columns = [it]
        oes = pd.concat((oes, oe),axis=1)
    oes = oes.sort_index()
    oes = (np.log10(oes)).replace(-np.inf, 0)

    fig,ax = plt.subplots(1,1)
    for i,v in oes.iterrows():
        ax.plot(v,marker="o",color="grey",linestyle='None')
    ax.set_xlabel("iteration number",fontsize=11)
    ax.set_ylabel("log $\phi$", fontsize=11)
    if finite_diff_grad is False:
        oes_mean = oes.mean()
        oes_mean = oes_mean.sort_index()
        ax.plot(oes_mean, color="k", linestyle='--', label="mean en phi")

    ylim = ax.get_ylim()
    try:
        hess_df = pd.read_csv("hess_progress.csv",index_col=0).T
        alpha_df = pd.read_csv("best_alpha.csv", index_col=0).T
        hess_df.columns, alpha_df.columns = ["hess"], ["alpha"]
        hess_and_alpha = pd.concat((hess_df, alpha_df), axis=1, sort=True)
        for i, v in hess_and_alpha.iterrows():
            #ax.text(x=float(i), y=(ylim[1] + (0.05 * (ylim[1] - ylim[0]))), s="{0};\nalpha: {1}".format(v[0], v[1]),
            #        fontsize=5, rotation=45, color='r', ha='center', va='center')
            ax.text(x=float(i), y=(ylim[1] + (0.05 * (ylim[1] - ylim[0]))), s="{0}".format(v[0]),
                    fontsize=5, rotation=45, color='r', ha='center', va='center')
    except FileNotFoundError:
        print("hess progress file missing")
    #plt.legend()
    #plt.show()
    plt.savefig(label)
    plt.close()
    os.chdir(os.path.join("..", ".."))

def invest(version,constraints=False):
    import shutil

    vars = {"initial_decvars": [[2.0,-1.0]], #[[1.0,-1.0],[-1.0,-1.0],[-2,-2],[-1.5,-0.5],[-2,1],[1.5,-1.5],[1.5,1.5],[0.5,2]], #[(x1, x2) for x1 in np.arange(-2.0,2.1,1.0) for x2 in np.arange(-2.0,2.1,1.0)]
            "draw_mult": [3e-5],
            "en_size": [10],
            "hess_self_scaling": [True],  #TODO: True for BFGS means once (it 2), for LBFGS means every one...
            "hess_update": [True],
            "damped": [False],
            "finite_diff_grad": [True],
            "derinc": [0.001],
            "nit": [30],
            "alg": ["BFGS"],
            "memory": [10],
            "strong_Wolfe": [False]
            }
    #"draw_mult": [3e-2,3e-33e-4]
    #"alpha_base": [0.1, 0.2]

    runs = [{'initial_decvars': a, 'draw_mult': b, 'en_size': c, 'hess_self_scaling': d, 'hess_update': e,
             'damped': f, 'finite_diff_grad': g, 'derinc': h, 'nit': i, 'alg': j, 'memory': k, 'strong_Wolfe': l}
            for a in vars['initial_decvars'] for b in vars['draw_mult'] for c in vars['en_size']
            for d in vars['hess_self_scaling'] for e in vars['hess_update'] for f in vars['damped']
            for g in vars['finite_diff_grad'] for h in vars['derinc'] for i in vars['nit']
            for j in vars['alg'] for k in vars['memory'] for l in vars['strong_Wolfe']]

    fails = []
    for i, v in enumerate(runs):
        rosenbrock_setup(version=version,initial_decvars=v['initial_decvars'],constraints=constraints)
        try:
            rosenbrock_multiple_update(version=version,nit=v['nit'],draw_mult=v['draw_mult'],en_size=v['en_size'],
                                       hess_self_scaling=v['hess_self_scaling'],hess_update=v['hess_update'],
                                       damped=v['damped'],constraints=constraints,
                                       finite_diff_grad=v['finite_diff_grad'],derinc=v['derinc'],
                                       alg=v['alg'],memory=v['memory'],strong_Wolfe=v['strong_Wolfe'])
        except:
            fails.append(v)
            os.chdir(os.path.join("..", ".."))
        # TODO: add constraint support to rosenbrock_phi_progress
        rosenbrock_phi_progress(version=version,finite_diff_grad=v['finite_diff_grad'],
                                label="phi_progress_ne{0}_initdv{1}_dm{2}_sca{3}_upd{4}_d{5}_fd{6}_der{7}_nit{8}_alg{9}_mem{10}_strw{11}.pdf"
                                .format(v['en_size'],v['initial_decvars'],v['draw_mult'],v['hess_self_scaling'],
                                        v['hess_update'],v['damped'],v['finite_diff_grad'], v['derinc'], v['nit'],
                                        v['alg'],v['memory'],v['strong_Wolfe']))

        if version == "2par":
            plot_2par_rosen(label="rosen_surf_ne{0}_idv{1}_dm{2}_scale{3}_update{4}_damp{5}_fd{6}_der{7}_nit{8}_alg{9}_mem{10}_strw{11}.pdf"
                            .format(v['en_size'],v['initial_decvars'],v['draw_mult'],v['hess_self_scaling'],
                                    v['hess_update'],v['damped'],v['finite_diff_grad'], v['derinc'], v['nit'],
                                    v['alg'], v['memory'], v['strong_Wolfe']),
                            constraints=constraints,finite_diff_grad=v['finite_diff_grad'])

        if version == "2par":
            os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
            #try:
             #   [shutil.copy(x,
              #               "phi_curv_it{0}_ne{1}_idv{2}_dm{3}_scale{4}_update{5}_damp{6}_fd{7}_derinc{8}_nit{9}.csv"
               #              .format(int(x.split("it")[1].split(".")[0]),v['en_size'],v['initial_decvars'],
                #                     v['draw_mult'],v['hess_self_scaling'],v['hess_update'],v['damped'],
                 #                    v['finite_diff_grad'], v['derinc'], v['nit']))
                 #for x in os.listdir() if ("_per_" in x) and ("_alpha_" in x)]
            #except FileNotFoundError:
             #   print("missing!")
            try:
                [shutil.copy(x, x.split(".")[0] +
                             "_ne{0}_idv{1}_dm{2}_scale{3}_update{4}_damp{5}_fd{6}_derinc{7}_nit{8}_alg{9}_mem{10}_strw{11}.csv"
                             .format(v['en_size'],v['initial_decvars'],v['draw_mult'],v['hess_self_scaling'],
                                     v['hess_update'],v['damped'],v['finite_diff_grad'], v['derinc'], v['nit'],
                                     v['alg'], v['memory'], v['strong_Wolfe']))
                 for x in os.listdir() if x == "hess_progress.csv"]
            except FileNotFoundError:
                print("missing!")
            os.chdir(os.path.join("..",".."))


def cma_invest(version, constraints=False):

    vars = {"initial_decvars": [[-1.5,0.5]],
            "draw_mult": [3e-3],
            "en_size": [10],
            "learning_rate": [0.5],
            "mu_prop": [0.25],
            "dist_mean": [False],
            "rank_one": [False],
            "mu_learning_prop": [0.5],
            "cma": [False],
            }

    runs = [{'initial_decvars': a, 'draw_mult': b, 'en_size': c,
             'learning_rate': d, 'mu_prop': e, 'dist_mean': f, 'rank_one': g, 'mu_learning_prop': h, 'cma': i}
            for a in vars['initial_decvars'] for b in vars['draw_mult'] for c in vars['en_size']
            for d in vars['learning_rate'] for e in vars['mu_prop'] for f in vars['dist_mean']
            for g in vars['rank_one'] for h in vars['mu_learning_prop'] for i in vars['cma']]
    # remove all runs with cma == False except one
    #runs = [x for x in runs if x['cma'] is True] + \
     #      [x for i, x in enumerate(runs) if x['cma'] is False and i == 1]

    fails = []
    for i,v in enumerate(runs):
        rosenbrock_setup(version=version, initial_decvars=v['initial_decvars'], constraints=constraints)
        try:
            rosenbrock_multiple_update(version=version,nit=20,draw_mult=v['draw_mult'],en_size=v['en_size'],
                                       hess_self_scaling=2,hess_update=True,
                                       damped=True,constraints=constraints,
                                       cma=v['cma'],
                                       learning_rate=v['learning_rate'], mu_prop=v['mu_prop'],
                                       use_dist_mean_for_delta=v['dist_mean'], rank_one=v['rank_one'],
                                       mu_learning_prop=v['mu_learning_prop'])
        except:
            fails.append(v)
            os.chdir(os.path.join("..", ".."))
        # TODO: add constraint support to rosenbrock_phi_progress
        rosenbrock_phi_progress(version=version,
                                label="phi_progress_ne{0}_idv{1}_dm{2}_cma{3}_lr{4}_mu_p{5}_mean{6}_r1{7}.pdf"
                                .format(v['en_size'],v['initial_decvars'],v['draw_mult'],
                                        v['cma'],v['learning_rate'],v['mu_prop'],v['dist_mean'],v['rank_one']))

        if version == "2par":
            plot_2par_rosen(label="rosen_surf_ne{0}_idv{1}_dm{2}_cma{3}_lr{4}_mu_p{5}_mean{6}_r1{7}.pdf"
                            .format(v['en_size'], v['initial_decvars'], v['draw_mult'],
                                    v['cma'], v['learning_rate'], v['mu_prop'], v['dist_mean'], v['rank_one']),
                            constraints=constraints)

            #os.chdir(os.path.join("..",".."))


# TODO: copy test dirs and make changes in there...
# TODO: test for switching between en and finite diffs

def phi_curv_tradeoff_invest():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    d = os.path.join("ennlouu","rosenbrock_2par")
    fig, ax = plt.subplots()
    nit = 13
    for it in range(12,nit):
        df = pd.read_csv(os.path.join(d, "curv_and_phi_per_alpha_it{}.csv".format(it)), index_col=0)
        ax.scatter(df['curv_cond'], np.log10(df['mean_en_phi']), c='purple', alpha=it/nit)
    ax.set_xlabel("$y^{T}s$", fontsize=14)
    ax.set_ylabel(r"log $\phi$", fontsize=14)
    ax.axvline(0, color="r", linestyle="--")
    plt.show()
    plt.savefig(os.path.join(d,"tmp.pdf"))
    plt.close()

def test_pestpp_on_rosen():
    import pyemu
    import pandas as pd
    d = os.path.join("ennlouu", "rosenbrock_2par")
    prefix = "rosenbrock_2par"
    pst = pyemu.Pst(os.path.join(d, prefix + ".pst"))
    # glm
    #[os.remove() for x in os.listdir(os.path.join(d)) if x.startswith("{}".format(prefix + "_glm.par") and ] # clean old
    pst.control_data.noptmax = 30
    pst.write(os.path.join(d, prefix + "_glm.pst"))
    pyemu.os_utils.run("pestpp-glm {}_glm.pst".format(prefix), cwd=d)
    # process results
    #parfile_dict = {}
    #for num_iter in range(1, pst.control_data.noptmax + 1):
        #df = pyemu.pst_utils.read_parfile(os.path.join(d, prefix + "_glm.par{}".format(num_iter)))  # need bpa
        #parfile_dict[num_iter] = df
    #plot_2par_rosen(parfile_dict=parfile_dict)
    df = pd.read_csv(os.path.join(d, prefix + "_glm.ipar"))
    plot_2par_rosen(ipar=df, label="rosen_2par_surf_glm_{0}its_[{1},{2}].pdf"
                    .format(pst.control_data.noptmax, pst.parameter_data.loc[pst.par_names[0],"parval1"],
                            pst.parameter_data.loc[pst.par_names[1],"parval1"]))

def natural_sort_key(s):
    import re
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def filter_plot(problem,constraints,log_phi=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if problem == "2par":
        if constraints:
            os.chdir(os.path.join("ennlouu", "rosenbrock_2par_constrained"))
        else:
            os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
    elif problem == "high_dim":
        if constraints:
            raise Exception
        else:
            os.chdir(os.path.join("ennlouu", "rosenbrock_high_dim"))
    elif problem == "supply2":
        os.chdir(os.path.join("ennlouu", "supply2_deterministic", "temp"))

    filter_per_it = [x for x in os.listdir() if "filter." in x and ".csv" in x]
    filter_per_it.sort(key=natural_sort_key)
    fig,ax = plt.subplots()
    for i,f in enumerate(filter_per_it):
        alpha = (i / len(filter_per_it)) * 1.0
        if alpha == 0:
            alpha += 0.1
        df = pd.read_csv(f)
        if log_phi:
            ax.scatter(df['beta'],np.log10(df['phi']),c='purple',alpha=alpha)
        else:
            ax.scatter(df['beta'], df['phi'], c='purple', alpha=alpha)
    ax.set_xlabel(r"$\beta$",fontsize=14)
    if log_phi:
        ax.set_ylabel("log $\phi$", fontsize=14)
        plt.gcf().subplots_adjust(left=0.15)
    else:
        ax.set_ylabel(r"$\phi$", fontsize=14)

    if log_phi:
        plt.savefig("filter_plot_log10.pdf")
    else:
        plt.savefig("filter_plot.pdf")
    os.chdir(os.path.join("..", ".."))


def supply2_setup():
    import pyemu
    os.chdir(os.path.join("ennlouu", "supply2_deterministic", "temp"))
    # from "template_from_pestpp-opt_benchmarks"
    pst = pyemu.Pst("supply2_pest.base.pst")
    # TODO: opt_direction variable
    # pst.parameter_data.loc[pst.parameter_data.pargp == pst.pestpp_options['opt_dec_var_groups'].split(",")[1], :]
    # we want phi and constraints to appear in obsen; like PESTPP-IES, we will ignore pi eqs
    # phi as an obs - see `convert.py` (from pst.prior_information.loc["pi_obj_func", "equation"])
    # modify frun
    # write convert.py
    # write ins
    # pst.write()
    # pst.prior_information.drop(pst.prior_information.index, inplace=True)


def supply2_update(nit=20,draw_mult=1e-5,en_size=20,biobj_weight=1.0,constraints=True):
    #TODO: populate constraint bool on fly
    import pyemu
    import numpy as np
    os.chdir(os.path.join("ennlouu","supply2_deterministic","temp"))
    [os.remove(x) for x in os.listdir() if (x.endswith("obsensemble.0000.csv"))]
    [os.remove(x) for x in os.listdir() if (x.startswith("filter.") and "csv" in x)]
    prefix = "supply2_pest.base"
    esqp = pyemu.EnsembleSQP(pst="{}.pst".format(prefix),)
    esqp.initialize(num_reals=en_size,draw_mult=draw_mult,constraints=True)
    for it in range(nit):
        esqp.update(step_mult=list(np.logspace(-6,0,7)),constraints=constraints,biobj_weight=biobj_weight,
                    opt_direction="max",biobj_transf=True)
    os.chdir(os.path.join("..", ".."))
    #TODO: s1r21_11 less than constraint modified..


def get_gwm_decvar_vals(filename):
    '''
    Untouched from pestpp benchmark `opt_supply2_chance` dir: `risk_sweep.py`
    '''
    dv_vals = {}
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                raise Exception()
            if "OPTIMAL RATES FOR EACH FLOW VARIABLE" in line:
                [f.readline() for _ in range(4)]
                while "----" not in line:
                    line = f.readline()
                    if "----" in line:
                        break
                    # print(line)
                    raw = line.lower().strip().split()
                    dv_vals[raw[0]] = float(raw[1])
                [f.readline() for _ in range(7)]
                while True:
                    line = f.readline()
                    if "----" in line:
                        break
                    raw = line.lower().strip().split()
                    dv_vals[raw[0]] = float(raw[2])

                return dv_vals

def plot_mean_dev_var_bar(opt_par_en="supply2_pest.parensemble.0000.csv",three_risk_cols=True,include_gwm=True):
    import os
    import numpy as np
    import pyemu
    import pandas as pd
    import matplotlib.pyplot as plt

    os.chdir(os.path.join("ennlouu","supply2_deterministic"))

    # parse the gwm output file for dec var values
    if include_gwm:
        gwm_dvs = get_gwm_decvar_vals(os.path.join("baseline_opt","supply2.gwmout"))

    fig = plt.figure(figsize=(190.0 / 25.4, 70.0 / 25.4))
    if three_risk_cols:
        fig = plt.figure(figsize=(190.0 / 25.4, 70.0 / 25.4))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
    else:
        fig = plt.figure(figsize=(190.0 / 25.4 / 3, 70.0 / 25.4))
        ax1 = plt.subplot(111)

    #rdir = os.path.join(wdir_base, results_dir)
    wdir = os.path.join("temp")
    pst = pyemu.Pst(os.path.join(wdir, "supply2_pest.base.pst"))
    dvs = pst.parameter_data.loc[pst.parameter_data.pargp.apply(lambda x: x in ["pumping", "external"]), "parnme"]

    #par_files = [f for f in os.listdir(rdir) if f.endswith(".par")]
    #print(par_files)
    #dfs = [pd.read_csv(os.path.join(rdir, f), skiprows=1, header=None, names=["parnme", "parval1"],
    #                   usecols=[0, 1], delim_whitespace=True, index_col=0).loc[dvs] for f in par_files]
    #print(dfs[0])

    df = pd.read_csv(os.path.join("temp",opt_par_en),index_col=0)
    df = df.loc[:,dvs]
    opt_mean_dec_var_df = df.mean()

    if three_risk_cols:
        risk_vals = np.array([float(f.split('_')[1]) for f in par_files])

        # obj_func = np.array([df.loc["pi_obj_func","modelled"] for df in dfs])
        #infeas = np.array([check_infeas(os.path.join(rdir, f.replace(".par", ".rec"))) for f in par_files])
        #print(infeas)
        # obj_func[infeas==True] = np.NaN
        # ax.plot(risk_vals,obj_func,'b',lw=0.5,marker=".")

        nu_idx = np.argwhere(risk_vals == 0.5)
        in_idx = np.argwhere(infeas == 1)[0] - 1

        labels = ["A) risk = {0}".format(risk_vals[0]), "B) risk = 0.5", "C) risk = {0}".format(risk_vals[in_idx][0])]

        for ax, df, l in zip([ax1, ax2, ax3], [dfs[0], dfs[nu_idx], dfs[in_idx]], labels):
            if "0.5" in l:
                print(gwm_dvs)
                df.loc[:, "gwm"] = df.index.map(lambda x: gwm_dvs[x])
                df.plot(kind="bar", ax=ax, legend=False, alpha=0.5)
            else:
                df.plot(kind="bar", ax=ax, legend=False, alpha=0.5)
            ax.set_xlabel("decision variable")
            ax.text(0.01, 1.01, l, transform=ax.transAxes)
    else:
        opt_mean_dec_var_df.plot(kind="bar",ax=ax1,color="blue",legend=False,alpha=0.5)
        ax1.set_xlabel("decision variable")
        ax1.text(0.01, 1.01, "(a)", transform=ax1.transAxes)

    ax1.set_ylabel("pumping rate ($\\frac{m^3}{d}$)")
    ax1.set_ylim(0, 55000)
    ax1.grid()
    if three_risk_cols:
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        for ax in [ax1, ax2, ax3]:
            ax.set_ylim(0, 55000)
            ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig("dec_vars.pdf")

def plot_2par_rosen(label="rosen_2par_surf.pdf",constraints=False,finite_diff_grad=False,
                    plot_init_decvars=False,ipar=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyemu

    if constraints:
        os.chdir(os.path.join("ennlouu", "rosenbrock_2par_constrained"))
        pst = pyemu.Pst("rosenbrock_2par_constrained.pst")
    else:
        os.chdir(os.path.join("ennlouu", "rosenbrock_2par"))
        pst = pyemu.Pst("rosenbrock_2par.pst")

    f = lambda x1, x2: 100.0 * (x2 - x1 ** 2.0) ** 2.0 + (1 - x1) ** 2.0  # from rosenbrock_2par.py

    n = 200  # dx and dy

    X, Y = np.meshgrid(np.linspace(pst.parameter_data.parlbnd[0], pst.parameter_data.parubnd[0], n),
                       np.linspace(pst.parameter_data.parlbnd[1], pst.parameter_data.parubnd[1], n))
    Z = f(X, Y)

    plt.contour(X, Y, Z, np.logspace(-0.5, 3.5, 20, base=10), cmap='gray')

    if constraints:
        #theta = np.linspace(0, 2 * np.pi)
        #r = 1.0
        #xx = r * np.cos(theta)
        #yy = r * np.sin(theta)
        # TODO: read from *.py
        xx = np.linspace(-2.2, 2.2, 100)
        yy = -6 * xx + 10
        plt.plot(xx, yy, 'r-')
        plt.ylim(-2.2, 2.2)

    #plt.scatter(x=[1.0],y=[1.0],marker="*",s=80)  # optimum

    if finite_diff_grad is False:
        par_ens = [x for x in os.listdir() if x.endswith(".parensemble.0000.csv")]
    else:
        par_ens = [x for x in os.listdir() if (x.startswith(pst.filename + "."))
                   and (x.endswith(".csv")) and ("phi" not in x) and ("obs" not in x)]

    if ipar is not None:  # testing PESTPP-GLM
        #for k,df in parfile_dict.items():
         #   plt.scatter(x=df.loc[pst.par_names[0],"parval1"], y=df.loc[pst.par_names[0],"parval1"], c="b", alpha=1.0)
        plt.scatter(x=ipar.loc[:, pst.par_names[0]], y=ipar.loc[:, pst.par_names[1]], c="b", alpha=1.0)
        for i,k in ipar.iterrows():
            plt.text(x=k[pst.par_names[0]],y=k[pst.par_names[1]],s=int(k["iteration"]))
        # TODO: Note that GLM requires sum of squares objective function - therefore need to re-contour by squares above if ipar is True
    else:
        for pe in par_ens:
            if pe.split(".")[2] == "parensemble" or "sweep" in pe.split(".")[1]:
                it = 0
            else:
                it = int(pe.split(".")[2])
            df = pd.read_csv(pe,index_col=0)
            #if len(par_ens) > 30:
             #   al = (np.log10(it + 1) / np.log10(len(par_ens) + 1))
            #else:
             #   al = (it + 1) / (len(par_ens) + 1)
            plt.scatter(x=df[pst.parameter_data.parnme[0]], y=df[pst.parameter_data.parnme[1]], c="b", alpha=1.0)
            #plt.text(x=df[pst.parameter_data.parnme[0]],y=df[pst.parameter_data.parnme[1]],s="{}".format(it))

    if finite_diff_grad or ipar is not None:
        plot_init_decvars = True
    if plot_init_decvars:
        #if len(par_ens) > 30:
         #   al = (np.log10(1 + 1) / np.log10(len(par_ens) + 1)) / 2
        #else:
         #   al = ((1 + 1) / (len(par_ens) + 1)) / 2
        #plt.scatter(x=[pst.parameter_data.parval1[0]],y=pst.parameter_data.parval1[1],
         #           marker="s",c="b",alpha=al)  # plot initial when using fds
        plt.scatter(x=[pst.parameter_data.parval1[0]],y=pst.parameter_data.parval1[1],
                    marker="x",c="b",alpha=0.5)  # plot initial when using fds

    plt.savefig(label)
    #plt.show()
    plt.close()
    os.chdir(os.path.join("..",".."))




if __name__ == "__main__":
    #rosenbrock_setup(version="2par",initial_decvars=[2.0,-2.0])
    #rosenbrock_2par_initialize()
    #rosenbrock_2par_initialize_diff_args_test()
    #rosenbrock_2par_single_update()
    #rosenbrock_multiple_update(version="2par",nit=10,en_size=10,draw_mult=3e-3,finite_diff_grad=True,
     #                         hess_update=True,hess_self_scaling=False)
    #rosenbrock_phi_progress(version="2par",finite_diff_grad=True)
    #rosenbrock_2par_grad_approx_invest()

    #rosenbrock_multiple_update(version="2par", nit=2, finite_diff_grad=True,)#alg="LBFGS")

    #plot_2par_rosen(finite_diff_grad=True)

    #rosenbrock_setup(version="high_dim")
    #rosenbrock_multiple_update(version="high_dim")
    #rosenbrock_phi_progress(version="high_dim")

    #invest(version="2par")
    #invest(version="high_dim")

    #rosenbrock_setup(version="2par", constraints=True)
    #invest(version="2par", constraints=True)

    #phi_curv_tradeoff_invest()

    #for idv in [[1.0,-1.0],[-1.0,-1.0],[-2,-2],[-1.5,-0.5],[-2,1],[1.5,-1.5],[1.5,1.5],[0.5,2]]:
     #   rosenbrock_setup(version="2par", initial_decvars=idv)
      #  test_pestpp_on_rosen()
    #test_pestpp_on_rosen()

    #rosenbrock_setup(version="2par",constraints=True,initial_decvars=[2.0,-2.0])
    rosenbrock_multiple_update(version="2par",constraints=True,finite_diff_grad=True,nit=30) #biobj_weight=5.0,alg="LBFGS",damped=False)
    #filter_plot(problem="2par", constraints=True, log_phi=True)
    plot_2par_rosen(finite_diff_grad=True,constraints=True)

    #supply2_setup()
    #supply2_update(en_size=20,draw_mult=1e-6)
    #filter_plot(problem="supply2", constraints=True, log_phi=True)
    #plot_mean_dev_var_bar(opt_par_en="supply2_pest.base.pst.5.2.0490312236469134e-07.parensemble.0000.csv",three_risk_cols=False,include_gwm=False)

    #rosenbrock_multiple_update(version="2par",cma=True,nit=10)
    #rosenbrock_phi_progress(version="2par",label="phi_progress_cma.pdf")

    #cma_invest(version="2par")