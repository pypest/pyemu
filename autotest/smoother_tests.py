import os
if not os.path.exists("temp"):
    os.mkdir("temp")

def henry_setup():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("smoother","henry_pc","pest.pst"))
    par = pst.parameter_data
    par.loc[:,"parlbnd"] = 20.0
    par.loc[:,"parubnd"] = 2000.0
    par.loc["mult1","parlbnd"] = 0.9
    par.loc["mult1","parubnd"] = 1.1

    # obs = pst.observation_data
    # head_groups = obs.groupby(obs.apply(lambda x: x.obgnme=="head" and x.weight>0.0, axis=1)).groups[True]
    # obs.loc[head_groups,"weight"] = 1.0
    # conc_groups = obs.groupby(obs.apply(lambda x: x.obgnme=="conc" and x.weight>0.0, axis=1)).groups[True]
    # obs.loc[conc_groups,"weight"] = 0.5

    pst.pestpp_options["sweep_parameter_csv_file"] = "sweep_in.csv"
    pst.write(pst.filename.replace("pest.pst","henry.pst"))


def henry():
    import os
    import pyemu
    os.chdir(os.path.join("smoother", "henry_pc"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst(os.path.join("henry.pst"))
    es = pyemu.EnsembleSmoother(pst, num_slaves=15,verbose="ies.log")
    es.initialize(210, init_lambda=1.0)
    for i in range(10):
        es.update(lambda_mults=[0.2,5.0],run_subset=45)
    os.chdir(os.path.join("..", ".."))


def henry_plot():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","henry_pc")
    pst = Pst(os.path.join(d,"henry.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    par_dfs = [pd.read_csv(par_file,index_col=0).apply(np.log10) for par_file in par_files]
    par_names = ["mult1"]
    mx = (pst.parameter_data.loc[:,"parubnd"] * 1.1).apply(np.log10)
    mn = (pst.parameter_data.loc[:,"parlbnd"] * 0.9).apply(np.log10)

    obj_df = pd.read_csv(os.path.join(d,"henry.pst.iobj.csv"),index_col=0)
    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    obj_df.loc[:,real_cols] = obj_df.loc[:,real_cols].apply(np.log10)
    obj_df.loc[:,"mean"] = obj_df.loc[:,"mean"].apply(np.log10)
    obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    axt = plt.twinx()
    obj_df.loc[:, real_cols].plot(ax=ax, lw=0.5, color="0.5", alpha=0.5, legend=False)
    ax.plot(obj_df.index, obj_df.loc[:, "mean"], 'b', lw=2.5,marker='.',markersize=5)
    #ax.fill_between(obj_df.index, obj_df.loc[:, "mean"] - (1.96 * obj_df.loc[:, "std"]),
    #                obj_df.loc[:, "mean"] + (1.96 * obj_df.loc[:, "std"]),
    #                facecolor="b", edgecolor="none", alpha=0.25)
    axt.plot(obj_df.index,obj_df.loc[:,"lambda"],"k",dashes=(2,1),lw=2.5)
    ax.set_ylabel("log$_10$ phi")
    axt.set_ylabel("lambda")
    ax.set_title("total runs:{0}".format(obj_df.total_runs.max()))
    plt.savefig(os.path.join(plt_dir,"iobj.pdf"))
    plt.close()

    with PdfPages(os.path.join(plt_dir,"parensemble.pdf")) as pdf:

        for par_file,par_df in zip(par_files,par_dfs):
            print(par_file)
            fig = plt.figure(figsize=(20,10))

            plt.figtext(0.5,0.975,par_file,ha="center")
            axes = [plt.subplot(1,1,i+1) for i in range(len(par_names))]
            for par_name,ax in zip(par_names,axes):
                mean = par_df.loc[:,par_name].mean()
                std = par_df.loc[:,par_name].std()
                par_df.loc[:,par_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.5,grid=False)
                ax.set_yticklabels([])
                ax.set_title("{0}, {1:6.2f}".\
                             format(par_name,10.0**mean))
                ax.set_xlim(mn[par_name],mx[par_name])
                ylim = ax.get_ylim()
                if "mult1" in par_name:
                    val = np.log10(1.0)
                else:
                    val = np.log10(200.0)
                ticks = ["{0:2.1f}".format(x) for x in 10.0**ax.get_xticks()]
                ax.set_xticklabels(ticks,rotation=90)
                ax.plot([val,val],ylim,"k-",lw=2.0)

                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
            pdf.savefig()
            plt.close()




    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    #print(obs_files)
    #mx = max([obs_df.obs.max() for obs_df in obs_dfs])
    #mn = min([obs_df.obs.min() for obs_df in obs_dfs])
    #print(mn,mx)
    obs_names = pst.nnz_obs_names
    obs_names.extend(["pd_one","pd_ten","pd_half"])
    print(len(obs_names))
    #print(obs_files)
    obs_dfs = [obs_df.loc[:,obs_names] for obs_df in obs_dfs]
    mx = {obs_name:max([obs_df.loc[:,obs_name].max() for obs_df in obs_dfs]) for obs_name in obs_names}
    mn = {obs_name:min([obs_df.loc[:,obs_name].min() for obs_df in obs_dfs]) for obs_name in obs_names}

    with PdfPages(os.path.join(plt_dir,"obsensemble.pdf")) as pdf:
        for obs_file,obs_df in zip(obs_files,obs_dfs):
            fig = plt.figure(figsize=(30,20))
            plt.figtext(0.5,0.975,obs_file,ha="center")
            print(obs_file)
            axes = [plt.subplot(8,5,i+1) for i in range(len(obs_names))]
            for ax,obs_name in zip(axes,obs_names):
                mean = obs_df.loc[:,obs_name].mean()
                std = obs_df.loc[:,obs_name].std()
                obs_df.loc[:,obs_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.5,grid=False)
                ax.set_yticklabels([])
                #print(ax.get_xlim(),mn[obs_name],mx[obs_name])
                ax.set_title("{0}, {1:6.2f}:{2:6.2f}".format(obs_name,mean,std))
                ax.set_xlim(mn[obs_name],mx[obs_name])
                #ax.set_xlim(0.0,20.0)
                ylim = ax.get_ylim()
                oval = pst.observation_data.loc[obs_name,"obsval"]
                ax.plot([oval,oval],ylim,"k-",lw=2)
                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
                ax.set_xticklabels([])
            pdf.savefig()
            plt.close()


def freyberg_check_phi_calc():
    import os
    import pandas as pd
    import pyemu
    import shutil
    os.chdir(os.path.join("smoother","freyberg"))
    xy = pd.read_csv("freyberg.xy")
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]

    pst = pyemu.Pst(os.path.join("freyberg.pst"))
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    nothk_names = [pname for pname in pst.adj_par_names if "hk" not in pname]
    parcov_nothk = dia_parcov.get(row_names=nothk_names)
    gs = pyemu.utils.geostats.read_struct_file(os.path.join("template","structure.dat"))
    print(gs.variograms[0].a,gs.variograms[0].contribution)
    #gs.variograms[0].a *= 10.0
    #gs.variograms[0].contribution *= 10.0
    gs.nugget = 0.0
    print(gs.variograms[0].a,gs.variograms[0].contribution)

    full_parcov = gs.covariance_matrix(xy.x,xy.y,xy.name)
    parcov = parcov_nothk.extend(full_parcov)
    #print(parcov.to_pearson().x[-1,:])

    pst.observation_data.loc[:,"weight"] /= 10.0
    #pst.write("temp.pst")
    obscov = pyemu.Cov.from_observation_data(pst)

    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,num_slaves=1,
                                verbose=True)

    es.initialize(num_reals=3)
    print(es.parensemble.loc[:,"hkr00c07"])
    pst.parameter_data.loc[:,"parval1"] = es.parensemble.iloc[0,:]

    pst.observation_data.loc[pst.nnz_obs_names,"obsval"] = es.obsensemble_0.loc[0,pst.nnz_obs_names]
    pst.control_data.noptmax = 0
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    shutil.copytree("template","temp")
    pst.write(os.path.join("temp","temp.pst"))

    os.chdir("temp")
    os.system("pestpp temp.pst")
    os.chdir("..")

    p = pyemu.Pst(os.path.join("temp","temp.pst"))
    print(p.phi)

    os.chdir(os.path.join("..",".."))

def freyberg():
    import os
    import pandas as pd
    import pyemu

    os.chdir(os.path.join("smoother","freyberg"))

    if not os.path.exists("freyberg.xy"):
        import flopy

        ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws="template",
                                        load_only=[])
        xy = pd.DataFrame([(x,y) for x,y in zip(ml.sr.xcentergrid.flatten(),ml.sr.ycentergrid.flatten())],
                          columns=['x','y'])
        names = []
        for i in range(ml.nrow):
            for j in range(ml.ncol ):
                names.append("hkr{0:02d}c{1:02d}".format(i,j))
        xy.loc[:,"name"] = names
        xy.to_csv("freyberg.xy")
    else:
        xy = pd.read_csv("freyberg.xy")
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]

    pst = pyemu.Pst(os.path.join("freyberg.pst"))
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    nothk_names = [pname for pname in pst.adj_par_names if "hk" not in pname]
    parcov_nothk = dia_parcov.get(row_names=nothk_names)
    gs = pyemu.utils.geostats.read_struct_file(os.path.join("template","structure.dat"))
    print(gs.variograms[0].a,gs.variograms[0].contribution)
    #gs.variograms[0].a *= 10.0
    #gs.variograms[0].contribution *= 10.0
    gs.nugget = 0.0
    print(gs.variograms[0].a,gs.variograms[0].contribution)

    full_parcov = gs.covariance_matrix(xy.x,xy.y,xy.name)
    parcov = parcov_nothk.extend(full_parcov)
    #print(parcov.to_pearson().x[-1,:])
    parcov.to_binary("freyberg_prior.jcb")
    parcov.to_ascii("freyberg_prior.cov")
    return
    pst.observation_data.loc[:,"weight"] /= 10.0
    pst.write("temp.pst")
    obscov = pyemu.Cov.from_obsweights(os.path.join("temp.pst"))

    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,num_slaves=20,
                                verbose=True)

    #gs.variograms[0].a=10000
    #gs.variograms[0].contribution=0.01
    #gs.variograms[0].anisotropy = 10.0
    # pp_df = pyemu.utils.gw_utils.pp_file_to_dataframe("points1.dat")
    # parcov_hk = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)
    # parcov_full = parcov_hk.extend(parcov_rch)

    es.initialize(100,init_lambda=100.0,enforce_bounds="reset")
    for i in range(10):
        es.update(lambda_mults=[0.01,0.2,5.0,100.0],run_subset=20)
    os.chdir(os.path.join("..",".."))

def freyberg_condor():
    import os
    import pandas as pd
    import pyemu

    os.chdir(os.path.join("smoother","freyberg"))

    if not os.path.exists("freyberg.xy"):
        import flopy

        ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws="template",
                                        load_only=[])
        xy = pd.DataFrame([(x,y) for x,y in zip(ml.sr.xcentergrid.flatten(),ml.sr.ycentergrid.flatten())],
                          columns=['x','y'])
        names = []
        for i in range(ml.nrow):
            for j in range(ml.ncol ):
                names.append("hkr{0:02d}c{1:02d}".format(i,j))
        xy.loc[:,"name"] = names
        xy.to_csv("freyberg.xy")
    else:
        xy = pd.read_csv("freyberg.xy")
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]

    pst = pyemu.Pst(os.path.join("freyberg.pst"))
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    nothk_names = [pname for pname in pst.adj_par_names if "hk" not in pname]
    parcov_nothk = dia_parcov.get(row_names=nothk_names)
    gs = pyemu.utils.geostats.read_struct_file(os.path.join("template","structure.dat"))
    print(gs.variograms[0].a,gs.variograms[0].contribution)
    #gs.variograms[0].a *= 10.0
    #gs.variograms[0].contribution *= 10.0
    gs.nugget = 0.0
    print(gs.variograms[0].a,gs.variograms[0].contribution)

    full_parcov = gs.covariance_matrix(xy.x,xy.y,xy.name)
    parcov = parcov_nothk.extend(full_parcov)
    #print(parcov.to_pearson().x[-1,:])

    pst.observation_data.loc[:,"weight"] /= 10.0
    pst.write("temp.pst")
    obscov = pyemu.Cov.from_obsweights(os.path.join("temp.pst"))

    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,num_slaves=20,
                                verbose=True,submit_file="freyberg.sub")

    #gs.variograms[0].a=10000
    #gs.variograms[0].contribution=0.01
    #gs.variograms[0].anisotropy = 10.0
    # pp_df = pyemu.utils.gw_utils.pp_file_to_dataframe("points1.dat")
    # parcov_hk = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)
    # parcov_full = parcov_hk.extend(parcov_rch)

    es.initialize(300,init_lambda=10000.0,enforce_bounds="reset")
    for i in range(10):
        es.update(lambda_mults=[0.2,5.0],run_subset=40)
    os.chdir(os.path.join("..",".."))


def freyberg_pars_to_array(par_df):
    import numpy as np
    #print(par_df.index)
    real_col = par_df.columns[0]
    hk_names = par_df.index.map(lambda x:x.startswith("hk"))
    hk_df = par_df.loc[hk_names,:]
    hk_df.loc[:,"row"] = hk_df.index.map(lambda x: int(x[3:5]))
    hk_df.loc[:,"column"] = hk_df.index.map(lambda x: int(x[-2:]))
    nrow,ncol = hk_df.row.max() + 1, hk_df.column.max() + 1
    arr = np.zeros((nrow,ncol)) - 999.0
    for r,c,v in zip(hk_df.row,hk_df.column,hk_df.loc[:,real_col]):
        arr[r-1,c-1] = v
    arr = np.ma.masked_where(arr==-999.,arr)
    return arr

def freyberg_plot_par_seq():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","freyberg")
    pst = Pst(os.path.join(d,"freyberg.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    par_dfs = [pd.read_csv(par_file,index_col=0).apply(np.log10) for par_file in par_files]
    #par_names = list(par_dfs[0].columns)
    par_names = ["rch_1","rch_2"]
    mx = (pst.parameter_data.loc[:,"parubnd"] * 1.1).apply(np.log10)
    mn = (pst.parameter_data.loc[:,"parlbnd"] * 0.9).apply(np.log10)
    f_count = 0
    for par_file,par_df in zip(par_files,par_dfs):
        #print(par_file)
        fig = plt.figure(figsize=(4.5,3.5))

        plt.figtext(0.5,0.95,"iteration {0}".format(f_count),ha="center")
        axes = [plt.subplot(3,4,i+1) for i in range(12)]
        arrs = []
        for ireal in range(12):
            arrs.append(freyberg_pars_to_array(par_df.iloc[[ireal],:].T))
        amx = max([arr.max() for arr in arrs])
        amn = max([arr.min() for arr in arrs])
        for ireal,arr in enumerate(arrs):
            axes[ireal].imshow(arr,vmax=amx,vmin=amn,interpolation="nearest")
            axes[ireal].set_xticklabels([])
            axes[ireal].set_yticklabels([])
        plt.savefig(os.path.join(plt_dir,"par_{0:03d}.png".format(f_count)))
        f_count += 1
        plt.close()
    bdir = os.getcwd()
    os.chdir(plt_dir)
    #os.system("ffmpeg -r 1 -i par_%03d.png -vcodec libx264  -pix_fmt yuv420p freyberg_pars.mp4")
    os.system("ffmpeg -r 2 -i par_%03d.png -loop 0 -final_delay 100 -y freyberg_pars.gif")
    os.chdir(bdir)

def freyberg_plot_obs_seq():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","freyberg")
    pst = Pst(os.path.join(d,"freyberg.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    obs_names = pst.nnz_obs_names
    obs_names.extend(pst.pestpp_options["forecasts"].split(',')[:-1])
    print(obs_names)
    print(len(obs_names))
    #print(obs_files)
    obs_dfs = [obs_df.loc[:,obs_names] for obs_df in obs_dfs]
    mx = {obs_name:max([obs_df.loc[:,obs_name].max() for obs_df in obs_dfs]) for obs_name in obs_names}
    mn = {obs_name:min([obs_df.loc[:,obs_name].min() for obs_df in obs_dfs]) for obs_name in obs_names}


    f_count = 0
    for obs_df in obs_dfs[1:]:
        fig = plt.figure(figsize=(4.5,3.5))
        plt.figtext(0.5,0.95,"iteration {0}".format(f_count),ha="center",fontsize=8)

        #print(obs_file)
        axes = [plt.subplot(3,4,i+1) for i in range(len(obs_names))]
        for ax,obs_name in zip(axes,obs_names):
            mean = obs_df.loc[:,obs_name].mean()
            std = obs_df.loc[:,obs_name].std()
            obs_df.loc[:,obs_name].hist(ax=ax,edgecolor="none",
                                        alpha=0.25,grid=False)
            ax.set_yticklabels([])
            #print(ax.get_xlim(),mn[obs_name],mx[obs_name])
            ax.set_title(obs_name,fontsize=6)
            ttl = ax.title
            ttl.set_position([.5, 1.00])
            ax.set_xlim(mn[obs_name],mx[obs_name])
            #ax.set_xlim(0.0,20.0)
            ylim = ax.get_ylim()
            oval = pst.observation_data.loc[obs_name,"obsval"]
            ax.plot([oval,oval],ylim,"k--",lw=0.5)
            #ax.plot([mean,mean],ylim,"b-",lw=0.5)
            #ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=0.5)
            #ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(plt_dir,"obs_{0:03d}.png".format(f_count)))
        f_count += 1
        plt.close()
    bdir = os.getcwd()
    os.chdir(plt_dir)
    #os.system("ffmpeg -r 1 -i obs_%03d.png -vcodec libx264  -pix_fmt yuv420p freyberg_obs.mp4")
    os.system("ffmpeg -r 2 -i obs_%03d.png -loop 0 -final_delay 100 -y freyberg_obs.gif")
    os.chdir(bdir)

def freyberg_plot_iobj():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","freyberg")
    pst = Pst(os.path.join(d,"freyberg.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    obj_df = pd.read_csv(os.path.join(d, "freyberg.pst.iobj.csv"), index_col=0)
    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    obj_df.loc[:, real_cols] = obj_df.loc[:, real_cols].apply(np.log10)
    obj_df.loc[:, "mean"] = obj_df.loc[:, "mean"].apply(np.log10)
    obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    obj_df.index = obj_df.total_runs
    obj_df.loc[:, real_cols].plot(ax=ax, lw=0.5, color="0.5", alpha=0.5, legend=False)
    ax.plot(obj_df.index, obj_df.loc[:, "mean"], 'b', lw=2.5, marker='.', markersize=5)
    # ax.fill_between(obj_df.index, obj_df.loc[:, "mean"] - (1.96 * obj_df.loc[:, "std"]),
    #                obj_df.loc[:, "mean"] + (1.96 * obj_df.loc[:, "std"]),
    #                facecolor="b", edgecolor="none", alpha=0.25)
    #axt = plt.twinx()
    #axt.plot(obj_df.index, obj_df.loc[:, "lambda"], "k", dashes=(2, 1), lw=2.5)
    pobj_df = pd.read_csv(os.path.join(d,"pest_master","freyberg.iobj"),index_col=0)
    #print(pobj_df.total_phi)
    #print(pobj_df.model_runs_completed)
    ax.plot(pobj_df.model_runs_completed.values,pobj_df.total_phi.apply(np.log10).values,"m",lw=2.5)
    #pobj_reg_df = pd.read_csv(os.path.join(d,"pest_master_reg","freyberg_reg.iobj"),index_col=0)
    #ax.plot(pobj_reg_df.model_runs_completed.values,pobj_reg_df.measurement_phi.apply(np.log10).values,"m",lw=2.5)

    ax.set_ylabel("log$_{10}$ $\phi$")
    #axt.set_ylabel("lambda")
    ax.set_xlabel("total runs")
    ax.grid()
    #ax.set_title("EnsembleSmoother $\phi$ summary; {0} realizations in ensemble".\
    #             format(obj_df.shape[1]-7))
    #ax.set_xticks(obj_df.index.values)
    #ax.set_xticklabels(["{0}".format(tr) for tr in obj_df.total_runs])

    plt.savefig(os.path.join(plt_dir, "iobj.png"))
    plt.close()



def freyberg_plot():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","freyberg")
    pst = Pst(os.path.join(d,"freyberg.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    obj_df = pd.read_csv(os.path.join(d, "freyberg.pst.iobj.csv"), index_col=0)
    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    obj_df.loc[:, real_cols] = obj_df.loc[:, real_cols].apply(np.log10)
    obj_df.loc[:, "mean"] = obj_df.loc[:, "mean"].apply(np.log10)
    obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    obj_df.loc[:, real_cols].plot(ax=ax, lw=0.5, color="0.5", alpha=0.5, legend=False)
    ax.plot(obj_df.index, obj_df.loc[:, "mean"], 'b', lw=2.5, marker='.', markersize=5)
    ax.set_xticks(obj_df.index.values)
    ax.set_xticklabels(["{0}".format(tr) for tr in obj_df.total_runs])
    # ax.fill_between(obj_df.index, obj_df.loc[:, "mean"] - (1.96 * obj_df.loc[:, "std"]),
    #                obj_df.loc[:, "mean"] + (1.96 * obj_df.loc[:, "std"]),
    #                facecolor="b", edgecolor="none", alpha=0.25)
    axt = plt.twinx()
    axt.plot(obj_df.index, obj_df.loc[:, "lambda"], "k", dashes=(2, 1), lw=2.5)
    ax.set_ylabel("log$_10$ $\phi$")
    axt.set_ylabel("lambda")
    ax.set_xlabel("total runs")
    ax.set_title("EnsembleSmoother $\phi$ summary; {0} realizations in ensemble".\
                 format(obj_df.shape[1]-7))
    plt.savefig(os.path.join(plt_dir, "iobj.pdf"))
    plt.close()

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    par_dfs = [pd.read_csv(par_file,index_col=0).apply(np.log10) for par_file in par_files]
    #par_names = list(par_dfs[0].columns)
    par_names = ["rch_1","rch_2"]
    mx = (pst.parameter_data.loc[:,"parubnd"] * 1.1).apply(np.log10)
    mn = (pst.parameter_data.loc[:,"parlbnd"] * 0.9).apply(np.log10)

    with PdfPages(os.path.join(plt_dir,"parensemble.pdf")) as pdf:
        for par_file,par_df in zip(par_files,par_dfs):
            #print(par_file)
            fig = plt.figure(figsize=(20,10))

            plt.figtext(0.5,0.975,par_file,ha="center")
            axes = [plt.subplot(2,6,i+1) for i in range(12)]
            arrs = []
            for ireal in range(10):
                arrs.append(freyberg_pars_to_array(par_df.iloc[[ireal],:].T))
            amx = max([arr.max() for arr in arrs])
            amn = max([arr.min() for arr in arrs])
            for ireal,arr in enumerate(arrs):
                axes[ireal].imshow(arr,vmax=amx,vmin=amn,interpolation="nearest")
            for par_name,ax in zip(par_names,axes[-2:]):
                mean = par_df.loc[:,par_name].mean()
                std = par_df.loc[:,par_name].std()
                par_df.loc[:,par_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.25,grid=False)
                ax.set_yticklabels([])
                ax.set_title("{0}, {1:6.2f}".\
                             format(par_name,10.0**mean))
                ax.set_xlim(mn[par_name],mx[par_name])
                ylim = ax.get_ylim()
                if "stage" in par_name:
                    val = np.log10(1.5)
                else:
                    val = np.log10(2.5)
                ticks = ["{0:2.1f}".format(x) for x in 10.0**ax.get_xticks()]
                ax.set_xticklabels(ticks,rotation=90)
                ax.plot([val,val],ylim,"k-",lw=2.0)

                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
            pdf.savefig()
            plt.close()


    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    obs_names = pst.nnz_obs_names
    obs_names.extend(pst.pestpp_options["forecasts"].split(',')[:-1])
    print(obs_names)
    print(len(obs_names))
    #print(obs_files)
    obs_dfs = [obs_df.loc[:,obs_names] for obs_df in obs_dfs]
    mx = {obs_name:max([obs_df.loc[:,obs_name].max() for obs_df in obs_dfs]) for obs_name in obs_names}
    mn = {obs_name:min([obs_df.loc[:,obs_name].min() for obs_df in obs_dfs]) for obs_name in obs_names}



    with PdfPages(os.path.join(plt_dir,"obsensemble.pdf")) as pdf:
        for obs_file,obs_df in zip(obs_files,obs_dfs):
            fig = plt.figure(figsize=(30,40))
            plt.figtext(0.5,0.975,obs_file,ha="center")
            print(obs_file)
            axes = [plt.subplot(3,4,i+1) for i in range(len(obs_names))]
            for ax,obs_name in zip(axes,obs_names):
                mean = obs_df.loc[:,obs_name].mean()
                std = obs_df.loc[:,obs_name].std()
                obs_df.loc[:,obs_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.25,grid=False)
                ax.set_yticklabels([])
                #print(ax.get_xlim(),mn[obs_name],mx[obs_name])
                ax.set_title("{0}, {1:6.2f}:{2:6.2f}".format(obs_name,mean,std))
                ax.set_xlim(mn[obs_name],mx[obs_name])
                #ax.set_xlim(0.0,20.0)
                ylim = ax.get_ylim()
                oval = pst.observation_data.loc[obs_name,"obsval"]
                ax.plot([oval,oval],ylim,"k-",lw=2)
                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
            pdf.savefig()
            plt.close()

def chenoliver_setup():
    import pyemu
    os.chdir(os.path.join("smoother","chenoliver"))
    in_file = os.path.join("par.dat")
    tpl_file = in_file+".tpl"
    out_file = os.path.join("obs.dat")
    ins_file = out_file+".ins"
    pst = pyemu.pst_utils.pst_from_io_files(tpl_file,in_file,ins_file,out_file)
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    par.loc[:,"parval1"] = -2.0
    par.loc[:,"parubnd"] = 20.0
    par.loc[:,"parlbnd"] = -20.0
    obs = pst.observation_data
    obs.loc[:,"obsval"] = 48.0
    obs.loc[:,"weight"] = 1.0
    pst.model_command = ["python chenoliver.py"]
    pst.control_data.noptmax = 0
    pst.pestpp_options["sweep_parameter_csv_file"] = os.path.join("sweep_in.csv")
    pst.write(os.path.join("chenoliver.pst"))

    os.chdir(os.path.join("..",".."))

def chenoliver_func_plot(ax=None):
    def func(par):
        return ((7.0/12.0) * par**3) - ((7.0/2.0) * par**2) + (8.0 * par)
    import numpy as np
    import matplotlib.pyplot as plt
    par = np.arange(-5.0,10.0,0.1)
    obs = func(par)
    if ax is None:
        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(111)
    ax.plot(par,obs,"0.5",dashes=(3,2),lw=4.0)

    ax.scatter(-2.0,func(-2.0),marker='^',s=175,color="b",label="prior mean",zorder=4)
    ax.scatter(5.9,func(5.9),marker='*',s=175,color="m",label="posterior mean",zorder=4)

    ax.set_xlabel("parameter value")
    ax.set_ylabel("observation value")
    ax.grid()
    plt.savefig(os.path.join("smoother","chenoliver","function.png"))

    #plt.show()

def chenoliver_plot_sidebyside():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle as rect
    import pandas as pd
    d = os.path.join("smoother","chenoliver")
    bins = 20
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    #print(obs_files)
    omx = max([obs_df.obs.max() for obs_df in obs_dfs])
    omn = min([obs_df.obs.min() for obs_df in obs_dfs])

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    par_dfs = [pd.read_csv(par_file) for par_file in par_files]
    #mx = max([par_df.par.max() for par_df in par_dfs])
    #mn = min([par_df.par.min() for par_df in par_dfs])
    pmx = 7
    pmn = -5
    figsize = (10,3)
    fcount = 1
    for pdf, odf in zip(par_dfs,obs_dfs[1:]):
        fig = plt.figure(figsize=figsize)
        plt.figtext(0.5,0.95,"iteration {0}".format(fcount),ha="center",fontsize=8)

        #axp = plt.subplot(1,3,1)
        #axo = plt.subplot(1,3,2)
        #axf = plt.subplot(1,3,3)
        axp = plt.axes((0.05,0.075,0.25,0.825))
        axo = plt.axes((0.375,0.075,0.25,0.825))
        axf = plt.axes((0.7,0.075,0.25,0.825))
        chenoliver_func_plot(axf)
        pdf.par.hist(ax=axp,bins=bins,edgecolor="none",grid=False)
        odf.obs.hist(ax=axo,bins=bins,edgecolor="none",grid=False)
        axf.scatter(pdf.par.values,odf.obs.values,marker='.',color="c",s=100)
        ylim = axf.get_ylim()
        r = rect((0.0,ylim[0]),4,ylim[1]-ylim[0],facecolor='0.5',alpha=0.25)
        axf.add_patch(r)
        axp.set_yticks([])
        axo.set_yticks([])
        ylim = axp.get_ylim()
        axp.plot([5.9,5.9],ylim,"k--")
        r = rect((0.0,ylim[0]),4,ylim[1]-ylim[0],facecolor='0.5',alpha=0.25)
        axp.add_patch(r)
        ylim = axo.get_ylim()
        axo.plot([48,48],ylim,"k--")
        axp.set_xlim(pmn,pmx)
        axo.set_xlim(omn,omx)
        axp.set_title("parameter",fontsize=6)
        axo.set_title("observation",fontsize=6)
        axf.set_ylabel("")
        axf.set_xlabel("")
        axf.set_title("par vs obs",fontsize=6)
        plt.savefig(os.path.join(plt_dir,"sbs_{0:03d}.png".format(fcount)))
        #plt.tight_layout()
        plt.close(fig)
        fcount += 1
        #if fcount > 15:
        #    break
    bdir = os.getcwd()
    os.chdir(plt_dir)
    #os.system("ffmpeg -r 6 -i sbs_%03d.png -vcodec libx264  -pix_fmt yuv420p chenoliver.mp4")
    os.system("ffmpeg -r 2 -i sbs_%03d.png -loop 0 -final_delay 100 -y chenoliver.gif")

    os.chdir(bdir)

def chenoliver_obj_plot():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    d = os.path.join("smoother","chenoliver")
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    obj_df = pd.read_csv(os.path.join(d,"chenoliver.pst.iobj.csv"),index_col=0)
    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    obj_df.loc[:,real_cols] = obj_df.loc[:,real_cols].apply(np.log10)
    obj_df.loc[:,"mean"] = obj_df.loc[:,"mean"].apply(np.log10)
    obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    #obj_df.loc[:, real_cols] = obj_df.loc[:, real_cols].apply(np.log10)
    #obj_df.loc[:, "mean"] = obj_df.loc[:, "mean"].apply(np.log10)
    #obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    obj_df.loc[:, real_cols].plot(ax=ax, lw=0.5, color="0.5", alpha=0.5, legend=False)
    ax.plot(obj_df.index, obj_df.loc[:, "mean"], 'b', lw=1.5, marker='.', markersize=5,label="ensemble mean")
    ax.set_ylabel("log$_{10}$ $\phi$")
    ax.set_xlabel("iteration")
    pobj_df = pd.read_csv(os.path.join(d,"pest","chenoliver.iobj"),index_col=0)
    ax.plot(pobj_df.index,pobj_df.total_phi.apply(np.log10),"m",lw=2.5,label="pest++")
    #ax.legend(loc="upper left")
    ax.grid()
    plt.savefig(os.path.join(plt_dir, "iobj.png"))
    plt.close()

def chenoliver_plot():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    d = os.path.join("smoother","chenoliver")
    bins = 20
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    #print(obs_files)
    mx = max([obs_df.obs.max() for obs_df in obs_dfs])
    mn = min([obs_df.obs.min() for obs_df in obs_dfs])
    #print(mn,mx)
    with PdfPages(os.path.join(plt_dir,"obsensemble.pdf")) as pdf:
        for obs_file,obs_df in zip(obs_files,obs_dfs):
            #fig = plt.figure(figsize=(10,10))
            ax = plt.subplot(111)
            obs_df.loc[:,["obs"]].hist(ax=ax,bins=bins,edgecolor="none")
            ax.set_xlim(mn,mx)
            ax.set_title("{0}".format(obs_file))
            #plt.savefig(os.path.join(plt_dir,os.path.split(obs_file)[-1]+".png"))
            #plt.close("all")
            pdf.savefig()
            plt.close()

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    par_dfs = [pd.read_csv(par_file) for par_file in par_files]
    #mx = max([par_df.par.max() for par_df in par_dfs])
    #mn = min([par_df.par.min() for par_df in par_dfs])
    mx = 7
    mn = -5

    with PdfPages(os.path.join(plt_dir,"parensemble.pdf")) as pdf:
        for par_file in par_files:
            par_df = pd.read_csv(par_file)
            fig = plt.figure(figsize=(10,10))
            ax = plt.subplot(111)
            par_df.loc[:,["par"]].hist(ax=ax,bins=bins,edgecolor="none")
            #ax.set_xlim(-10,10)
            ax.set_xlim(mn,mx)

            ax.set_xticks(np.arange(mn,mx+0.25,0.25))
            ax.set_xticklabels(["{0:2.2f}".format(x) for x in np.arange(mn,mx+0.25,0.25)], rotation=90)
            ax.set_title("{0}".format(par_file))
            #plt.savefig(os.path.join(plt_dir,os.path.split(par_file)[-1]+".png"))
            #plt.close("all")
            pdf.savefig()
            plt.close()

def chenoliver():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","chenoliver"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv") and "bak" not in f]
    [os.remove(csv_file) for csv_file in csv_files]

    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    pst = pyemu.Pst("chenoliver.pst")
    #obscov = pyemu.Cov(x=np.ones((1,1))*16.0,names=["obs"],isdiagonal=True)
    obscov = pyemu.Cov(x=np.ones((1,1)),names=["obs"],isdiagonal=True)

    num_reals = 100
    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,
                                num_slaves=15,verbose=True)
    es.initialize(num_reals=num_reals,enforce_bounds=None,init_lambda=10.0)
    for it in range(25):
        es.update(use_approx=False)
    os.chdir(os.path.join("..",".."))


def chenoliver_existing():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","chenoliver"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv") and "bak" not in f]
    [os.remove(csv_file) for csv_file in csv_files]

    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    pst = pyemu.Pst("chenoliver.pst")
    obscov = pyemu.Cov(x=np.ones((1,1))*16.0,names=["obs"],isdiagonal=True)
    #obscov = pyemu.Cov(x=np.ones((1,1))*16.0,names=["obs"],isdiagonal=True)

    num_reals = 100
    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,
                                num_slaves=10,verbose=True)
    es.initialize(num_reals=num_reals,enforce_bounds=None)

    obs1 = es.obsensemble.copy()
    es.parensemble_0.to_csv("paren.csv")
    es.obsensemble_0.to_csv("obsen.csv")

    #es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,
    #                            num_slaves=1,verbose=True)
    es.initialize(parensemble="paren.csv",obsensemble="obsen.csv")

    obs2 = es.obsensemble.copy()
    print(obs1.shape,obs2.shape)
    print(obs1)
    print(obs2)
    assert (obs1 - obs2).loc[:,"obs"].sum() == 0.0

    for it in range(40):
        es.update(lambda_mults=[0.1,1.0,10.0],use_approx=False,run_subset=30)
    os.chdir(os.path.join("..",".."))

def chenoliver_condor():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","chenoliver"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv") and "bak" not in f]
    [os.remove(csv_file) for csv_file in csv_files]

    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    pst = pyemu.Pst("chenoliver.pst")
    obscov = pyemu.Cov(x=np.ones((1,1))*16.0,names=["obs"],isdiagonal=True)
    
    num_reals = 100
    es = pyemu.EnsembleSmoother(pst,parcov=parcov,obscov=obscov,
                                num_slaves=10,verbose=True,
                                submit_file="chenoliver.sub")
    es.initialize(num_reals=num_reals,enforce_bounds=None)
    for it in range(40):
        es.update(lambda_mults=[1.0],use_approx=True)
    os.chdir(os.path.join("..",".."))

def tenpar():
    import os
    import numpy as np
    import flopy
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst("10par_xsec.pst")
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    v = pyemu.utils.ExpVario(contribution=0.25,a=60.0)
    gs = pyemu.utils.GeoStruct(variograms=[v],transform="log")
    par = pst.parameter_data
    k_names = par.loc[par.parnme.apply(lambda x: x.startswith('k')),"parnme"]
    sr = flopy.utils.SpatialReference(delc=[10],delr=np.zeros((10))+10.0)

    full_cov = gs.covariance_matrix(sr.xcentergrid[0,:],sr.ycentergrid[0,:],k_names)
    dia_parcov.drop(list(k_names),axis=1)
    cov = dia_parcov.extend(full_cov)

    es = pyemu.EnsembleSmoother("10par_xsec.pst",parcov=cov,
                                num_slaves=10,port=4005,verbose=True,
                                drop_bad_reals=14000.)
    lz = es.get_localizer().to_dataframe()
    #the k pars upgrad of h01_04 and h01_06 are localized
    upgrad_pars = [pname for pname in lz.columns if "_" in pname and\
                   int(pname.split('_')[1]) > 4]
    lz.loc["h01_04",upgrad_pars] = 0.0
    upgrad_pars = [pname for pname in lz.columns if '_' in pname and \
                   int(pname.split('_')[1]) > 6]
    lz.loc["h01_06", upgrad_pars] = 0.0
    lz = pyemu.Matrix.from_dataframe(lz).T
    print(lz)
    es.initialize(num_reals=100,init_lambda=10000.0)

    for it in range(1):
        #es.update(lambda_mults=[0.1,1.0,10.0],localizer=lz,run_subset=20)
        #es.update(lambda_mults=[0.1,1.0,10.0],run_subset=30)
        es.update(lambda_mults=[.1,1000.0])
    os.chdir(os.path.join("..",".."))


def tenpar_opt():
    import os
    import numpy as np
    import flopy
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst("10par_xsec.pst")
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    v = pyemu.utils.ExpVario(contribution=0.25,a=60.0)
    gs = pyemu.utils.GeoStruct(variograms=[v],transform="log")
    par = pst.parameter_data
    k_names = par.loc[par.parnme.apply(lambda x: x.startswith('k')),"parnme"]
    sr = flopy.utils.SpatialReference(delc=[10],delr=np.zeros((10))+10.0)

    full_cov = gs.covariance_matrix(sr.xcentergrid[0,:],sr.ycentergrid[0,:],k_names)
    dia_parcov.drop(list(k_names),axis=1)
    cov = dia_parcov.extend(full_cov)

    obs = pst.observation_data
    # obs.loc["h01_02","weight"] = 10.0
    # obs.loc["h01_02","obgnme"] = "lt_test"
    # obs.loc["h01_02", "obsval"] = 1.0
    obs.loc["h01_09","weight"] = 100.0
    obs.loc["h01_09",'obgnme'] = "lt_test"
    obs.loc["h01_09", 'obsval'] = 3.0
    print(obs)
    #return()
    es = pyemu.EnsembleSmoother(pst,parcov=cov,
                                num_slaves=10,port=4005,verbose=True,
                                drop_bad_reals=140000.)
    lz = es.get_localizer().to_dataframe()
    #the k pars upgrad of h01_04 and h01_06 are localized
    upgrad_pars = [pname for pname in lz.columns if "_" in pname and\
                   int(pname.split('_')[1]) > 4]
    lz.loc["h01_04",upgrad_pars] = 0.0
    upgrad_pars = [pname for pname in lz.columns if '_' in pname and \
                   int(pname.split('_')[1]) > 6]
    lz.loc["h01_06", upgrad_pars] = 0.0
    lz = pyemu.Matrix.from_dataframe(lz).T
    print(lz)
    es.initialize(num_reals=10,init_lambda=10000.0)

    for it in range(10):
        #es.update(lambda_mults=[0.1,1.0,10.0],localizer=lz,run_subset=20)
        #es.update(lambda_mults=[0.1,1.0,10.0],run_subset=30)
        es.update(lambda_mults=[.1,1000.0])
    os.chdir(os.path.join("..",".."))

def tenpar_restart():
    import os
    import numpy as np
    import flopy
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))

    pst = pyemu.Pst("10par_xsec.pst")
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    v = pyemu.utils.ExpVario(contribution=0.25,a=60.0)
    gs = pyemu.utils.GeoStruct(variograms=[v],transform="log")
    par = pst.parameter_data
    k_names = par.loc[par.parnme.apply(lambda x: x.startswith('k')),"parnme"]
    sr = flopy.utils.SpatialReference(delc=[10],delr=np.zeros((10))+10.0)

    full_cov = gs.covariance_matrix(sr.xcentergrid[0,:],sr.ycentergrid[0,:],k_names)
    dia_parcov.drop(list(k_names),axis=1)
    cov = dia_parcov.extend(full_cov)

    es = pyemu.EnsembleSmoother("10par_xsec.pst",parcov=cov,
                                num_slaves=10,port=4005,verbose=True)
    lz = es.get_localizer().to_dataframe()
    #the k pars upgrad of h01_04 and h01_06 are localized
    upgrad_pars = [pname for pname in lz.columns if "_" in pname and\
                   int(pname.split('_')[1]) > 4]
    lz.loc["h01_04",upgrad_pars] = 0.0
    upgrad_pars = [pname for pname in lz.columns if '_' in pname and \
                   int(pname.split('_')[1]) > 6]
    lz.loc["h01_06", upgrad_pars] = 0.0
    lz = pyemu.Matrix.from_dataframe(lz).T
    print(lz)
    es.initialize(parensemble="par_start.csv",obsensemble="obs_start.csv",
                  restart_obsensemble="obs_restart.csv",init_lambda=10000.0)

    for it in range(1):
        #es.update(lambda_mults=[0.1,1.0,10.0],localizer=lz,run_subset=20)
        es.update(lambda_mults=[0.1,1.0,10.0],run_subset=30)
    os.chdir(os.path.join("..",".."))


def tenpar_failed_runs():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))
    #csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    #[os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst("10par_xsec.pst")
    dia_parcov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)

    v = pyemu.utils.ExpVario(contribution=0.25,a=60.0)
    gs = pyemu.utils.GeoStruct(variograms=[v],transform="log")
    par = pst.parameter_data
    k_names = par.loc[par.parnme.apply(lambda x: x.startswith('k')),"parnme"]
    sr = pyemu.utils.SpatialReference(delc=[10],delr=np.zeros((10))+10.0)

    full_cov = gs.covariance_matrix(sr.xcentergrid[0,:],sr.ycentergrid[0,:],k_names)
    dia_parcov.drop(list(k_names),axis=1)
    cov = dia_parcov.extend(full_cov)

    es = pyemu.EnsembleSmoother("10par_xsec.pst",parcov=cov,
                                num_slaves=2,
                                verbose=True)
    lz = es.get_localizer().to_dataframe()
    #the k pars upgrad of h01_04 and h01_06 are localized
    upgrad_pars = [pname for pname in lz.columns if "_" in pname and\
                   int(pname.split('_')[1]) > 4]
    lz.loc["h01_04",upgrad_pars] = 0.0
    upgrad_pars = [pname for pname in lz.columns if '_' in pname and \
                   int(pname.split('_')[1]) > 6]
    lz.loc["h01_06", upgrad_pars] = 0.0
    lz = pyemu.Matrix.from_dataframe(lz).T
    print(lz)
    #es.initialize(num_reals=10,init_lambda=10000.0)
    es.initialize(parensemble="par_start.csv",obsensemble="obs_start.csv")

    for it in range(10):
        #es.update(lambda_mults=[0.1,1.0,10.0],localizer=lz,run_subset=20)
        #es.update(lambda_mults=[0.1,1.0,10.0],run_subset=7)
        es.update(use_approx=False,lambda_mults=[0.1,1.0,10.0])
    os.chdir(os.path.join("..",".."))


def tenpar_plot():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    import pandas as pd
    from pyemu import Pst
    d = os.path.join("smoother","10par_xsec")
    pst = Pst(os.path.join(d,"10par_xsec.pst"))
    plt_dir = os.path.join(d,"plot")
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]

    par_dfs = [pd.read_csv(par_file,index_col=0) for par_file in par_files]

    par_names = list(par_dfs[0].columns)
    #mx = (pst.parameter_data.loc[:,"parubnd"] * 1.1)
    #mn = (pst.parameter_data.loc[:,"parlbnd"] * 0.9)

    mx = max([pdf.max().max() for pdf in par_dfs])


    num_reals_plot = 12
    plot_rows = 2
    plot_cols = 6
    assert plot_rows * plot_cols == num_reals_plot
    figsize = (20,10)
    with PdfPages(os.path.join(plt_dir,"parensemble_reals.pdf")) as pdf:

        for par_file,par_df in zip(par_files,par_dfs):
            #print(par_file)
            fig = plt.figure(figsize=figsize)

            plt.figtext(0.5,0.975,par_file,ha="center")
            axes = [plt.subplot(plot_rows,plot_cols,i+1) for i in range(num_reals_plot)]
            for ireal in range(num_reals_plot):
                real_df = par_df.iloc[ireal,:]
                #print(real_df)

                real_df.plot(kind="bar",ax=axes[ireal])
                axes[ireal].set_ylim(0,mx.max())
            pdf.savefig()
            plt.close()


    obj_df = pd.read_csv(os.path.join(d,"10par_xsec.pst.iobj.csv"),index_col=0)
    real_cols = [col for col in obj_df.columns if col.startswith("0")]
    obj_df.loc[:,real_cols] = obj_df.loc[:,real_cols].apply(np.log10)
    obj_df.loc[:,"mean"] = obj_df.loc[:,"mean"].apply(np.log10)
    obj_df.loc[:, "std"] = obj_df.loc[:, "std"].apply(np.log10)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    axt = plt.twinx()
    obj_df.loc[:, real_cols].plot(ax=ax, lw=0.5, color="0.5", alpha=0.5, legend=False)
    ax.plot(obj_df.index, obj_df.loc[:, "mean"], 'b', lw=2.5,marker='.',markersize=5)
    #ax.fill_between(obj_df.index, obj_df.loc[:, "mean"] - (1.96 * obj_df.loc[:, "std"]),
    #                obj_df.loc[:, "mean"] + (1.96 * obj_df.loc[:, "std"]),
    #                facecolor="b", edgecolor="none", alpha=0.25)
    axt.plot(obj_df.index,obj_df.loc[:,"lambda"],"k",dashes=(2,1),lw=2.5)
    ax.set_ylabel("log$_10$ phi")
    axt.set_ylabel("lambda")
    ax.set_title("total runs:{0}".format(obj_df.total_runs.max()))
    plt.savefig(os.path.join(plt_dir,"iobj.pdf"))
    plt.close()

    mx = (pst.parameter_data.loc[:,"parubnd"] * 1.1)
    mn = (pst.parameter_data.loc[:,"parlbnd"] * 0.9)

    with PdfPages(os.path.join(plt_dir,"parensemble.pdf")) as pdf:

        for par_file,par_df in zip(par_files,par_dfs):
            print(par_file)
            fig = plt.figure(figsize=(20,10))

            plt.figtext(0.5,0.975,par_file,ha="center")
            axes = [plt.subplot(2,6,i+1) for i in range(len(par_names))]
            for par_name,ax in zip(par_names,axes):
                mean = par_df.loc[:,par_name].mean()
                std = par_df.loc[:,par_name].std()
                par_df.loc[:,par_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.5,grid=False)
                ax.set_yticklabels([])
                ax.set_title("{0}, {1:6.2f}".\
                             format(par_name,10.0**mean))
                ax.set_xlim(mn[par_name],mx[par_name])
                ylim = ax.get_ylim()
                if "stage" in par_name:
                    val = 1.5
                else:
                    val = 2.5
                ticks = ["{0:2.1f}".format(x) for x in ax.get_xticks()]
                ax.set_xticklabels(ticks,rotation=90)
                ax.plot([val,val],ylim,"k-",lw=2.0)

                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
            pdf.savefig()
            plt.close()




    obs_files = [os.path.join(d,f) for f in os.listdir(d) if "obsensemble." in f
                 and ".png" not in f]
    obs_dfs = [pd.read_csv(obs_file) for obs_file in obs_files]
    #print(obs_files)
    #mx = max([obs_df.obs.max() for obs_df in obs_dfs])
    #mn = min([obs_df.obs.min() for obs_df in obs_dfs])
    #print(mn,mx)
    obs_names = ["h01_04","h01_06","h01_08","h02_08"]
    #print(obs_files)
    obs_dfs = [obs_df.loc[:,obs_names] for obs_df in obs_dfs]
    mx = {obs_name:max([obs_df.loc[:,obs_name].max() for obs_df in obs_dfs]) for obs_name in obs_names}
    mn = {obs_name:min([obs_df.loc[:,obs_name].min() for obs_df in obs_dfs]) for obs_name in obs_names}

    with PdfPages(os.path.join(plt_dir,"obsensemble.pdf")) as pdf:
        for obs_file,obs_df in zip(obs_files,obs_dfs):
            fig = plt.figure(figsize=(10,10))
            plt.figtext(0.5,0.975,obs_file,ha="center")
            print(obs_file)
            axes = [plt.subplot(2,2,i+1) for i in range(len(obs_names))]
            for ax,obs_name in zip(axes,obs_names):
                mean = obs_df.loc[:,obs_name].mean()
                std = obs_df.loc[:,obs_name].std()
                obs_df.loc[:,obs_name].hist(ax=ax,edgecolor="none",
                                            alpha=0.5,grid=False)
                ax.set_yticklabels([])
                #print(ax.get_xlim(),mn[obs_name],mx[obs_name])
                ax.set_title("{0}, {1:6.2f}:{2:6.2f}".format(obs_name,mean,std))
                #ax.set_xlim(mn[obs_name],mx[obs_name])
                ax.set_xlim(0.0,20.0)
                ylim = ax.get_ylim()
                oval = pst.observation_data.loc[obs_name,"obsval"]
                ax.plot([oval,oval],ylim,"k-",lw=2)
                ax.plot([mean,mean],ylim,"b-",lw=1.5)
                ax.plot([mean+(2.0*std),mean+(2.0*std)],ylim,"b--",lw=1.5)
                ax.plot([mean-(2.0*std),mean-(2.0*std)],ylim,"b--",lw=1.5)
            pdf.savefig()
            plt.close()


def setup_lorenz():
    import os
    import shutil
    import pandas as pd
    import pyemu

    state_file = "lorenz.dat"
    d = os.path.join("smoother", "lorenz","template")
    dt = 1.0

    prev = [1.0,1.0,1.05,dt]
    if os.path.exists(d):
        shutil.rmtree(d)
    #os.mkdir(d)
    os.makedirs(d)

    df = pd.DataFrame({"variable":['x','y','z','dt']},index=['x','y','z','dt'])
    df.loc[:,"prev"] = prev
    df.loc[:,"new"] = prev

    df.to_csv(os.path.join(d,state_file),sep=' ',index=False)

    df.loc[:,"prev"] = df.variable.apply(lambda x: "~    {0}    ~".format(x))
    with open(os.path.join(d,state_file+".tpl"),'w') as f:
        f.write("ptf ~\n")
        df.to_csv(f,sep=' ',index=False)

    with open(os.path.join(d,state_file+".ins"),'w') as f:
        f.write("pif ~\nl1\n")
        for v in df.variable:
            f.write("l1 w !prev_{0}! !{0}!\n".format(v))

    with open(os.path.join(d,"forward_run.py"),'w') as f:
        f.write("import os\nimport numpy as np\nimport pandas as pd\n")
        f.write("sigma,rho,beta = 10.0,28.0,2.66667\n")

        f.write("df = pd.read_csv('{0}',delim_whitespace=True,index_col=0)\n".format(state_file))
        f.write("x,y,z,dt = df.loc[:,'prev'].values\n")

        f.write("df.loc['x','new'] = sigma * (y - x)\n")
        f.write("df.loc['y','new'] = (rho * x) - y - (x * z)\n")
        f.write("df.loc['z','new'] = (x * y) - (beta * z)\n")
        f.write("df.loc[:,'new'] *= dt\n")
        f.write("df.to_csv('{0}',sep=' ')\n".format(state_file))

    #with open(os.path.join(d,"par.tpl"),'w') as f:
    #    f.write("ptf ~\n")
    #    f.write("dum ~ dum   ~\n")

    base_dir = os.getcwd()
    os.chdir(d)
    pst = pyemu.Pst.from_io_files(*pyemu.helpers.parse_dir_for_io_files('.'))
    os.chdir(base_dir)
    pst.parameter_data.loc[:,"parval1"] = prev
    pst.parameter_data.loc['y',"parlbnd"] = -40.0
    pst.parameter_data.loc['y', "parubnd"] = 40.0
    pst.parameter_data.loc['x', "parlbnd"] = -40.0
    pst.parameter_data.loc['x', "parubnd"] = 40.0
    pst.parameter_data.loc['z', "parlbnd"] = 0.0
    pst.parameter_data.loc['z', "parubnd"] = 50.0
    pst.parameter_data.loc[:,"partrans"] = "none"
    pst.parameter_data.loc['dt','partrans'] = 'fixed'

    pst.observation_data.loc[:,"weight"] = 0.0
    pst.observation_data.loc[['x','y','z'],'weight'] = 1.0

    pst.model_command = "python forward_run.py"
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.control_data.noptmax = 10

    pst.write(os.path.join(d,"lorenz.pst"))

    print(pst.parameter_data)

    pyemu.helpers.run("pestpp lorenz.pst",cwd=d)

if __name__ == "__main__":

    #setup_lorenz()
    #henry_setup()
    #henry()
    #henry_plot()
    #freyberg()
    #freyberg_plot()
    #freyberg_plot_iobj()
    #freyberg_plot_par_seq()
    #freyberg_plot_obs_seq()
    #chenoliver_func_plot()
    #chenoliver_plot_sidebyside()
    #chenoliver_obj_plot()
    #chenoliver_setup()
    #chenoliver_condor()
    #chenoliver()
    #chenoliver_existing()
    #chenoliver_plot()
    #chenoliver_func_plot()
    #chenoliver_plot_sidebyside()
    #chenoliver_obj_plot()
    #tenpar()
    tenpar_opt()
    #tenpar_restart()
    #tenpar_plot()
    #tenpar_failed_runs()
    #freyberg()
    #freyberg_check_phi_calc()
    #freyberg_condor()
    #freyberg_plot()
    #freyberg_plot_iobj()
    #freyberg_plotuse_iobj()
    #freyberg_plot_par_seq()
    #freyberg_plot_obs_seq()
