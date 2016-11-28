import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def freyberg():
    import os
    import pyemu

    os.chdir(os.path.join("smoother","freyberg"))
    pst = pyemu.Pst(os.path.join("freyberg.pst"))
    #mc = pyemu.MonteCarlo(pst=pst)
    #mc.draw(2)
    #print(mc.parensemble)
    num_reals = 5
    es = pyemu.EnsembleSmoother(pst)
    es.initialize(num_reals)
    es.update()
    os.chdir(os.path.join("..",".."))

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
    par.loc[:,"parval1"] = 10.0
    par.loc[:,"parubnd"] = -1.0
    par.loc[:,"parlbnd"] = -10.0
    obs = pst.observation_data
    obs.loc[:,"obsval"] = 48.0
    obs.loc[:,"weight"] = 1.0
    pst.model_command = ["python chenoliver.py"]
    pst.control_data.noptmax = 0
    pst.pestpp_options["sweep_parameter_csv_file"] = os.path.join("sweep_in.csv")
    pst.write(os.path.join("chenoliver.pst"))

    os.chdir(os.path.join("..",".."))

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
    print(obs_files)
    mx = max([obs_df.obs.max() for obs_df in obs_dfs])
    mn = min([obs_df.obs.min() for obs_df in obs_dfs])
    print(mn,mx)
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
    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    pst = pyemu.Pst("chenoliver.pst")
    #pst.observation_data.loc[:,"weight"] = 1.0/8.0
    obscov = pyemu.Cov(x=np.ones((1,1)),names=["obs"],isdiagonal=True)
    es = pyemu.EnsembleSmoother(pst,parcov=parcov,num_slaves=10,use_approx=False)
    es.initialize(num_reals=100)
    for it in range(10):
        es.update()
    os.chdir(os.path.join("..",".."))


def tenpar():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))
    es = pyemu.EnsembleSmoother("pest.pst")
    es.initialize(num_reals=10)
    for it in range(1):
        es.update()
    os.chdir(os.path.join("..",".."))



if __name__ == "__main__":
    #freyberg_smoother_test()
    #chenoliver_setup()
    chenoliver()
    #tenpar_test()
    chenoliver_plot()