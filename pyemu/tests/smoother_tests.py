import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def freyberg_smoother_test():
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
    import matplotlib.pyplot as plt
    import pandas as pd
    d = os.path.join("smoother","chenoliver")
    par_files = [os.path.join(d,f) for f in os.listdir(d) if "parensemble." in f
                 and ".png" not in f]
    for par_file in par_files:
        par_df = pd.read_csv(par_file)
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        par_df.loc[:,["par"]].hist(ax=ax)
        ax.set_xlim(-10,10)

        plt.savefig(par_file+".png")
        plt.close("all")


def chenoliver_test():
    import os
    import numpy as np
    import pyemu

    os.chdir(os.path.join("smoother","chenoliver"))
    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    pst = pyemu.Pst("chenoliver.pst")
    pst.observation_data.loc[:,"weight"] = 1.0
    es = pyemu.EnsembleSmoother(pst,parcov=parcov)
    es.initialize(num_reals=50)
    for it in range(10):
        es.update()
    os.chdir(os.path.join("..",".."))


def tenpar_test():
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
    #chenoliver_test()
    #tenpar_test()
    chenoliver_plot()