import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def freyberg_smoother_test():
    import os
    import pyemu

    os.chdir("smoother")
    pst = pyemu.Pst(os.path.join("freyberg.pst"))
    #mc = pyemu.MonteCarlo(pst=pst)
    #mc.draw(2)
    #print(mc.parensemble)
    num_reals = 5
    es = pyemu.EnsembleSmoother(pst)
    es.initialize(num_reals)
    es.update()
    os.chdir("..")


def chenoliver_setup():
    import pyemu
    os.chdir("smoother")
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

    os.chdir("..")


def chenoliver_test():
    import os
    import numpy as np
    import pyemu

    os.chdir("smoother")
    parcov = pyemu.Cov(x=np.ones((1,1)),names=["par"],isdiagonal=True)
    es = pyemu.EnsembleSmoother("chenoliver.pst",parcov=parcov)
    es.initialize(num_reals=10)
    es.update()
    os.chdir("..")




if __name__ == "__main__":
    #freyberg_smoother_test()
    #chenoliver_setup()
    chenoliver_test()