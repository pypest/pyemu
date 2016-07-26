import os
if not os.path.exists("temp"):
    os.mkdir("temp")

def freyberg_smoother_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("smoother","freyberg.pst"))
    #mc = pyemu.MonteCarlo(pst=pst)
    #mc.draw(2)
    #print(mc.parensemble)
    num_reals = 5
    es = pyemu.EnsembleSmoother(pst)
    es.initialize(num_reals)
    es.update()



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
    par.loc[:,"parval1"] = -2.0
    par.loc[:,"parubnd"] = -1.0
    par.loc[:,"parlbnd"] = -3.0
    obs = pst.observation_data
    obs.loc[:,"obsval"] = 48.0
    obs.loc[:,"weight"] = 1.0
    pst.model_command = ["python chenoliver.py"]
    pst.control_data.noptmax = 0
    pst.write(os.path.join("chenoliver.pst"))
    os.chdir("..")



if __name__ == "__main__":
    #freyberg_smoother_test()
    chenoliver_setup()