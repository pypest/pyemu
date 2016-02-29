
def mc_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")

    out_dir = os.path.join("mc")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #write testing
    mc = MonteCarlo(jco=jco,verbose=True)
    mc.draw(10,obs=True)
    mc.write_psts(os.path.join("mc","real_"))
    mc.parensemble.to_parfiles(os.path.join("mc","real_"))
    mc = MonteCarlo(jco=jco,verbose=True)
    mc.draw(500,obs=True)
    print("prior ensemble variance:",
          np.var(mc.parensemble.loc[:,"mult1"]))
    projected_en = mc.project_parensemble(inplace=False)
    print("projected ensemble variance:",
          np.var(projected_en.loc[:,"mult1"]))

    import pyemu
    sc = pyemu.Schur(jco=jco)

    mc = MonteCarlo(pst=pst,parcov=sc.posterior_parameter,verbose=True)
    mc.draw(500)
    print("posterior ensemble variance:",
          np.var(mc.parensemble.loc[:,"mult1"]))

def fixed_par_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    mc = MonteCarlo(jco=jco,pst=pst)
    mc.pst.parameter_data.loc["mult1","partrans"] = "fixed"
    mc.draw(10)
    assert np.all(mc.parensemble.loc[:,"mult1"] ==
                  mc.pst.parameter_data.loc["mult1","parval1"])


def uniform_draw_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    mc = MonteCarlo(jco=jco,pst=pst)
    from datetime import datetime
    start = datetime.now()
    mc.draw(num_reals=1000,how="uniform")
    print(datetime.now() - start)
    import matplotlib.pyplot as plt
    ax = mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,alpha=0.5)
    mc.draw(num_reals=1000)
    mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,ax=ax,alpha=0.5)
    #plt.show()


def write_regul_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo

    mc = MonteCarlo(jco=os.path.join("verf_results","freyberg_ord.jco"))
    mc.pst.control_data.pestmode = "regularization"
    mc.draw(10)
    mc.write_psts(os.path.join("temp","freyberg_real"),existing_jco="freyberg_ord.jco")






if __name__ == "__main__":
    #mc_test()
    #fixed_par_test()
    #uniform_draw_test()
    write_regul_test()
