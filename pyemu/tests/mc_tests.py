
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

if __name__ == "__main__":
    mc_test()