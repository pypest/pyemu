import os
if not os.path.exists("temp"):
    os.mkdir("temp")


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
    mc.draw(10,obs=True)
    print("prior ensemble variance:",
          np.var(mc.parensemble.loc[:,"mult1"]))
    projected_en = mc.project_parensemble(inplace=False)
    print("projected ensemble variance:",
          np.var(projected_en.loc[:,"mult1"]))

    import pyemu
    sc = pyemu.Schur(jco=jco)

    mc = MonteCarlo(pst=pst,parcov=sc.posterior_parameter,verbose=True)
    mc.draw(10)
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
    mc.draw(num_reals=5000,how="uniform")
    print(datetime.now() - start)
    #print(mc.parensemble)
    #import matplotlib.pyplot as plt
    #ax = mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,alpha=0.5)
    mc.draw(num_reals=100)
    #mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,ax=ax,alpha=0.5)
    #plt.show()
    start = datetime.now()
    pe = mc.parensemble.from_uniform_draw(mc.parensemble,5000)
    print(datetime.now() - start)
    print(pe)


def gaussian_draw_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo,Cov,ParameterEnsemble
    from datetime import datetime
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")

    mc = MonteCarlo(jco=jco,pst=pst)
    num_reals = 100

    start = datetime.now()
    mc.draw(num_reals=num_reals,how="gaussian")
    print(mc.parensemble.head())
    print(datetime.now() - start)
    vals = mc.pst.parameter_data.parval1.values
    cov = Cov.from_parameter_data(mc.pst)
    start = datetime.now()
    val_array = np.random.multivariate_normal(vals, cov.as_2d,num_reals)
    print(datetime.now() - start)

    start = datetime.now()
    pe = ParameterEnsemble.from_gaussian_draw(mc.parensemble,cov,num_reals=num_reals)
    print(datetime.now() - start)
    print(mc.parensemble.head())
    print(pe.head())

def write_regul_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo

    mc = MonteCarlo(jco=os.path.join("verf_results","freyberg_ord.jco"))
    mc.pst.control_data.pestmode = "regularization"
    mc.draw(10)
    mc.write_psts(os.path.join("temp","freyberg_real"),existing_jco="freyberg_ord.jco")


def from_dataframe_test():
    import os
    import numpy as np
    import pandas as pd
    from pyemu import MonteCarlo,Ensemble,ParameterEnsemble,Pst

    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    mc = MonteCarlo(jco=jco,pst=pst)
    names = ["par_{0}".format(_) for _ in range(10)]
    df = pd.DataFrame(np.random.random((10,mc.pst.npar)),columns=mc.pst.par_names)
    mc.parensemble = ParameterEnsemble.from_dataframe(df=df,pst=mc.pst)
    print(mc.parensemble.shape)
    mc.project_parensemble()
    mc.parensemble.to_csv(os.path.join("temp","test.csv"))

def scale_offset_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","scale_offest_test.pst"))
    par = pst.parameter_data
    print(par)
    en1 = pyemu.ParameterEnsemble(pst)
    en1.draw(pyemu.Cov.from_parameter_data(pst),num_reals=1000)
    en2 = pyemu.ParameterEnsemble(pst)
    en2.draw(cov=None,num_reals=1000,how="uniform")
    print(en1)
    print(en2)
    # import matplotlib.pyplot as plt
    #
    # for par in en1.columns:
    #     ax = en1.loc[:,par].plot(kind="hist",bins=50)
    #     en2.loc[:,par].plot(kind="hist",bins=50,ax=ax)
    #     ax.set_title(par)
    #     plt.show()


def ensemble_seed_test():
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    cov = pyemu.Cov.from_parameter_data(pst)
    pe1 = pyemu.ParameterEnsemble(pyemu.Pst(os.path.join("pst","pest.pst")))
    pe2 = pyemu.ParameterEnsemble(pyemu.Pst(os.path.join("pst","pest.pst")))

    pe1.reseed()
    pe1.draw(cov,num_reals=10)
    #np.random.seed(1111)
    pe2.reseed()
    pe2.draw(cov,num_reals=10)
    assert (pe1-pe2).apply(np.abs).as_matrix().max() == 0.0
    print(pe1.head())
    print(pe2.head())
    pe2.draw(cov,num_reals=10)
    print(pe2.head())
    assert (pe1-pe2).apply(np.abs).as_matrix().max() != 0.0

    pe1.reseed()
    pe1.draw(cov,num_reals=10,how="uniform")
    pe2.reseed()
    pe2.draw(cov,num_reals=10,how="uniform")
    assert (pe1-pe2).apply(np.abs).as_matrix().max() == 0.0


def pnulpar_test():
    import os
    import pyemu
    dir = "mc"
    mc = pyemu.MonteCarlo(jco=os.path.join("mc","freyberg_ord.jco"))
    par_dir = os.path.join("mc","prior_par_draws")
    par_files = [os.path.join(par_dir,f) for f in os.listdir(par_dir) if f.endswith('.par')]
    mc.parensemble.read_parfiles(par_files)
    real_num = [int(os.path.split(f)[-1].split('.')[0].split('_')[1]) for f in par_files]
    mc.parensemble.index = real_num
    #print(mc.parensemble)
    print(mc.parensemble.istransformed)
    en = mc.project_parensemble(nsing=1,inplace=False, enforce_bounds='reset')
    #en.index = [i+1 for i in en.index]
    print(mc.parensemble.istransformed)

    par_dir = os.path.join("mc", "proj_par_draws")
    par_files = [os.path.join(par_dir, f) for f in os.listdir(par_dir) if f.endswith('.par')]
    real_num = [int(os.path.split(f)[-1].split('.')[0].split('_')[1]) for f in par_files]

    en_pnul = pyemu.ParameterEnsemble(mc.pst)
    en_pnul.read_parfiles(par_files)
    en_pnul.index = real_num
    en.sort_index(axis=1, inplace=True)
    en.sort_index(axis=0, inplace=True)
    en_pnul.sort_index(axis=1, inplace=True)
    en_pnul.sort_index(axis=0, inplace=True)
    diff = 100.0 * ((en - en_pnul) / en)
    assert max(diff.max()) < 1.0e-4


def enforce_test():
    import os
    import pyemu

    mc = pyemu.MonteCarlo(jco=os.path.join("mc","freyberg_ord.jco"),verbose=True)

    cov = pyemu.Cov(x=mc.parcov.x * 0.1,names=mc.parcov.row_names,isdiagonal=True)
    mc = pyemu.MonteCarlo(jco=os.path.join("mc","freyberg_ord.jco"),
                          parcov=cov)
    mc.draw(num_reals=100,enforce_bounds='drop')
    assert mc.parensemble.shape[0] == 100.0

    mc = pyemu.MonteCarlo(jco=os.path.join("mc","freyberg_ord.jco"))
    mc.draw(num_reals=100,enforce_bounds='drop')
    assert mc.parensemble.shape[0] == 0


def enforce_scale():
    import os
    import pyemu
    from pyemu import MonteCarlo
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    pst = pyemu.Pst(pst)
    pst.parameter_data = pst.parameter_data.iloc[:2,:]
    pst.parameter_data.loc["mult1","partrans"] = "none"
    pst.parameter_data.loc["mult1","parval1"] = -1.0

    mc = MonteCarlo(pst=pst,verbose=True)
    mc.draw(1,enforce_bounds="scale")


# def tied_test():
#     import os
#     import pyemu
#     pst_dir = os.path.join('..','tests',"pst")
#     pst = pyemu.Pst(os.path.join(pst_dir,"br_opt_no_zero_weighted.pst"))
#     mc = pyemu.MonteCarlo(pst=pst)
#     mc.draw(num_reals=2)
#     par = pst.parameter_data
#     tied = pst.tied
#     for pname,tname in zip(tied.parnme,tied.partied):
#         pval = par.loc[pname,"parval1"]
#         tval = par.loc[tname,"parval1"]
#         rat = pval / tval
#         rats = mc.parensemble.loc[:,pname] / mc.parensemble.loc[:,tname]
#
#         assert rats.mean() == rat

def pe_to_csv_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    from pyemu import MonteCarlo
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    pst = pyemu.Pst(pst)
    #pst.parameter_data = pst.parameter_data.iloc[:2,:]
    pst.parameter_data.loc["mult1","partrans"] = "none"
    pst.parameter_data.loc["mult1","parval1"] = -1.0

    mc = MonteCarlo(pst=pst,verbose=True)
    mc.draw(1,enforce_bounds="reset")
    if not mc.parensemble.istransformed:
        mc.parensemble._transform()
    fname = os.path.join("mc","test.csv")
    mc.parensemble.to_csv(fname)
    df = pd.read_csv(fname)
    pe = pyemu.ParameterEnsemble.from_dataframe(pst=pst,df=df)
    pe1 = pe.copy()
    pe.enforce()

    assert np.allclose(pe1.as_matrix(),pe.as_matrix())

def diagonal_cov_draw_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo,Cov,Pst
    jco = os.path.join("pst","pest.jcb")
    pst = Pst(jco.replace(".jcb",".pst"))

    mc = MonteCarlo(jco=jco,pst=pst)
    num_reals = 10
    mc.draw(num_reals,obs=True)
    print(mc.obsensemble)
    pe1 = mc.parensemble.copy()

    cov = Cov(x=mc.parcov.as_2d,names=mc.parcov.row_names)
    #print(type(cov))
    mc = MonteCarlo(jco=jco,pst=pst)
    mc.parensemble.reseed()
    mc.draw(num_reals,cov=cov)
    pe2 = mc.parensemble
    #print(pe1-pe2)

def obs_id_draw_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo,ObservationEnsemble
    from datetime import datetime
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")

    mc = MonteCarlo(jco=jco,pst=pst)
    num_reals = 100
    oe = ObservationEnsemble.from_id_gaussian_draw(mc.obsensemble,num_reals=num_reals)
    print(oe.shape)
    print(oe.head())

if __name__ == "__main__":
    obs_id_draw_test()
    #diagonal_cov_draw_test()
    #pe_to_csv_test()
    #scale_offset_test()
    #mc_test()
    #fixed_par_test()
    #uniform_draw_test()
    #gaussian_draw_test()
    #write_regul_test()
    #from_dataframe_test()
    #ensemble_seed_test()
    #pnulpar_test()
    #enforce_test()
    #tied_test()
    #enforce_scale_test()
    #freyberg_verf_test()