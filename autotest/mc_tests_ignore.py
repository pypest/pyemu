import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def mc_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo, Cov
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")

    out_dir = os.path.join("mc")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #write testing
    mc = MonteCarlo(jco=jco,verbose=True,sigma_range=6)
    cov = Cov.from_parameter_data(mc.pst,sigma_range=6)
    assert np.abs((mc.parcov.x - cov.x).sum()) == 0.0
    mc.draw(10,obs=True)
    mc.write_psts(os.path.join("temp","real_"))
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
    from pyemu import MonteCarlo,ParameterEnsemble
    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    mc = MonteCarlo(jco=jco,pst=pst)
    mc.pst.parameter_data.loc["mult1","partrans"] = "fixed"
    mc.draw(10)
    assert np.all(mc.parensemble.loc[:,"mult1"] ==
                  mc.pst.parameter_data.loc["mult1","parval1"])
    pe = ParameterEnsemble.from_gaussian_draw(mc.pst,mc.parcov,2)


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
    pe = mc.parensemble.from_uniform_draw(mc.pst,5000)
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
    pe = ParameterEnsemble.from_gaussian_draw(mc.pst,cov,num_reals=num_reals)
    pet = pe._transform(inplace=False)

    pe = pet._back_transform(inplace=False)
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
    from pyemu import MonteCarlo,Ensemble,ParameterEnsemble,Pst, Cov

    jco = os.path.join("pst","pest.jcb")
    pst = jco.replace(".jcb",".pst")
    mc = MonteCarlo(jco=jco,pst=pst)
    names = ["par_{0}".format(_) for _ in range(10)]
    df = pd.DataFrame(np.random.random((10,mc.pst.npar)),columns=mc.pst.par_names)
    mc.parensemble = ParameterEnsemble.from_dataframe(df=df,pst=mc.pst)
    print(mc.parensemble.shape)
    mc.project_parensemble()
    mc.parensemble.to_csv(os.path.join("temp","test.csv"))

    pstc = Pst(pst)
    par = pstc.parameter_data
    par.sort_index(ascending=False,inplace=True)
    cov = Cov.from_parameter_data(pstc)
    pe = ParameterEnsemble.from_gaussian_draw(pst=mc.pst,cov=cov)


def parfile_test():
    import os
    import numpy as np
    import pandas as pd
    from pyemu import MonteCarlo, Ensemble, ParameterEnsemble, Pst, Cov

    jco = os.path.join("pst", "pest.jcb")
    pst = jco.replace(".jcb", ".pst")

    mc = MonteCarlo(jco=jco, pst=pst)
    mc.pst.parameter_data.loc[mc.pst.par_names[1], "scale"] = 0.001
    mc.draw(10)
    mc.parensemble.to_parfiles(os.path.join("temp","testpar"))

    pst = Pst(pst)
    pst.parameter_data = pst.parameter_data.iloc[1:]
    pst.parameter_data["test","parmne"] = "test"

    parfiles = [os.path.join("temp",f) for f in os.listdir("temp") if "testpar" in f]
    rnames = ["test{0}".format(i) for i in range(len(parfiles))]

    pe = ParameterEnsemble.from_parfiles(pst=pst,parfile_names=parfiles,real_names=rnames)

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
    #mc.parensemble.read_parfiles(par_files)
    mc.parensemble = pyemu.ParameterEnsemble.from_parfiles(pst=mc.pst,parfile_names=par_files)
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

    en_pnul = pyemu.ParameterEnsemble.from_parfiles(pst=mc.pst,parfile_names=par_files)
    #en_pnul.read_parfiles(par_files)
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

    mc = pyemu.MonteCarlo(jco=os.path.join("mc","freyberg_ord.jco"))
    mc.draw(100,enforce_bounds="reset")
    diff = mc.parensemble.ubnd - mc.parensemble.max(axis=0)
    assert diff.min() == 0.0

    diff = mc.parensemble.lbnd - mc.parensemble.min(axis=0)
    assert diff.max() == 0.0
    #print(mc.parensemble.max(axis=0))
    #print(mc.parensemble.iloc[:,0])

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
    fname = os.path.join("temp","test.csv")
    mc.parensemble.to_csv(fname)
    df = pd.read_csv(fname,index_col=0)
    pe = pyemu.ParameterEnsemble.from_dataframe(pst=pst,df=df)
    pe1 = pe.copy()
    pe.enforce()

    assert np.allclose(pe1.as_matrix(),pe.as_matrix())

def diagonal_cov_draw_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo,Cov,Pst,ParameterEnsemble
    jco = os.path.join("pst","pest.jcb")
    pst = Pst(jco.replace(".jcb",".pst"))

    mc = MonteCarlo(jco=jco,pst=pst)
    num_reals = 100
    mc.draw(num_reals,obs=True)
    print(mc.obsensemble)
    pe1 = mc.parensemble.copy()

    cov = Cov(x=mc.parcov.as_2d,names=mc.parcov.row_names)
    #print(type(cov))
    mc = MonteCarlo(jco=jco,pst=pst)
    mc.parensemble.reseed()
    mc.draw(num_reals,cov=cov)
    pe2 = mc.parensemble

    pe3 = ParameterEnsemble.from_gaussian_draw(mc.pst,num_reals=num_reals,cov=mc.parcov)

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
    oe = ObservationEnsemble.from_id_gaussian_draw(mc.pst,num_reals=num_reals)
    print(oe.shape)
    print(oe.head())


def par_diagonal_draw_test():
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
    pe = ParameterEnsemble.from_gaussian_draw(mc.pst,cov,num_reals=num_reals)
    print(datetime.now() - start)
    print(mc.parensemble.head())
    print(pe.head())


def phi_vector_test():
    import os
    import pyemu
    jco = os.path.join("pst","pest.jcb")
    pst = pyemu.Pst(jco.replace(".jcb",".pst"))

    mc = pyemu.MonteCarlo(pst=pst)
    num_reals = 15
    mc.draw(num_reals,obs=True)
    print(mc.obsensemble.phi_vector)
    print(float(mc.obsensemble.phi_vector.mean()))


def change_weights_test():
    import os
    import numpy as np
    import pyemu
    from pyemu import MonteCarlo, ObservationEnsemble
    from datetime import datetime
    jco = os.path.join("pst", "pest.jcb")
    pst = jco.replace(".jcb", ".pst")

    mc = MonteCarlo(jco=jco, pst=pst)
    print(mc.pst.nnz_obs_names)
    ogcov = mc.obscov.to_dataframe().loc[mc.pst.nnz_obs_names,mc.pst.nnz_obs_names]

    num_reals = 10000
    oe = ObservationEnsemble.from_id_gaussian_draw(mc.pst, num_reals=num_reals)
    for oname in mc.pst.nnz_obs_names:
        w = mc.pst.observation_data.loc[oname,"weight"]
        v = ogcov.loc[oname,oname]
        est = np.std(oe.loc[:,oname])**2
        pd = 100.0 * (np.abs(v-est)) / v
        print(oname,np.std(oe.loc[:,oname])**2,ogcov.loc[oname,oname],pd,(1.0/w)**2)
        assert pd < 10.0,"{0},{1},{2},{3}".format(oname,v,est,pd)
        assert (1.0/w)**2 == v,"{0},{1},{2}".format(oname,v,(1.0/w)**2)

    mc.pst.observation_data.loc[mc.pst.nnz_obs_names,"weight"] = 1000.0
    mc.reset_obscov(pyemu.Cov.from_observation_data(mc.pst))
    newcov = mc.obscov.to_dataframe().loc[mc.pst.nnz_obs_names,mc.pst.nnz_obs_names]
    #print(mc.obsensemble.pst.observation_data.loc[mc.pst.nnz_obs_names,"weight"])

    num_reals = 10000
    oe = ObservationEnsemble.from_id_gaussian_draw(mc.pst, num_reals=num_reals)
    for oname in mc.pst.nnz_obs_names:
        w = mc.pst.observation_data.loc[oname, "weight"]
        v = newcov.loc[oname, oname]
        est = np.std(oe.loc[:, oname]) ** 2
        pd = 100.0 * (np.abs(v - est)) / v
        # print(oname,np.std(oe.loc[:,oname])**2,ogcov.loc[oname,oname],pd,(1.0/w)**2)
        assert pd < 10.0, "{0},{1},{2},{3}".format(oname, v, est, pd)
        assert (1.0 / w) ** 2 == v, "{0},{1},{2}".format(oname, v, (1.0 / w) ** 2)


def homegrown_draw_test():

    import os
    import numpy as np
    import pyemu
    from datetime import datetime

    v = pyemu.geostats.ExpVario(contribution=1.0,a=1.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])

    npar = 20
    pst = pyemu.pst_utils.generic_pst(["p{0:010d}".format(i) for i in range(npar)],["o1"])


    pst.parameter_data.loc[:,"partrans"] = "none"
    par = pst.parameter_data
    par.loc[:,"x"] = np.random.random(npar) * 10.0
    par.loc[:, "y"] = np.random.random(npar) * 10.0

    par.loc[pst.par_names[0], "pargp"] = "zero"
    par.loc[pst.par_names[1:10],"pargp"] = "one"
    par.loc[pst.par_names[11:20], "pargp"] = "two"
    print(pst.parameter_data.pargp.unique())

    cov = gs.covariance_matrix(par.x,par.y,par.parnme)
    num_reals = 100

    s = datetime.now()
    pe_chunk = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals,use_homegrown=True,group_chunks=True)
    print(pe_chunk.iloc[:,0])
    return
    d3 = (datetime.now() - s).total_seconds()

    mc = pyemu.MonteCarlo(pst=pst)

    s = datetime.now()
    #print(s)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals)
    mc.draw(num_reals=num_reals,cov=cov)
    pe = mc.parensemble
    d1 = (datetime.now() - s).total_seconds()
    #print(d1)

    s = datetime.now()
    #print(s)
    peh = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals,use_homegrown=True)
    d2 = (datetime.now() - s).total_seconds()
    #print(d2)

    #import matplotlib.pyplot as plt

    for pname in peh.names:
        #ax = plt.subplot(111)
        m1 = pe.loc[:,pname].mean()
        m2 = peh.loc[:, pname].mean()
        m3 = pe_chunk.loc[:,pname].mean()
        print(par.loc[pname,"parval1"],m2,m1,m3)
        #pe.loc[:,pname].hist(ax=ax,bins=10,alpha=0.5)
        #peh.loc[:,pname].hist(ax=ax,bins=10,alpha=0.5)
        #plt.show()
        #break

    print(d2,d1,d3)

def ensemble_covariance_test():
    import os
    import numpy as np
    import pyemu
    from datetime import datetime

    v = pyemu.geostats.ExpVario(contribution=1.0, a=1.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])

    npar = 10
    pst = pyemu.pst_utils.generic_pst(["p{0:010d}".format(i) for i in range(npar)], ["o1"])

    pst.parameter_data.loc[:, "partrans"] = "none"
    par = pst.parameter_data
    par.loc[:, "x"] = np.random.random(npar) * 10.0
    par.loc[:, "y"] = np.random.random(npar) * 10.0

    cov = gs.covariance_matrix(par.x, par.y, par.parnme)
    num_reals = 100000

    mc = pyemu.MonteCarlo(pst=pst)

    peh = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals,use_homegrown=True)

    localizer = np.ones_like(cov.x)
    localizer[cov.x<1.0e-1] = 0.0

    cov = cov.hadamard_product(localizer)

    ecov = peh.covariance_matrix(localizer=localizer)

    d = 100.0 * (np.abs((cov - ecov).x) / cov.x)
    d[localizer==0.0] = np.nan

    assert np.nanmax(d) < 10.0

    # import matplotlib.pyplot as plt
    #
    # cov = cov.x
    # cov[localizer == 0.0] = np.nan
    # ecov = ecov.x
    # ecov[localizer == 0.0] = np.nan
    #
    # ax = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)
    # vmax = cov.max()
    # vmin = cov.min()
    # ax.imshow(cov,vmax=vmax,vmin=vmin)
    # ax2.imshow(ecov,vmax=vmax,vmin=vmin)
    # p = ax3.imshow(d)
    # plt.colorbar(p)
    # plt.show()

def binary_ensemble_dev():
    import os
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import pyemu

    d = os.path.join("..","misc")
    pst = os.path.join(d,"pest.pst")
    csv = os.path.join(d,"par.csv")
    jcb = csv+".jcb"
    pst = pyemu.Pst(pst)

    start = datetime.now()
    print(start,"loading csv")
    df = pd.read_csv(csv)
    end = datetime.now()
    print("csv load took:",(end-start).total_seconds())

    pe = pyemu.ParameterEnsemble.from_dataframe(pst=pst,df=df)
    start = datetime.now()
    print(start,"writing binary")
    pe.as_pyemu_matrix().to_binary(jcb)
    end = datetime.now()
    print("binary write took:",(end-start).total_seconds())

    start = datetime.now()
    print(start,"loading jcb")
    m = pyemu.Matrix.from_binary(jcb)
    end = datetime.now()
    print("jcb load took:",(end-start).total_seconds())


def to_from_binary_test():
    import os
    import numpy as np
    import pyemu
    from datetime import datetime

    v = pyemu.geostats.ExpVario(contribution=1.0, a=1.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])

    npar = 1000
    pst = pyemu.pst_utils.generic_pst(["p{0:010d}".format(i) for i in range(npar)], ["o1"])

    pst.parameter_data.loc[:, "partrans"] = "none"
    par = pst.parameter_data
    par.loc[:, "x"] = np.random.random(npar) * 10.0
    par.loc[:, "y"] = np.random.random(npar) * 10.0

    cov = gs.covariance_matrix(par.x, par.y, par.parnme)
    num_reals = 1000

    mc = pyemu.MonteCarlo(pst=pst)

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals, use_homegrown=True)
    oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst,num_reals=num_reals)


    pe_name = os.path.join("temp","pe.jcb")
    oe_name = os.path.join("temp","oe.jcb")
    pe.to_binary(pe_name)
    oe.to_binary(oe_name)

    pe1 = pyemu.ParameterEnsemble.from_binary(mc.pst,pe_name)
    oe1 = pyemu.ObservationEnsemble.from_binary(mc.pst,oe_name)
    pe1.index = pe1.index.map(np.int64)
    oe1.index = oe1.index.map(np.int64)
    d = (oe - oe1).apply(np.abs)
    assert d.max().max() == 0.0
    d = (pe - pe1).apply(np.abs)
    assert d.max().max() == 0.0, d


def add_base_test():
    import os
    import numpy as np
    from pyemu import MonteCarlo, Cov, ParameterEnsemble
    from datetime import datetime
    jco = os.path.join("pst", "pest.jcb")
    pst = jco.replace(".jcb", ".pst")

    mc = MonteCarlo(jco=jco, pst=pst)
    num_reals = 100

    mc.draw(num_reals=num_reals, how="gaussian",obs=True)
    mc.parensemble.add_base()
    diff = mc.parensemble.loc["base",:] - mc.pst.parameter_data.parval1
    assert diff.sum() == 0.0
    try:
        mc.parensemble.add_base()
    except:
        pass
    else:
        raise  Exception()

    mc.obsensemble.add_base()
    diff = mc.obsensemble.loc["base", :] - mc.pst.observation_data.obsval
    assert diff.sum() == 0.0
    try:
        mc.obsensemble.add_base()
    except:
        pass
    else:
        raise Exception()



def sparse_draw_test():
    import os
    import numpy as np
    import pyemu
    from datetime import datetime

    v = pyemu.geostats.ExpVario(contribution=1.0, a=1.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])

    npar = 20
    pst = pyemu.pst_utils.generic_pst(["p{0:010d}".format(i) for i in range(npar)], ["o1"])

    pst.parameter_data.loc[:, "partrans"] = "none"
    par = pst.parameter_data
    par.loc[:, "x"] = np.random.random(npar) * 10.0
    par.loc[:, "y"] = np.random.random(npar) * 10.0

    par.loc[pst.par_names[0], "pargp"] = "zero"
    par.loc[pst.par_names[1:10], "pargp"] = "one"
    par.loc[pst.par_names[11:20], "pargp"] = "two"
    print(pst.parameter_data.pargp.unique())

    cov = gs.covariance_matrix(par.x, par.y, par.parnme)

    num_reals = 100000

    pe_base = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=num_reals,group_chunks=True,
                                                         use_homegrown=True)

    scov = pyemu.SparseMatrix.from_matrix(cov)
    pe_sparse = pyemu.ParameterEnsemble.from_sparse_gaussian_draw(pst=pst,cov=scov,num_reals=num_reals)

    d = pe_base.mean() - pe_sparse.mean()
    assert d.apply(np.abs).max() < 0.05
    d = pe_base.std() - pe_sparse.std()
    assert d.apply(np.abs).max() < 0.05


def triangular_draw_test():
    import os
    import matplotlib.pyplot as plt
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.parameter_data.loc[:,"partrans"] = "none"
    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst,1000)
    #print(pst.par_names)
    #pe.iloc[:,0].hist()

    #plt.show()


def invest():

    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyemu

    df = pd.read_csv(os.path.join("temp", "sweep_in.csv"), index_col=0)
    print(df.shape)
    #df = df.loc[:, df.std() != 0]
    print(df.shape)

    pyemu.plot.plot_utils.ensemble_helper(df.iloc[:, [0, 1]])
    plt.show()


def mixed_par_draw_test():
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pname = pst.par_names[0]
    pst.parameter_data.loc[pname,"partrans"] = "none"
    npar = pst.npar
    num_reals = 100
    pe1 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, num_reals=num_reals)

    pe2 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {},default="uniform",num_reals=num_reals)
    pe3 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, default="triangular", num_reals=num_reals)

    # ax = plt.subplot(111)
    # pe1.loc[:,pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe2.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe3.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # plt.show()

    how = {}

    for p in pst.adj_par_names[:10]:
        how[p] = "gaussian"
    for p in pst.adj_par_names[12:30]:
        how[p] = "uniform"
    for p in pst.adj_par_names[40:100]:
        how[p] = "triangular"
    #for pnames in how.keys():
    #    pst.parameter_data.loc[::3,"partrans"] = "fixed"

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    pst.parameter_data.loc[pname, "partrans"] = "none"
    how = {p:"uniform" for p in pst.par_names}
    how["junk"] = "uniform"
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{p:"junk" for p in pst.par_names})
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{},default="junk")
    except:
        pass
    else:
        raise Exception("should have failed")

    cov = pyemu.Cov.from_parameter_data(pst)
    cov.drop(pst.par_names[:2],0)
    how = {p:"gaussian" for p in pst.par_names}
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how,cov=cov)
    except:
        pass
    else:
        raise Exception("should have failed")

    how = {p: "uniform" for p in pst.par_names}
    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how, cov=cov)


    #cov.drop(pst.par_names[:2], 0)
    assert pst.npar == npar


def ensemble_deviations_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))

    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst,num_reals=10)
    dev = pe.get_deviations()
    assert dev.shape == pe.shape
    assert type(dev) == type(pe)

    oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst,num_reals=10)
    dev = oe.get_deviations()
    assert dev.shape == oe.shape
    assert type(dev) == type(oe)

    df = pd.DataFrame(index=np.arange(10), columns=pst.par_names)
    df.loc[:,:]= 1.0
    pe = pyemu.ParameterEnsemble.from_dataframe(pst=pst,df=df)
    dev = pe.get_deviations()
    assert dev.max().max() == 0.0
    assert dev.min().min() == 0.0

    df = pd.DataFrame(index=np.arange(10), columns=pst.obs_names)
    df.loc[:, :] = 1.0
    oe = pyemu.ObservationEnsemble.from_dataframe(pst=pst, df=df)
    dev = oe.get_deviations()
    assert dev.max().max() == 0.0
    assert dev.min().min() == 0.0


if __name__ == "__main__":
    # ensemble_deviations_test()
    mixed_par_draw_test()
    # triangular_draw_test()
    # sparse_draw_test()
    # binary_ensemble_dev()
    # to_from_binary_test()
    # ensemble_covariance_test()
    # homegrown_draw_test()
    # change_weights_test()
    # phi_vector_test()
    # par_diagonal_draw_test()
    # obs_id_draw_test()
    # diagonal_cov_draw_test()
    # pe_to_csv_test()
    # scale_offset_test()
    # mc_test()
    # fixed_par_test()
    # uniform_draw_test()
    #gaussian_draw_test()
    # parfile_test()
    # write_regul_test()
    #from_dataframe_test()
    # ensemble_seed_test()
    # pnulpar_test()
    # enforce_test()
    # add_base_test()