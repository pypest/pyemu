import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def schur_test_nonpest():
    import numpy as np
    from pyemu import Matrix, Cov, Schur, Jco
    #non-pest
    pnames = ["p1","p2","p3"]
    onames = ["o1","o2","o3","o4"]
    npar = len(pnames)
    nobs = len(onames)
    j_arr = np.random.random((nobs,npar))
    jco = Jco(x=j_arr,row_names=onames,col_names=pnames)
    parcov = Cov(x=np.eye(npar),names=pnames)
    obscov = Cov(x=np.eye(nobs),names=onames)
    forecasts = "o2"

    s = Schur(jco=jco,parcov=parcov,obscov=obscov,forecasts=forecasts)
    print(s.get_parameter_summary())
    print(s.get_forecast_summary())

    #this should fail
    passed = False
    try:
        print(s.get_par_group_contribution())
        passed = True
    except Exception as e:
        print(str(e))
    if passed:
        raise Exception("should have failed")

    #this should fail
    passed = False
    try:
        print(s.get_removed_obs_group_importance())
        passed = True
    except Exception as e:
        print(str(e))
    if passed:
        raise Exception("should have failed")

    print(s.get_par_contribution({"group1":["p1","p3"]}))

    print(s.get_removed_obs_importance({"group1":["o1","o3"]}))


def schur_test():
    import os
    from pyemu import Schur, Cov, Pst
    w_dir = os.path.join("..","..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    pst = Pst(os.path.join(w_dir,"pest.pst"))
    cov = Cov.from_parameter_data(pst)
    cov.to_uncfile(os.path.join("temp","pest.unc"),covmat_file=None)
    cov2 = Cov.from_uncfile(os.path.join("temp","pest.unc"))
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts,
               parcov=cov2)
    print(sc.prior_forecast)
    print(sc.posterior_forecast)
    print(sc.get_par_group_contribution())
    print(sc.get_removed_obs_group_importance())

def sigma_range_tests():


def errvar_test_nonpest():
    import numpy as np
    from pyemu import ErrVar, Matrix, Cov
    #non-pest
    pnames = ["p1","p2","p3"]
    onames = ["o1","o2","o3","o4"]
    npar = len(pnames)
    nobs = len(onames)
    j_arr = np.random.random((nobs,npar))
    jco = Matrix(x=j_arr,row_names=onames,col_names=pnames)
    parcov = Cov(x=np.eye(npar),names=pnames)
    obscov = Cov(x=np.eye(nobs),names=onames)
    forecasts = "o2"

    omitted = "p3"

    e = ErrVar(jco=jco,parcov=parcov,obscov=obscov,forecasts=forecasts,
               omitted_parameters=omitted)
    svs = [0,1,2,3,4,5]
    print(e.get_errvar_dataframe(svs))


def errvar_test():
    import os
    from pyemu import ErrVar
    w_dir = os.path.join("..","..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    ev = ErrVar(jco=os.path.join(w_dir,"pest.jcb"),forecasts=forecasts)
    print(ev.prior_forecast)
    print(ev.get_errvar_dataframe())


def dataworth_test():
    import os
    import numpy as np
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","Freyberg")
    forecasts = ["travel_time","sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    sc.pst.observation_data.index = sc.pst.observation_data.obsnme
    base_obs = sc.pst.nnz_obs_names
    zw_names = [name for name in sc.pst.zero_weight_obs_names if name not in forecasts ]
    #print(sc.get_removed_obs_importance(obslist_dict={"test":zw_names}))
    oname = "or00c00_0"
    names = {"test":oname}
    added = sc.get_added_obs_importance(base_obslist=base_obs,
                                      obslist_dict=names,reset_zero_weight=True)
    removed = sc.get_removed_obs_importance(obslist_dict=names)
    print("removed",removed)
    print("added",added)
    for fname in forecasts:
        diff = np.abs(removed.loc["test",fname] - added.loc["base",fname])
        assert diff < 0.01,"{0},{1},{2}".format(fname,removed.loc["test",fname],added.loc["base",fname])

    names = {"test1":oname,"test2":["or00c00_1","or00c00_2"]}
    sc.next_most_important_added_obs(forecast="travel_time",obslist_dict=names,
                                     base_obslist=sc.pst.nnz_obs_names)


def dataworth_next_test():
    import os
    import numpy as np
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","Freyberg")
    #w_dir = os.path.join("..","..","examples","freyberg")
    forecasts = ["sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    next_test = sc.next_most_important_added_obs(forecast="sw_gw_0",
                                           base_obslist=sc.pst.nnz_obs_names,
                                           obslist_dict={"test":sc.pst.nnz_obs_names})
    # the returned dataframe should only have one row since the 'base' case
    # should be the same as the 'test' case
    assert next_test.shape[0] == 1

    obs = sc.pst.observation_data
    obs.index = obs.obsnme
    row_groups = obs.groupby([lambda x: x.startswith("or"), lambda x: x in sc.pst.nnz_obs_names,
                              lambda x: x[:4]]).groups
    obslist_dict = {}
    for key,idxs in row_groups.items():
        if not key[0] or key[1]:
            continue
        obslist_dict[key[2]] = list(idxs)

    imp_df = sc.get_added_obs_importance(base_obslist=sc.pst.nnz_obs_names,
                                         obslist_dict=obslist_dict,
                                         reset_zero_weight=1.0)
    next_test = sc.next_most_important_added_obs(forecast="sw_gw_0",
                                                 base_obslist=sc.pst.nnz_obs_names,
                                                 obslist_dict=obslist_dict,
                                                 reset_zero_weight=1.0,
                                                 niter=4)
    print(next_test)
    print(imp_df.sort_index())
    assert next_test.shape[0] == 4


def par_contrib_test():
    import os
    import numpy as np
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","Freyberg")
    forecasts = ["travel_time","sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    par = sc.pst.parameter_data
    par.index = par.parnme
    groups = {name:list(idxs) for name,idxs in par.groupby(par.pargp).groups.items()}

    parlist_dict = {}
    print(sc.next_most_par_contribution(forecast="travel_time",
                                        parlist_dict=groups))


def map_test():
    import os
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","10par_xsec","master_opt0")
    forecasts = ["h01_08","h02_08"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts)
    print(sc.map_parameter_estimate)
    print(sc.map_forecast_estimate)


def forecast_pestpp_load_test():
    import os
    import pyemu
    pst_name = os.path.join("pst","forecast.pst")
    jco_name = pst_name.replace(".pst",".jcb")
    pst = pyemu.Pst(pst_name)
    print(pst.pestpp_options)
    sc = pyemu.Schur(jco=jco_name)

    print(sc.get_forecast_summary())


def css_test():
    import os
    import numpy as np
    import pandas as pd
    from pyemu import Schur
    #w_dir = os.path.join("..","..","verification","10par_xsec","master_opt0")
    w_dir = "la"
    forecasts = ["h01_08","h02_08"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"))
    css = sc.get_par_css_dataframe()
    css_pestpp = pd.read_csv(os.path.join(w_dir,"pest.isen"))
    diff = (css_pestpp - css.pest_css).apply(np.abs).sum(axis=1)[0]
    assert diff < 0.001,diff

def inf():
    import os
    import numpy as np
    import pandas as pd
    from pyemu import Influence
    w_dir = os.path.join("..","..","verification","10par_xsec","master_opt0")
    inf = Influence(jco=os.path.join(w_dir,"pest.jcb"),
                    resfile=os.path.join(w_dir,"pest.rei"))
    print(inf.cooks_d)


def inf2():

    #non-pest
    from pyemu.mat import mat_handler as mhand
    from pyemu.pst import Pst
    from pyemu import Influence
    import numpy as np

    inpst = Pst(os.path.join("..","..","verification","Freyberg",
                             "Freyberg_pp","freyberg_pp.pst"))

    pnames = inpst.par_names
    onames = inpst.obs_names
    npar = inpst.npar
    nobs = inpst.nobs
    j_arr = np.random.random((nobs,npar))
    parcov = mhand.Cov(x=np.eye(npar),names=pnames)
    obscov = mhand.Cov(x=np.eye(nobs),names=onames)
    jco = mhand.Jco.from_binary(inpst.filename.replace(".pst",".jcb"))
    resf = inpst.filename.replace(".pst",".rei")
    s = Influence(jco=jco,obscov=obscov, pst=inpst,resfile=resf)
    print(s.hat)
    print(s.observation_leverage)
    #v = s.studentized_res
    print(s.estimated_err_var)
    print(s.studentized_res)


def freyberg_verf_test():
    import os
    import pyemu
    import numpy as np
    import pandas as pd
    wdir = os.path.join("..","..","verification","Freyberg")
    post_pd7 = pyemu.Cov.from_ascii(os.path.join(wdir,"post.cov"))
    sc = pyemu.Schur(os.path.join(wdir,"freyberg.jcb"))
    post_pyemu = sc.posterior_parameter
    diff = (post_pd7 - post_pyemu).to_dataframe()
    diff = (diff / sc.pst.parameter_data.parval1 * 100.0).apply(np.abs)
    assert diff.max().max() < 1.0

    pd1_file = os.path.join(wdir,"predunc1_textable.dat")
    names = ["forecasts","pd1_pr","py_pr","pd1_pt","py_pt"]
    pd1_df = pd.read_csv(pd1_file,sep='&',header=None,names=names)
    pd1_df.index = pd1_df.forecasts
    fsum = sc.get_forecast_summary()
    pd1_cols = ["pd1_pr","pd1_pt"]
    py_cols = ["prior_var","post_var"]
    forecasts = ["sw_gw_0","sw_gw_1"]
    for fname in forecasts:
        for pd1_col,py_col in zip(pd1_cols,py_cols):
            pd1_pr = pd1_df.loc[fname,pd1_col]
            py_pr = np.sqrt(fsum.loc[fname,py_col])
            assert np.abs(pd1_pr - py_pr) < 1.0e-1,"{0},{1}".format(pd1_pr,py_pr)

    out_files = [os.path.join(wdir,f) for f in os.listdir(wdir)\
                 if ".predvar1b.out" in f and f.split('.')[0] in forecasts]
    print(out_files)
    pv1b_results = {}
    for out_file in out_files:
        pred_name = os.path.split(out_file)[-1].split('.')[0]
        f = open(out_file,'r')
        for _ in range(3):
            f.readline()
        arr = np.loadtxt(f)
        pv1b_results[pred_name] = arr

    omitted_parameters = [pname for pname in sc.pst.parameter_data.parnme if\
                          pname.startswith("wf")]
    obs_names = sc.pst.nnz_obs_names
    obs_names.extend(forecasts)
    jco = pyemu.Jco.from_binary(os.path.join(wdir,"freyberg.jcb")).\
        get(obs_names,sc.pst.adj_par_names)
    ev = pyemu.ErrVar(jco=jco,pst=sc.pst,forecasts=forecasts,
                      omitted_parameters=omitted_parameters,verbose=False)
    #print(ev.jco.shape)
    df = ev.get_errvar_dataframe(np.arange(36))
    #print(df)
    max_idx = 12
    #print(pv1b_results.keys())
    for ipred,pred in enumerate(forecasts):
        arr = pv1b_results[pred][:max_idx,:]
        first = df[("first", pred)][:max_idx]
        second = df[("second", pred)][:max_idx]
        third = df[("third", pred)][:max_idx]
        #print(arr[:,1])
        #print(first)
        diff = np.abs(arr[:,1] - first) / arr[:,1] * 100.0
        assert diff.max() < 0.01
        diff = np.abs(arr[:,2] - second) / arr[:,1] * 100.0
        assert diff.max() < 0.01
        diff = np.abs(arr[:,3] - third) / arr[:,1] * 100.0
        assert diff.max() < 0.01



if __name__ == "__main__":
    freyberg_verf_test()
    #forecast_pestpp_load_test()
    #map_test()
    #par_contrib_test()
    #dataworth_test()
    #dataworth_next_test()
    #schur_test_nonpest()
    #schur_test()
    #errvar_test_nonpest()
    #errvar_test()
    #css_test()
    #inf_test()
    #inf2_test()