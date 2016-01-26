def schur_test_nonpest():
    import numpy as np
    from pyemu import Matrix, Cov, Schur
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
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),forecasts=forecasts)
    print(sc.prior_forecast)
    print(sc.posterior_forecast)
    print(sc.get_par_group_contribution())
    print(sc.get_removed_obs_group_importance())


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
    forecasts = ["travel_time","sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    next_test = sc.next_most_important_added_obs(forecast="travel_time",
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
        obslist_dict[key[2]] = idxs

    imp_df = sc.get_added_obs_importance(base_obslist=sc.pst.nnz_obs_names,
                                         obslist_dict=obslist_dict,
                                         reset_zero_weight=1.0)
    next_test = sc.next_most_important_added_obs(forecast="travel_time",
                                                 base_obslist=sc.pst.nnz_obs_names,
                                                 obslist_dict=obslist_dict,
                                                 reset_zero_weight=1.0)
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
    groups = par.groupby(par.pargp).groups

    parlist_dict = {}
    print(sc.next_most_par_contribution(forecast="travel_time",
                                        parlist_dict=groups))



if __name__ == "__main__":
    par_contrib_test()
    #dataworth_test()
    #dataworth_next_test()
    #schur_test_nonpest()
    #schur_test()
    #errvar_test_nonpest()
    #errvar_test()