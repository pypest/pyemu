import os
import copy
from pathlib import Path
import shutil
from pst_from_tests import setup_tmp, ies_exe_path, _get_port

import pytest


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
    print(s.get_added_obs_importance({"group1": ["o1", "o3"]},reset_zero_weight=0.0))
    print(s.get_removed_obs_importance({"group1":["o1","o3"]}))

    forecasts = Matrix(x=np.random.random((1,npar)),row_names=[forecasts],col_names=pnames)

    sc = Schur(jco=jco,forecasts=forecasts.T,parcov=parcov,obscov=obscov)
    ffile = os.path.join("temp","forecasts.jcb")
    forecasts.to_binary(ffile)

    sc = Schur(jco=jco, forecasts=ffile, parcov=parcov, obscov=obscov)


def schur_test(tmp_path):
    import os
    import numpy as np
    from pyemu import Schur, Cov, Pst
    w_dir = os.path.join("..","verification","henry")
    w_dir = setup_tmp(w_dir, tmp_path)
    forecasts = ["pd_ten","c_obs10_2"]
    pst = Pst(os.path.join(w_dir,"pest.pst"))
    cov = Cov.from_parameter_data(pst)
    cov.to_uncfile(os.path.join(w_dir,"pest.unc"),covmat_file=None)
    cov2 = Cov.from_uncfile(os.path.join(w_dir,"pest.unc"))
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts,
               parcov=cov2)
    print(sc.prior_forecast)
    print(sc.posterior_forecast)
    print(sc.get_par_group_contribution())

    df = sc.get_par_group_contribution(include_prior_results=True)
    levels = list(df.columns.levels[1])
    assert "prior" in levels,levels
    assert "post" in levels,levels

    print(sc.get_removed_obs_importance(reset_zero_weight=True))

    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts,
               sigma_range=6.0)
    cov = Cov.from_parameter_data(pst,sigma_range=6.0)

    assert np.abs((sc.parcov.x - cov.x).sum()) == 0.0

    sc = Schur(jco=os.path.join(w_dir, "pest.jcb"),
               forecasts=forecasts,
               sigma_range=6.0,scale_offset=False)
    assert np.abs((sc.parcov.x - cov.x).sum()) == 0.0

    pst.parameter_data.loc[:,"offset"] = 100.0
    cov = Cov.from_parameter_data(pst)
    sc = Schur(jco=os.path.join(w_dir, "pest.jcb"),
               pst=pst,
               forecasts=forecasts,
               sigma_range=6.0, scale_offset=False)
    assert np.abs((sc.parcov.x - cov.x).sum()) != 0.0

    cov = Cov.from_parameter_data(pst,scale_offset=False,sigma_range=6.0)
    assert np.abs((sc.parcov.x - cov.x).sum()) == 0.0


def la_test_io():
    from pyemu import Schur, Cov, Pst
    w_dir = os.path.join("..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    pst = Pst(os.path.join(w_dir,"pest.pst"))
    cov = Cov.from_parameter_data(pst)
    cov.to_binary(os.path.join("temp","pest.bin.cov"))
    cov.to_ascii(os.path.join("temp","pest.txt.cov"))
    sc_bin = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts,
               parcov=os.path.join("temp","pest.bin.cov"))

    sc_ascii = Schur(jco=os.path.join(w_dir,"pest.jcb"),
               forecasts=forecasts,
               parcov=os.path.join("temp","pest.txt.cov"))


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
    w_dir = os.path.join("..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    ev = ErrVar(jco=os.path.join(w_dir,"pest.jcb"),
                forecasts=forecasts)
    print(ev.prior_forecast)
    print(ev.get_errvar_dataframe())


def dataworth_test(tmp_path):
    import os
    import numpy as np
    from pyemu import Schur,Cov
    w_dir = os.path.join("..","verification","Freyberg")
    w_dir = setup_tmp(w_dir, tmp_path)
    forecasts = ["travel_time","sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    #sc.pst.observation_data.loc[sc.pst.nnz_obs_names[3:],"weight"] = 0.0
    base_obs = sc.pst.nnz_obs_names
    zw_names = [name for name in sc.pst.zero_weight_obs_names if name not in forecasts ]

    #print(sc.get_removed_obs_importance(obslist_dict={"test":zw_names}))
    oname = "or00c00_0"

    names = {"test":oname}
    added = sc.get_added_obs_importance(base_obslist=base_obs,
                                      obslist_dict=names,reset_zero_weight=1.0)
    sc.pst.observation_data.loc[oname,"weight"] = 1.0
    #sc.reset_obscov()
    cov = Cov.from_observation_data(sc.pst)
    scm = Schur(jco=os.path.join(w_dir, "freyberg.jcb"), forecasts=forecasts, verbose=True,
                obscov=cov)
    scm.pst.observation_data.loc[oname, "weight"] = 1.0
    removed = sc.get_removed_obs_importance(obslist_dict=names)
    for fname in forecasts:
        diff = np.abs(removed.loc["test",fname] - added.loc["base",fname])
        print("add,{0},{1},{2}".format(fname, added.loc["test", fname], added.loc["base", fname]))
        print("rem,{0},{1},{2}".format(fname, removed.loc["test", fname], removed.loc["base", fname]))
        assert diff < 0.01,"{0},{1},{2}".format(fname,removed.loc["test",fname],added.loc["base",fname])

    names = {"test1":oname,"test2":["or00c00_1","or00c00_2"]}
    scm.pst.observation_data.loc[oname, "weight"] = 0.0
    scm.next_most_important_added_obs(forecast="travel_time",obslist_dict=names,
                                     base_obslist=scm.pst.nnz_obs_names,reset_zero_weight=1.0)


def dataworth_next_test(tmp_path):
    import os
    import numpy as np
    from pyemu import Schur
    w_dir = os.path.join("..","verification","Freyberg")
    w_dir = setup_tmp(w_dir, tmp_path)
    #w_dir = os.path.join("..","..","examples","freyberg")
    forecasts = ["sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    next_test = sc.next_most_important_added_obs(forecast="sw_gw_0",
                                           base_obslist=sc.pst.nnz_obs_names,
                                           obslist_dict={"test":sc.pst.nnz_obs_names},reset_zero_weight=0.0)

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
    #print(imp_df.sort_index())
    assert next_test.shape[0] == 4,next_test.shape


def par_contrib_speed_test():
    import pyemu

    npar = 1800
    nobs = 1000
    nfore = 1000

    par_names = ["par{0}".format(i) for i in range(npar)]
    obs_names = ["obs{0}".format(i) for i in range(nobs)]
    fore_names = ["fore{0}".format(i) for i in range(nfore)]

    all_names = copy.deepcopy(obs_names)
    all_names.extend(fore_names)
    pst = pyemu.Pst.from_par_obs_names(par_names,all_names)
    cal_jco = pyemu.Jco.from_names(obs_names,par_names,random=True)
    fore_jco = pyemu.Jco.from_names(par_names,fore_names,random=True)
    pst.observation_data.loc[obs_names,"weight"] = 1.0
    pst.observation_data.loc[fore_names,"weight"] = 0.0

    sc = pyemu.Schur(jco=cal_jco,pst=pst,forecasts=fore_jco, verbose=True)
    sc.get_par_contribution(parlist_dict={par_names[0]:par_names[0]})


def par_contrib_test():
    import os
    import numpy as np
    from pyemu import Schur
    w_dir = os.path.join("..","verification","Freyberg")
    forecasts = ["travel_time","sw_gw_0","sw_gw_1"]
    sc = Schur(jco=os.path.join(w_dir,"freyberg.jcb"),forecasts=forecasts,verbose=True)
    par = sc.pst.parameter_data
    par.index = par.parnme
    groups = {name:list(idxs) for name,idxs in par.groupby(par.pargp).groups.items()}

    parlist_dict = {}
    print(sc.next_most_par_contribution(forecast="travel_time",
                                        parlist_dict=groups))


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
    w_dir = os.path.join("..","verification","10par_xsec","master_opt0")
    inf = Influence(jco=os.path.join(w_dir,"pest.jcb"),
                    resfile=os.path.join(w_dir,"pest.rei"))
    print(inf.cooks_d)


def inf2():

    #non-pest
    from pyemu.mat import mat_handler as mhand
    from pyemu.pst import Pst
    from pyemu import Influence
    import numpy as np

    inpst = Pst(os.path.join("..","verification","Freyberg",
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


def freyberg_verf_test(tmp_path):
    import os
    import pyemu
    import numpy as np
    import pandas as pd
    wdir = os.path.join("..","verification","Freyberg")
    wdir = setup_tmp(wdir, tmp_path)
    post_pd7 = pyemu.Cov.from_ascii(os.path.join(wdir,"post.cov"))
    sc = pyemu.Schur(os.path.join(wdir,"freyberg.jcb"))
    post_pyemu = sc.posterior_parameter
    diff = (post_pd7 - post_pyemu).to_dataframe()
    diff = (diff / sc.pst.parameter_data.parval1 * 100.0).apply(np.abs)
    print(diff.max().max())
    assert diff.max().max() < 10.0

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


def alternative_dw():
    import os
    import pyemu
    import numpy as np
    import pandas as pd
    wdir = os.path.join("..","verification","Freyberg")
    sc = pyemu.Schur(os.path.join(wdir,"freyberg.jcb"))
    print(sc.pst.nnz_obs)

    obs = sc.pst.observation_data
    zw_obs = obs.loc[obs.weight==0,"obsnme"].iloc[:5].values
    test_obs = {o:o for o in zw_obs}
    base = sc.get_added_obs_importance(obslist_dict=test_obs)
    zw_pst = pyemu.Pst(os.path.join(wdir,"freyberg.pst"))
    zw_pst.observation_data.loc[:,"weight"] = 0.0
    for o in zw_obs:
        ojcb = sc.jco.get(row_names=[o])
        zw_pst.observation_data.loc[:,"weight"] = 0.0
        zw_pst.observation_data.loc[o,"weight"] = 1.0
        sc_o = pyemu.Schur(jco=ojcb,pst=zw_pst,parcov=sc.posterior_parameter,forecasts=sc.forecasts)
        print(sc_o.get_forecast_summary())

def obscomp_test():
    import os
    import numpy as np
    from pyemu import LinearAnalysis
    w_dir = os.path.join("..", "verification", "Freyberg")
    forecasts = ["travel_time", "sw_gw_0", "sw_gw_1"]
    la = LinearAnalysis(jco=os.path.join(w_dir, "freyberg.jcb"), forecasts=forecasts, verbose=True)
    df = la.get_obs_competition_dataframe()
    print(df)


def ends_freyberg_dev():
    import numpy as np
    import pyemu
    test_d = "ends_master"
    case = "freyberg6_run_ies"
    pst_name = os.path.join(test_d,case+".pst")
    pst = pyemu.Pst(pst_name)
    predictions = ["headwater_20171130","tailwater_20161130","trgw_0_9_1_20161130"]
    pst.pestpp_options["predictions"] = predictions

    # build up the obs to test info
    obs = pst.observation_data
    zobs = [o for o,w in zip(obs.obsnme,obs.weight) if w > 0 and o not in predictions]
    zgps = obs.loc[zobs,"obgnme"].unique()
    obslist_dict = {zg: obs.loc[obs.obgnme==zg,"obsnme"].tolist() for zg in zgps}
    # now reset these obs to have weights that reflect expected noise
    for zg,onames in obslist_dict.items():
        if "gage" in zg:
            obs.loc[onames,"weight"] = 1./(obs.loc[onames,"obsval"] * 0.1)
        else:
            obs.loc[onames,"weight"] = 2.0

    oe_name = pst_name.replace(".pst",".0.obs.csv")
    oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=oe_name)
    #oe = oe.iloc[:10,:]
    ends = pyemu.EnDS(pst=pst,sim_ensemble=oe)
    dfmean,dfstd,dfper = ends.get_posterior_prediction_moments(obslist_dict=obslist_dict,include_first_moment=False)
    print(dfstd)
    print(dfmean)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ax = dfstd.plot(kind='bar')
    ax.set_ylabel("percent variance reduction")
    ax.set_xlabel("new obs group")
    ax.set_title("ensemble variance analysis for three Freyberg predictions",loc="left")
    plt.tight_layout()
    plt.savefig("percent.pdf")
    plt.close("all")


    real_seq = np.linspace(3,200,20,dtype=int)
    print(real_seq)
    rep_seq = np.ones_like(real_seq) * 7
    results = ends.get_posterior_prediction_convergence(real_seq,rep_seq,obslist_dict=obslist_dict)
    fig,axes = plt.subplots(len(predictions),1,figsize=(10,10))
    groups = list(obslist_dict.keys())
    groups.sort()
    colors = ["c", "m", "b"]
    for p,ax in zip(predictions,axes):
        for i,nreal in enumerate(real_seq):

            for group,c in zip(groups,colors):
                label = None
                if i == 0:
                    label = group
                ax.scatter(nreal,results[nreal].loc[group,p],marker=".",s=30,c=c,label=label)
        ax.legend(loc="upper left")
        ax.set_title("prediction: "+p,loc="left")
        ax.set_xlabel("num reals")
        ax.set_ylabel("std")
    plt.tight_layout()
    plt.savefig("converge.pdf")
    plt.close(fig)

    mean_dfs, dfstd,dfper = ends.get_posterior_prediction_moments(obslist_dict=obslist_dict,sim_ensemble=oe.iloc[:100,:])
    fig,axes = plt.subplots(len(predictions),1,figsize=(10,10))
    groups = list(mean_dfs.keys())
    groups.sort()
    colors = ["c","m","b"]
    for p,ax in zip(predictions,axes):
        for group,c in zip(groups,colors):
            print(p,group,c)
            for i,mn in enumerate(mean_dfs[group].loc[:,p]):
                x,y = pyemu.plot_utils.gaussian_distribution(mn,dfstd.loc[group,p],100)
                label=None
                if i == 0:
                    label = "possible posterior with "+group
                #ax.fill_between(x,0,y,facecolor=c,alpha=0.05,label=label)
                ax.plot(x,y,color=c,lw=1.0,dashes=(1,1),label=label,zorder=1,alpha=0.5)
            axt = plt.twinx(ax)
        axt.hist(oe.loc[:,p],bins=20,facecolor="0.5",alpha=0.5,density=True,zorder=4)
        xlim,ylim = ax.get_xlim(),ax.get_ylim()

        r = Rectangle((xlim[0],ylim[0]),0,0,facecolor="0.5",alpha=0.5,label="Prior")
        ax.add_patch(r)
        ax.set_yticks([])
        axt.set_yticks([])
        ax.set_ylim(0,ax.get_ylim()[1])
        axt.set_ylim(0, axt.get_ylim()[1])

        ax.legend(loc="upper left")
        ax.set_title("prediction:"+p,loc="left")
    plt.tight_layout()
    plt.savefig("post.pdf")
    plt.close(fig)

    return
    ends = pyemu.EnDS(pst=pst_name, ensemble=oe_name)
    cov = pyemu.Cov.from_observation_data(pst)
    cov.to_uncfile(os.path.join(test_d, "obs.unc"), covmat_file=None)
    cov.to_binary(os.path.join(test_d,"cov.jcb"))
    cov.to_ascii(os.path.join(test_d, "cov.mat"))

    ends = pyemu.EnDS(pst=pst, ensemble=oe,obscov=cov)
    ends = pyemu.EnDS(pst=pst, ensemble=oe, obscov=os.path.join(test_d, "cov.mat"))
    ends = pyemu.EnDS(pst=pst, ensemble=oe, obscov=os.path.join(test_d, "obs.unc"))


def ends_freyberg_test(tmp_path):
    import pyemu
    test_d = "ends_master"
    test_d = setup_tmp(test_d, tmp_path)
    case = "freyberg6_run_ies"
    pst_name = os.path.join(test_d, case + ".pst")
    pst = pyemu.Pst(pst_name)
    predictions = ["headwater_20171130", "tailwater_20161130", "trgw_0_9_1_20161130"]
    pst.pestpp_options["predictions"] = predictions

    # build up the obs to test info
    obs = pst.observation_data
    zobs = [o for o, w in zip(obs.obsnme, obs.weight) if w > 0 and o not in predictions]
    zgps = obs.loc[zobs, "obgnme"].unique()
    obslist_dict = {zg: obs.loc[obs.obgnme == zg, "obsnme"].tolist() for zg in zgps}
    # now reset these obs to have weights that reflect expected noise
    for zg, onames in obslist_dict.items():
        if "gage" in zg:
            obs.loc[onames, "weight"] = 1. / (obs.loc[onames, "obsval"] * 0.1)
        else:
            obs.loc[onames, "weight"] = 2.0

    oe_name = pst_name.replace(".pst", ".0.obs.csv")
    oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=oe_name).iloc[:100,:]
    # oe = oe.iloc[:10,:]
    ends = pyemu.EnDS(pst=pst, sim_ensemble=oe)
    dfmean, dfstd, dfper = ends.get_posterior_prediction_moments(obslist_dict=obslist_dict,
                                                                 include_first_moment=True)
    assert len(dfmean) == len(obslist_dict),len(dfmean)
    for gp,df in dfmean.items():
        assert df.shape[0] == oe.shape[0]
    assert dfstd.columns.tolist() == predictions
    assert len(set(dfstd.index.tolist()[1:]).symmetric_difference(set(obslist_dict.keys()))) == 0

    ends = pyemu.EnDS(pst=pst_name, sim_ensemble=oe_name,predictions=predictions)
    cov = pyemu.Cov.from_observation_data(pst)
    cov.to_uncfile(os.path.join(test_d, "obs.unc"), covmat_file=None)
    cov.to_coo(os.path.join(test_d, "cov.jcb"))
    cov.to_ascii(os.path.join(test_d, "cov.mat"))

    ends = pyemu.EnDS(pst=pst, sim_ensemble=oe, obscov=cov,predictions=predictions)
    ends = pyemu.EnDS(pst=pst, sim_ensemble=oe, obscov=os.path.join(test_d, "cov.mat"),predictions=predictions)
    ends = pyemu.EnDS(pst=pst, sim_ensemble=oe, obscov=os.path.join(test_d, "obs.unc"),predictions=predictions)




if __name__ == "__main__":
    ends_freyberg_test("temp")
    #ends_freyberg_dev()
    #obscomp_test()
    #alternative_dw()
    #freyberg_verf_test()
    #forecast_pestpp_load_test()
    #map_test()
    #par_contrib_speed_test()
    #schur_test()
    #par_contrib_test()
    #dataworth_test()
    #dataworth_next_test()
    #schur_test_nonpest()
    #la_test_io()
    #errvar_test_nonpest()
    #errvar_test()
    #css_test()
    #inf_test()
    #inf2_test()
