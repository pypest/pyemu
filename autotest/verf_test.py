import os
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pyemu

predictions =  ["sw_gw_0","sw_gw_1","or28c05_0","or28c05_1"]
post_mat = os.path.join("verf_results","post.cov")
verf_dir = "verf_results"
ord_base = os.path.join(verf_dir,"freyberg_ord")

if not os.path.exists("temp"):
    os.mkdir("temp")


def predunc7_test():
    post_pd7 = pyemu.Cov.from_ascii(post_mat)

    la_ord = pyemu.Schur(jco=ord_base+".jco",predictions=predictions)
    post_pyemu = la_ord.posterior_parameter
    delta_sum = np.abs((post_pd7 - post_pyemu).x).sum()
    print("delta matrix sum: {0:15.6E}".format(delta_sum))
    assert delta_sum < 1.0e-4


def predunc1_test():
    la_ord = pyemu.Schur(jco=ord_base+".jco",predictions=predictions)
    fsum = la_ord.get_forecast_summary()
    fsum.loc[:,["prior_var","post_var"]] = fsum.loc[:,["prior_var","post_var"]].apply(np.sqrt)
    # load the predunc1 results
    pd1_results = pd.read_csv(os.path.join(verf_dir,"predunc1_results.dat"))
    pd1_results.index = ["prior_var","post_var"]

    for forecast_name in fsum.index:
        pd1_pr,pd1_pt = pd1_results.loc[:,forecast_name]
        pr,pt = fsum.loc[forecast_name,["prior_var","post_var"]].values
        pr_diff = np.abs(pr - pd1_pr)
        pt_diff = np.abs(pt - pd1_pt)
        print("forecast:",forecast_name,"prior diff:{0:15.6E}".format(pr_diff),\
              "post diff:{0:15.6E}".format(pt_diff))
        assert pr_diff < 1.0e-3
        assert pt_diff < 1.0e-3


def predvar1b_test():

    out_files = [os.path.join(verf_dir,f) for f in os.listdir(verf_dir) if f.endswith(".out") and "ident" not in f]
    pv1b_results = {}
    for out_file in out_files:
        pred_name = os.path.split(out_file)[-1].split('.')[0]
        f = open(out_file,'r')
        for _ in range(3):
            f.readline()
        arr = np.loadtxt(f)
        pv1b_results[pred_name] = arr

    pst = pyemu.Pst(ord_base+".pst")
    omitted_parameters = [pname for pname in pst.parameter_data.parnme if pname.startswith("wf")]
    la_ord_errvar = pyemu.ErrVar(jco=ord_base+".jco",
                                 predictions=predictions,
                                 omitted_parameters=omitted_parameters,
                                 verbose=False)
    df = la_ord_errvar.get_errvar_dataframe(np.arange(36))

    max_idx = 13
    idx = np.arange(max_idx)
    for ipred,pred in enumerate(predictions):
        arr = pv1b_results[pred][:max_idx,:]
        first = df[("first", pred)][:max_idx]
        second = df[("second", pred)][:max_idx]
        third = df[("third", pred)][:max_idx]

        first_diff = (np.abs(arr[:,1] - first)).sum()
        second_diff = (np.abs(arr[:,2] - second)).sum()
        third_diff = (np.abs(arr[:,3] - third)).sum()
        print(pred,first_diff,second_diff,third_diff)
        assert first_diff < 1.5
        assert second_diff < 1.5
        assert third_diff < 1.5

def ident_test():

    idf = pd.read_csv(os.path.join(verf_dir, "ident.out"),
                      sep=r"\s+", index_col="parameter")

    la_ord_errvar = pyemu.ErrVar(jco=ord_base+".jco",
                                 predictions=predictions,
                                 verbose=False)
    df = la_ord_errvar.get_identifiability_dataframe(5)
    for pname in idf.index:
        ival = idf.loc[pname,"identifiability"]
        val = df.loc[pname,"ident"]
        diff = np.abs(ival - val)
        print(pname,ival,val)
        assert diff < 1.0E-3,"{0}:{1}".format(pname,diff)


def pnulpar_test():
    pst = pyemu.Pst(ord_base+".pst")
    # load the pnulpar projected ensemble
    d = os.path.join(verf_dir,"proj_par_draws")
    par_files = [ os.path.join(d,f) for f in os.listdir(d) if f.startswith("draw_")]
    pnul_en = pyemu.ParameterEnsemble.from_parfiles(pst=pst,parfile_names=par_files)
    #pnul_en.read_parfiles_prefix(os.path.join(verf_dir,"proj_par_draws","draw_"))
    pnul_en.loc[:,"fname"] = pnul_en.index
    #pnul_en.index = pnul_en.fname.apply(lambda x:str(int(x.split('.')[0].split('_')[-1])))
    f = pnul_en.pop("fname")

    mc = pyemu.MonteCarlo(jco=ord_base+".jco")
    d = os.path.join(verf_dir, "prior_par_draws")
    par_files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("draw_")]
    #mc.parensemble.read_parfiles_prefix(os.path.join(verf_dir,"prior_par_draws","draw_"))
    mc.parensemble = pyemu.ParameterEnsemble.from_parfiles(pst=mc.pst,parfile_names=par_files)
    mc.parensemble.loc[:,"fname"] = mc.parensemble.index
    #mc.parensemble.index = mc.parensemble.fname.apply(lambda x:str(int(x.split('.')[0].split('_')[-1])))
    f = mc.parensemble.pop("fname")

    en = mc.project_parensemble(nsing=1,inplace=False)

    diff = 100 * (np.abs(pnul_en - en) / en)
    assert max(diff.max()) < 1.0e-3


if __name__ == "__main__":
    #predunc7_test()
    #predunc1_test()
    #predvar1b_test()
    #ident_test()
    pnulpar_test()

