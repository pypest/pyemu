import os
import sys

#sys.path.append(os.path.join("..","pyemu"))
import pyemu
from pyemu.prototypes.moouu import EvolAlg, EliteDiffEvol, ParetoObjFunc
#path = os.path.join(os.getcwd(), 'autotest', 'moouu', '10par_xsec', '10par_xsec.pst')


if not os.path.exists("temp1"):
    os.mkdir("temp1")

def test_paretoObjFunc():
    import pyemu
    import os
    os.chdir(os.path.join('moouu', 'StochasticProblemSuite'))
    pst = pyemu.Pst('SRN.pst')
    obj_dict = {pst.obs_names[0]: 'min', pst.obs_names[1]: 'min'}
    dv_names = pst.par_names
    d_vars = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={dv: 'uniform'for dv in dv_names},
                                                      partial=True, num_reals=5)
    ga = EliteDiffEvol(pst=pst, num_slaves=4, verbose=True)
    ga.initialize(obj_func_dict=obj_dict, num_par_reals=5, num_dv_reals=5, dv_ensemble=d_vars)


def tenpar_test():
    import os
    import numpy as np
    import flopy
    import pyemu

    bd = os.getcwd()
    try:
        #os.chdir(os.path.join("moouu","10par_xsec"))
        os.chdir(os.path.join("moouu", 'StochasticProblemSuite'))
        csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
        [os.remove(csv_file) for csv_file in csv_files]
        #pst = pyemu.Pst("10par_xsec.pst")
        pst = pyemu.Pst('SRN.pst')
        obj_names = pst.nnz_obs_names
        # pst.observation_data.loc[pst.obs_names[0],"obgnme"] = "greaterthan"
        # pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0
        # pst.observation_data.loc[pst.obs_names[0], "obsval"] *= 0.85
        # pst.observation_data.loc[pst.obs_names[-1], "obgnme"] = "greaterthan"
        # pst.observation_data.loc[pst.obs_names[-1], "weight"] = 1.0
        # pst.observation_data.loc[pst.obs_names[-1], "obsval"] *= 0.85

        # pst.observation_data.loc["h01_10", "obgnme"] = "greaterthan"
        # pst.observation_data.loc["h01_10", "weight"] = 1.0
        #pst.observation_data.loc["h01_10", "obsval"] *= 0.85


        par = pst.parameter_data
        #par.loc[:,"partrans"] = "none"

        obj_dict = {}
        obj_dict[obj_names[0]] = "min"
        obj_dict[obj_names[1]] = "min"



        # testing for reduce method
        # oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst, num_reals=5000)
        # logger = pyemu.Logger("temp.log")
        # obj_func = evol_proto.ParetoObjFunc(pst,obj_dict,logger)
        # df = obj_func.reduce_stack_with_risk_shift(oe,50,0.05)
        #
        # import matplotlib.pyplot as plt
        # ax = plt.subplot(111)
        # oe.iloc[:, -1].hist(ax=ax)
        # ylim = ax.get_ylim()
        # val = df.iloc[0,-1]
        # ax.plot([val, val], ylim)
        # ax.set_ylim(ylim)
        # plt.show()
        # print(df.shape)
        # return
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "gaussian" for p in pst.adj_par_names[2:]},
                                                      num_reals=5,
                                                      partial=False)
        ea = EliteDiffEvol(pst, num_slaves=8, port=4005, verbose=True)

        dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in pst.adj_par_names[:2]},
                                                      num_reals=5,
                                                      partial=True)

        ea.initialize(obj_dict,num_dv_reals=5,num_par_reals=5,risk=0.5)
        ea.initialize(obj_dict, par_ensemble=pe, dv_ensemble=dv, risk=0.5)

        #ea.update()


        # test the infeas calcs
    #     oe = ea.obs_ensemble
    #     ea.obj_func.is_nondominated_continuous(oe)
    #     ea.obj_func.is_nondominated_kung(oe)
    #     is_feasible = ea.obj_func.is_feasible(oe)
    #     oe.loc[is_feasible.index,"feas"] = is_feasible
    #     obs = pst.observation_data
    #     for lt_obs in pst.less_than_obs_constraints:
    #         val = obs.loc[lt_obs,"obsval"]
    #         infeas = oe.loc[:,lt_obs] >= val
    #         assert np.all(~is_feasible.loc[infeas])
    #
    #     for gt_obs in pst.greater_than_obs_constraints:
    #         val = obs.loc[gt_obs,"obsval"]
    #         infeas = oe.loc[:,gt_obs] <= val
    #         assert np.all(~is_feasible.loc[infeas])
    #
    #     # test that the end members are getting max distance
    #     crowd_distance = ea.obj_func.crowd_distance(oe)
    #     for name,direction in ea.obj_func.obs_dict.items():
    #         assert crowd_distance.loc[oe.loc[:,name].idxmax()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:,name].idxmax()]
    #         assert crowd_distance.loc[oe.loc[:, name].idxmin()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:, name].idxmin()]
    except Exception as e:
        os.chdir(os.path.join("..",".."))
        raise Exception(str(e))

    os.chdir(os.path.join("..",".."))


def tenpar_dev():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import flopy
    import pyemu

    os.chdir(os.path.join("moouu","10par_xsec"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst("10par_xsec.pst")
    #obj_names = pst.nnz_obs_names
    obj_names = ["h01_04", "h01_06"]

    # pst.observation_data.loc[pst.obs_names[0],"obgnme"] = "greaterthan"
    # pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0
    # pst.observation_data.loc[pst.obs_names[0], "obsval"] *= 0.85
    # pst.observation_data.loc[pst.obs_names[-1], "obgnme"] = "greaterthan"
    # pst.observation_data.loc[pst.obs_names[-1], "weight"] = 1.0
    # pst.observation_data.loc[pst.obs_names[-1], "obsval"] *= 0.85

    # pst.observation_data.loc["h01_10", "obgnme"] = "lessthan"
    # pst.observation_data.loc["h01_10", "weight"] = 1.0
    # pst.observation_data.loc["h01_10", "obsval"] *= 0.85


    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"


    obj_dict = {}
    obj_dict[obj_names[0]] = "min"
    obj_dict[obj_names[1]] = "max"
    #obj_dict[obj_names[2]] = "max"

    # testing for reduce method
    # oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst, num_reals=5000)
    # logger = pyemu.Logger("temp.log")
    # obj_func = evol_proto.ParetoObjFunc(pst,obj_dict,logger)
    # df = obj_func.reduce_stack_with_risk_shift(oe,50,0.05)
    #
    # import matplotlib.pyplot as plt
    # ax = plt.subplot(111)
    # oe.iloc[:, -1].hist(ax=ax)
    # ylim = ax.get_ylim()
    # val = df.iloc[0,-1]
    # ax.plot([val, val], ylim)
    # ax.set_ylim(ylim)
    # plt.show()
    # print(df.shape)
    # return
    dv_names = pst.adj_par_names[2:]
    par_names = pst.adj_par_names[:2]
    par.loc[dv_names, "parlbnd"] = 1.0
    par.loc[dv_names, "parubnd"] = 5.0

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in par_names},
                                                 num_reals=1,
                                                 partial=False)


    dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in dv_names},
                                                  num_reals=10,
                                                  partial=True)

    dv.index = ["p_{0}".format(i) for i in range(dv.shape[0])]
    ea = EliteDiffEvol(pst, num_slaves=5, port=4005, verbose=True)

    ea.initialize(obj_dict,par_ensemble=pe,dv_ensemble=dv,risk=0.5)



    #ax = plt.subplot(111)
    obj_org = ea.obs_ensemble.loc[:, obj_names].copy()
    dom_org = ea.obj_func.is_nondominated(obj_org)


    axes = pd.plotting.scatter_matrix(obj_org)
    print(axes)
    axes[0,0].set_xlim(0,4)
    axes[1, 1].set_xlim(0, 4)
    axes[0,1].set_xlim(0,4)
    axes[0,1].set_ylim(0,4)
    axes[1, 0].set_xlim(0, 4)
    axes[1, 0].set_ylim(0, 4)

    #plt.show()
    plt.savefig("iter_{0:03d}.png".format(0))
    plt.close("all")
    for i in range(30):

        ea.update()

        obj = ea.obs_ensemble.loc[:, obj_names]
        axes = pd.plotting.scatter_matrix(obj)
        print(axes)
        axes[0, 0].set_xlim(0, 4)
        axes[1, 1].set_xlim(0, 4)
        axes[0, 1].set_xlim(0, 4)
        axes[0, 1].set_ylim(0, 4)
        axes[1, 0].set_xlim(0, 4)
        axes[1, 0].set_ylim(0, 4)

        plt.savefig("iter_{0:03d}.png".format(i+1))
        plt.close("all")

    os.system("ffmpeg -r 2 -i iter_%03d.png -loop 0 -final_delay 100 -y shhh.mp4")
    return
    ax = plt.subplot(111)
    colors = ['r','y','g','b','m']
    risks = [0.05,0.25,0.51,0.75,0.95]
    for risk,color in zip(risks,colors):
        ea.initialize(obj_dict,par_ensemble=pe,dv_ensemble=dv,risk=risk,dv_names=pst.adj_par_names[2:])
        oe = ea.obs_ensemble
        # call the nondominated sorting
        is_nondom = ea.obj_func.is_nondominated(oe)
        obj = oe.loc[:,obj_names]
        obj.loc[is_nondom,"is_nondom"] = is_nondom
        #print(obj)

        stack = ea.last_stack
        plt.scatter(stack.loc[:, obj_names[0]], stack.loc[:, obj_names[1]], color="0.5", marker='.',s=10, alpha=0.25)

        plt.scatter(obj.loc[:,obj_names[0]],obj.loc[:,obj_names[1]],color=color,marker='.',alpha=0.25,s=8)
        ind = obj.loc[is_nondom,:]
        #plt.scatter(ind.iloc[:, 0], ind.iloc[:, 1], color="m", marker='.',s=8,alpha=0.5)
        isfeas = ea.obj_func.is_feasible(oe)

        isf = obj.loc[isfeas,:]
        #plt.scatter(isf.iloc[:, 0], isf.iloc[:, 1], color="g", marker='.', s=30, alpha=0.5)
        both = [True if s and d else False for s,d in zip(is_nondom,isfeas)]
        both = obj.loc[both,:]
        plt.scatter(both.loc[:, obj_names[0]], both.loc[:, obj_names[1]], color=color, marker='+', s=90,alpha=0.5)

    ax.set_xlabel("{0}, dir:{1}".format(obj_names[0],obj_dict[obj_names[0]]))
    ax.set_ylabel("{0}, dir:{1}".format(obj_names[1], obj_dict[obj_names[1]]))
    plt.savefig("risk_compare.pdf")
    #plt.show()

    os.chdir(os.path.join("..",".."))


def setup_freyberg_transport():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import flopy
    import pyemu


    org_model_ws = os.path.join("..","examples","freyberg_sfr_reaches")
    new_model_ws = os.path.join("moouu","freyberg","temp")
    mf_nam = "freyberg.nam"
    mt_nam = "freyberg_mt.nam"

    mf = flopy.modflow.Modflow.load(mf_nam,model_ws=org_model_ws,verbose=True,version="mfnwt",exe_name="mfnwt")
    mf.dis.nper = 1
    mf.dis.perlen = 3650.0
    mf.external_path = '.'
    rdata = mf.sfr.reach_data
    #print(rdata)
    upstrm = 33
    dwstrm = 32.5
    total_length = mf.dis.delc.array.max() * mf.nrow
    slope = (upstrm - dwstrm) / total_length
    #print(rdata.dtype,slope)
    strtop = np.linspace(upstrm,dwstrm,40)
    #print(strtop)
    rdata["strtop"] = strtop
    rdata["slope"] = slope


    sdata = mf.sfr.segment_data[0]
    print(sdata)
    print(sdata.dtype)
    sdata["flow"][0] = 10000
    sdata["width1"][:] = 5.
    sdata["width2"][:] = 5.
    #sdata["hcond1"][:] *= 10.0

    mf.change_model_ws(new_model_ws,reset_external=True)
    mf.rch.rech[0] *= 0.1
    mf.wel.stress_period_data[0]["flux"][:] *= 0.1

    ib = mf.bas6.ibound[0].array
    drn_data = []
    for i in range(mf.nrow):
        for j in range(mf.ncol):
            if j == 15:
                continue
            if ib[i,j] < 0:
                print(mf.dis.botm.array[0,i,j])
                drn_data.append([0,i,j,33,10000.0])
    flopy.modflow.ModflowDrn(mf,stress_period_data=drn_data)
    ib[ib<0] = 1
    mf.bas6.ibound = ib

    #mf.upw.vka[1] *= 100.0
    mf.upw.hk *= 5.0

    mf.write_input()
    mf.run_model()

    hds = flopy.utils.HeadFile(os.path.join(new_model_ws,mf_nam.replace(".nam",".hds")),model=mf)
    hds.plot(colorbar=True)
    plt.show()

    mlist = flopy.utils.MfListBudget(os.path.join(new_model_ws,mf_nam.replace(".nam",".list")))
    df = mlist.get_dataframes(diff=True)[1]
    df.plot(kind="bar")
    plt.show()

    mt = flopy.mt3d.Mt3dms.load(mt_nam,model_ws=org_model_ws,verbose=True,exe_name="mt3dusgs",modflowmodel=mf)

    mt.btn.nper = 1
    mt.btn.perlen = 3650.0
    #mt.external_path = '.'
    mt.remove_package("SSM")
    spd = []
    ib = mf.bas6.ibound[0].array
    for i in range(mf.nrow):
        for j in range(mf.ncol):
            if ib[i,j] <= 0:
                continue
            spd.append([0,i,j,2.0,15])

    flopy.mt3d.Mt3dSsm(mt,crch=0.0,stress_period_data=spd,mxss=10000)
    flopy.mt3d.Mt3dRct(mt,ireact=1,rc1=0.0075,igetsc=0)
    mt.change_model_ws(new_model_ws,reset_external=True)
    mt.sft.nsfinit = 40
    mt.sft.nobssf = 40
    mt.sft.obs_sf = np.arange(mt.sft.nsfinit) + 1
    mt.write_input()
    mt.run_model()

    unc = flopy.utils.UcnFile(os.path.join(new_model_ws,"MT3D001.UCN"),model=mf)
    unc.plot(colorbar=True,masked_values=[1.0e30])
    plt.show()
    return new_model_ws


def setup_freyberg_pest_interface():
    import os
    import shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import flopy
    import pyemu

    def set_par_bounds(pst,new_model_ws):
        df_zn = pd.read_csv(os.path.join(new_model_ws, "arr_pars.csv"))
        df_zn = df_zn.loc[df_zn.suffix == "_zn", :]

        par = pst.parameter_data

        pr_pars = par.loc[par.parnme.apply(lambda x: "prst" in x), "parnme"]
        par.loc[pr_pars, "parubnd"] = 10.0
        par.loc[pr_pars, "parlbnd"] = 0.1

        pr_pars = par.loc[par.parnme.apply(lambda x: "al" in x), "parnme"]
        par.loc[pr_pars, "parubnd"] = 10.0
        par.loc[pr_pars, "parlbnd"] = 0.1

        rc_pars = par.loc[par.parnme.apply(lambda x: "rc1" in x), "parnme"]
        par.loc[rc_pars, "parubnd"] = 10.0
        par.loc[rc_pars, "parlbnd"] = 0.1

        rc_pars = par.loc[par.parnme.apply(lambda x: "scn" in x), "parnme"]
        par.loc[rc_pars, "parubnd"] = 1.5
        par.loc[rc_pars, "parlbnd"] = 0.5

        # vka zone bounds
        vka_names = par.loc[par.parnme.apply(lambda x: "vka" in x), "parnme"]
        par.loc[vka_names, "parubnd"] = 10.0
        par.loc[vka_names, "parlbnd"] = 0.1  # org model value is 10, so this is means lower bound is 1.0

        # rch zone bounds
        rch_names = par.loc[par.parnme.apply(lambda x: "rech" in x), 'parnme']
        par.loc[rch_names, "parlbnd"] = 0.8
        par.loc[rch_names, "parubnd"] = 1.2



    def write_ssm_tpl(ssm_file):

        f_in = open(ssm_file, 'r')
        tpl_file = ssm_file + ".tpl"
        f_tpl = open(tpl_file, 'w')
        f_tpl.write("ptf ~\n")
        while True:
            line = f_in.readline()
            if line == '':
                break
            f_tpl.write(line)
            if "stress period" in line.lower():
                # f_tpl.write(line)
                while True:
                    line = f_in.readline()
                    if line == '':
                        break
                    raw = line.strip().split()
                    i = int(raw[1]) - 1
                    j = int(raw[2]) - 1
                    pname = "k{0:02d}_{1:02d}".format(i, j)
                    tpl_str = " ~ {0}~".format(pname)
                    line = line[:30] + tpl_str + line[40:]
                    # print(line)
                    f_tpl.write(line)
        return tpl_file

    org_model_ws = setup_freyberg_transport()

    props = []
    paks = ["upw.hk","upw.vka","extra.prst","extra.rc11","extra.scn1"]
    for k in range(3):
        for p in paks:
            props.append([p,k])
    props.append(["rch.rech",0])


    ph = pyemu.helpers.PstFromFlopyModel("freyberg.nam",org_model_ws=org_model_ws,new_model_ws="template",remove_existing=True,
                                         grid_props=props,spatial_bc_props=["wel.flux",2],hds_kperk=[[0,0],[0,1],[0,2]],
                                         mflist_waterbudget=True,sfr_pars=True,build_prior=False,
                                         extra_post_cmds=["mt3dusgs freyberg_mt.nam >mt_stdout"],
                                         model_exe_name="mfnwt")

    pyemu.helpers.run("mfnwt freyberg.nam", cwd=ph.m.model_ws)
    mt = flopy.mt3d.Mt3dms.load("freyberg_mt.nam", model_ws=org_model_ws, verbose=True, exe_name="mt3dusgs")
    mt.external_path = '.'
    mt.change_model_ws("template",reset_external=True)
    mt.write_input()
    pyemu.helpers.run("mt3dusgs freyberg_mt.nam",cwd="template")

    tpl_file = write_ssm_tpl(os.path.join("template","freyberg_mt.ssm"))
    df_ssm = ph.pst.add_parameters(tpl_file,pst_path='.')
    ph.pst.parameter_data.loc[df_ssm.parnme,"partrans"] = "none"
    ph.pst.parameter_data.loc[df_ssm.parnme, "parval1"] = 1.0
    ph.pst.parameter_data.loc[df_ssm.parnme, "parlbnd"] = 0.0
    ph.pst.parameter_data.loc[df_ssm.parnme, "parubnd"] = 10.0

    # copy the original prsity array to arr_org
    new_model_ws = ph.m.model_ws
    [shutil.copy2(os.path.join(new_model_ws, f), os.path.join(new_model_ws, 'arr_org', f)) for f in
     os.listdir(new_model_ws) if "prsity" in f.lower()]

    [shutil.copy2(os.path.join(new_model_ws, f), os.path.join(new_model_ws, 'arr_org', f)) for f in
     os.listdir(new_model_ws) if "rc11" in f.lower()]

    [shutil.copy2(os.path.join(new_model_ws, f), os.path.join(new_model_ws, 'arr_org', f)) for f in
     os.listdir(new_model_ws) if "sconc" in f.lower()]


    # mod arr pars csv for prsity name len issue - 12 chars, seriously?
    df = pd.read_csv(os.path.join(new_model_ws, "arr_pars.csv"))
    df.loc[:, "model_file"] = df.model_file.apply(lambda x: x.replace("prst", "prsity"))
    df.loc[:, "org_file"] = df.org_file.apply(lambda x: x.replace("prst", "prsity"))
    df.loc[:, "model_file"] = df.model_file.apply(lambda x: x.replace("scn", "sconc"))
    df.loc[:, "org_file"] = df.org_file.apply(lambda x: x.replace("scn", "sconc"))
    df.loc[:, "upper_bound"] = np.NaN
    pr_rows = df.model_file.apply(lambda x: "prsity" in x)
    df.loc[pr_rows, "upper_bound"] = 0.35
    df.loc[pr_rows, "lower_bound"] = 0.01
    df.to_csv(os.path.join(new_model_ws, "arr_pars.csv"))

    bdir = os.getcwd()
    os.chdir(new_model_ws)

    mt_list = mt.name + ".list"
    frun_line, ins_files, df = pyemu.gw_utils.setup_mtlist_budget_obs(mt_list,
                                                                      start_datetime=ph.m.start_datetime)
    pst = ph.pst
    for ins_file in ins_files:
        pst.add_observations(ins_file, out_file=ins_file.replace(".ins", ""))
    pst.observation_data.loc[df.obsnme, "weight"] = 0.0
    pst.observation_data.loc[df.obsnme, "obgnme"] = "mtlist"
    ph.frun_post_lines.append(frun_line)
    pst.observation_data.loc[df.obsnme, "weight"] = 0.0
    pst.observation_data.loc[df.obsnme, "obgnme"] = "mtlist"

    ucn_kperk = []
    for k in range(ph.m.nlay):
        ucn_kperk.append([mt.nper - 1, k])

    fline, df = pyemu.gw_utils.setup_hds_obs("mt3d001.ucn", skip=1.0e+30, kperk_pairs=ucn_kperk,
                                             prefix="ucn1")
    ph.frun_post_lines.append(fline)

    ph.tmp_files.append("mt3d001.ucn")
    ucn1 = pst.add_observations(os.path.join("mt3d001.ucn.dat.ins"), os.path.join("mt3d001.ucn.dat"))

    all_times = np.cumsum(mt.btn.perlen.array)
    times = []
    for time in all_times:
        times.append(time)

    sft_obs = "freyberg_mt.sftcobs.out"
    df_sft = pyemu.gw_utils.setup_sft_obs(sft_obs, times=times, ncomp=1)
    new_obs_sft = pst.add_observations(sft_obs + '.processed.ins', sft_obs + ".processed")
    ph.frun_post_lines.append("pyemu.gw_utils.apply_sft_obs()")

    os.chdir(bdir)

    set_par_bounds(ph.pst,ph.m.model_ws)
    print(ph.pst.npar)

    ph.write_forward_run()
    ph.pst.parameter_data.loc[df_ssm.parnme,"partrans"] = "fixed"
    pe = ph.draw(1000)

    pe.enforce()
    pe.to_csv(os.path.join(ph.m.model_ws,"sweep_in.csv"))
    # tie nitrate loading rates by zones


    df_ssm.loc[:, "i"] = df_ssm.parnme.apply(lambda x: int(x[1:3]))
    df_ssm.loc[:, "j"] = df_ssm.parnme.apply(lambda x: int(x[-2:]))

    df_ssm.loc[:,"zone"] = df_ssm.apply(lambda x: zarr[x.i,x.j],axis=1)
    z_grps = df_ssm.groupby(df_ssm.zone).groups
    for z,znames in z_grps.items():
        ph.pst.parameter_data.loc[znames[0],"partrans"] = "none"
        ph.pst.parameter_data.loc[znames[1:],"partrans"] = "tied"
        ph.pst.parameter_data.loc[znames[1:], "partied"] = znames[0]
    print(np.unique(zarr))
    ph.pst.write(os.path.join("template", "freyberg.pst"))

    pyemu.os_utils.run("pestpp freyberg.pst",cwd=ph.m.model_ws)


def run_freyberg_par_sweep():

    pyemu.os_utils.start_slaves("template","pestpp-swp","freyberg.pst",20,master_dir="master_par_sweep")


def process_freyberg_par_sweep():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join("master_par_sweep","sweep_out.csv"),index_col=0)
    df.columns = df.columns.str.lower()

    oname = "sfrc40_1_03650.00"
    df.loc[:,"sfrc40_1_03650.00"].hist(bins=20)
    plt.show()
    vals = df.loc[:,oname].copy()
    vals.sort_values(inplace=True)
    print(vals)
    ninetyfith = int(df.shape[0] * 0.95)
    idx = vals.index.values
    idx_nf = idx[ninetyfith]
    val = vals.loc[idx_nf]
    par_df = pd.read_csv(os.path.join("template","sweep_in.csv"),index_col=0)
    pars = par_df.iloc[idx_nf,:]
    pst = pyemu.Pst(os.path.join("template","freyberg.pst"))
    pst.parameter_data.loc[:,"parval1"] = pars.loc[pst.par_names]

    pst.write(os.path.join("template","freyberg_nf.pst"))
    pyemu.os_utils.run("pestpp freyberg_nf.pst",cwd="template")
    pst = pyemu.Pst(os.path.join("template","freyberg_nf.pst"))

    assert np.abs(val - pst.res.loc[oname,"modelled"]) < 0.0001
    pst.observation_data.loc[:,"obsval"] = pst.res.loc[pst.obs_names,"modelled"]
    pst.write(os.path.join("template","freyberg_nf.pst"))
    pyemu.os_utils.run("pestpp freyberg_nf.pst",cwd="template")

    par = pst.parameter_data

    load_pars = set(
        par.loc[par.apply(lambda x: x.pargp == "pargp" and x.parnme.startswith("k"), axis=1), "parnme"].values)
    par.loc[par.parnme.apply(lambda x: x not in load_pars), "partrans"] = "fixed"
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, num_reals=100000)
    pe.to_csv(os.path.join("template", "dec_var_sweep_in.csv"))
    pst.pestpp_options["sweep_parameter_csv_file"] = "dec_var_sweep_in.csv"
    pst.write(os.path.join("template", "freyberg_nf.pst"))

def write_ssm_tpl(ssm_file):

    f_in = open(ssm_file,'r')
    tpl_file = ssm_file + ".tpl"
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    while True:
        line = f_in.readline()
        if line == '':
            break
        f_tpl.write(line)
        if "stress period" in line.lower():
            #f_tpl.write(line)
            while True:
                line = f_in.readline()
                if line == '':
                    break
                raw = line.strip().split()
                i = int(raw[1]) - 1
                j = int(raw[2]) - 1
                pname = "k{0:02d}_{1:02d}".format(i,j)
                tpl_str = "~{0}~ ".format(pname)
                line = line[:39] + tpl_str + line[48:]
                #print(line)
                f_tpl.write(line)




def run_freyberg_dec_var_sweep():
    import os
    import shutil
    import pyemu
    m_d = "master_dec_var_sweep"
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree("template",m_d)
    if os.path.exists("template_temp"):
        shutil.rmtree("template_temp")
    shutil.copytree("template","template_temp")
    os.remove(os.path.join("template_temp","dec_var_sweep_in.csv"))
    os.chdir(m_d)
    pyemu.os_utils.start_slaves(os.path.join("..","template_temp"),"pestpp-swp","freyberg_nf.pst",num_slaves=20,master_dir='.')


def process_freyberg_dec_var_sweep():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyemu

    df = pd.read_csv(os.path.join("master_dec_var_sweep","sweep_out.csv"),index_col=0)
    df.columns = df.columns.str.lower()
    print(df.shape)
    oname = "sfrc40_1_03650.00"
    oname2 = "gw_malo1c_19791230"
    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    ax.scatter(df.loc[:,oname],df.loc[:,oname2],marker='.',s=5,color="0.5",alpha=0.5)
    ax.set_xlabel("reach 40 concentration ($\\frac{mg}{l}$)")
    ax.set_ylabel("total nitrate mass loading ($kg$)")
    # plt.show()
    odict = {oname:"min",oname2:"max"}
    pst = pyemu.Pst(os.path.join("template","freyberg.pst"))
    logger = pyemu.Logger("temp.log")
    obj = pyemu.moouu.ParetoObjFunc(pst=pst,obj_function_dict=odict,logger=logger)
    nondom = obj.is_nondominated_kung(df)
    ax.scatter(df.loc[nondom,oname],df.loc[nondom,oname2],marker=".",s=12,color='b')
    #df.loc[:,oname].hist()
    plt.tight_layout()
    plt.savefig("freyberg_bruteforce_truth.pdf")
    plt.show()



def plot_freyberg_domain():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import flopy
    import pyemu

    pst = pyemu.Pst(os.path.join("template","freyberg.pst"))
    par = pst.parameter_data
    load_pars = set(par.loc[par.apply(lambda x: x.pargp == "pargp" and x.parnme.startswith("k"), axis=1), "parnme"].values)
    pst.parameter_data = par.loc[par.parnme.apply(lambda x: x not in load_pars), :]
    pst.parameter_data.loc[:,"partrans"] = "log"
    group_names = {}
    for group in pst.par_groups:
        k = None
        if "hk" in group:
            k = int(group[-1])
            tag = "horizontal hydraulic conductivity"
        elif "vk" in group:
            k = int(group[-1])
            tag = "vertical hydraulic conductivity"
        elif "prst" in group:
            k = int(group[-1])
            tag = "porosity"
        elif "scn" in group:
            k = int(group[-1])
            tag = "initial nitrate concentration"
        elif "rech" in group:
            k = 0
            tag = "recharge"
        elif "pargp" in group:
            tag = "surface-water/groundwater exchange conductance"
        elif "welflux" in group:
            k = 2
            tag = "abstraction rate"
        elif "flow" in group:
            tag = "surface-water inflow"
        elif "rc1" in group:
            k = int(group[-1])
            tag = "first-order nitrate decay"
        else:
            raise Exception(group)
        if k is not None:
            tag = tag + ' layer {0}'.format(k+1)
        group_names[group] = tag
    pst.write_par_summary_table("freyberg.pars.tex",sigma_range=6.0,group_names=group_names)


    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws="template",check=False,verbose=True)
    fig = plt.figure(figsize=(2.25, 3))
    ax = plt.subplot(111,aspect="equal")
    mm = flopy.plot.ModelMap(model=m,ax=ax)
    ib = m.bas6.ibound[0].array
    zarr = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "hk.zones"), dtype=int)
    zarr = np.ma.masked_where(ib==0,zarr)
    cmap = plt.get_cmap("viridis")
    cmap.set_bad("k",0.0)
    ibmask = ib.copy()
    ibmask = np.ma.masked_where(ibmask!=0,ibmask)


    mm.plot_array(zarr,cmap=cmap,alpha=0.5,edgecolor="none")
    cmap = plt.get_cmap("Greys_r")
    cmap.set_bad("k",0.0)
    mm.plot_array(ibmask,cmap=cmap)
    mm.plot_bc(package=m.wel,plotAll=True)
    mm.plot_bc(package=m.sfr,color='m')
    mm.plot_bc(package=m.drn,color='b')

    plt.tight_layout()
    plt.savefig("freyberg_domain.pdf")
    plt.show()



if __name__ == "__main__":
    #test_paretoObjFunc()


    #tenpar_test()
    #quick_tests()
    #tenpar_test()
    #tenpar_dev()
    #setup_freyberg_transport()
    setup_freyberg_pest_interface()
    #run_freyberg_par_sweep()
    #process_freyberg_par_sweep()
    #setup_freyberg_transport()
    #setup_freyberg_pest_interface()
    #run_freyberg_dec_var_sweep()
    process_freyberg_dec_var_sweep()
    plot_freyberg_domain()