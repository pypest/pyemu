import os
import shutil
from pathlib import Path
import pyemu
from pyemu.legacy import PstFromFlopyModel


def setup_tmp(od, tmp_path, sub=None):
    basename = Path(od).name
    if sub is not None:
        new_d = Path(tmp_path, basename, sub)
    else:
        new_d = Path(tmp_path, basename)
    if new_d.exists():
        shutil.rmtree(new_d)
    Path(tmp_path).mkdir(exist_ok=True)
    # creation functionality
    shutil.copytree(od, new_d)
    return new_d


def from_flopy_kl_test(tmp_path):
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        nam_file = "freyberg.nam"
        m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
        flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0]]})
        hfb_data = []
        jcol1, jcol2 = 14, 15
        for i in range(m.nrow):
            hfb_data.append([0, i, jcol1, i, jcol2, 0.001])
        flopy.modflow.ModflowHfb(m, 0, 0, len(hfb_data), hfb_data=hfb_data)

        m.external_path = '.'
        m.write_input()
        setattr(m,"sr",pyemu.helpers.SpatialReference(delc=m.dis.delc.array,delr=m.dis.delr.array))
        new_model_ws = "temp_pst_from_flopy1"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)

        hds_kperk = []
        for k in range(m.nlay):
            for kper in range(m.nper):
                hds_kperk.append([kper, k])
        temp_list_props = [["wel.flux", None]]
        spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
        kl_props = [["upw.hk", 0], ["upw.vka", 0], ["rch.rech", 0]]
        ph = PstFromFlopyModel(m, new_model_ws=new_model_ws,
                               org_model_ws=tmp_model_ws,
                               kl_props=kl_props,
                               remove_existing=True,
                               model_exe_name="mfnwt")
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def from_flopy(tmp_path):
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    hfb_data = []
    jcol1, jcol2 = 14, 15
    for i in range(m.nrow):
        hfb_data.append([0, i, jcol1, i, jcol2, 0.001])
    flopy.modflow.ModflowHfb(m, 0, 0, len(hfb_data), hfb_data=hfb_data)

    m.external_path = '.'
    m.write_input()

    new_model_ws = "temp_pst_from_flopy2"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)

    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    temp_list_props = [["wel.flux", None]]
    spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
    ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                           org_model_ws=tmp_model_ws,
                           zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                           remove_existing=True,
                           model_exe_name="mfnwt",
                           temporal_list_props=temp_list_props,
                           spatial_list_props=spat_list_props, hfb_pars=True)
    csv = os.path.join(new_model_ws, "arr_pars.csv")
    df = pd.read_csv(csv, index_col=0)
    mults_not_linked_to_pst = [f for f in df.mlt_file.unique()
                               if f not in ph.pst.input_files]
    assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)
    par = ph.pst.parameter_data
    pe = ph.draw(100)

    par.loc["welflux_000", 'parval1'] = 2.0

    os.chdir(new_model_ws)
    ph.pst.write_input_files()
    pyemu.helpers.apply_list_pars()
    os.chdir(tmp_path)
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)

    ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                           org_model_ws=tmp_model_ws,
                                         zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt",
                                         spatial_list_props=spat_list_props)
    pe = ph.draw(100)

    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=tmp_model_ws,
                                         zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt", temporal_list_props=temp_list_props)
    pe = ph.draw(100)
    ph.pst.parameter_data.loc["rech0_zn1", "parval1"] = 2.0

    os.chdir(new_model_ws)
    # try:
    ph.pst.write_input_files()
    csv = os.path.join("arr_pars.csv")
    df = pd.read_csv(csv,index_col=0)
    df.loc[:, "upper_bound"] = np.nan
    df.loc[:, "lower_bound"] = np.nan
    df.to_csv(csv)
    pyemu.helpers.apply_array_pars()

    # jwhite 21 sept 2019 - the except here is no longer being
    # caught because of multiprocessing...
    # #df.loc[:, "org_file"] = df.org_file.iloc[0]
    # #df.loc[:, "model_file"] = df.org_file
    # df.loc[:, "upper_bound"] = np.arange(df.shape[0])
    # df.loc[:, "lower_bound"] = np.nan
    # print(df)
    # df.to_csv(csv)
    # try:
    #     pyemu.helpers.apply_array_pars()
    # except:
    #     pass
    # else:
    #     raise Exception()
    # df.loc[:, "lower_bound"] = np.arange(df.shape[0])
    # df.loc[:, "upper_bound"] = np.nan
    # print(df)
    # df.to_csv(csv)
    # try:
    #     pyemu.helpers.apply_array_pars()
    # except:
    #     pass
    # else:
    #     raise Exception()

    df.loc[:, "lower_bound"] = 0.1
    df.loc[:, "upper_bound"] = 0.9
    print(df)
    df.to_csv(csv)

    pyemu.helpers.apply_array_pars()
    arr = np.loadtxt(df.model_file.iloc[0])
    assert arr.min() >= df.lower_bound.iloc[0]
    assert arr.max() <= df.upper_bound.iloc[0]
    os.chdir(bd)

    zn_arr = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "hk.zones"), dtype=int)
    org_model_ws = Path("..", "examples", "freyberg_sfr_update").absolute()
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    os.chdir(tmp_path)
    tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
    nam_file = "freyberg.nam"

    #m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    os.chdir(tmp_path)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt", sfr_pars=True, sfr_obs=True,
                                             temporal_sfr_pars=True)
    pe = helper.draw(100)

    # go again testing passing list to sfr_pars
    #m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    new_model_ws = "temp_pst_from_flopy2a"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt",
                                             sfr_pars=['flow', 'not_a_par'],
                                             temporal_sfr_pars=True,
                                             sfr_obs=True)
    try:
        pe = helper.draw(100)
    except:
        pass
    else:
        raise Exception()




    # go again passing bumph to sfr_par
    #m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    new_model_ws = "temp_pst_from_flopy2b"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt", sfr_pars=['not_a_par0', 'not_a_par1'], sfr_obs=True)
    try:
        pe = helper.draw(100)
    except:
        pass
    else:
        raise Exception()

    pp_props = [["upw.ss", [0, 1]], ["upw.ss", 1], ["upw.ss", 2], ["extra.prsity", 0], \
                ["rch.rech", 0], ["rch.rech", [1, 2]]]
    new_model_ws = "temp_pst_from_flopy2c"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             pp_props=pp_props, hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt")

    m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, exe_name="mfnwt", check=False)
    const_props = [["rch.rech", i] for i in range(m.nper)]
    new_model_ws = "temp_pst_from_flopy2d"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(m, new_model_ws,
                                             const_props=const_props, hds_kperk=[0, 0], remove_existing=True)
    pe = helper.draw(100)
    grid_props = [["extra.pr", 0]]
    for k in range(3):
        # grid scale pars for hk in all layers
        grid_props.append(["upw.hk", k])
        # const par for hk, ss, sy in all layers
        const_props.append(["upw.hk", k])
        const_props.append(["upw.ss", k])
        const_props.append(["upw.sy", k])
    new_model_ws = "temp_pst_from_flopy2e"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)

    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             grid_props=grid_props, hds_kperk=[0, 0], remove_existing=True)
    pe = helper.draw(100)
    # zones using ibound values - vka in layer 2
    zone_props = ["upw.vka", 1]
    new_model_ws = "temp_pst_from_flopy2f"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             zone_props=zone_props, hds_kperk=[0, 0], remove_existing=True)
    pe = helper.draw(100)
    # kper-level multipliers for boundary conditions
    list_props = []
    for iper in range(m.nper):
        list_props.append(["wel.flux", iper])
        # list_props.append(["drn.elev",iper])
    new_model_ws = "temp_pst_from_flopy2g"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             temporal_list_props=list_props, hds_kperk=[0, 0], remove_existing=True)

    pe = helper.draw(100)

    k_zone_dict = {k: zn_arr for k in range(3)}

    obssim_smp_pairs = None
    new_model_ws = "temp_pst_from_flopy2h"
    if os.path.exists(new_model_ws):
        shutil.rmtree(new_model_ws,ignore_errors=True)
    helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                             pp_props=pp_props,
                                             const_props=const_props,
                                             grid_props=grid_props,
                                             zone_props=zone_props,
                                             temporal_list_props=list_props,
                                             spatial_list_props=list_props,
                                             remove_existing=True,
                                             obssim_smp_pairs=obssim_smp_pairs,
                                             pp_space=4,
                                             use_pp_zones=False,
                                             k_zone_dict=k_zone_dict,
                                             hds_kperk=[0, 0], build_prior=False)
    pst = helper.pst
    par = pst.parameter_data
    par.loc[par.parubnd>100,"pariubnd"] = 100.0
    par.loc[par.parlbnd<0.1,"parlbnd"] = 0.1
    pe = helper.draw(100)
    obs = pst.observation_data
    obs.loc[:, "weight"] = 0.0
    obs.loc[obs.obsnme.apply(lambda x: x.startswith("cr")), "weight"] = 1.0
    obs.loc[obs.weight > 0.0, "obsval"] += np.random.normal(0.0, 2.0, pst.nnz_obs)
    pst.control_data.noptmax = 0
    pst.write(os.path.join(new_model_ws, "freyberg_pest.pst"))
    cov = helper.build_prior(fmt="none")
    cov.to_coo(os.path.join(new_model_ws, "cov.coo"))


def from_flopy_zone_pars_test(tmp_path):
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = Path("..", "examples", "freyberg_sfr_update").absolute()
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    zn_arr = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "hk.zones"), dtype=int)
    zn_arr2 = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "rand.zones"), dtype=int)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        nam_file = "freyberg.nam"
        m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
        new_model_ws = "temp_pst_from_flopy3"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)
        grid_props = [["upw.ss", [0, 1]], ["upw.ss", 1], ["upw.ss", 2], ["extra.prsity", 0],
                    ["rch.rech", 0], ["rch.rech", [1, 2]]]
        const_props = [["rch.rech", i] for i in range(m.nper)]
        grid_props = grid_props.extend(["extra.prsity", 0])
        zone_props = [["extra.prsity", 0], ["extra.prsity", 2], ["upw.vka", 1], ["upw.vka", 2]]

        pp_props = [["upw.hk", [0, 1]], ["extra.prsity", 1], ["upw.ss", 1], ["upw.ss", 2], ["upw.vka", 2]]
        k_zone_dict = {"upw.hk": {k: zn_arr for k in range(3)}, "extra.prsity": {k: zn_arr2 for k in range(3)},
                       "general_zn": {k: zn_arr for k in range(3)}}
        obssim_smp_pairs = None
        helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                                 const_props=const_props,
                                                 grid_props=grid_props,
                                                 zone_props=zone_props,
                                                 pp_props=pp_props,
                                                 remove_existing=True,
                                                 obssim_smp_pairs=obssim_smp_pairs,
                                                 pp_space=4,
                                                 use_pp_zones=True,
                                                 k_zone_dict=k_zone_dict,
                                                 hds_kperk=[0, 0], build_prior=False)

        k_zone_dict = {"upw.vka": {k: zn_arr for k in range(3)}, "extra.prsity": {k: zn_arr2 for k in range(3)}}
        new_model_ws = "temp_pst_from_flopy3b"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws, ignore_errors=True)
        helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                                 const_props=const_props,
                                                 grid_props=grid_props,
                                                 zone_props=zone_props,
                                                 pp_props=pp_props,
                                                 remove_existing=True,
                                                 obssim_smp_pairs=obssim_smp_pairs,
                                                 pp_space=4,
                                                 use_pp_zones=True,
                                                 k_zone_dict=k_zone_dict,
                                                 hds_kperk=[0, 0], build_prior=False)
        print(helper.pst.par_groups)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def from_flopy_test(tmp_path):
    bd = os.getcwd()
    try:
        from_flopy(tmp_path)
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in from_flopy: " + str(e))
    os.chdir(bd)
    # print(os.getcwd())


def from_flopy_test_reachinput_test(tmp_path):
    bd = os.getcwd()
    try:
        from_flopy_reachinput(tmp_path)
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in from_flopy_reachinput: " + str(e))
    os.chdir(bd)
    # print(os.getcwd())


def from_flopy_reachinput(tmp_path):
    import pandas as pd
    """ test for building sfr pars from reachinput sfr and seg pars across all kper"""
    try:
        import flopy
    except:
        return
    import pyemu

    # if platform.platform().lower().startswith('win'):
    #     tempchek = os.path.join("..", "..", "bin", "win", "tempchek.exe")
    # else:
    #     tempchek = None  # os.path.join("..", "..", "bin", "linux", "tempchek")

    bd = os.getcwd()
    org_model_ws = Path("..", "examples", "freyberg_sfr_reaches").absolute()
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    os.chdir(tmp_path)
    tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)

    # test passing different arguments
    args_to_test = [True,
                    ["strhc1", "flow"],
                    ["flow", "runoff"],
                    ["not_a_par", "not_a_par2"],
                    "strhc1",
                    ["strhc1", "flow", "runoff"]]
    for i, sfr_par in enumerate(args_to_test):  # if i=2 no reach pars, i==3 no pars, i=4 no seg pars
        for f in ["sfr_reach_pars.config", "sfr_seg_pars.config"]:  # clean up
            if os.path.exists(f):
                os.remove(f)
        if i < 5:
            include_temporal_pars = False
        else:
            include_temporal_pars = {'flow': [0], 'runoff': [2]}
        new_model_ws = "temp_pst_from_flopy_reachesa{0}".format(i)
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws, ignore_errors=True)
        helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                                 hds_kperk=[0, 0], remove_existing=True,
                                                 model_exe_name="mfnwt", sfr_pars=sfr_par,
                                                 temporal_sfr_pars=include_temporal_pars,
                                                 sfr_obs=True)
        os.chdir(new_model_ws)
        mult_files = []
        spars = {}
        try:  # read seg pars config file
            with open("sfr_seg_pars.config", 'r') as f:
                for line in f:
                    line = line.strip().split()
                    spars[line[0]] = line[1]
            mult_files.append(spars["mult_file"])
        except:
            if i in [3, 4]:  # for scenario 3 or 4 not expecting any seg pars
                pass
            else:
                raise Exception()
        rpars = {}
        try:  # read reach pars config file
            with open("sfr_reach_pars.config", 'r') as f:
                for line in f:
                    line = line.strip().split()
                    rpars[line[0]] = line[1]
            mult_files.append(rpars["mult_file"])
        except:
            if i in [2, 3]:  # for scenario 2 or 3 not expecting any reach pars
                pass
            else:
                raise Exception()
        try:
            # actually write out files to check template file
            helper.pst.write_input_files()
            try:
                exec(helper.frun_pre_lines[0])
            except Exception as e:
                raise Exception("error applying sfr pars, check tpl(s) and datafiles: {0}".format(str(e)))

            # test using tempchek for writing tpl file
            # par = helper.pst.parameter_data
            # if rpars == {}:
            #     par_file = "{}.par".format(spars['nam_file'])
            # else:
            #     par_file = "{}.par".format(rpars['nam_file'])
            # with open(par_file, 'w') as f:
            #     f.write('single point\n')
            #     f.flush()
            #     par[['parnme', 'parval1', 'scale', 'offset']].to_csv(f, sep=' ', header=False, index=False, mode='a')
            # if tempchek is not None:
                # for mult in mult_files:
                #     tpl_file = "{}.tpl".format(mult)
                #     try:
                #         pyemu.os_utils.run("{} {} {} {}".format(tempchek, tpl_file, mult, par_file))
                #     except Exception as e:
                #         raise Exception("error running tempchek on template file {1} and data file {0} : {2}".
                #                         format(mult, tpl_file, str(e)))
                # try:
                #     exec(helper.frun_pre_lines[0])
                # except Exception as e:
                #     raise Exception("error applying sfr pars, check tpl(s) and datafiles: {0}".format(str(e)))
        except Exception as e:
            if i == 3:  # scenario 3 should not set up any parameters
                pass
            else:
                raise e
        os.chdir(tmp_path)


def run_array_pars():
    import os
    import pyemu
    new_model_ws = "temp_pst_from_flopy4"
    os.chdir(new_model_ws)
    pyemu.helpers.apply_array_pars()
    os.chdir('..')


def pst_from_flopy_geo_draw_test(tmp_path):
    import shutil
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        nam_file = "freyberg.nam"
        m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
        flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0]]})
        m.external_path = '.'
        m.write_input()

        new_model_ws = "temp_pst_from_flopy5"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)

        hds_kperk = []
        for k in range(m.nlay):
            for kper in range(m.nper):
                hds_kperk.append([kper, k])
        temp_list_props = [["wel.flux", None]]
        spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
        ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                             org_model_ws=tmp_model_ws,
                                             zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                             remove_existing=True,
                                             model_exe_name="mfnwt", temporal_list_props=temp_list_props,
                                             spatial_list_props=spat_list_props)

        num_reals = 100000
        pe1 = ph.draw(num_reals=num_reals, sigma_range=6)
        pyemu.Ensemble.reseed()
        pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(ph.pst, ph.build_prior(sigma_range=6), num_reals=num_reals)

        mn1, mn2 = pe1.mean(), pe2.mean()
        sd1, sd2 = pe1.std(), pe2.std()

        diff_mn = mn1 - mn2
        diff_sd = sd1 - sd2
        # print(mn1,mn2)
        print(diff_mn)
        assert diff_mn.apply(np.abs).max() < 0.1
        print(diff_sd)
        assert diff_sd.apply(np.abs).max() < 0.1
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def from_flopy_pp_test(tmp_path):
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        nam_file = "freyberg.nam"
        m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
        ib = m.bas6.ibound.array
        ib[ib>0] = 3
        m.bas6.ibound = ib
        m.external_path = '.'
        m.write_input()

        new_model_ws = "temp_pst_from_flopy6"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)
        pp_props = [["upw.ss", [0, 1]],["upw.hk",[1,0]],["upw.vka",1]]

        obssim_smp_pairs = None
        helper = PstFromFlopyModel(nam_file, new_model_ws, tmp_model_ws,
                                                 pp_props=pp_props,
                                                 remove_existing=True,
                                                 pp_space=4,
                                                 use_pp_zones=False,
                                                build_prior=False)

        os.chdir(new_model_ws)
        pyemu.helpers.apply_array_pars()
        os.chdir(tmp_path)

        mlt_dir = os.path.join(new_model_ws,"arr_mlt")
        for f in os.listdir(mlt_dir):
            arr = np.loadtxt(os.path.join(mlt_dir,f))
            assert np.all(arr==1)
        df = pd.read_csv(os.path.join(new_model_ws, "arr_pars.csv"), index_col=0)
        assert np.all(df.pp_fill_value.values == 1)

        new_model_ws = "temp_pst_from_flopy7"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)
        props = ["upw.ss","upw.hk","upw.vka"]
        pp_props = []
        for k in range(m.nlay):
            for p in props:
                pp_props.append([p,k])
        #pp_props = [["upw.ss", [0,], ["upw.hk", [1, 0]], ["upw.vka", 1]]

        obssim_smp_pairs = None
        helper = PstFromFlopyModel(nam_file, new_model_ws,
                                                 tmp_model_ws,
                                                 pp_props=pp_props,
                                                 remove_existing=True,
                                                 pp_space=4,
                                                 use_pp_zones=False,
                                                 build_prior=True)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def pst_from_flopy_specsim_draw_test(tmp_path):
    import shutil
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        nam_file = "freyberg.nam"
        m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws, check=False)
        flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0],
                                                            [0, 0, 1, 31.0, 1.0, 25.0]]})
        m.external_path = '.'
        m.write_input()

        new_model_ws = "temp_pst_from_flopy8"
        if os.path.exists(new_model_ws):
            shutil.rmtree(new_model_ws,ignore_errors=True)

        hds_kperk = []
        for k in range(m.nlay):
            for kper in range(m.nper):
                hds_kperk.append([kper, k])
        temp_list_props = [["wel.flux", None]]
        spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
        v = pyemu.geostats.ExpVario(a=2500,contribution=1.0)
        gs = pyemu.geostats.GeoStruct(variograms=[v],transform="log")
        ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                             org_model_ws=tmp_model_ws,
                                             grid_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                             remove_existing=True,
                                             model_exe_name="mfnwt", temporal_list_props=temp_list_props,
                                             spatial_list_props=spat_list_props,build_prior=False,
                                             grid_geostruct=gs)

        num_reals = 5000
        par = ph.pst.parameter_data
        par.loc[:,"parval1"] = 1
        par.loc[:, "parubnd"] = 10
        par.loc[:, "parlbnd"] = .1

        #gr_par = par.loc[par.pargp.apply(lambda x: "gr" in x),:]
        #par.loc[gr_par.parnme,"parval1"] = 20#np.arange(1,gr_par.shape[0]+1)

        #par.loc[gr_par.parnme,"parubnd"] = 30#par.loc[gr_par.parnme,"parval1"].max()
        #par.loc[gr_par.parnme, "parlbnd"] = 0.001#par.loc[gr_par.parnme,"parval1"].min()
        #print(par.loc[gr_par.parnme,"parval1"])
        li = par.partrans == "log"

        pe1 = ph.draw(num_reals=num_reals, sigma_range=2,use_specsim=True)
        gr_df = ph.par_dfs[ph.gr_suffix]
        grps = gr_df.pargp.unique()
        gr_par = gr_df.loc[gr_df.pargp==grps[0],:]
        pe1.transform()
        mn1 = pe1.mean()
        sd1 = pe1.std()
        real1 = pe1.loc[pe1.index[-1],gr_par.parnme].copy()
        del pe1

        pyemu.Ensemble.reseed()
        #print(ph.pst.parameter_data.loc[gr_par.parnme,"parval1"])
        #pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(ph.pst, ph.build_prior(sigma_range=2), num_reals=num_reals)
        pe2 = ph.draw(num_reals=num_reals,sigma_range=2)
        pe2.transform()
        # real2 = pe2.loc[0, gr_par.parnme]
        mn2 = pe2.mean()
        sd2 = pe2.std()
        del pe2

        arr = np.zeros((ph.m.nrow,ph.m.ncol))
        arr[gr_par.i,gr_par.j] = real1

        par_vals = par.parval1.copy()
        par_vals.loc[li] = par_vals.loc[li].apply(np.log10)

        # diag = pyemu.Cov.from_parameter_data(ph.pst,sigma_range=2.0)
        # var_vals = {p:np.sqrt(v) for p,v in zip(diag.row_names,diag.x)}
        # for pname in par_vals.index:
        #     print(pname,par_vals[pname],mn1[pname],mn2[pname],var_vals[pname],sd1[pname],sd2[pname])

        diff_mn = mn1 - mn2
        diff_sd = sd1 - sd2
        #print(diff_mn)
        assert diff_mn.apply(np.abs).max() < 0.1, diff_mn.apply(np.abs).max()
        #print(sd1)
        #print(sd2)
        #print(diff_sd)
        assert diff_sd.apply(np.abs).max() < 0.1,diff_sd.apply(np.abs).sort_values()
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def at_bounds_test():
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    par = pst.parameter_data
    par.loc[pst.par_names[0],"parval1"] = par.parubnd[pst.par_names[0]] + 1.0
    par.loc[pst.par_names[1], "parval1"] = par.parlbnd[pst.par_names[1]]

    lb,ub = pst.get_adj_pars_at_bounds()
    assert len(lb) == 1
    assert len(ub) == 1


def ineq_phi_test():
    import pyemu
    import numpy as np

    def _check_adjust(cf, compgp, new, reset):
        cf.res.loc[pst.nnz_obs_names, "group"] = compgp
        cf.adjust_weights(obsgrp_dict={compgp: new})
        assert np.isclose(cf.phi_components[compgp], new)
        cf.adjust_weights(obsgrp_dict={compgp: reset})
        assert np.isclose(cf.phi_components[compgp], reset)

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    phi_comp=pst.phi_components
    #print(pst.res.loc[pst.nnz_obs_names,"residual"])
    res = pst.res
    swrgt = ((res.loc[res.residual > 0, 'residual'] *
              res.loc[res.residual > 0, 'weight'])**2).sum()
    swrlt = ((res.loc[res.residual < 0, 'residual'] *
              res.loc[res.residual < 0, 'weight'])**2).sum()
    for s in ["g_test", "greater_test", "<@"]:
        pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = s
        assert np.isclose(pst.phi_components[s], swrgt)
        _check_adjust(pst, s, 1000, swrgt)
    for s in ["l_test", "less_test", ">@"]:
        pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = s
        assert np.isclose(pst.phi_components[s], swrlt)
        _check_adjust(pst, s, 1000, swrlt)

    pst.observation_data.loc[
        pst.nnz_obs_names, "obsval"
    ] = pst.res.loc[pst.nnz_obs_names,"modelled"] - 1
    for s in ["g_test", "greater_test", "<@"]:
        pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = s
        assert pst.phi < 1.0e-6

    pst.observation_data.loc[
        pst.nnz_obs_names, "obsval"
    ] = pst.res.loc[pst.nnz_obs_names, "modelled"] + 1
    for s in ["l_test", "less_test", ">@"]:
        pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = s
        assert pst.phi < 1.0e-6

    #pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "l_test"
    #print(org_phi, pst.phi)


def test_pstfromflopy_deprecation():
    try:
        pyemu.helpers.PstFromFlopyModel()
    except DeprecationWarning:
        pass


def results_ies_1_test():
    import pyemu
    m_d = os.path.join("pst", "master_ies1")
    r = pyemu.Results(m_d=m_d)
    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    r = pst.master_ies1
    r = pst.r0
    ies = pst.ies
    mou = pst.mou

    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"),result_dir=m_d)

    df = pst.ies.get("paren",0)
    
    df = r.ies.rmr
    print(df)
    assert df is not None

    # get all change sum files in an multiindex df
    df = r.ies.pcs
    assert df is not None

    # same for conflicts across iterations
    df = r.ies.pdc
    assert df is not None

    # weights
    df = r.ies.weights
    
    assert df is not None
    assert df.index.dtype=='object'

    # various phi dfs
    df = r.ies.philambda
    assert df is not None
    df = r.ies.phigroup
    assert df is not None
    df = r.ies.phiactual
    assert df is not None
    #print(df)
    df = r.ies.phimeas
    assert df is not None
    # noise
    df = r.ies.noise
    assert df is not None
    # get the prior par en
    df = r.ies.paren0
    assert df is not None
    # get the 1st iter obs en
    df = r.ies.obsen1
    assert df is not None
    # get the combined par en across all iters
    df = r.ies.paren
    assert df is not None
    #print(df)


def results_ies_3_test():
    m_d1 = os.path.join("pst","master_ies1")
    m_d2 = os.path.join("pst", "master_ies2")
    pst = pyemu.Pst(os.path.join(m_d1,"pest.pst"))
    #pst.add_results(m_d1)
    pst.add_results(m_d2)

    ies0 = pst.r0.ies
    ies1 = pst.r1.ies
    ies = pst.ies
    assert len(ies) == 2
    ies00 = ies[0]

    pst = pyemu.Pst(os.path.join(m_d1, "pest.pst"))
    pst.add_results(m_d2)
    try:
        pst.add_results(m_d2)
    except Exception as e:
        pass
    else:
        raise Exception("should have failed...")

    pst = pyemu.Pst(os.path.join(m_d1, "pest.pst"))
    pst.add_results([m_d2],cases=["test"])
    #print(pst.r0.ies.paren)
    #print(pst.r0.ies.obsen)
    #print(pst.r1.ies.files_loaded)
    #print(pst.r1.ies.obsen)

    #print(pst.r1.ies.files_loaded)

def results_ies_2_test():
    import pyemu
    m_d = os.path.join("pst", "master_ies2")

    for case in ["test"]:
        r = pyemu.Results(m_d=m_d, case=case)

        # get all change sum files in an multiindex df
        df = r.ies.pcs
        assert df is not None

        # same for conflicts across iterations
        df = r.ies.pdc
        assert df is not None

        # weights
        df = r.ies.weights
        assert df is not None

        # various phi dfs
        df = r.ies.philambda
        assert df is not None
        df = r.ies.phigroup
        assert df is not None
        df = r.ies.phiactual
        assert df is not None
        df = r.ies.phimeas
        assert df is not None
        # noise
        df = r.ies.noise
        assert df is not None
        # get the prior par en
        df = r.ies.paren0
        assert df is not None
        # get the 1st iter obs en
        df = r.ies.obsen1
        assert df is not None
        # get the combined par en across all iters
        df = r.ies.paren
        assert df is not None

def results_mou_1_test():
    import pyemu
    for m_d in [os.path.join("pst", "zdt1_bin"),os.path.join("pst", "zdt1_ascii")]:
        r = pyemu.Results(m_d=m_d)

        df = r.mou.nestedparstack000
        #print(df)

        assert df is not None

        df = r.mou.parstack0
        #print(df)
        assert df is not None

        df = r.mou.stack_summary0
        #print(df)
        assert df is not None


        df = r.mou.chanceobspop1
        #print(df)
        assert df is not None

        df = r.mou.chanceobspop
        #print(df)
        assert df is not None

        df = r.mou.chancedvpop1
        #print(df)
        assert df is not None

        df = r.mou.chancedvpop
        #print(df)
        assert df is not None

        df = r.mou.dvpop
        assert df is not None

        df = r.mou.dvpop0
        #print(df)
        assert df is not None

        df = r.mou.obspop
        #print(df)
        assert df is not None

        df = r.mou.obspop3
        # print(df)
        assert df is not None

        df = r.mou.paretosum_archive
        #print(df)
        assert df is not None

        df = r.mou.paretosum
        #print(df)
        assert df is not None

        df = r.mou.archivedvpop
        #print(df)
        assert df is not None

        df = r.mou.archiveobspop
        #print(df)
        assert df is not None

def dialate_bound_test():
    import pyemu



if __name__ == "__main__":
    results_ies_3_test()
    results_ies_1_test()
    results_ies_2_test()
    results_mou_1_test()
    #at_bounds_test()

    #pst_from_flopy_geo_draw_test()
    #pst_from_flopy_specsim_draw_test()
    # run_array_pars()
    # from_flopy_zone_pars()
    #from_flopy_pp_test()
    #from_flopy()
    #parrep_test(".")
    #from_flopy_kl_test()
    #from_flopy_reachinput()
    #ineq_phi_test()



