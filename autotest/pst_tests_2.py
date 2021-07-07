import os
import platform

if not os.path.exists("temp"):
    os.mkdir("temp")

def from_flopy_kl_test():
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    hfb_data = []
    jcol1, jcol2 = 14, 15
    for i in range(m.nrow):
        hfb_data.append([0, i, jcol1, i, jcol2, 0.001])
    flopy.modflow.ModflowHfb(m, 0, 0, len(hfb_data), hfb_data=hfb_data)

    org_model_ws = "temp"
    m.change_model_ws(org_model_ws)
    m.write_input()
    setattr(m,"sr",pyemu.helpers.SpatialReference(delc=m.dis.delc.array,delr=m.dis.delr.array))
    new_model_ws = "temp_pst_from_flopy"

    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    temp_list_props = [["wel.flux", None]]
    spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
    kl_props = [["upw.hk", 0], ["upw.vka", 0], ["rch.rech", 0]]
    ph = pyemu.helpers.PstFromFlopyModel(m, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         kl_props=kl_props,
                                         remove_existing=True,
                                         model_exe_name="mfnwt")


def from_flopy():
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    hfb_data = []
    jcol1, jcol2 = 14, 15
    for i in range(m.nrow):
        hfb_data.append([0, i, jcol1, i, jcol2, 0.001])
    flopy.modflow.ModflowHfb(m, 0, 0, len(hfb_data), hfb_data=hfb_data)

    org_model_ws = "temp"
    m.change_model_ws(org_model_ws)
    m.write_input()

    new_model_ws = "temp_pst_from_flopy"

    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    temp_list_props = [["wel.flux", None]]
    spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
    ph = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt", temporal_list_props=temp_list_props,
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
    os.chdir("..")

    ph = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt",
                                         spatial_list_props=spat_list_props)
    pe = ph.draw(100)

    ph = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         zone_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt", temporal_list_props=temp_list_props)
    pe = ph.draw(100)
    ph.pst.parameter_data.loc["rech0_zn1", "parval1"] = 2.0

    bd = os.getcwd()
    os.chdir(new_model_ws)
    # try:
    ph.pst.write_input_files()
    csv = os.path.join("arr_pars.csv")
    df = pd.read_csv(csv,index_col=0)
    df.loc[:, "upper_bound"] = np.NaN
    df.loc[:, "lower_bound"] = np.NaN
    df.to_csv(csv)
    pyemu.helpers.apply_array_pars()

    # jwhite 21 sept 2019 - the except here is no longer being
    # caught because of multiprocessing...
    # #df.loc[:, "org_file"] = df.org_file.iloc[0]
    # #df.loc[:, "model_file"] = df.org_file
    # df.loc[:, "upper_bound"] = np.arange(df.shape[0])
    # df.loc[:, "lower_bound"] = np.NaN
    # print(df)
    # df.to_csv(csv)
    # try:
    #     pyemu.helpers.apply_array_pars()
    # except:
    #     pass
    # else:
    #     raise Exception()
    # df.loc[:, "lower_bound"] = np.arange(df.shape[0])
    # df.loc[:, "upper_bound"] = np.NaN
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

    # except:
    #     pass
    os.chdir(bd)

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)

    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt", sfr_pars=True, sfr_obs=True,
                                             temporal_sfr_pars=True)
    pe = helper.draw(100)

    # go again testing passing list to sfr_pars
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)

    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)

    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             pp_props=pp_props, hds_kperk=[0, 0], remove_existing=True,
                                             model_exe_name="mfnwt")

    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, exe_name="mfnwt", check=False)
    const_props = [["rch.rech", i] for i in range(m.nper)]
    helper = pyemu.helpers.PstFromFlopyModel(m, new_model_ws,
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
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             grid_props=grid_props, hds_kperk=[0, 0], remove_existing=True)
    pe = helper.draw(100)
    # zones using ibound values - vka in layer 2
    zone_props = ["upw.vka", 1]
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             zone_props=zone_props, hds_kperk=[0, 0], remove_existing=True)
    pe = helper.draw(100)
    # kper-level multipliers for boundary conditions
    list_props = []
    for iper in range(m.nper):
        list_props.append(["wel.flux", iper])
        # list_props.append(["drn.elev",iper])
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             temporal_list_props=list_props, hds_kperk=[0, 0], remove_existing=True)

    pe = helper.draw(100)
    zn_arr = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "hk.zones"), dtype=int)
    k_zone_dict = {k: zn_arr for k in range(3)}

    obssim_smp_pairs = None
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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

    from_flopy_zone_pars()


def from_flopy_zone_pars():
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    m.change_model_ws(org_model_ws)
    m.write_input()

    new_model_ws = "temp_pst_from_flopy"
    grid_props = [["upw.ss", [0, 1]], ["upw.ss", 1], ["upw.ss", 2], ["extra.prsity", 0],
                ["rch.rech", 0], ["rch.rech", [1, 2]]]
    const_props = [["rch.rech", i] for i in range(m.nper)]
    grid_props = grid_props.extend(["extra.prsity", 0])
    zone_props = [["extra.prsity", 0], ["extra.prsity", 2], ["upw.vka", 1], ["upw.vka", 2]]

    zn_arr = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "hk.zones"), dtype=int)
    zn_arr2 = np.loadtxt(os.path.join("..", "examples", "Freyberg_Truth", "rand.zones"), dtype=int)

    pp_props = [["upw.hk", [0, 1]], ["extra.prsity", 1], ["upw.ss", 1], ["upw.ss", 2], ["upw.vka", 2]]
    k_zone_dict = {"upw.hk": {k: zn_arr for k in range(3)}, "extra.prsity": {k: zn_arr2 for k in range(3)},
                   "general_zn": {k: zn_arr for k in range(3)}}
    obssim_smp_pairs = None
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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



def from_flopy_test():
    bd = os.getcwd()
    try:
        from_flopy()
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in from_flopy: " + str(e))
    # print(os.getcwd())


def from_flopy_test_reachinput_test():
    bd = os.getcwd()
    try:
        from_flopy_reachinput()
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in from_flopy_reachinput: " + str(e))
    # print(os.getcwd())


def from_flopy_reachinput():
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
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_reaches")
    nam_file = "freyberg.nam"
    new_model_ws = "temp_pst_from_flopy_reaches"

    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
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
        helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
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
                raise Exception(str(e))
        os.chdir(bd)


def run_array_pars():
    import os
    import pyemu
    new_model_ws = "temp_pst_from_flopy"
    os.chdir(new_model_ws)
    pyemu.helpers.apply_array_pars()
    os.chdir('..')

def parrep_test():
    import pyemu
    import pandas as pd
    import numpy as np
    # make some fake parnames and values
    parnames = ['p_{0:03}'.format(i) for i in range(20)]
    np.random.seed(42)
    parvals = np.random.random(20) + 5
    parvals[0] = 0.001

    # make a fake parfile
    with open('fake.par','w') as ofp:
        ofp.write('single point\n')
        [ofp.write('{0:10s} {1:12.6f} 1.00 0.0\n'.format(i,j)) for i,j in zip(parnames,parvals)]
    
    # make a fake ensemble parameter file
    np.random.seed(99)
    parens = pd.DataFrame(np.tile(parvals,(5,1))+np.random.randn(5,20)*.5, columns=parnames)
    parens.index = list(range(4)) + ['base']
    parens.index.name = 'real_name'
    parens.loc['base'] = parvals[::-1]
    # get cheeky and reverse the column names to test updating
    parens.columns = parens.columns.sort_values(ascending = False)
    parens.to_csv('fake.par.0.csv')
    
    parens.drop('base').to_csv('fake.par.0.nobase.csv')
    # and make a fake pst file
    pst = pyemu.pst_utils.generic_pst(par_names=parnames)
    pst.parameter_data['parval1'] = [float(i+1) for i in range(len(parvals))]
    pst.parameter_data['parlbnd'] = 0.01
    pst.parameter_data['parubnd'] = 100.01
    
    pyemu.ParameterEnsemble(pst=pst,df=parens).to_binary('fake_parens.jcb')
    # test the parfile style
    pst.parrep('fake.par')
    assert pst.parameter_data.parval1[0] == pst.parameter_data.parlbnd[0]
    assert np.allclose(pst.parameter_data.iloc[1:].parval1.values,parvals[1:],atol=0.0001)
    assert pst.control_data.noptmax == 0
    pst.parrep('fake.par', noptmax=99, enforce_bounds=False)
    assert np.allclose(pst.parameter_data.parval1.values,parvals,atol=0.0001)
    assert pst.control_data.noptmax == 99
    
    # now test the ensemble style
    pst.parrep('fake.par.0.csv')
    assert pst.parameter_data.parval1[0] == pst.parameter_data.parlbnd[0]
    assert np.allclose(pst.parameter_data.iloc[1:].parval1.values,parvals[1:],atol=0.0001)

    pst.parrep('fake.par.0.nobase.csv')
    # flip the parameter ensemble back around
    parens = parens[parens.columns.sort_values()]
    assert np.allclose(pst.parameter_data.parval1.values[:-1],parens.T[0].values[:-1],atol=0.0001)

    pst.parrep('fake.par.0.csv', real_name=3)
    # flip the parameter ensemble back around
    parens = parens[parens.columns.sort_values()]
    assert np.allclose(pst.parameter_data.parval1.values[:-1],parens.T[3].values[:-1],atol=0.0001)

    pst.parrep('fake_parens.jcb', real_name=2)
    # confirm binary format works as csv did
    assert np.allclose(pst.parameter_data.parval1.values[:-1],parens.T[2].values[:-1],atol=0.0001)

def pst_from_flopy_geo_draw_test():
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    org_model_ws = "temp"
    m.change_model_ws(org_model_ws)
    m.write_input()

    new_model_ws = "temp_pst_from_flopy"

    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    temp_list_props = [["wel.flux", None]]
    spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
    ph = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
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




def from_flopy_pp_test():
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    m.change_model_ws("temp")
    ib = m.bas6.ibound.array
    ib[ib>0] = 3
    m.bas6.ibound = ib
    m.write_input()

    new_model_ws = "temp_pst_from_flopy"
    pp_props = [["upw.ss", [0, 1]],["upw.hk",[1,0]],["upw.vka",1]]

    obssim_smp_pairs = None
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, "temp",
                                             pp_props=pp_props,
                                             remove_existing=True,
                                             pp_space=4,
                                             use_pp_zones=False,
                                            build_prior=False)

    b_d = os.getcwd()
    os.chdir(new_model_ws)
    try:
        pyemu.helpers.apply_array_pars()
    except Exception as e:
        os.chdir(b_d)
        raise (str(e))
    os.chdir(b_d)


    mlt_dir = os.path.join(new_model_ws,"arr_mlt")
    for f in os.listdir(mlt_dir):
        arr = np.loadtxt(os.path.join(mlt_dir,f))
        assert np.all(arr==1)
    df = pd.read_csv(os.path.join(new_model_ws, "arr_pars.csv"), index_col=0)
    assert np.all(df.pp_fill_value.values == 1)

    new_model_ws = "temp_pst_from_flopy"
    props = ["upw.ss","upw.hk","upw.vka"]
    pp_props = []
    for k in range(m.nlay):
        for p in props:
            pp_props.append([p,k])
    #pp_props = [["upw.ss", [0,], ["upw.hk", [1, 0]], ["upw.vka", 1]]

    obssim_smp_pairs = None
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, "temp",
                                             pp_props=pp_props,
                                             remove_existing=True,
                                             pp_space=4,
                                             use_pp_zones=False,
                                             build_prior=True)



def pst_from_flopy_specsim_draw_test():
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    org_model_ws = "temp"
    m.change_model_ws(org_model_ws)
    m.write_input()

    new_model_ws = "temp_pst_from_flopy"

    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    temp_list_props = [["wel.flux", None]]
    spat_list_props = [["riv.cond", 0], ["riv.stage", 0]]
    v = pyemu.geostats.ExpVario(a=2500,contribution=1.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v],transform="log")
    ph = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         grid_props=[["rch.rech", 0], ["rch.rech", [1, 2]]],
                                         remove_existing=True,
                                         model_exe_name="mfnwt", temporal_list_props=temp_list_props,
                                         spatial_list_props=spat_list_props,build_prior=False,
                                         grid_geostruct=gs)

    num_reals = 10000
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

    pyemu.Ensemble.reseed()
    #print(ph.pst.parameter_data.loc[gr_par.parnme,"parval1"])
    #pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(ph.pst, ph.build_prior(sigma_range=2), num_reals=num_reals)
    pe2 = ph.draw(num_reals=num_reals,sigma_range=2)

    pe1.transform()
    pe2.transform()
    gr_df = ph.par_dfs[ph.gr_suffix]
    grps = gr_df.pargp.unique()
    gr_par = gr_df.loc[gr_df.pargp==grps[0],:]
    real1 = pe1.loc[pe1.index[-1],gr_par.parnme]
    real2 = pe2.loc[0, gr_par.parnme]

    arr = np.zeros((ph.m.nrow,ph.m.ncol))
    arr[gr_par.i,gr_par.j] = real1

    par_vals = par.parval1.copy()
    par_vals.loc[li] = par_vals.loc[li].apply(np.log10)
    mn1, mn2 = pe1.mean(), pe2.mean()
    sd1, sd2 = pe1.std(), pe2.std()
    diag = pyemu.Cov.from_parameter_data(ph.pst,sigma_range=2.0)
    var_vals = {p:np.sqrt(v) for p,v in zip(diag.row_names,diag.x)}
    for pname in par_vals.index:
        print(pname,par_vals[pname],mn1[pname],mn2[pname],var_vals[pname],sd1[pname],sd2[pname])

    diff_mn = mn1 - mn2
    diff_sd = sd1 - sd2
    #print(diff_mn)
    assert diff_mn.apply(np.abs).max() < 0.1, diff_mn.apply(np.abs).max()
    #print(sd1)
    #print(sd2)
    #print(diff_sd)
    assert diff_sd.apply(np.abs).max() < 0.1,diff_sd.apply(np.abs).sort_values()

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
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    org_phi = pst.phi
    #print(pst.res.loc[pst.nnz_obs_names,"residual"])
    pst.observation_data.loc[pst.nnz_obs_names, "obsval"] = pst.res.loc[pst.nnz_obs_names,"modelled"] - 1
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "g_test"
    assert pst.phi < 1.0e-6
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "greater_test"
    assert pst.phi < 1.0e-6
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "<@"
    assert pst.phi < 1.0e-6


    pst.observation_data.loc[pst.nnz_obs_names, "obsval"] = pst.res.loc[pst.nnz_obs_names, "modelled"] + 1
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "l_test"
    assert pst.phi < 1.0e-6
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "less_"
    assert pst.phi < 1.0e-6
    pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = ">@"
    assert pst.phi < 1.0e-6

    #pst.observation_data.loc[pst.nnz_obs_names, "obgnme"] = "l_test"
    #print(org_phi, pst.phi)


if __name__ == "__main__":

    #at_bounds_test()

    #pst_from_flopy_geo_draw_test()
    #pst_from_flopy_specsim_draw_test()
    # run_array_pars()
    # from_flopy_zone_pars()
    #from_flopy_pp_test()
    from_flopy()
    #parrep_test()
    #from_flopy_kl_test()
    #from_flopy_reachinput()
    #ineq_phi_test()


