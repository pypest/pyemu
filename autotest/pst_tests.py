import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def from_io_with_inschek_test():
    import os
    from pyemu import Pst,pst_utils
    # creation functionality
    dir = os.path.join("..","verification","10par_xsec","template_mac")
    pst = Pst(os.path.join(dir,"pest.pst"))


    tpl_files = [os.path.join(dir,f) for f in pst.template_files]
    out_files = [os.path.join(dir,f) for f in pst.output_files]
    ins_files = [os.path.join(dir,f) for f in pst.instruction_files]
    in_files = [os.path.join(dir,f) for f in pst.input_files]


    new_pst = Pst.from_io_files(tpl_files, in_files,
                                ins_files, out_files,
                                pst_filename=os.path.join("temp","test.pst"))
    print(new_pst.observation_data)
    return

def tpl_ins_test():
    import os
    from pyemu import Pst,pst_utils
    # creation functionality
    dir = os.path.join("..","verification","henry","misc")
    files = os.listdir(dir)
    tpl_files,ins_files = [],[]
    for f in files:
        if f.lower().endswith(".tpl") and "coarse" not in f:
            tpl_files.append(os.path.join(dir,f))
        if f.lower().endswith(".ins"):
            ins_files.append(os.path.join(dir,f))

    out_files = [f.replace(".ins",".junk") for f in ins_files]
    in_files = [f.replace(".tpl",".junk") for f in tpl_files]

    pst_utils.pst_from_io_files(tpl_files, in_files,
                                ins_files, out_files,
                                pst_filename=os.path.join("temp","test.pst"))
    return


def res_test():
    import os
    import numpy as np
    from pyemu import Pst,pst_utils
    # residual functionality testing
    pst_dir = os.path.join("pst")

    p = Pst(os.path.join(pst_dir,"pest.pst"))
    phi_comp = p.phi_components
    assert "regul_p" in phi_comp
    assert "regul_m" in phi_comp

    p.adjust_weights_resfile()

    d = np.abs(p.phi - p.nnz_obs)
    assert d < 1.0E-5
    p.adjust_weights(obsgrp_dict={"head": 50})
    assert np.abs(p.phi_components["head"] - 50) < 1.0e-6

    # get()
    new_p = p.get()
    new_p.prior_information = p.prior_information
    new_file = os.path.join("temp", "new.pst")
    new_p.write(new_file)

    p_load = Pst(new_file,resfile=p.resfile)
    for gname in p.phi_components:
        d = np.abs(p.phi_components[gname] - p_load.phi_components[gname])
        assert d < 1.0e-5

def pst_manip_test():
    import os
    from pyemu import Pst
    pst_dir = os.path.join("pst")
    org_path = os.path.join(pst_dir,"pest.pst")
    new_path = os.path.join("temp","pest1.pst")
    pst = Pst(org_path)
    pst.control_data.pestmode = "regularisation"
    pst.write(new_path)
    pst = Pst(new_path)
    pst.svd_data.maxsing = 1
    pst.write(new_path,update_regul=True)


def load_test():
    import os
    from pyemu import Pst,pst_utils
    pst_dir = os.path.join("pst")
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    # just testing all sorts of different pst files
    pst_files = os.listdir(pst_dir)
    exceptions = []
    load_fails = []
    for pst_file in pst_files:
        if pst_file.endswith(".pst"):
            print(pst_file)
            try:
                p = Pst(os.path.join(pst_dir,pst_file))
            except Exception as e:
                exceptions.append(pst_file + " read fail: " + str(e))
                load_fails.append(pst_file)
                continue
            out_name = os.path.join(temp_dir,pst_file)
           #p.write(out_name,update_regul=True)
            try:
                p.write(out_name,update_regul=True)
            except Exception as e:
                exceptions.append(pst_file + " write fail: " + str(e))
                continue
            try:
                p = Pst(out_name)
            except Exception as e:
                exceptions.append(pst_file + " reload fail: " + str(e))
                continue

    #with open("load_fails.txt",'w') as f:
    #    [f.write(pst_file+'\n') for pst_file in load_fails]
    if len(exceptions) > 0:
        raise Exception('\n'.join(exceptions))

def smp_test():
    import os
    from pyemu.pst.pst_utils import smp_to_dataframe,dataframe_to_smp,\
        parse_ins_file,smp_to_ins
    
    smp_filename = os.path.join("misc","gainloss.smp")
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df,smp_filename+".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename+".ins")
    print(len(obs_names))

    smp_filename = os.path.join("misc","sim_hds_v6.smp")
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df,smp_filename+".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename+".ins")
    print(len(obs_names))


def smp_dateparser_test():
    import os
    from pyemu.pst.pst_utils import smp_to_dataframe,dataframe_to_smp,\
        parse_ins_file,smp_to_ins

    smp_filename = os.path.join("misc","gainloss.smp")
    df = smp_to_dataframe(smp_filename,datetime_format= "%d/%m/%Y %H:%M:%S")
    print(df.dtypes)
    dataframe_to_smp(df,smp_filename+".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename+".ins")
    print(len(obs_names))

    smp_filename = os.path.join("misc","sim_hds_v6.smp")
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df,smp_filename+".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename+".ins")
    print(len(obs_names))


def tied_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir,"br_opt_no_zero_weighted.pst"))
    print(pst.tied_lines)
    pst.write(os.path.join("temp","pest_tied_tester_1.pst"))
    mc = pyemu.MonteCarlo(pst=pst)
    mc.draw(1)
    mc.write_psts(os.path.join("temp","tiedtest_"))

def derivative_increment_tests():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","inctest.pst"))
    pst.calculate_pertubations()



def pestpp_args_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir,"br_opt_no_zero_weighted.pst"))
    pst.pestpp_options["lambdas"] = "0.1,0.2,0.3"
    pst.write(os.path.join("temp","temp.pst"))
    pst = pyemu.Pst(os.path.join("temp","temp.pst"))
    print(pst.pestpp_options)


def reweight_test():
    import os
    import numpy as np
    from pyemu import Pst,pst_utils
    pst_dir = os.path.join("pst")
    p = Pst(os.path.join(pst_dir,"pest.pst"))
    obsgrp_dict = {"pred":1.0,"head":1.0,"conc":1.0}
    p.adjust_weights(obsgrp_dict=obsgrp_dict)
    assert np.abs(p.phi - 3.0) < 1.0e-5,p.phi

    obs = p.observation_data
    obs.loc[obs.obgnme=="pred","weight"] = 0.0
    assert np.abs(p.phi - 2.0) < 1.0e-5,p.phi

    obs_dict = {"pd_one":1.0,"pd_ten":1.0}
    p.adjust_weights(obs_dict=obs_dict)
    assert np.abs(p.phi - 4.0) < 1.0e-5,p.phi


def reweight_res_test():
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    print(pst.res.loc[pst.nnz_obs_names,:])
    print(pst.phi,pst.nnz_obs)
    pst.adjust_weights_resfile()
    print(pst.phi,pst.nnz_obs)
    assert np.abs(pst.phi - pst.nnz_obs) < 1.0e-6


def regul_rectify_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","inctest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    assert pst.prior_information.shape[0] == pst.npar_adj
    pst._update_control_section()
    assert pst.control_data.nprior == pst.prior_information.shape[0]
    fix_names = pst.adj_par_names[::2]
    pst.parameter_data.loc[fix_names,"partrans"] = "fixed"
    pst.rectify_pi()
    assert pst.prior_information.shape[0] == pst.npar_adj
    pst._update_control_section()
    assert pst.control_data.nprior == pst.prior_information.shape[0]


def nnz_groups_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir,"br_opt_no_zero_weighted.pst"))
    org_og = pst.obs_groups
    org_nnz_og = pst.nnz_obs_groups
    obs = pst.observation_data
    obs.loc[obs.obgnme==org_og[0],"weight"] = 0.0
    new_og = pst.obs_groups
    new_nnz_og = pst.nnz_obs_groups
    assert org_og[0] not in new_nnz_og


def regdata_test():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    phimlim = 10.0
    pst.reg_data.phimlim = phimlim
    pst.control_data.pestmode = "regularization"
    pst.write(os.path.join("temp","pest_regultest.pst"))
    pst_new = pyemu.Pst(os.path.join("temp","pest_regultest.pst"))
    assert pst_new.reg_data.phimlim == phimlim


def plot_flopy_par_ensemble_test():
    import shutil
    import numpy as np
    try:
        import flopy
    except:
        return
    try:
        import matplotlib.pyplot as plt
    except:
        print("error importing pyplot")
        return
    import pyemu
    org_model_ws = os.path.join("..", "examples", "Freyberg_transient")
    nam_file = "freyberg.nam"

    new_model_ws = "temp_pst_from_flopy"
    pp_props = [["upw.hk", 0], ["upw.hk", 1]]
    helper = pyemu.helpers.PstFromFlopyModel(nam_file, new_model_ws, org_model_ws,
                                             grid_props=pp_props, remove_existing=True,
                                             model_exe_name="mfnwt")
    mc = pyemu.MonteCarlo(pst=helper.pst)
    os.chdir(new_model_ws)
    mc.draw(10,cov=helper.parcov)

    pyemu.helpers.plot_flopy_par_ensemble(mc.pst, mc.parensemble, num_reals=None, model=helper.m)
    pyemu.helpers.plot_flopy_par_ensemble(mc.pst, mc.parensemble, num_reals=None)

    # try:
    #     import cartopy.crs as ccrs
    #     import cartopy.io.img_tiles as cimgt
    #
    #     import pyproj
    # except:
    #     return
    #
    # stamen_terrain = cimgt.StamenTerrain()
    # zoom = 8
    #
    # def fig_ax_gen():
    #     fig = plt.figure(figsize=(20,20))
    #     nrow,ncol = 5,4
    #     axes = []
    #     for i in range(nrow*ncol):
    #         ax = plt.subplot(nrow,ncol,i+1,projection=stamen_terrain.crs)
    #         ax.set_extent([97, 98.5, 29.5, 31.])
    #         ax.add_image(stamen_terrain,zoom=zoom)
    #
    #         axes.append(ax)
    #     return fig, axes
    #
    # pcolormesh_trans = ccrs.UTM(zone=14)
    # pyemu.helpers.plot_flopy_par_ensemble(mc.pst, mc.parensemble, num_reals=1,fig_axes_generator=fig_ax_gen,
    #                                       pcolormesh_transform=pcolormesh_trans)

    os.chdir("..")

def from_flopy_test():
    import shutil
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu
    org_model_ws = os.path.join("..","examples","Freyberg_transient")
    nam_file = "freyberg.nam"

    new_model_ws = "temp_pst_from_flopy"
    pp_props = [["upw.ss",[0,1]],["upw.ss",1],["upw.ss",2],["extra.prsity",0],\
                ["rch.rech",np.arange(182)],["rch.rech",np.arange(183,365)]]
    helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                    pp_props=pp_props,hds_kperk=[0,0],remove_existing=True,
                                             model_exe_name="mfnwt")

    m = flopy.modflow.Modflow.load(nam_file,model_ws=org_model_ws,exe_name="mfnwt")
    const_props = [["rch.rech",i] for i in range(365)]
    helper = pyemu.helpers.PstFromFlopyModel(m,new_model_ws,
                                    const_props=const_props,hds_kperk=[0,0],remove_existing=True)

    grid_props = [["extra.pr",0]]
    for k in range(3):
        #grid scale pars for hk in all layers
        grid_props.append(["upw.hk",k])
        # const par for hk, ss, sy in all layers
        const_props.append(["upw.hk",k])
        const_props.append(["upw.ss",k])
        const_props.append(["upw.sy",k])
    helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                    grid_props=grid_props,hds_kperk=[0,0],remove_existing=True)

    # zones using ibound values - vka in layer 2
    zone_props = ["upw.vka",1]
    helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                   zone_props=zone_props,hds_kperk=[0,0],remove_existing=True)

    # kper-level multipliers for boundary conditions
    bc_props = [["drn.cond",None]]
    for iper in range(365):
        bc_props.append(["wel.flux",iper])
        bc_props.append(["drn.elev",iper])
    helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                    bc_props=bc_props,hds_kperk=[0,0],remove_existing=True)


    zn_arr = np.loadtxt(os.path.join("..","examples","Freyberg_Truth","hk.zones"),dtype=int)
    k_zone_dict = {k:zn_arr for k in range(3)}


    obssim_smp_pairs = None
    helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                    pp_props=pp_props,
                                    const_props=const_props,
                                    grid_props=grid_props,
                                    zone_props=zone_props,
                                    bc_props=bc_props,
                                    remove_existing=True,
                                    obssim_smp_pairs=obssim_smp_pairs,
                                    pp_space=4,
                                    use_pp_zones=False,
                                    k_zone_dict=k_zone_dict,
                                    hds_kperk=[0,0])
    pst = helper.pst
    obs = pst.observation_data
    obs.loc[:,"weight"] = 0.0
    obs.loc[obs.obsnme.apply(lambda x: x.startswith("cr")),"weight"] = 1.0
    obs.loc[obs.weight>0.0,"obsval"] += np.random.normal(0.0,2.0,pst.nnz_obs)
    pst.control_data.noptmax = 0
    pst.write(os.path.join(new_model_ws,"freyberg_pest.pst"))



def run_array_pars():
    import os
    import pyemu
    new_model_ws = "temp_pst_from_flopy"
    os.chdir(new_model_ws)
    pyemu.helpers.apply_array_pars()
    os.chdir('..')

def add_pi_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.prior_information = pst.null_prior
    par_names = pst.parameter_data.parnme[:10]
    pst.add_pi_equation(par_names,coef_dict={par_names[1]:-1.0})
    pst.write(os.path.join("temp","test.pst"))

    pst = pyemu.Pst(os.path.join("temp","test.pst"))
    print(pst.prior_information)

def setattr_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.model_command = 'test'
    assert isinstance(pst.model_command,list)
    pst.model_command = ["test","test1"]
    assert isinstance(pst.model_command,list)
    pst.write(os.path.join("temp","test.pst"))
    pst = pyemu.Pst(os.path.join("temp","test.pst"))
    assert isinstance(pst.model_command,list)


def add_pars_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    npar = pst.npar
    tpl_file = os.path.join("temp","crap.in.tpl")
    with open(tpl_file,'w') as f:
        f.write("ptf ~\n")
        f.write("  ~junk1   ~\n")
        f.write("  ~ {0}  ~\n".format(pst.parameter_data.parnme[0]))
    pst.add_parameters(tpl_file,"crap.in",pst_path="temp")
    assert npar + 1 == pst.npar
    assert "junk1" in pst.parameter_data.parnme
    assert os.path.join("temp","crap.in") in pst.input_files
    assert os.path.join("temp","crap.in.tpl") in pst.template_files

def add_obs_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    nobs = pst.nobs
    ins_file = os.path.join("temp", "crap.out.ins")
    out_file = os.path.join("temp","crap.out")
    oval = 1234.56
    with open(ins_file, 'w') as f:
        f.write("pif ~\n")
        #f.write("  ~junk1   ~\n")
        #f.write("  ~ {0}  ~\n".format(pst.parameter_data.parnme[0]))
        f.write("l1 w  !{0}!\n".format("crap1"))
    with open(out_file,"w") as f:
        f.write("junk1  {0:8.2f} \n".format(oval))
    pst.add_observations(ins_file,out_file, pst_path="temp")
    assert nobs + 1 == pst.nobs
    assert "crap1" in pst.observation_data.obsnme
    assert os.path.join("temp", "crap.out") in pst.output_files,str(pst.output_files)
    assert os.path.join("temp", "crap.out.ins") in pst.instruction_files
    print(pst.observation_data.loc["crap1","obsval"], oval)

def test_write_input_files():
    import os
    import shutil
    import numpy as np
    import pyemu
    from pyemu import Pst, pst_utils
    # creation functionality
    dir = os.path.join("..", "verification", "10par_xsec", "template_mac")
    if os.path.exists("temp_dir"):
        shutil.rmtree("temp_dir")
    shutil.copytree(dir,"temp_dir")
    os.chdir("temp_dir")
    pst = Pst(os.path.join("pest.pst"))
    pst.write_input_files()
    arr1 = np.loadtxt(pst.input_files[0])
    print(pst.parameter_data.parval1)
    pst.parameter_data.loc[:,"parval1"] *= 10.0
    pst.write_input_files()
    arr2 = np.loadtxt(pst.input_files[0])
    assert (arr1 * 10).sum() == arr2.sum()
    os.chdir("..")


def res_stats_test():
    import os
    import pyemu

    import os
    import numpy as np
    from pyemu import Pst, pst_utils
    # residual functionality testing
    pst_dir = os.path.join("pst")

    p = pyemu.pst_utils.generic_pst(["p1"],["o1"])
    try:
        p.get_res_stats()
    except:
        pass
    else:
        raise Exception()

    p = Pst(os.path.join(pst_dir, "pest.pst"))
    phi_comp = p.phi_components
    #print(phi_comp)
    df = p.get_res_stats()
    assert np.abs(df.loc["rss","all"] - p.phi) < 1.0e-6,"{0},{1}".format(df.loc["rss","all"],p.phi)
    for pc in phi_comp.keys():
        assert phi_comp[pc] == p.phi_components[pc]

if __name__ == "__main__":
    #res_stats_test()
    #test_write_input_files()
    #add_obs_test()
    #add_pars_test()
    #setattr_test()
    # run_array_pars()
    #from_flopy_test()
    plot_flopy_par_ensemble_test()
    #add_pi_test()
    # regdata_test()
    # nnz_groups_test()
    # regul_rectify_test()
    # derivative_increment_tests()
    #tied_test()
    # smp_test()
    # smp_dateparser_test()
    #pst_manip_test()
    #tpl_ins_test()
    #load_test()
    #res_test()
    #smp_test()
    #from_io_with_inschek_test()
    #pestpp_args_test()
    #reweight_test()
    #reweight_res_test()