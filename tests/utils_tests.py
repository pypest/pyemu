import os
if not os.path.exists("temp"):
    os.mkdir("temp")

def add_pi_obj_func_test():
    import os
    import pyemu

    pst = os.path.join("utils","dewater_pest.pst")
    pst = pyemu.optimization.add_pi_obj_func(pst,out_pst_name=os.path.join("utils","dewater_pest.piobj.pst"))
    print(pst.prior_information.loc["pi_obj_func","equation"])
    #pst._update_control_section()
    assert pst.control_data.nprior == 1


def fac2real_test():
    import os
    import numpy as np
    import pyemu
    # pp_file = os.path.join("utils","points1.dat")
    # factors_file = os.path.join("utils","factors1.dat")
    # pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
    #                               out_file=os.path.join("utils","test.ref"))

    pp_file = os.path.join("utils", "points2.dat")
    factors_file = os.path.join("utils", "factors2.dat")
    pyemu.utils.gw_utils.fac2real(pp_file, factors_file,
                                  out_file=os.path.join("utils", "test.ref"))
    arr1 = np.loadtxt(os.path.join("utils","fac2real_points2.ref"))
    arr2 = np.loadtxt(os.path.join("utils","test.ref"))

    #print(np.nansum(np.abs(arr1-arr2)))
    #print(np.nanmax(np.abs(arr1-arr2)))
    assert np.nanmax(np.abs(arr1-arr2)) < 0.01

    # import matplotlib.pyplot as plt
    # diff = (arr1-arr2)/arr1 * 100.0
    # diff[np.isnan(arr1)] = np.nan
    # p = plt.imshow(diff,interpolation='n')
    # plt.colorbar(p)
    # plt.show()

def vario_test():
    import numpy as np
    import pyemu
    contribution = 0.1
    a = 2.0
    for const in [pyemu.utils.geostats.ExpVario,pyemu.utils.geostats.GauVario,
                  pyemu.utils.geostats.SphVario]:

        v = const(contribution,a)
        h = v._h_function(np.array([0.0]))
        assert h == contribution
        h = v._h_function(np.array([a*1000]))
        assert h == 0.0

        v2 = const(contribution,a,anisotropy=2.0,bearing=90.0)
        print(v2._h_function(np.array([a])))


def aniso_test():

    import pyemu
    contribution = 0.1
    a = 2.0
    for const in [pyemu.utils.geostats.ExpVario,pyemu.utils.geostats.GauVario,
                  pyemu.utils.geostats.SphVario]:

        v = const(contribution,a)
        v2 = const(contribution,a,anisotropy=2.0,bearing=90.0)
        v3 = const(contribution,a,anisotropy=2.0,bearing=0.0)
        pt0 = (0,0)
        pt1 = (1,0)
        assert v.covariance(pt0,pt1) == v2.covariance(pt0,pt1)

        pt0 = (0,0)
        pt1 = (0,1)
        assert v.covariance(pt0,pt1) == v3.covariance(pt0,pt1)


def geostruct_test():
    import pyemu
    v1 = pyemu.utils.geostats.ExpVario(0.1,2.0)
    v2 = pyemu.utils.geostats.GauVario(0.1,2.0)
    v3 = pyemu.utils.geostats.SphVario(0.1,2.0)

    g = pyemu.utils.geostats.GeoStruct(0.2,[v1,v2,v3])
    pt0 = (0,0)
    pt1 = (0,0)
    print(g.covariance(pt0,pt1))
    assert g.covariance(pt0,pt1) == 0.5

    pt0 = (0,0)
    pt1 = (1.0e+10,0)
    assert g.covariance(pt0,pt1) == 0.2


def struct_file_test():
    import os
    import pyemu
    structs = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils","struct.dat"))
    #print(structs[0])
    pt0 = (0,0)
    pt1 = (0,0)
    for s in structs:
        assert s.covariance(pt0,pt1) == s.nugget + \
                                             s.variograms[0].contribution
    with open(os.path.join("utils","struct_out.dat"),'w') as f:
        for s in structs:
            s.to_struct_file(f)
    structs1 = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils","struct_out.dat"))
    for s in structs1:
        assert s.covariance(pt0,pt1) == s.nugget + \
                                             s.variograms[0].contribution


def covariance_matrix_test():
    import os
    import pandas as pd
    import pyemu

    pts = pd.read_csv(os.path.join("utils","points1.dat"),delim_whitespace=True,
                      header=None,names=["name","x","y"],usecols=[0,1,2])
    struct = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils","struct.dat"))[0]
    struct.variograms[0].covariance_matrix(pts.x,pts.y,names=pts.name)

    print(struct.covariance_matrix(pts.x,pts.y,names=pts.name).x)


def setup_ppcov_simple():
    import os
    import platform

    exe_file = os.path.join("utils","ppcov.exe")
    print(platform.platform())
    if not os.path.exists(exe_file) or not platform.platform().lower().startswith("win"):
        print("can't run ppcov setup")
        return
    pts_file = os.path.join("utils","points1_test.dat")
    str_file = os.path.join("utils","struct_test.dat")

    args1 = [pts_file,'0.0',str_file,"struct1",os.path.join("utils","ppcov.struct1.out"),'','']
    args2 = [pts_file,'0.0',str_file,"struct2",os.path.join("utils","ppcov.struct2.out"),'','']
    args3 = [pts_file,'0.0',str_file,"struct3",os.path.join("utils","ppcov.struct3.out"),'','']


    for args in [args1,args2,args3]:
        in_file = os.path.join("utils","ppcov.in")
        with open(in_file,'w') as f:
            f.write('\n'.join(args))
        os.system(exe_file + '<' + in_file)


def ppcov_simple_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    pts_file = os.path.join("utils","points1_test.dat")
    str_file = os.path.join("utils","struct_test.dat")

    mat1_file = os.path.join("utils","ppcov.struct1.out")
    mat2_file = os.path.join("utils","ppcov.struct2.out")
    mat3_file = os.path.join("utils","ppcov.struct3.out")

    ppc_mat1 = pyemu.Cov.from_ascii(mat1_file)
    ppc_mat2 = pyemu.Cov.from_ascii(mat2_file)
    ppc_mat3 = pyemu.Cov.from_ascii(mat3_file)

    pts = pd.read_csv(pts_file,header=None,names=["name","x","y"],usecols=[0,1,2],
                      delim_whitespace=True)

    struct1,struct2,struct3 = pyemu.utils.geostats.read_struct_file(str_file)
    print(struct1)
    print(struct2)
    print(struct3)

    for mat,struct in zip([ppc_mat1,ppc_mat2,ppc_mat3],[struct1,struct2,struct3]):

        str_mat = struct.covariance_matrix(x=pts.x,y=pts.y,names=pts.name)
        print(str_mat.row_names)
        delt = mat.x - str_mat.x
        assert np.abs(delt).max() < 1.0e-7


def setup_ppcov_complex():
    import os
    import platform

    exe_file = os.path.join("utils","ppcov.exe")
    print(platform.platform())
    if not os.path.exists(exe_file) or not platform.platform().lower().startswith("win"):
        print("can't run ppcov setup")
        return
    pts_file = os.path.join("utils","points1_test.dat")
    str_file = os.path.join("utils","struct_complex.dat")

    args1 = [pts_file,'0.0',str_file,"struct1",os.path.join("utils","ppcov.complex.struct1.out"),'','']
    args2 = [pts_file,'0.0',str_file,"struct2",os.path.join("utils","ppcov.complex.struct2.out"),'','']

    for args in [args1,args2]:
        in_file = os.path.join("utils","ppcov.in")
        with open(in_file,'w') as f:
            f.write('\n'.join(args))
        os.system(exe_file + '<' + in_file)


def ppcov_complex_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    pts_file = os.path.join("utils","points1_test.dat")
    str_file = os.path.join("utils","struct_complex.dat")

    mat1_file = os.path.join("utils","ppcov.complex.struct1.out")
    mat2_file = os.path.join("utils","ppcov.complex.struct2.out")

    ppc_mat1 = pyemu.Cov.from_ascii(mat1_file)
    ppc_mat2 = pyemu.Cov.from_ascii(mat2_file)

    pts = pd.read_csv(pts_file,header=None,names=["name","x","y"],usecols=[0,1,2],
                      delim_whitespace=True)

    struct1,struct2 = pyemu.utils.geostats.read_struct_file(str_file)
    print(struct1)
    print(struct2)

    for mat,struct in zip([ppc_mat1,ppc_mat2],[struct1,struct2]):

        str_mat = struct.covariance_matrix(x=pts.x,y=pts.y,names=pts.name)
        delt = mat.x - str_mat.x
        print(mat.x[:,0])
        print(str_mat.x[:,0])


        print(np.abs(delt).max())

        assert np.abs(delt).max() < 1.0e-7
        #break

def pp_to_tpl_test():
    import os
    import pyemu
    pp_file = os.path.join("utils","points1.dat")
    pp_df = pyemu.gw_utils.pilot_points_to_tpl(pp_file,name_prefix="test_")
    print(pp_df.columns)


def tpl_to_dataframe_test():
    import os
    import pyemu
    pp_file = os.path.join("utils","points1.dat")
    pp_df = pyemu.gw_utils.pilot_points_to_tpl(pp_file,name_prefix="test_")
    df_tpl = pyemu.gw_utils.pp_tpl_to_dataframe(pp_file+".tpl")
    assert df_tpl.shape[0] == pp_df.shape[0]

# def to_mps_test():
#     import os
#     import pyemu
#     jco_file = os.path.join("utils","dewater_pest.jcb")
#     jco = pyemu.Jco.from_binary(jco_file)
#     #print(jco.x)
#     pst = pyemu.Pst(jco_file.replace(".jcb",".pst"))
#     #print(pst.nnz_obs_names)
#     oc_dict = {oc:"l" for oc in pst.nnz_obs_names}
#     obj_func = {name:1.0 for name in pst.par_names}
#
#     #pyemu.optimization.to_mps(jco=jco_file)
#     #pyemu.optimization.to_mps(jco=jco_file,obs_constraint_sense=oc_dict)
#     #pyemu.optimization.to_mps(jco=jco_file,obj_func="h00_00")
#     decision_var_names = pst.parameter_data.loc[pst.parameter_data.pargp=="q","parnme"].tolist()
#     pyemu.optimization.to_mps(jco=jco_file,obj_func=obj_func,decision_var_names=decision_var_names,
#                               risk=0.975)

def setup_pp_test():
    import os
    import pyemu
    try:
        import flopy
    except:
        return
    model_ws = os.path.join("..","examples","Freyberg","extra_crispy")
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws)

    pp_dir = os.path.join("utils")
    ml.export(os.path.join(pp_dir,"test_unrot_grid.shp"))

    par_info_unrot = pyemu.gw_utils.setup_pilotpoints_grid(ml,prefix_dict={0:["hk1_","sy1_","rch_"]},
                                                     every_n_cell=2,pp_dir=pp_dir,tpl_dir=pp_dir,
                                                     shapename=os.path.join(pp_dir,"test_unrot.shp"))

    ml.sr.rotation = 15
    ml.export(os.path.join(pp_dir,"test_rot_grid.shp"))
    #pyemu.gw_utils.setup_pilotpoints_grid(ml)

    par_info_rot = pyemu.gw_utils.setup_pilotpoints_grid(ml,every_n_cell=2, pp_dir=pp_dir, tpl_dir=pp_dir,
                                                     shapename=os.path.join(pp_dir, "test_rot.shp"))

    print(par_info_unrot.x)
    print(par_info_rot.x)


def read_hob_test():
    import os
    import pyemu
    hob_file = os.path.join("utils","HOB.txt")
    pyemu.gw_utils.modflow_hob_to_instruction_file(hob_file)


def read_pval_test():
    import os
    import pyemu
    pval_file = os.path.join("utils", "meras_trEnhance.pval")
    pyemu.gw_utils.modflow_pval_to_template_file(pval_file)


def pp_to_shapefile_test():
    import os
    import pyemu
    try:
        import shapefile
    except:
        print("no pyshp")
        return
    pp_file = os.path.join("utils","points1.dat")
    shp_file = pp_file+".shp"
    pyemu.gw_utils.write_pp_shapfile(pp_file)

def write_tpl_test():
    import os
    import pyemu
    tpl_file = os.path.join("utils","test_write.tpl")
    in_file = os.path.join("utils","tpl_test.dat")
    par_vals = {"q{0}".format(i+1):12345678.90123456 for i in range(7)}
    pyemu.pst_utils.write_to_template(par_vals,tpl_file,in_file)


def read_pestpp_runstorage_file_test():
    import os
    import pyemu
    rnj_file = os.path.join("utils","freyberg.rnj")
    #rnj_file = os.path.join("..", "..", "verification", "10par_xsec", "master_opt1","pest.rnj")
    p1,o1 = pyemu.helpers.read_pestpp_runstorage(rnj_file)
    p2,o2 = pyemu.helpers.read_pestpp_runstorage(rnj_file,9)
    diff = p1 - p2
    diff.sort_values("parval1",inplace=True)

def smp_to_ins_test():
    import os
    import pyemu
    smp = os.path.join("utils","TWDB_wells.smp")
    ins = os.path.join('temp',"test.ins")
    try:
        pyemu.pst_utils.smp_to_ins(smp,ins)
    except:
        pass
    else:
        raise Exception("should have failed")
    pyemu.pst_utils.smp_to_ins(smp,ins,True)

def master_and_slaves():
    import shutil
    import pyemu
    slave_dir = os.path.join("..","verification","10par_xsec","template_mac")
    master_dir = os.path.join("temp","master")
    assert os.path.exists(slave_dir)
    #pyemu.helpers.start_slaves(slave_dir,"pestpp","pest.pst",1,
    #                           slave_root="temp",master_dir=master_dir)

    #now try it from within the master dir
    base_cwd = os.getcwd()
    os.chdir(master_dir)
    pyemu.helpers.start_slaves(os.path.join("..","..",slave_dir),
                              "pestpp","pest.pst",3,
                              master_dir='.')
    os.chdir(base_cwd)


def first_order_pearson_regul_test():
    import os
    from pyemu import Schur
    from pyemu.utils.helpers import first_order_pearson_tikhonov,zero_order_tikhonov
    w_dir = "la"
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"))
    zero_order_tikhonov(sc.pst)
    first_order_pearson_tikhonov(sc.pst,sc.posterior_parameter,reset=False)
    print(sc.pst.prior_information)
    sc.pst.rectify_pi()

def zero_order_regul_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","inctest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    print(pst.prior_information)

def kl_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    try:
        import flopy
    except:
        print("flopy not imported...")
        return
    model_ws = os.path.join("..","verification","Freyberg","extra_crispy")
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws)
    str_file = os.path.join("..","verification","Freyberg","structure.dat")
    arr_dict = {"test":np.ones((ml.nrow,ml.ncol))}
    arr_dict["hk_tru"] = np.loadtxt(os.path.join("..","verification",
                                                 "Freyberg","extra_crispy",
                                                 "hk.truth.ref"))
    basis_file = os.path.join("utils","basis.dat")
    tpl_file = os.path.join("utils","test.tpl")
    back_dict = pyemu.utils.helpers.kl_setup(num_eig=800,sr=ml.sr,
                                             struct_file =str_file,
                                             array_dict=arr_dict,
                                             basis_file=basis_file,
                                             tpl_file=tpl_file)
    for name in back_dict.keys():
        diff = np.abs((arr_dict[name] - back_dict[name])).sum()
        print(diff)
        assert np.abs(diff) < 1.0e-2
    df = pd.read_csv(tpl_file,skiprows=1)
    df.loc[:,"new_val"] = df.org_val
    df.to_csv(tpl_file.replace(".tpl",".csv"),index=False)
    #kl_appy(par_file, basis_file,par_to_file_dict)
    par_to_file_dict = {"test":os.path.join("utils","test.ref"),\
                        "hk_tru":os.path.join("utils","hk_tru.ref")}
    pyemu.utils.helpers.kl_apply(tpl_file.replace(".tpl",".csv"),basis_file,
                                 par_to_file_dict,(ml.nrow,ml.ncol))
    #import matplotlib.pyplot as plt
    for par,filename in par_to_file_dict.items():
        arr = np.loadtxt(filename)
        arr1 = arr_dict[par]
        # mx = max(arr.max(),arr1.max())
        # mn = min(arr.min(),arr1.min())
        # fig = plt.figure()
        # ax1,ax2 = plt.subplot(211),plt.subplot(212)
        # c1 = ax1.imshow(arr,vmin=mn,vmax=mx)
        # plt.colorbar(c1,ax=ax1)
        # c2 = ax2.imshow(arr_dict[par],vmin=mn,vmax=mx)
        # plt.colorbar(c2,ax=ax2)
        # plt.show()
        diff = np.abs((arr - arr1)).sum()
        print(diff)
        assert np.abs(diff) < 1.0e-2

def ok_test():
    import os
    import pandas as pd
    import pyemu
    str_file = os.path.join("utils","struct_test.dat")
    pts_data = pd.DataFrame({"x":[1.0,2.0,3.0],"y":[0.,0.,0.],"name":["p1","p2","p3"]})
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    interp_points = pts_data.copy()
    kf = ok.calc_factors(interp_points.x,interp_points.y)
    for ptname in pts_data.name:
        assert kf.loc[ptname,"inames"][0] == ptname
        assert kf.loc[ptname,"ifacts"][0] == 1.0
        assert sum(kf.loc[ptname,"ifacts"]) == 1.0
    print(kf)

def ok_grid_test():

    try:
        import flopy
    except:
        return

    import numpy as np
    import pandas as pd
    import pyemu
    nrow,ncol = 10,5
    delr = np.ones((ncol)) * 1.0/float(ncol)
    delc = np.ones((nrow)) * 1.0/float(nrow)

    num_pts = 0
    ptx = np.random.random(num_pts)
    pty = np.random.random(num_pts)
    ptname = ["p{0}".format(i) for i in range(num_pts)]
    pts_data = pd.DataFrame({"x":ptx,"y":pty,"name":ptname})
    pts_data.index = pts_data.name
    pts_data = pts_data.loc[:,["x","y","name"]]


    sr = flopy.utils.SpatialReference(delr=delr,delc=delc)
    pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
    pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
    str_file = os.path.join("utils","struct_test.dat")
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    kf = ok.calc_factors_grid(sr,verbose=False,var_filename=os.path.join("utils","test_var.ref"),minpts_interp=1)
    ok.to_grid_factors_file(os.path.join("utils","test.fac"))

def ok_grid_zone_test():

    try:
        import flopy
    except:
        return

    import numpy as np
    import pandas as pd
    import pyemu
    nrow,ncol = 10,5
    delr = np.ones((ncol)) * 1.0/float(ncol)
    delc = np.ones((nrow)) * 1.0/float(nrow)

    num_pts = 0
    ptx = np.random.random(num_pts)
    pty = np.random.random(num_pts)
    ptname = ["p{0}".format(i) for i in range(num_pts)]
    pts_data = pd.DataFrame({"x":ptx,"y":pty,"name":ptname})
    pts_data.index = pts_data.name
    pts_data = pts_data.loc[:,["x","y","name"]]


    sr = flopy.utils.SpatialReference(delr=delr,delc=delc)
    pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
    pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
    pts_data.loc[:,"zone"] = 1
    pts_data.zone.iloc[1] = 2
    print(pts_data.zone.unique())
    str_file = os.path.join("utils","struct_test.dat")
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    zone_array = np.ones((nrow,ncol))
    zone_array[0,0] = 2
    kf = ok.calc_factors_grid(sr,verbose=False,
                              var_filename=os.path.join("utils","test_var.ref"),
                              minpts_interp=1,zone_array=zone_array)
    ok.to_grid_factors_file(os.path.join("utils","test.fac"))


def ppk2fac_verf_test():
    import os
    import numpy as np
    import pyemu
    ws = os.path.join("..","verification","Freyberg")
    gspc_file = os.path.join(ws,"grid.spc")
    pp_file = os.path.join(ws,"pp_00_pp.dat")
    str_file = os.path.join(ws,"structure.complex.dat")
    ppk2fac_facfile = os.path.join(ws,"ppk2fac_fac.dat")
    pyemu_facfile = os.path.join("utils","pyemu_facfile.dat")
    sr = pyemu.utils.SpatialReference.from_gridspec(gspc_file)
    ok = pyemu.utils.OrdinaryKrige(str_file,pp_file)
    ok.calc_factors_grid(sr,maxpts_interp=10)
    ok.to_grid_factors_file(pyemu_facfile)
    zone_arr = np.loadtxt(os.path.join(ws,"extra_crispy","ref","ibound.ref"))

    pyemu_arr = pyemu.utils.fac2real(pp_file,pyemu_facfile,out_file=None)
    ppk2fac_arr = pyemu.utils.fac2real(pp_file,ppk2fac_facfile,out_file=None)
    pyemu_arr[zone_arr == 0] = np.NaN
    pyemu_arr[zone_arr == -1] = np.NaN
    ppk2fac_arr[zone_arr == 0] = np.NaN
    ppk2fac_arr[zone_arr == -1] = np.NaN

    diff = np.abs(pyemu_arr - ppk2fac_arr)
    print(diff)

    assert np.nansum(diff) < 1.0e-6,np.nansum(diff)
    


def opt_obs_worth():
    import os
    import pyemu
    wdir = os.path.join("utils")
    os.chdir(wdir)
    pst = pyemu.Pst(os.path.join("supply2_pest.fosm.pst"))
    zero_weight_names = [n for n,w in zip(pst.observation_data.obsnme,pst.observation_data.weight) if w == 0.0]
    #print(zero_weight_names)
    #for attr in ["base_jacobian","hotstart_resfile"]:
    #    pst.pestpp_options[attr] = os.path.join(wdir,pst.pestpp_options[attr])
    #pst.template_files = [os.path.join(wdir,f) for f in pst.template_files]
    #pst.instruction_files = [os.path.join(wdir,f) for f in pst.instruction_files]
    #print(pst.template_files)
    df = pyemu.optimization.get_added_obs_importance(pst,obslist_dict={"zeros":zero_weight_names})
    os.chdir("..")
    print(df)


def mflist_budget_test():
    import pyemu
    import os
    try:
        import flopy
    except:
        print("no flopy...")
        return
    model_ws = os.path.join("..","examples","Freyberg_transient")
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,check=False)
    list_filename = os.path.join(model_ws,"freyberg.list")
    assert os.path.exists(list_filename)
    pyemu.gw_utils.setup_mflist_budget_obs(list_filename,start_datetime=ml.start_datetime)


def pp_prior_builder_test():
    import os
    import pyemu
    pst = os.path.join("pst","pest.pst")
    tpl_file = os.path.join("utils","pp_locs.tpl")
    str_file = os.path.join("utils","structure.dat")
    pyemu.helpers.pilotpoint_prior_builder(pst,{str_file:tpl_file})

def linearuniversal_krige_test():
    try:
        import flopy
    except:
        return

    import numpy as np
    import pandas as pd
    import pyemu
    nrow,ncol = 10,5
    delr = np.ones((ncol)) * 1.0/float(ncol)
    delc = np.ones((nrow)) * 1.0/float(nrow)

    num_pts = 0
    ptx = np.random.random(num_pts)
    pty = np.random.random(num_pts)
    ptname = ["p{0}".format(i) for i in range(num_pts)]
    pts_data = pd.DataFrame({"x":ptx,"y":pty,"name":ptname})
    pts_data.index = pts_data.name
    pts_data = pts_data.loc[:,["x","y","name"]]


    sr = flopy.utils.SpatialReference(delr=delr,delc=delc)
    pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
    pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
    pts_data.loc["i0j0","value"] = 1.0
    pts_data.loc["imxjmx","value"] = 0.0

    str_file = os.path.join("utils","struct_test.dat")
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    luk = pyemu.utils.geostats.LinearUniversalKrige(gs,pts_data)
    df = luk.estimate_grid(sr,verbose=True,
                               var_filename=os.path.join("utils","test_var.ref"),
                               minpts_interp=1)

def gslib_2_dataframe_test():
    import os
    import pyemu
    gslib_file = os.path.join("utils","ch91pt.shp.gslib")
    df = pyemu.geostats.gslib_2_dataframe(gslib_file)
    print(df)

def sgems_to_geostruct_test():
    import os
    import pyemu
    xml_file = os.path.join("utils", "ch00")
    gs = pyemu.geostats.read_sgems_variogram_xml(xml_file)

def load_sgems_expvar_test():
    import os
    import numpy as np
    #import matplotlib.pyplot as plt
    import pyemu
    dfs = pyemu.geostats.load_sgems_exp_var(os.path.join("utils","ch00_expvar"))
    xmn,xmx = 1.0e+10,-1.0e+10
    for d,df in dfs.items():
        xmn = min(xmn,df.x.min())
        xmx = max(xmx,df.x.max())

    xml_file = os.path.join("utils", "ch00")
    gs = pyemu.geostats.read_sgems_variogram_xml(xml_file)
    v = gs.variograms[0]
    #ax = gs.plot(ls="--")
    #plt.show()
    #x = np.linspace(xmn,xmx,100)
    #y = v.inv_h(x)

    #
    #plt.plot(x,y)
    #plt.show()

def read_hydmod_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    try:
        import flopy
    except:
        return
    df, outfile = pyemu.gw_utils.modflow_read_hydmod_file(os.path.join('utils','freyberg.hyd.bin'),
                                                          os.path.join('utils','freyberg.hyd.bin.dat'))
    df = pd.read_csv(os.path.join('utils', 'freyberg.hyd.bin.dat'), delim_whitespace=True)
    dftrue = pd.read_csv(os.path.join('utils', 'freyberg.hyd.bin.dat.true'), delim_whitespace=True)

    assert np.allclose(df.obsval.as_matrix(), dftrue.obsval.as_matrix())

def make_hydmod_insfile_test():
    import os
    import pyemu
    try:
        import flopy
    except:
        return
    pyemu.gw_utils.modflow_hydmod_to_instruction_file(os.path.join('utils','freyberg.hyd.bin'))

    #assert open(os.path.join('utils','freyberg.hyd.bin.dat.ins'),'r').read() == open('freyberg.hyd.dat.ins', 'r').read()
    assert os.path.exists(os.path.join('utils','freyberg.hyd.bin.dat.ins'))

if __name__ == "__main__":
    #load_sgems_expvar_test()
    read_hydmod_test()
    make_hydmod_insfile_test()
    #gslib_2_dataframe_test()
    #sgems_to_geostruct_test()
    #linearuniversal_krige_test()
    #pp_prior_builder_test()
    #mflist_budget_test()
    #tpl_to_dataframe_test()
    #kl_test()
    # zero_order_regul_test()
    # first_order_pearson_regul_test()
    # master_and_slaves()
    # smp_to_ins_test()
    # read_pestpp_runstorage_file_test()
    # write_tpl_test()
    # pp_to_shapefile_test()
    # read_pval_test()
    # read_hob_test()
    # setup_pp_test()
    # to_mps_test()
    # pp_to_tpl_test()
    # setup_ppcov_complex()
    # ppcov_complex_test()
    # setup_ppcov_simple()
    # ppcov_simple_test()
    # fac2real_test()
    # vario_test()
    # geostruct_test()
    # aniso_test()
    # struct_file_test()
    # covariance_matrix_test()
    # add_pi_obj_func_test()
    # ok_test()
    #ok_grid_test()
    #ok_grid_zone_test()
    #opt_obs_worth()
    #ppk2fac_verf_test()