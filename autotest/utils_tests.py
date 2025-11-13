import os
import shutil
import pytest
import platform
# if not os.path.exists("temp"):
#     os.mkdir("temp")
from pathlib import Path
import pandas as pd
import sys
sys.path.append("..")
import pyemu
from pst_from_tests import _get_port

from conftest import full_exe_ref_dict

exepath_dict = full_exe_ref_dict()
ies_exe_path = exepath_dict['pestpp-ies']
mf_exe_path = exepath_dict['mfnwt']
mf6_exe_path = exepath_dict['mf6']
pp_exe_path = exepath_dict['pestpp-glm']
usg_exe_path = exepath_dict["mfusg_gsi"]
mou_exe_path = exepath_dict["pestpp-mou"]

def add_pi_obj_func_test(tmp_path):
    import os

    pst = os.path.join("utils","dewater_pest.pst")
    pst = pyemu.optimization.add_pi_obj_func(
        pst,
        out_pst_name=os.path.join(tmp_path,"dewater_pest.piobj.pst")
    )
    print(pst.prior_information.loc["pi_obj_func","equation"])
    #pst._update_control_section()
    assert pst.control_data.nprior == 1

def fac2real_test(tmp_path):
    import os
    import numpy as np
    # pp_file = os.path.join("utils","points1.dat")
    # factors_file = os.path.join("utils","factors1.dat")
    # pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
    #                               out_file=os.path.join("utils","test.ref"))

    pp_file = os.path.join("utils", "points2.dat")
    factors_file = os.path.join("utils", "factors2.dat")
    pyemu.geostats.fac2real(pp_file, factors_file,
                            out_file=os.path.join(tmp_path, "test.ref"))
    arr1 = np.loadtxt(os.path.join("utils", "fac2real_points2.ref"))
    arr2 = np.loadtxt(os.path.join(tmp_path, "test.ref"))

    #print(np.nansum(np.abs(arr1-arr2)))
    #print(np.nanmax(np.abs(arr1-arr2)))
    nmax = np.nanmax(np.abs(arr1-arr2))
    assert nmax < 0.01

    # import matplotlib.pyplot as plt
    # diff = (arr1-arr2)/arr1 * 100.0
    # diff[np.isnan(arr1)] = np.nan
    # p = plt.imshow(diff,interpolation='n')
    # plt.colorbar(p)
    # plt.show()

def fac2real_wrapped_test(tmp_path):
    import os
    import numpy as np
    # pp_file = os.path.join("utils","points1.dat")
    # factors_file = os.path.join("utils","factors1.dat")
    # pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
    #                               out_file=os.path.join("utils","test.ref"))

    pp_file = os.path.join("misc", "pp.pts")
    factors_file = os.path.join("misc", "pp.fac")
    pyemu.geostats.fac2real(pp_file, factors_file,
                            out_file=os.path.join(tmp_path, "test.ref"))



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


def struct_file_test(tmp_path):
    import os
    import pyemu
    o_str_file = os.path.join("utils","struct.dat")
    str_file = os.path.join(tmp_path,"struct.dat")
    shutil.copy(o_str_file, str_file)
    structs = pyemu.utils.geostats.read_struct_file(str_file)
    #print(structs[0])
    pt0 = (0,0)
    pt1 = (0,0)
    for s in structs:
        assert s.covariance(pt0,pt1) == s.nugget + \
                                             s.variograms[0].contribution
    with open(os.path.join(tmp_path, "struct_out.dat"),'w') as f:
        for s in structs:
            s.to_struct_file(f)
    structs1 = pyemu.utils.geostats.read_struct_file(
            os.path.join(tmp_path,"struct_out.dat"))
    for s in structs1:
        assert s.covariance(pt0,pt1) == s.nugget + \
                                             s.variograms[0].contribution


def covariance_matrix_test():
    import os
    import pandas as pd
    import pyemu

    pts = pd.read_csv(os.path.join("utils", "points1.dat"), 
                      sep=r"\s+",
                      header=None, 
                      names=["name", "x", "y"], 
                      usecols=[0, 1, 2])
    struct = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils", "struct.dat"))[0]
    struct.variograms[0].covariance_matrix(pts.x, pts.y, 
                                           names=pts.name)

    print(struct.covariance_matrix(pts.x,pts.y,names=pts.name).x)


def setup_ppcov_simple(tmp_path):
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
        in_file = os.path.join(tmp_path, "utils", "ppcov.in")
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

    pts = pd.read_csv(pts_file, header=None,
                      names=["name", "x", "y"],
                      usecols=[0, 1, 2],
                      sep=r"\s+")

    struct1,struct2,struct3 = pyemu.utils.geostats.read_struct_file(str_file)
    print(struct1)
    print(struct2)
    print(struct3)

    for mat,struct in zip([ppc_mat1,ppc_mat2,ppc_mat3],[struct1,struct2,struct3]):

        str_mat = struct.covariance_matrix(x=pts.x,y=pts.y,names=pts.name)
        print(str_mat.row_names)
        delt = mat.x - str_mat.x
        assert np.abs(delt).max() < 1.0e-7



def setup_ppcov_complex(tmp_path):
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
        in_file = os.path.join(tmp_path, "utils","ppcov.in")
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

    pts = pd.read_csv(pts_file,
                      header=None,
                      names=["name", "x", "y"],
                      usecols=[0, 1, 2],
                      sep=r"\s+",)

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

def pp_to_tpl_test(tmp_path):
    import os
    import pyemu
    o_pp_file = os.path.join("utils","points1.dat")
    pp_file = os.path.join(tmp_path, "points1.dat")
    shutil.copy(o_pp_file, pp_file)
    pp_df = pyemu.pp_utils.pilot_points_to_tpl(pp_file,name_prefix="test_")
    print(pp_df.columns)


def tpl_to_dataframe_test(tmp_path):
    import os
    import pyemu
    o_pp_file = os.path.join("utils","points1.dat")
    pp_file = os.path.join(tmp_path, "points1.dat")
    shutil.copy(o_pp_file, pp_file)
    pp_df = pyemu.pp_utils.pilot_points_to_tpl(pp_file, name_prefix="test_")
    df_tpl = pyemu.pp_utils.pp_tpl_to_dataframe(pp_file+".tpl")
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

def setup_pp_test(tmp_path):
    import os
    import pyemu
    try:
        import flopy
    except:
        return
    o_model_ws = os.path.join("..","examples","Freyberg","extra_crispy")
    model_ws = os.path.join(tmp_path, "extra_crispy")
    shutil.copytree(o_model_ws, model_ws)
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,check=False)
    pp_dir = os.path.join(tmp_path)
    #ml.export(os.path.join("temp","test_unrot_grid.shp"))
    sr = pyemu.helpers.SpatialReference().from_namfile(
        os.path.join(ml.model_ws, ml.namefile),
        delc=ml.dis.delc, delr=ml.dis.delr)
    sr.rotation = 0.
    par_info_unrot = pyemu.pp_utils.setup_pilotpoints_grid(sr=sr, prefix_dict={0: "hk1",1:"hk2"},
                                                           every_n_cell=2, pp_dir=pp_dir, tpl_dir=pp_dir,
                                                           shapename=os.path.join(tmp_path, "test_unrot.shp"),
                                                           )
    #print(par_info_unrot.parnme.value_counts())
    gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(a=1000,contribution=1.0))
    ok = pyemu.geostats.OrdinaryKrige(gs,par_info_unrot)
    ok.calc_factors_grid(sr)
    
    sr2 = pyemu.helpers.SpatialReference.from_gridspec(
        os.path.join(ml.model_ws, "test.spc"), lenuni=2)
    par_info_drot = pyemu.pp_utils.setup_pilotpoints_grid(sr=sr2, prefix_dict={0: ["hk1_", "sy1_", "rch_"]},
                                                           every_n_cell=2, pp_dir=pp_dir, tpl_dir=pp_dir,
                                                           shapename=os.path.join(tmp_path, "test_unrot.shp"),
                                                           )
    ok = pyemu.geostats.OrdinaryKrige(gs, par_info_unrot)
    ok.calc_factors_grid(sr2)

    par_info_mrot = pyemu.pp_utils.setup_pilotpoints_grid(ml,prefix_dict={0:["hk1_","sy1_","rch_"]},
                                                     every_n_cell=2,pp_dir=pp_dir,tpl_dir=pp_dir,
                                                     shapename=os.path.join(tmp_path,"test_unrot.shp"))
    ok = pyemu.geostats.OrdinaryKrige(gs, par_info_unrot)
    ok.calc_factors_grid(sr)



    sr.rotation = 15
    #ml.export(os.path.join("temp","test_rot_grid.shp"))

    #pyemu.gw_utils.setup_pilotpoints_grid(ml)

    par_info_rot = pyemu.pp_utils.setup_pilotpoints_grid(sr=sr,every_n_cell=2, pp_dir=pp_dir, tpl_dir=pp_dir,
                                                     shapename=os.path.join(tmp_path, "test_rot.shp"))
    ok = pyemu.geostats.OrdinaryKrige(gs, par_info_unrot)
    ok.calc_factors_grid(sr)
    print(par_info_unrot.x)
    print(par_info_drot.x)
    print(par_info_mrot.x)
    print(par_info_rot.x)


def read_hob_test(tmp_path):
    import os
    import pyemu
    o_hob_file = os.path.join("utils","HOB.txt")
    hob_file = os.path.join(tmp_path,"HOB.txt")
    shutil.copy(o_hob_file, hob_file)
    df = pyemu.gw_utils.modflow_hob_to_instruction_file(hob_file)
    print(df.obsnme)


def read_pval_test(tmp_path):
    import os
    import pyemu
    import shutil
    o_pval_file = os.path.join("utils", "meras_trEnhance.pval")
    pval_file = os.path.join(tmp_path, "meras_trEnhance.pval")
    shutil.copy(o_pval_file, pval_file)
    pyemu.gw_utils.modflow_pval_to_template_file(pval_file)


def pp_to_shapefile_test(tmp_path):
    import os
    import pyemu
    try:
        import shapefile
    except:
        print("no pyshp")
        return
    o_pp_file = os.path.join("utils", "points1.dat")
    pp_file = os.path.join(tmp_path, "points1.dat")
    shutil.copy(o_pp_file, pp_file)
    shp_file = os.path.join(tmp_path, "points1.dat.shp")
    pyemu.pp_utils.write_pp_shapfile(pp_file, shp_file)
    df = pyemu.pp_utils.pilot_points_from_shapefile(shp_file)


def write_tpl_test(tmp_path):
    import os
    import pyemu
    o_tpl_file = os.path.join("utils","test_write.tpl")
    tpl_file = os.path.join(tmp_path, "test_write.tpl")
    shutil.copy(o_tpl_file,tpl_file)
    in_file = os.path.join(tmp_path,"tpl_test.dat")
    par_vals = {"q{0}".format(i+1):12345678.90123456 for i in range(7)}
    pyemu.pst_utils.write_to_template(par_vals,tpl_file,in_file)


def read_pestpp_runstorage_file_test(tmp_path):
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    #rnj_file = os.path.join("utils","freyberg.rnj")
    #rnj_file = os.path.join("..", "..", "verification", "10par_xsec", "master_opt1","pest.rnj")
    rns_file = os.path.join("utils","runstor.rns")
    p1,o1 = pyemu.helpers.read_pestpp_runstorage(rns_file,irun="all")
    p2 = pd.read_csv(os.path.join("utils","runstor.0.par.csv"),index_col=0)

    diff = np.abs(p1.loc[:,p2.columns].values - p2.values)
    print(diff.max())
    assert diff.max() < 1.0e-7

def smp_to_ins_test(tmp_path):
    import os
    import pyemu
    smp = os.path.join("utils","TWDB_wells.smp")
    ins = os.path.join(tmp_path,"test.ins")
    try:
        pyemu.pst_utils.smp_to_ins(smp,ins)
    except:
        pass
    else:
        raise Exception("should have failed")
    pyemu.smp_utils.smp_to_ins(smp,ins,True)

def master_and_workers(tmp_path):  # not run?!?
    import pyemu
    worker_dir = os.path.join("..","verification","10par_xsec","template_mac")
    master_dir = os.path.join(tmp_path,"master")
    if not os.path.exists(master_dir):
        os.mkdir(master_dir)
    assert os.path.exists(worker_dir)
    pyemu.helpers.start_workers(worker_dir,"pestpp","pest.pst",1,
                               worker_root=tmp_path,master_dir=master_dir, port=_get_port())

    #now try it from within the master dir
    base_cwd = os.getcwd()
    os.chdir(master_dir)
    worker_dir = Path(worker_dir).relative_to(master_dir)
    pyemu.helpers.start_workers(worker_dir,
                              "pestpp","pest.pst",3,
                              master_dir='.', port=_get_port())
    os.chdir(base_cwd)


def first_order_pearson_regul_test(tmp_path):
    import os
    from pyemu import Schur
    from pyemu.utils.helpers import first_order_pearson_tikhonov,zero_order_tikhonov
    w_dir = "la"
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"))
    pt = sc.posterior_parameter
    zero_order_tikhonov(sc.pst)
    first_order_pearson_tikhonov(sc.pst,pt,reset=False)

    print(sc.pst.prior_information)
    sc.pst.rectify_pi()
    assert sc.pst.control_data.pestmode == "regularization"
    sc.pst.write(os.path.join(tmp_path, 'test.pst'))

def zero_order_regul_test(tmp_path):
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","inctest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    print(pst.prior_information)
    assert pst.control_data.pestmode == "regularization"
    pst.write(os.path.join(tmp_path,'test.pst'))

    pyemu.helpers.zero_order_tikhonov(pst,reset=False)
    assert pst.prior_information.shape[0] == pst.npar_adj * 2


def kl_test(tmp_path):
    import os
    import numpy as np
    import pyemu
    import matplotlib.pyplot as plt
    try:
        import flopy
    except:
        print("flopy not imported...")
        return
    o_model_ws = os.path.join("..","verification","Freyberg","extra_crispy")
    model_ws = Path(tmp_path, "extra_crispy")
    shutil.copytree(o_model_ws, model_ws)

    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,check=False)
    o_str_file = os.path.join("..","verification","Freyberg","structure.dat")
    str_file = "structure.dat"
    shutil.copy(o_str_file, os.path.join(tmp_path, str_file))

    arr_tru = np.loadtxt(Path(model_ws, "hk.truth.ref")) + 20
    basis_file = "basis.jco"
    tpl_file = "test.tpl"
    factors_file = "factors.dat"
    num_eig = 100
    prefixes = ["hk1"]
    sr = pyemu.helpers.SpatialReference(delc=ml.dis.delc.array,delr=ml.dis.delr.array)
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        df = pyemu.utils.helpers.kl_setup(num_eig=num_eig, sr=sr,
                                          struct=str_file,
                                          factors_file=factors_file,
                                          basis_file=basis_file,
                                          prefixes=prefixes, islog=False,
                                          tpl_dir='.')

        basis = pyemu.Matrix.from_binary(basis_file)
        basis = basis[:, :num_eig]
        arr_tru = np.atleast_2d(arr_tru.flatten()).transpose()
        proj = np.dot(basis.T.x, arr_tru)[:num_eig]
        # proj.autoalign = False
        back = np.dot(basis.x, proj)

        back = back.reshape(ml.nrow, ml.ncol)
        df.parval1 = proj
        arr = pyemu.geostats.fac2real(df, factors_file, out_file=None)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)

    fig = plt.figure(figsize=(10, 10))
    ax1, ax2 = plt.subplot(121),plt.subplot(122)
    mn,mx = arr_tru.min(),arr_tru.max()
    print(arr.max(), arr.min())
    print(back.max(),back.min())
    diff = np.abs(back - arr)
    print(diff.max())
    assert diff.max() < 1.0e-5


def ok_test(tmp_path):
    import os
    import pandas as pd
    import pyemu
    import numpy as np
    o_str_file = os.path.join("utils","struct_test.dat")
    str_file = os.path.join(tmp_path, "struct_test.dat")
    shutil.copy(o_str_file, str_file)
    pts_data = pd.DataFrame({"x":[1.0,2.0,3.0],"y":[0.,0.,0.],"name":["p1","p2","p3"]})
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    interp_points = pts_data.copy()
    kf = ok.calc_factors(interp_points.x,interp_points.y)
    #for ptname in pts_data.name:
    for i in kf.index:
        assert len(kf.loc[i,"inames"])== 1
        assert kf.loc[i,"ifacts"][0] == 1.0
        assert sum(kf.loc[i,"ifacts"]) == 1.0
    print(kf)

    # evaluate the negative factor correction
    # set up some points with a cluster far from a single points - this triggers some negative factors for testing
    pts_data = pd.DataFrame({"x":[1.0,1.0,1.0,1.0,2.0,3.0, 3500.0],"y":[0.,0.1,-0.1,0.001,0.,0.,0.],"name":["p1","p2","p3","p4","p5","p6","p7"]})     
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    # evaluate factors with and without negative correction
    kf2_nocorr = ok.calc_factors([1000.0],[0.],remove_negative_factors=False)
    kf2_corr = ok.calc_factors([1000.0],[0.])
    
    print(kf2_nocorr)
    print(kf2_corr)
    # do some checking here
    # do all the factors sum to unity with and without correction?
    assert np.isclose(kf2_nocorr.iloc[0].ifacts.sum(), 1,atol=1e-6)
    assert np.isclose(kf2_corr.iloc[0].ifacts.sum(), 1,atol=1e-6)
    
    # are the corrected factors still reasonably close to the uncorrected positive values?
    fcorr = kf2_corr.iloc[0].ifacts
    fnocorr = kf2_nocorr.iloc[0].ifacts
    np.allclose(fcorr,fnocorr[fnocorr>0], atol=1e-2)


def ok_grid_test(tmp_path):

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
    pts_data['name'] = pts_data.name.astype(str)

    sr = pyemu.helpers.SpatialReference(delr=delr,delc=delc)
    pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
    pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
    str_file = os.path.join("utils","struct_test.dat")
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    kf = ok.calc_factors_grid(sr,verbose=False,var_filename=os.path.join(tmp_path,"test_var.ref"),minpts_interp=1)
    ok.to_grid_factors_file(os.path.join(tmp_path,"test.fac"))


def ok_grid_zone_test(tmp_path):

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
    pts_data['name'] = pts_data.name.astype(str)

    sr = pyemu.helpers.SpatialReference(delr=delr,delc=delc)
    pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
    pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
    pts_data.loc[:, "zone"] = 1
    pts_data.loc[pts_data.index[1], "zone"] = 2
    print(pts_data.zone.unique())
    str_file = os.path.join("utils","struct_test.dat")
    gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
    ok = pyemu.utils.geostats.OrdinaryKrige(gs,pts_data)
    zone_array = np.ones((nrow,ncol))
    zone_array[0,0] = 2
    kf = ok.calc_factors_grid(sr,verbose=False,
                              var_filename=os.path.join(tmp_path,"test_var.ref"),
                              minpts_interp=1,zone_array=zone_array,num_threads=2)
    ok.to_grid_factors_file(os.path.join(tmp_path,"test.fac"))


def ppk2fac_verf_test(tmp_path):
    import os
    import numpy as np
    import pyemu
    try:
        import flopy
    except:
        return
    ws = os.path.join("..","verification","Freyberg")
    filedict = dict(gspc_file="grid.spc",
                    pp_file="pp_00_pp.dat",
                    str_file="structure.complex.dat",
                    ppk2fac_facfile="ppk2fac_fac.dat",
                    zone_arr=os.path.join("extra_crispy","ref","ibound.ref"))
    [shutil.copy(os.path.join(ws, f), tmp_path) for _, f in filedict.items()]
    gspc_file = os.path.join(tmp_path, filedict['gspc_file'])
    pp_file = os.path.join(tmp_path, filedict['pp_file'])
    str_file = os.path.join(tmp_path, filedict['str_file'])
    ppk2fac_facfile = os.path.join(tmp_path, filedict['ppk2fac_facfile'])
    zone_arr = np.loadtxt(os.path.join(tmp_path, os.path.basename(filedict["zone_arr"])))
    pyemu_facfile = os.path.join(tmp_path, "pyemu_facfile.dat")
    sr = pyemu.helpers.SpatialReference.from_gridspec(gspc_file)
    ok = pyemu.utils.OrdinaryKrige(str_file, pp_file)
    ok.calc_factors_grid(sr, maxpts_interp=10)
    ok.to_grid_factors_file(pyemu_facfile)

    pyemu_arr = pyemu.utils.fac2real(pp_file,pyemu_facfile,out_file=None)
    ppk2fac_arr = pyemu.utils.fac2real(pp_file,ppk2fac_facfile,out_file=None)
    pyemu_arr[zone_arr == 0] = np.nan
    pyemu_arr[zone_arr == -1] = np.nan
    ppk2fac_arr[zone_arr == 0] = np.nan
    ppk2fac_arr[zone_arr == -1] = np.nan

    diff = np.abs(pyemu_arr - ppk2fac_arr)
    print(diff)

    assert np.nansum(diff) < 1.0e-6,np.nansum(diff)


# def opt_obs_worth():
#     import os
#     import pyemu
#     wdir = os.path.join("utils")
#     os.chdir(wdir)
#     pst = pyemu.Pst(os.path.join("supply2_pest.fosm.pst"))
#     zero_weight_names = [n for n,w in zip(pst.observation_data.obsnme,pst.observation_data.weight) if w == 0.0]
#     #print(zero_weight_names)
#     #for attr in ["base_jacobian","hotstart_resfile"]:
#     #    pst.pestpp_options[attr] = os.path.join(wdir,pst.pestpp_options[attr])
#     #pst.template_files = [os.path.join(wdir,f) for f in pst.template_files]
#     #pst.instruction_files = [os.path.join(wdir,f) for f in pst.instruction_files]
#     #print(pst.template_files)
#     df = pyemu.optimization.get_added_obs_importance(pst,obslist_dict={"zeros":zero_weight_names})
#     os.chdir("..")
#     print(df)


def mflist_budget_test(tmp_path):
    import pyemu
    import os
    import pandas as pd
    import shutil
    try:
        import flopy
    except:
        print("no flopy...")
        return
    model_ws = os.path.join("..", "examples", "Freyberg_transient")
    shutil.copytree(model_ws, Path(tmp_path, "Freyberg_transient"))
    model_ws = Path(tmp_path, "Freyberg_transient")
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,check=False,load_only=[])
    list_filename = os.path.join(model_ws,"freyberg.list")
    assert os.path.exists(list_filename)
    df = pyemu.gw_utils.setup_mflist_budget_obs(list_filename,
                                                flx_filename=os.path.join(tmp_path, "flux.dat"),
                                                vol_filename=os.path.join(tmp_path, "vol.dat"),
                                                start_datetime=ml.start_datetime)
    print(df)

    times = df.loc[df.index.str.startswith('vol_wells')].index.str.split(
        '_', expand=True).get_level_values(2)[::100]
    times = pd.to_datetime(times, yearfirst=True)
    df = pyemu.gw_utils.setup_mflist_budget_obs(
        list_filename,
        flx_filename=os.path.join(tmp_path, "flux.dat"),
        vol_filename=os.path.join(tmp_path, "vol.dat"),
        start_datetime=ml.start_datetime, specify_times=times)
    flx, vol = pyemu.gw_utils.apply_mflist_budget_obs(
        list_filename, os.path.join(tmp_path, 'flux.dat'),
        os.path.join(tmp_path,'vol.dat'),
        start_datetime=ml.start_datetime,
        times=os.path.join(tmp_path, 'budget_times.config')
    )
    assert (flx.index == vol.index).all()
    assert (flx.index == times).all()


def mtlist_budget_test(tmp_path):
    import pyemu
    import shutil
    import os
    try:
        import flopy
    except:
        print("no flopy...")
        return

    list_filenames = [Path("utils","mt3d.list"), Path("utils", "mt3d_imm_sor.lst")]
    _ = [shutil.copy(list_filename, Path(tmp_path, list_filename.name))
         for list_filename in list_filenames]
    list_filename = "mt3d.list"
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert os.path.exists(list_filename)
        frun_line, ins_files, df = pyemu.gw_utils.setup_mtlist_budget_obs(
            list_filename, start_datetime='1-1-1970')
        assert len(ins_files) == 2

        frun_line, ins_files, df = pyemu.gw_utils.setup_mtlist_budget_obs(
            list_filename, start_datetime='1-1-1970', gw_prefix='')
        assert len(ins_files) == 2

        frun_line, ins_files, df = pyemu.gw_utils.setup_mtlist_budget_obs(
            list_filename, start_datetime=None)
        assert len(ins_files) == 2

        list_filename = "mt3d_imm_sor.lst"
        assert os.path.exists(list_filename)
        frun_line, ins_files, df = pyemu.gw_utils.setup_mtlist_budget_obs(
            list_filename, start_datetime='1-1-1970')
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


@pytest.mark.timeout(method='thread', timeout=90)
def test_geostat_prior_builder(tmp_path):
    import os
    import numpy as np
    import pyemu
    import pandas as pd
    import gc

    for fname in [Path("pst", "pest.pst"),
                  Path("utils", "pp_locs.tpl"),
                  Path("utils", "structure.dat")]:
        shutil.copy(fname, tmp_path)
    pst_file = os.path.join(tmp_path, "pest.pst")
    pst = pyemu.Pst(pst_file)
    # print(pst.parameter_data)
    tpl_file = os.path.join(tmp_path, "pp_locs.tpl")
    str_file = os.path.join(tmp_path, "structure.dat")

    cov = pyemu.helpers.geostatistical_prior_builder(pst_file,{str_file:tpl_file})
    d1 = np.diag(cov.x)
    del cov

    df = pyemu.pp_utils.pp_tpl_to_dataframe(tpl_file)
    df.loc[:,"zone"] = np.arange(df.shape[0])
    gs = pyemu.geostats.read_struct_file(str_file)
    cov = pyemu.helpers.geostatistical_prior_builder(pst_file,{gs:df},
                                               sigma_range=4)
    nnz = np.count_nonzero(cov.x)
    d2 = np.diag(cov.x)
    del cov
    assert nnz == pst.npar_adj
    assert np.array_equiv(d1, d2)

    pst.parameter_data.loc[pst.par_names[1:10], "partrans"] = "tied"
    pst.parameter_data['partied'] = pd.Series(
        np.nan, index=pst.parameter_data.index, dtype=object
    )
    pst.parameter_data.loc[pst.par_names[1:10], "partied"] = pst.par_names[0]
    cov = pyemu.helpers.geostatistical_prior_builder(pst, {gs: df},
                                                     sigma_range=4)
    nnz = np.count_nonzero(cov.x)
    del cov
    assert nnz == pst.npar_adj

    ttpl_file = os.path.join(tmp_path, "temp.dat.tpl")
    with open(ttpl_file, 'w') as f:
        f.write("ptf ~\n ~ temp1  ~\n")
    pst.add_parameters(ttpl_file, ttpl_file.replace(".tpl", ""))

    pst.parameter_data.loc["temp1", "parubnd"] = 1.1
    pst.parameter_data.loc["temp1", "parlbnd"] = 0.9

    cov = pyemu.helpers.geostatistical_prior_builder(pst, {str_file: tpl_file})
    assert cov.shape[0] == pst.npar_adj
    del cov
    gc.collect()


def geostat_draws_test(tmp_path):
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    pst_file = os.path.join("pst","pest.pst")
    pst = pyemu.Pst(pst_file)
    print(pst.parameter_data)
    tpl_file = os.path.join("utils", "pp_locs.tpl")
    str_file = os.path.join("utils", "structure.dat")

    #make a df with one entry
    df_one = pd.DataFrame({"parnme":"mult1","x":-999,"y":-9999,"zone":-9999},index=["mult1"])


    pe = pyemu.helpers.geostatistical_draws(pst_file,{str_file:tpl_file,str_file:df_one})
    assert (pe.shape == pe.dropna().shape)

    pst.parameter_data["partied"] = pd.Series(
        np.nan, index=pst.parameter_data.index, dtype=object
    )
    pst.parameter_data.loc[pst.par_names[1:10], "partrans"] = "tied"
    pst.parameter_data.loc[pst.par_names[1:10], "partied"] = pst.par_names[0]
    pe = pyemu.helpers.geostatistical_draws(pst, {str_file: tpl_file})
    assert (pe.shape == pe.dropna().shape)
    assert "mult1" in pe.columns

    df = pyemu.pp_utils.pp_tpl_to_dataframe(tpl_file)
    df.loc[:,"zone"] = np.arange(df.shape[0])
    gs = pyemu.geostats.read_struct_file(str_file)
    np.random.seed(pyemu.en.SEED)
    pe = pyemu.helpers.geostatistical_draws(pst_file,{gs:df},
                                          sigma_range=4)
    np.random.seed(pyemu.en.SEED)
    pe2 = pyemu.helpers.geostatistical_draws(pst_file,{gs:df},
                                          sigma_range=4)
    pe.to_csv(os.path.join(os.path.join("utils","geostat_pe.csv")))

    diff = pe - pe2
    print(diff.max())
    assert diff.max().max() == 0.0


    ttpl_file = os.path.join(tmp_path, "temp.dat.tpl")
    with open(ttpl_file, 'w') as f:
        f.write("ptf ~\n ~ temp1  ~\n")
    pst.add_parameters(ttpl_file, ttpl_file.replace(".tpl", ""))

    pst.parameter_data.loc["temp1", "parubnd"] = 1.1
    pst.parameter_data.loc["temp1", "parlbnd"] = 0.9
    pst.parameter_data.loc[pst.par_names[1:10],"partrans"] = "tied"
    pst.parameter_data.loc[pst.par_names[1:10], "partied"] = pst.par_names[0]
    pe = pyemu.helpers.geostatistical_draws(pst, {str_file: tpl_file})
    assert (pe.shape == pe.dropna().shape)


# def linearuniversal_krige_test():
#     try:
#         import flopy
#     except:
#         return
#
#     import numpy as np
#     import pandas as pd
#     import pyemu
#     nrow,ncol = 10,5
#     delr = np.ones((ncol)) * 1.0/float(ncol)
#     delc = np.ones((nrow)) * 1.0/float(nrow)
#
#     num_pts = 0
#     ptx = np.random.random(num_pts)
#     pty = np.random.random(num_pts)
#     ptname = ["p{0}".format(i) for i in range(num_pts)]
#     pts_data = pd.DataFrame({"x":ptx,"y":pty,"name":ptname})
#     pts_data.index = pts_data.name
#     pts_data = pts_data.loc[:,["x","y","name"]]
#
#
#     sr = flopy.utils.SpatialReference(delr=delr,delc=delc)
#     pts_data.loc["i0j0", :] = [sr.xcentergrid[0,0],sr.ycentergrid[0,0],"i0j0"]
#     pts_data.loc["imxjmx", :] = [sr.xcentergrid[-1, -1], sr.ycentergrid[-1, -1], "imxjmx"]
#     pts_data.loc["i0j0","value"] = 1.0
#     pts_data.loc["imxjmx","value"] = 0.0
#
#     str_file = os.path.join("utils","struct_test.dat")
#     gs = pyemu.utils.geostats.read_struct_file(str_file)[0]
#     luk = pyemu.utils.geostats.LinearUniversalKrige(gs,pts_data)
#     df = luk.estimate_grid(sr,verbose=True,
#                                var_filename=os.path.join("utils","test_var.ref"),
#                                minpts_interp=1)


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


def read_hydmod_test(tmp_path):
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    try:
        import flopy
    except:
        return
    df, outfile = pyemu.gw_utils.modflow_read_hydmod_file(os.path.join('utils','freyberg.hyd.bin'),
                                                          os.path.join(tmp_path,'freyberg.hyd.bin.dat'))
    df = pd.read_csv(os.path.join(tmp_path, 'freyberg.hyd.bin.dat'), sep=r"\s+")
    dftrue = pd.read_csv(os.path.join('utils', 'freyberg.hyd.bin.dat.true'), sep=r"\s+")

    assert np.allclose(df.obsval.values, dftrue.obsval.values)


def make_hydmod_insfile_test(tmp_path):
    import os
    import shutil
    import pyemu
    try:
        import flopy
    except:
        return
    shutil.copy2(os.path.join('utils','freyberg.hyd.bin'),os.path.join(tmp_path,'freyberg.hyd.bin'))
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pyemu.gw_utils.modflow_hydmod_to_instruction_file('freyberg.hyd.bin')
        #assert open(os.path.join('utils','freyberg.hyd.bin.dat.ins'),'r').read() == open('freyberg.hyd.dat.ins', 'r').read()
        assert os.path.exists('freyberg.hyd.bin.dat.ins')
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def plot_summary_test(tmp_path):
    import os
    import pandas as pd

    import pyemu
    try:
        import matplotlib.pyplot as plt
    except:
        return

    par_df = pd.read_csv(os.path.join("utils","freyberg_pp.par.usum.csv"),
                         index_col=0)
    idx = list(par_df.index.map(lambda x: x.startswith("HK")))
    par_df = par_df.loc[idx,:]
    ax = pyemu.plot_utils.plot_summary_distributions(par_df,label_post=True)
    plt.savefig(os.path.join(tmp_path,"hk_par.png"))
    plt.close()

    df = os.path.join("utils","freyberg_pp.pred.usum.csv")
    figs,axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    #plt.show()
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join(tmp_path,"test_pred_{0}.png".format(i)))
        plt.close(fig)
    df = os.path.join("utils","freyberg_pp.par.usum.csv")
    figs, axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join(tmp_path,"test_par_{0}.png".format(i)))
        plt.close(fig)


def hds_timeseries_test(tmp_path):
    import os
    import shutil
    import numpy as np
    try:
        import flopy
    except:
        return
    import pyemu

    model_ws =os.path.join("..","examples","Freyberg_transient")
    org_hds_file = os.path.join(model_ws, "freyberg.hds")
    hds_file = os.path.join(tmp_path, "freyberg.hds")

    org_cbc_file = org_hds_file.replace(".hds",".cbc")
    cbc_file = hds_file.replace(".hds", ".cbc")

    shutil.copy2(org_hds_file, hds_file)
    shutil.copy2(org_cbc_file, cbc_file)

    m = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, check=False)
    kij_dict = {"test1": [0, 0, 0], "test2": (1, 1, 1), "test": (0, 10, 14)}

    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, include_path=True)

    # m.change_model_ws("temp",reset_external=True)
    # m.write_input()
    # pyemu.os_utils.run("mfnwt freyberg.nam",cwd="temp")

    cmd, df1 = pyemu.gw_utils.setup_hds_timeseries(cbc_file, kij_dict, include_path=True, prefix="stor",
                                                   text="storage", fill=0.0)

    cmd,df2 = pyemu.gw_utils.setup_hds_timeseries(cbc_file, kij_dict, model=m, include_path=True, prefix="stor",
                                        text="storage",fill=0.0)

    print(df1)
    d = np.abs(df1.obsval.values - df2.obsval.values)
    print(d.max())
    assert d.max() == 0.0,d

    try:
        pyemu.gw_utils.setup_hds_timeseries(cbc_file, kij_dict, model=m, include_path=True, prefix="consthead",
                                            text="constant head")
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pyemu.gw_utils.setup_hds_timeseries(cbc_file, kij_dict, model=m, include_path=True, prefix="consthead",
                                            text="JUNK")
    except:
        pass
    else:
        raise Exception("should have failed")


    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, include_path=True,prefix="hds")

    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,load_only=[],check=False)
    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict,model=m,include_path=True)
    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, model=m, include_path=True,prefix="hds")

    org_hds_file = os.path.join("utils", "MT3D001.UCN")
    hds_file = os.path.join(tmp_path, "MT3D001.UCN")
    shutil.copy2(org_hds_file, hds_file)
    kij_dict = {"test1": [0, 0, 0], "test2": (1, 1, 1)}

    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, include_path=True)
    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, include_path=True, prefix="hds")

    m = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, load_only=[], check=False)
    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, model=m, include_path=True)
    pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, model=m, include_path=True, prefix="hds")

    # df1 = pd.read_csv(out_file, sep=r"\s+",)
    # pyemu.gw_utils.apply_hds_obs(hds_file)
    # df2 = pd.read_csv(out_file, sep=r"\s+",)
    # diff = df1.obsval - df2.obsval


def grid_obs_test(tmp_path):
    import os
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu

    m_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    shutil.copytree(m_ws, os.path.join(tmp_path, "freyberg_sfr_update"))
    m_ws = os.path.join(tmp_path, "freyberg_sfr_update")
    org_hds_file = os.path.join("..","examples","Freyberg_Truth","freyberg.hds")
    multlay_hds_file = os.path.join(m_ws, "freyberg.hds")  # 3 layer version
    ucn_file = os.path.join(m_ws, "MT3D001.UCN")  # mt example
    hds_file = os.path.join(tmp_path,"freyberg.hds")
    out_file = hds_file+".dat"
    multlay_out_file = multlay_hds_file+".dat"
    ucn_out_file = ucn_file+".dat"
    shutil.copy2(org_hds_file,hds_file)
    # todo filepaths might be relative (if not running it pytest)

    bd = os.getcwd()
    os.chdir(tmp_path)
    m_ws = "freyberg_sfr_update"
    try:
        pyemu.gw_utils.setup_hds_obs(hds_file)
        df1 = pd.read_csv(out_file, sep=r"\s+")
        pyemu.gw_utils.apply_hds_obs(hds_file)
        df2 = pd.read_csv(out_file, sep=r"\s+")
        diff = df1.obsval - df2.obsval
        assert abs(diff.max()) < 1.0e-6, abs(diff.max())

        pyemu.gw_utils.setup_hds_obs(multlay_hds_file)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+")
        assert len(df1) == 3*len(df2), "{} != 3*{}".format(len(df1), len(df2))
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+")
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval,df2.obsval), abs(diff.max())

        pyemu.gw_utils.setup_hds_obs(hds_file, skip=-999)
        df1 = pd.read_csv(out_file, sep=r"\s+")
        pyemu.gw_utils.apply_hds_obs(hds_file)
        df2 = pd.read_csv(out_file,sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert diff.max() < 1.0e-6

        pyemu.gw_utils.setup_hds_obs(ucn_file, skip=1.e30, prefix='ucn')
        df1 = pd.read_csv(ucn_out_file, sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(ucn_file)
        df2 = pd.read_csv(ucn_out_file, sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())

        # skip = lambda x : x < -888.0
        skip = lambda x: x if x > -888.0 else np.nan
        pyemu.gw_utils.setup_hds_obs(hds_file,skip=skip)
        df1 = pd.read_csv(out_file,sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(hds_file)
        df2 = pd.read_csv(out_file,sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert diff.max() < 1.0e-6

        kperk_pairs = (0,0)
        pyemu.gw_utils.setup_hds_obs(hds_file,kperk_pairs=kperk_pairs,
                                     skip=skip)
        df1 = pd.read_csv(out_file,sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(hds_file)
        df2 = pd.read_csv(out_file,sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert diff.max() < 1.0e-6

        kperk_pairs = [(0, 0), (0, 1), (0, 2)]
        pyemu.gw_utils.setup_hds_obs(multlay_hds_file, kperk_pairs=kperk_pairs,
                                     skip=skip)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        assert len(df1) == 3*len(df2), "{} != 3*{}".format(len(df1), len(df2))
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())

        kperk_pairs = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
        pyemu.gw_utils.setup_hds_obs(multlay_hds_file, kperk_pairs=kperk_pairs,
                                     skip=skip)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        assert len(df1) == 2 * len(df2), "{} != 2*{}".format(len(df1), len(df2))
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())

        m = flopy.modflow.Modflow.load("freyberg.nam", model_ws=m_ws, load_only=["BAS6"],forgive=False,verbose=True)
        kperk_pairs = [(0, 0), (0, 1), (0, 2)]
        skipmask = m.bas6.ibound.array
        pyemu.gw_utils.setup_hds_obs(multlay_hds_file, kperk_pairs=kperk_pairs,
                                     skip=skipmask)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        assert len(df1) == len(df2) == np.abs(skipmask).sum(), \
            "array skip failing, expecting {0} obs but returned {1}".format(np.abs(skipmask).sum(), len(df1))
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())

        kperk_pairs = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
        skipmask = m.bas6.ibound.array[0]
        pyemu.gw_utils.setup_hds_obs(multlay_hds_file, kperk_pairs=kperk_pairs,
                                     skip=skipmask)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        assert len(df1) == len(df2) == 2 * m.nlay * np.abs(skipmask).sum(), "array skip failing"
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())

        kperk_pairs = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
        skipmask = m.bas6.ibound.array
        pyemu.gw_utils.setup_hds_obs(multlay_hds_file, kperk_pairs=kperk_pairs,
                                     skip=skipmask)
        df1 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        pyemu.gw_utils.apply_hds_obs(multlay_hds_file)
        df2 = pd.read_csv(multlay_out_file, sep=r"\s+",)
        assert len(df1) == len(df2) == 2 * np.abs(skipmask).sum(), "array skip failing"
        diff = df1.obsval - df2.obsval
        assert np.allclose(df1.obsval, df2.obsval), abs(diff.max())
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def postprocess_inactive_conc_test(tmp_path):
    import os
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    bd = os.getcwd()
    model_ws = os.path.join("..", "examples", "Freyberg_transient")
    shutil.copytree(model_ws, os.path.join(tmp_path, "Freyberg_transient"))
    model_ws = os.path.join(tmp_path, "Freyberg_transient")
    org_hds_file = os.path.join("utils", "MT3D001.UCN")
    hds_file = os.path.join(tmp_path, "MT3D001.UCN")
    shutil.copy2(org_hds_file, hds_file)
    # todo filepaths might be relative (if not running it pytest)

    kij_dict = {"test1": [0, 0, 0], "test2": (1, 1, 1), "inact": [0, 81, 35]}
    os.chdir(tmp_path)
    try:
        m = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, load_only=[], check=False)
        frun_line, df = pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, model=m, include_path=True, 
                                                            prefix="hds",
                                                            postprocess_inact=1E30)
        df0 = pd.read_csv("{0}_timeseries.processed".format(os.path.split(hds_file)[-1]), sep=r"\s+").T
        df1 = pd.read_csv("{0}_timeseries.post_processed".format(os.path.split(hds_file)[-1]), sep=r"\s+").T
        eval(frun_line)
        df2 = pd.read_csv("{0}_timeseries.processed".format(os.path.split(hds_file)[-1]), sep=r"\s+").T
        df3 = pd.read_csv("{0}_timeseries.post_processed".format(os.path.split(hds_file)[-1]), sep=r"\s+").T
        assert np.allclose(df0, df2)
        assert np.allclose(df2.test1, df3.test1)
        assert np.allclose(df2.test2, df3.test2)
        assert np.allclose(df3, df1)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def gw_sft_ins_test(tmp_path):
    import os
    import pyemu
    import shutil

    sft_outfile = os.path.join("utils", "test_sft.out")
    shutil.copy(sft_outfile, Path(tmp_path, "test_sft.out"))
    sft_outfile = Path(tmp_path, "test_sft.out")
    #pyemu.gw_utils.setup_sft_obs(sft_outfile)
    #pyemu.gw_utils.setup_sft_obs(sft_outfile,start_datetime="1-1-1970")
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        df = pyemu.gw_utils.setup_sft_obs(str(sft_outfile), start_datetime="1-1-1970",times=[10950.00])
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)
    #print(df)


def sfr_helper_test(tmp_path):  # TODO: need attention to move IO to tmp_path (particularly writing)
    import os
    import pandas as pd
    import pyemu
    import flopy
    import numpy as np

    #setup the process
    m_ws = os.path.join("utils", 'supply2eg')
    shutil.copytree(m_ws, os.path.join(tmp_path, 'supply2eg'))
    m_ws = os.path.join(tmp_path, 'supply2eg')

    m = flopy.modflow.Modflow.load("supply2.nam",model_ws=m_ws,check=False,verbose=True,forgive=False,
                                   load_only=["dis","sfr"])
    sd = m.sfr.segment_data[0].copy()

    sd["flow"] = 1.0
    sd["pptsw"] = 1.0

    m.sfr.segment_data = {k:sd.copy() for k in range(m.nper)}

    df_sfr = pyemu.gw_utils.setup_sfr_seg_parameters(
        m, include_temporal_pars=['hcond1', 'flow'])
    print(df_sfr)
    bd = os.getcwd()
    os.chdir(m_ws)
    try:
        # change the name of the sfr file that will be created
        pars = {}
        with open("sfr_seg_pars.config") as f:
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        pars["sfr_filename"] = "test.sfr"
        with open("sfr_seg_pars.config", 'w') as f:
            for k, v in pars.items():
                f.write("{0} {1}\n".format(k, v))
                # change some hcond1 values
        df = pd.read_csv("sfr_seg_temporal_pars.dat", index_col=0)
        df.loc[:, "flow"] = 10.0
        df.to_csv("sfr_seg_temporal_pars.dat", sep=',')

        sd1 = pyemu.gw_utils.apply_sfr_seg_parameters().segment_data
        m1 = flopy.modflow.Modflow.load("supply2.nam", load_only=["sfr"], check=False)
        for kper,sd in m1.sfr.segment_data.items():
            #print(sd["flow"],sd1[kper]["flow"])
            for i1, i2 in zip(sd["flow"], sd1[kper]["flow"]):
                assert i1 * 10 == i2, "{0},{1}".format(i1, i2)

        df_sfr = pyemu.gw_utils.setup_sfr_seg_parameters("supply2.nam", model_ws=m_ws, include_temporal_pars=True)

        # change the name of the sfr file that will be created
        pars = {}
        with open("sfr_seg_pars.config") as f:
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        pars["sfr_filename"] = "test.sfr"
        with open("sfr_seg_pars.config", 'w') as f:
            for k, v in pars.items():
                f.write("{0} {1}\n".format(k, v))

        # change some hcond1 values
        df = pd.read_csv("sfr_seg_pars.dat",index_col=0)
        df.loc[:, "hcond1"] = 1.0
        df.to_csv("sfr_seg_pars.dat", sep=',')

        # make sure the hcond1 mult worked...
        sd1 = pyemu.gw_utils.apply_sfr_seg_parameters().segment_data[0]
        m1 = flopy.modflow.Modflow.load("supply2.nam", load_only=["sfr"], check=False)
        sd2 = m1.sfr.segment_data[0]

        sd1 = pd.DataFrame.from_records(sd1)
        sd2 = pd.DataFrame.from_records(sd2)

        # print(sd1.hcond1)
        # print(sd2.hcond2)

        assert sd1.hcond1.sum() == sd2.hcond1.sum()

        # change some hcond1 values
        df = pd.read_csv("sfr_seg_pars.dat",index_col=0)
        df.loc[:,"hcond1"] = 0.5
        df.to_csv("sfr_seg_pars.dat",sep=',')

        #change the name of the sfr file that will be created
        pars = {}
        with open("sfr_seg_pars.config") as f:
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        pars["sfr_filename"] = "test.sfr"
        with open("sfr_seg_pars.config",'w') as f:
            for k,v in pars.items():
                f.write("{0} {1}\n".format(k,v))

        #make sure the hcond1 mult worked...
        sd1 = pyemu.gw_utils.apply_sfr_seg_parameters().segment_data[0]
        m1 = flopy.modflow.Modflow.load("supply2.nam",load_only=["sfr"],check=False)
        sd2 = m1.sfr.segment_data[0]

        sd1 = pd.DataFrame.from_records(sd1)
        sd2 = pd.DataFrame.from_records(sd2)

        #print(sd1.hcond1)
        #print(sd2.hcond2)

        assert (sd1.hcond1 * 2.0).sum() == sd2.hcond1.sum()
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)

    #setup the process
    m_ws = os.path.join('..', 'examples', 'freyberg_sfr_reaches')
    shutil.copytree(m_ws, os.path.join(tmp_path, 'freyberg_sfr_reaches'))
    m_ws = os.path.join(tmp_path, 'freyberg_sfr_reaches')

    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=m_ws,check=False,
                                   verbose=False,forgive=False,
                                   load_only=["dis","sfr"])
    sd = m.sfr.segment_data[0].copy()

    sd["flow"] = 1.0
    sd["pptsw"] = 1.0

    m.sfr.segment_data = {k:sd.copy() for k in range(m.nper)}

    df_sfr = pyemu.gw_utils.setup_sfr_seg_parameters(
        m, include_temporal_pars=['hcond1', 'flow'])
    df_reaches = pyemu.gw_utils.setup_sfr_reach_parameters(m)
    print(df_sfr)
    os.chdir(m_ws)
    try:
        # change the name of the sfr file that will be created
        pars = {}
        with open("sfr_seg_pars.config") as f:
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        pars["sfr_filename"] = "test.sfr"
        with open("sfr_seg_pars.config", 'w') as f:
            for k, v in pars.items():
                f.write("{0} {1}\n".format(k, v))
                # change some hcond1 values
        df = pd.read_csv("sfr_seg_temporal_pars.dat", index_col=0)
        df.loc[:, "flow"] = 10.0
        df.to_csv("sfr_seg_temporal_pars.dat", sep=',')

        rdf = pd.read_csv("sfr_reach_pars.dat", index_col=0)
        rdf.loc[:, "strhc1"] = 10.0
        rdf.to_csv("sfr_reach_pars.dat", sep=',')

        newsfr = pyemu.gw_utils.apply_sfr_seg_parameters(True, True)
        sd1 = newsfr.segment_data
        rc1 = newsfr.reach_data
        m1 = flopy.modflow.Modflow.load("freyberg.nam", load_only=["sfr"], check=False)
        for kper,sd in m1.sfr.segment_data.items():
            #print(sd["flow"],sd1[kper]["flow"])
            for i1,i2 in zip(sd["flow"],sd1[kper]["flow"]):
                assert i1 * 10 == i2, "{0},{1}".format(i1, i2)
        assert all(np.isclose(rc1.strhc1, 10 * m1.sfr.reach_data.strhc1))
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def sfr_obs_test(tmp_path):
    import os
    import pyemu
    import flopy

    [shutil.copy(os.path.join("utils",f"freyberg.{ext}"), tmp_path)
     for ext in ["sfr.out", "nam", "dis", "bas"]]
    sfr_file = "freyberg.sfr.out"
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pyemu.gw_utils.setup_sfr_obs(sfr_file)
        pyemu.gw_utils.setup_sfr_obs(sfr_file,seg_group_dict={"obs1":[1,4],"obs2":[16,17,18,19,22,23]})

        m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=".",load_only=[],check=False)
        pyemu.gw_utils.setup_sfr_obs(sfr_file,model=m)
        pyemu.gw_utils.apply_sfr_obs()
        pyemu.gw_utils.setup_sfr_obs(sfr_file, seg_group_dict={"obs1": [1, 4], "obs2": [16, 17, 18, 19, 22, 23]},model=m)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)

def sfr_reach_obs_test(tmp_path):
    import os
    import pyemu
    import flopy
    import pandas as pd
    import numpy as np
    [shutil.copy(os.path.join("utils", f"freyberg.{ext}"), tmp_path)
     for ext in ["sfr.out", "nam", "dis", "bas"]]
    sfr_file = "freyberg.sfr.out"
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, seg_reach=[[1, 2], [4, 1], [2, 2]])
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, seg_reach=np.array([[1, 2], [4, 1], [2, 2]]))
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file)
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*40  # (nper*nobs)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file,seg_reach={"obs1": [1, 2], "obs2": [4, 1]})
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)
        seg_reach_df = pd.DataFrame.from_dict({"obs1": [1, 2], "obs2": [4, 1]}, columns=['segment', 'reach'], orient='index')
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, seg_reach=seg_reach_df)
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)

        m = flopy.modflow.Modflow.load("freyberg.nam", model_ws=".", load_only=[], check=False)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, model=m)
        pyemu.gw_utils.apply_sfr_reach_obs()
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*40  # (nper*nobs)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, seg_reach={"obs1": [1, 2], "obs2": [4, 1], "blah": [2, 1]}, model=m)
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)
        pyemu.gw_utils.setup_sfr_reach_obs(sfr_file, model=m, seg_reach=seg_reach_df)
        proc = pd.read_csv("{0}.reach_processed".format(sfr_file), sep=' ')
        assert proc.shape[0] == 3*2  # (nper*nobs)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)

def gage_obs_test(tmp_path):
    import os
    import pyemu
    import numpy as np

    bd = os.getcwd()
    gage_file = "RmSouth_pred_7d.gage1.go"
    shutil.copy(os.path.join("utils", gage_file), tmp_path)

    os.chdir(tmp_path)
    try:
        gage = pyemu.gw_utils.setup_gage_obs(gage_file, start_datetime='2007-04-11')
        if gage is not None:
            print(gage[1], gage[2])

        times = np.concatenate(([0], np.arange(7., 7. * 404, 7.)))
        gage = pyemu.gw_utils.setup_gage_obs(gage_file, start_datetime='2007-04-11', times=times)
        if gage is not None:
            print(gage[1], gage[2])
        pyemu.gw_utils.apply_gage_obs()
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def pst_from_parnames_obsnames_test(tmp_path):
    import pyemu
    import os

    parnames  = ['param1','par2','p3']
    obsnames  = ['obervation1','ob2','o6']
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pst = pyemu.helpers.pst_from_parnames_obsnames(parnames, obsnames)

        pst.write(os.path.join(tmp_path, 'simpletemp.pst'))

        newpst = pyemu.Pst(os.path.join(tmp_path, 'simpletemp.pst'))

        assert newpst.nobs == len(obsnames)
        assert newpst.npar == len(parnames)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def write_jactest_test(tmp_path):
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "5.pst"))
    print(pst.parameter_data)
    #return
    df = pyemu.helpers.build_jac_test_csv(pst,num_steps=5)
    print(df)


    df = pyemu.helpers.build_jac_test_csv(pst, num_steps=5,par_names=["par1"])
    print(df)

    df = pyemu.helpers.build_jac_test_csv(pst, num_steps=5,forward=False)
    print(df)
    df.to_csv(os.path.join(tmp_path,"sweep_in.csv"))
    print(pst.parameter_data)
    pst.write(os.path.join(tmp_path,"test.pst"))
    #pyemu.helpers.run("sweep test.pst",cwd="temp")


def plot_id_bar_test(tmp_path):
    import pyemu
    # import matplotlib.pyplot as plt
    w_dir = "la"
    shutil.copy(os.path.join(w_dir, "pest.jcb"), os.path.join(tmp_path, "pest.jcb"))
    shutil.copy(os.path.join(w_dir, "pest.pst"), os.path.join(tmp_path, "pest.pst"))
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ev = pyemu.ErrVar(jco="pest.jcb")
        id_df = ev.get_identifiability_dataframe(singular_value=15)
        pyemu.plot_utils.plot_id_bar(id_df)
        #plt.show()
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def jco_from_pestpp_runstorage_test(tmp_path):
    import os
    import pyemu

    jco_file = os.path.join("utils","pest.jcb")
    shutil.copy(jco_file, os.path.join(tmp_path, "pest.jcb"))
    pst_file = jco_file.replace(".jcb",".pst")
    shutil.copy(pst_file, os.path.join(tmp_path, "pest.pst"))
    rnj_file = jco_file.replace(".jcb",".rnj")
    shutil.copy(rnj_file, os.path.join(tmp_path, "pest.rnj"))

    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        jco = pyemu.Jco.from_binary("pest.jcb")
        jco2 = pyemu.helpers.jco_from_pestpp_runstorage("pest.rnj", "pest.pst")
        diff = (jco - jco2).to_dataframe()
        print(diff)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def hfb_test(tmp_path):
    import os
    try:
        import flopy
    except:
        return
    import pyemu

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    shutil.copytree(org_model_ws, os.path.join(tmp_path, "freyberg_sfr_update"))
    model_ws = os.path.join(tmp_path, "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False)
    try:
        pyemu.gw_utils.write_hfb_template(m)
    except:
        pass
    else:
        raise Exception()

    hfb_data = []
    jcol1, jcol2 = 14,15
    for i in range(m.nrow):
        hfb_data.append([0,i,jcol1,i,jcol2,0.001])
    flopy.modflow.ModflowHfb(m,0,0,len(hfb_data),hfb_data=hfb_data)
    m.write_input()
    m.exe_name = "mfnwt"
    try:
        m.run_model()
    except:
        pass

    tpl_file,df = pyemu.gw_utils.write_hfb_template(m)
    assert os.path.exists(tpl_file)
    assert df.shape[0] == m.hfb6.hfb_data.shape[0]


def hfb_zn_mult_test(tmp_path):
    import os
    try:
        import flopy
    except:
        return
    import pyemu
    import pandas as pd

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    shutil.copytree(org_model_ws, os.path.join(tmp_path, "freyberg_sfr_update"))
    model_ws = os.path.join(tmp_path, "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(
        nam_file, model_ws=model_ws, check=False)
    try:
        pyemu.gw_utils.write_hfb_template(m)
    except:
        pass
    else:
        raise Exception()

    hfb_data = []
    jcol1, jcol2 = 14, 15
    for i in range(m.nrow)[:11]:
        hfb_data.append([0, i, jcol1, i, jcol2, 0.001])
    for i in range(m.nrow)[11:21]:
        hfb_data.append([0, i, jcol1, i, jcol2, 0.002])
    for i in range(m.nrow)[21:]:
        hfb_data.append([0, i, jcol1, i, jcol2, 0.003])
    flopy.modflow.ModflowHfb(m, 0, 0, len(hfb_data), hfb_data=hfb_data)
    orig_len = len(m.hfb6.hfb_data)
    m.write_input()
    m.exe_name = "mfnwt"
    try:
        m.run_model()
    except:
        pass

    orig_vals, tpl_file = pyemu.gw_utils.write_hfb_zone_multipliers_template(m)
    assert os.path.exists(tpl_file)
    hfb_pars = pd.read_csv(os.path.join(m.model_ws, 'hfb6_pars.csv'))
    hfb_tpl_contents = open(tpl_file, 'r').readlines()
    mult_str = ''.join(hfb_tpl_contents[1:]).replace(
        '~ hbz_0000 ~', '0.1').replace(
        '~ hbz_0001 ~', '1.0').replace(
        '~ hbz_0002 ~', '10.0')
    with open(hfb_pars.mlt_file.values[0], 'w') as mfp:
        mfp.write(mult_str)
    pyemu.gw_utils.apply_hfb_pars(os.path.join(m.model_ws, 'hfb6_pars.csv'))
    with open(hfb_pars.mlt_file.values[0], 'r') as mfp:
        for i, line in enumerate(mfp):
            pass
    mhfb = flopy.modflow.ModflowHfb.load(hfb_pars.model_file.values[0], m)
    assert i-1 == orig_len == len(mhfb.hfb_data)


def read_runstor_test(tmp_path):
    import os
    import numpy as np
    import pandas as pd
    import pyemu
    d = os.path.join("utils","runstor")
    shutil.copytree(d, os.path.join(tmp_path, "runstor"))
    d = os.path.join(tmp_path, "runstor")
    pst = pyemu.Pst(os.path.join(d,"pest.pst"))

    par_df,obs_df = pyemu.helpers.read_pestpp_runstorage(os.path.join(d,"pest.rns"),"all")
    par_df2 = pd.read_csv(os.path.join(d,"sweep_in.csv"),index_col=0)
    obs_df2 = pd.read_csv(os.path.join(d,"sweep_out.csv"),index_col=0)
    obs_df2.columns = obs_df2.columns.str.lower()
    obs_df2 = obs_df2.loc[:,obs_df.columns]
    par_df2 = par_df2.loc[:,par_df.columns]
    pdif = np.abs(par_df.values - par_df2.values).max()
    odif = np.abs(obs_df.values - obs_df2.values).max()
    print(pdif,odif)
    assert pdif < 1.0e-6,pdif
    assert odif < 1.0e-6,odif
   
    try:
        pyemu.helpers.read_pestpp_runstorage(os.path.join(d, "pest.rns"), "junk")
    except:
        pass
    else:
        raise Exception()


def smp_test(tmp_path):
    import os
    from pyemu.utils import smp_to_dataframe, dataframe_to_smp, \
        smp_to_ins
    from pyemu.pst.pst_utils import parse_ins_file

    o_smp_filename = os.path.join("misc", "gainloss.smp")
    smp_filename = os.path.join(tmp_path, "gainloss.smp")
    shutil.copy(o_smp_filename, smp_filename)
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df, smp_filename + ".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename + ".ins")
    print(len(obs_names))

    o_smp_filename = os.path.join("misc", "sim_hds_v6.smp")
    smp_filename = os.path.join(tmp_path, "sim_hds_v6.smp")
    shutil.copy(o_smp_filename, smp_filename)
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df, smp_filename + ".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename + ".ins")
    print(len(obs_names))


def smp_dateparser_test(tmp_path):
    import os
    import pyemu
    from pyemu.utils import smp_to_dataframe, dataframe_to_smp, \
        smp_to_ins

    o_smp_filename = os.path.join("misc", "gainloss.smp")
    smp_filename = os.path.join(tmp_path, "gainloss.smp")
    shutil.copy(o_smp_filename, smp_filename)
    df = smp_to_dataframe(smp_filename, datetime_format="%d/%m/%Y %H:%M:%S")
    print(df.dtypes)
    dataframe_to_smp(df, smp_filename + ".test")
    smp_to_ins(smp_filename)
    obs_names = pyemu.pst_utils.parse_ins_file(smp_filename + ".ins")
    print(len(obs_names))

    o_smp_filename = os.path.join("misc", "sim_hds_v6.smp")
    smp_filename = os.path.join(tmp_path, "sim_hds_v6.smp")
    shutil.copy(o_smp_filename, smp_filename)
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df, smp_filename + ".test")
    smp_to_ins(smp_filename)
    obs_names = pyemu.pst_utils.parse_ins_file(smp_filename + ".ins")
    print(len(obs_names))



def fieldgen_dev(tmp_path):
    import shutil
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu
    from pyemu.legacy import PstFromFlopyModel

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False)
    flopy.modflow.ModflowRiv(m, stress_period_data={0: [[0, 0, 0, 30.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0],
                                                        [0, 0, 1, 31.0, 1.0, 25.0]]})
    org_model_ws = tmp_path
    m.change_model_ws(org_model_ws)
    m.write_input()

    new_model_ws = "temp_fieldgen"

    ph = PstFromFlopyModel(nam_file, new_model_ws=new_model_ws,
                                         org_model_ws=org_model_ws,
                                         grid_props=[["upw.hk", 0], ["rch.rech", 0]],
                                         remove_existing=True,build_prior=False)
    v = pyemu.geostats.ExpVario(1.0,1000,anisotropy=10,bearing=45)
    gs = pyemu.geostats.GeoStruct(nugget=0.0,variograms=v,name="aniso")
    struct_dict = {gs:["hk","ss"]}
    df = pyemu.helpers.run_fieldgen(m,10,struct_dict,cwd=new_model_ws)

    import matplotlib.pyplot as plt
    i = df.index.map(lambda x: int(x.split('_')[0]))
    j = df.index.map(lambda x: int(x.split('_')[1]))
    arr = np.zeros((m.nrow,m.ncol))
    arr[i,j] = df.iloc[:,0]
    plt.imshow(arr)
    plt.show()


def ok_grid_invest(tmp_path):

    try:
        import flopy
    except:
        return

    import numpy as np
    import pandas as pd
    import pyemu
    nrow,ncol = 200,200
    delr = np.ones((ncol)) * 1.0/float(ncol)
    delc = np.ones((nrow)) * 1.0/float(nrow)

    num_pts = 100
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
    kf = ok.calc_factors_grid(sr,verbose=False,var_filename=os.path.join(tmp_path,"test_var.ref"),minpts_interp=1,num_threads=1)
    kf2 = ok.calc_factors_grid(sr, verbose=False, var_filename=os.path.join(tmp_path, "test_var.ref"), minpts_interp=1,num_threads=10)
    ok.to_grid_factors_file(os.path.join(tmp_path,"test.fac"))
    diff = (kf.err_var - kf2.err_var).apply(np.abs).sum()
    assert diff < 1.0e-10

def specsim_test():
    try:
        import flopy
    except:
        return

    import numpy as np
    import pyemu
    num_reals = 100
    nrow,ncol = 40,20
    a = 2500
    contrib = 1.0
    nugget = 0
    delr = np.ones((ncol)) * 250
    delc = np.ones((nrow)) * 250
    variograms = [pyemu.geostats.ExpVario(contribution=contrib,a=a,anisotropy=1,bearing=10)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms,transform="none",nugget=nugget)
    broke_delr = delr.copy()
    broke_delr[0] = 0.0
    broke_delc = delc.copy()
    broke_delc[0] = 0.0

    try:
        ss = pyemu.geostats.SpecSim2d(geostruct=gs,delx=broke_delr,dely=delc)
    except Exception as e:
        pass
    else:
        raise Exception("should have failed")

    variograms = [pyemu.geostats.ExpVario(contribution=contrib, a=a, anisotropy=1, bearing=00)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms, transform="none", nugget=nugget)
    try:
        ss = pyemu.geostats.SpecSim2d(geostruct=gs,delx=broke_delr,dely=delc)
    except Exception as e:
        pass
    else:
        raise Exception("should have failed")

    try:
        ss = pyemu.geostats.SpecSim2d(geostruct=gs,delx=delr,dely=broke_delc)
    except Exception as e:
        pass
    else:
        raise Exception("should have failed")

    variograms = [pyemu.geostats.ExpVario(contribution=contrib, a=a, anisotropy=10, bearing=0)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms, transform="log", nugget=nugget)
    np.random.seed(1)

    ss = pyemu.geostats.SpecSim2d(geostruct=gs, delx=delr, dely=delc)
    mean_value = 15.0
    reals = ss.draw_arrays(num_reals=num_reals, mean_value=mean_value)
    assert reals.shape == (num_reals, nrow, ncol),reals.shape
    reals = np.log10(reals)
    mean_value = np.log10(mean_value)
    var = np.var(reals, axis=0).mean()

    mean = reals.mean()

    theo_var = ss.geostruct.sill
    print(var, theo_var)
    print(mean, mean_value)
    assert np.abs(var - theo_var) < 0.1
    assert np.abs(mean - mean_value) < 0.1

    np.random.seed(1)
    variograms = [pyemu.geostats.ExpVario(contribution=contrib, a=a, anisotropy=10, bearing=0)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms, transform="none", nugget=nugget)

    ss = pyemu.geostats.SpecSim2d(geostruct=gs, delx=delr, dely=delc)
    mean_value = 25.0
    reals = ss.draw_arrays(num_reals=num_reals,mean_value=mean_value)
    assert reals.shape == (num_reals,nrow,ncol)
    var = np.var(reals,axis=0).mean()
    mean = reals.mean()

    theo_var = ss.geostruct.sill
    print(var,theo_var)
    print(mean,mean_value)
    assert np.abs(var - theo_var) < 0.1
    assert np.abs(mean - mean_value) < 0.1

def aniso_invest():

    try:
        import flopy
    except:
        return

    import numpy as np
    import pandas as pd
    import pyemu
    from  datetime import datetime
    nrow,ncol = 40,20
    delr = np.ones((ncol)) * 250
    delc = np.ones((nrow)) * 250
    variograms = [pyemu.geostats.ExpVario(contribution=2.5,a=2500.0,anisotropy=10,bearing=90)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms,transform="none",nugget=0.0)

    np.random.seed(1)
    num_reals = 100
    start = datetime.now()
    ss = pyemu.geostats.SpecSim2d(geostruct=gs, delx=delr, dely=delc)
    mean_value = 1.0
    reals1 = ss.draw_arrays(num_reals=num_reals,mean_value=mean_value)
    print((datetime.now() - start).total_seconds())

    variograms = [pyemu.geostats.ExpVario(contribution=2.5, a=2000.0, anisotropy=10, bearing=0)]
    gs = pyemu.geostats.GeoStruct(variograms=variograms, transform="none", nugget=0.0)
    ss = pyemu.geostats.SpecSim2d(geostruct=gs, delx=delr, dely=delc)
    reals2 = ss.draw_arrays(num_reals=num_reals, mean_value=mean_value)

    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(6,3))
    axes[0].imshow(reals2[0])
    axes[1].imshow(reals1[0])
    #axes[0].set_title("bearing: 10")
    #axes[1].set_title("bearing: 95")
    plt.show()

def run_test():
    import pyemu
    import platform

    if "window" in platform.platform().lower():
        pyemu.os_utils.run("echo test")
    else:
        pyemu.os_utils.run("ls")
    try:
        pyemu.os_utils.run("junk")
    except:
        pass
    else:
        raise Exception("should have failed")

def run_sp_success_test():
    import platform
    if "window" in platform.platform().lower():
        pyemu.os_utils.run("echo test", use_sp=True, shell=True)
    else:
        pyemu.os_utils.run("ls", use_sp=True, shell=True)
    assert True

def run_sp_failure_test():
    with pytest.raises(Exception):
        pyemu.os_utils.run("junk_command", use_sp=True, 
                           shell=False, logfile=False)

def run_sp_capture_output_test(tmp_path):
    import platform
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False
    log_file = os.path.join(tmp_path, "pyemu.log")
    pyemu.os_utils.run("echo Hello World", 
                       verbose=False, use_sp=True, 
                       shell=shell, cwd=tmp_path, logfile=True)
    
    with open(log_file, 'r') as f:
        content = f.read()
    assert "Hello World" in content

def run_sp_verbose_test(capsys):
    import platform
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False
    pyemu.os_utils.run("echo test", use_sp=True, 
                       shell=shell, verbose=True)
    captured = capsys.readouterr()
    assert "test" in captured.out

@pytest.mark.skip(reason="slow as atm -- was stomped on by maha_pdc_test previously")
def maha_pdc_summary_test(tmp_path):  # todo add back in? currently super slowww
    import pyemu
    Path(tmp_path).mkdir(exist_ok=True)
    l1_critical_value = 6.4 #chi squared value at df=1,p=0.01
    l2_critical_value = 9.2 #chi squared value at df=2,p=0.01
    pst_file = os.path.join("la", "pest.pst")
    shutil.copy(pst_file, tmp_path)
    pst = pyemu.Pst(os.path.join(tmp_path, "pest.pst"))
    pst.observation_data.loc[:,"weight"] = 1.0
    en = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst,num_reals=20)
    level_1,level_2 = pyemu.helpers.get_maha_obs_summary(en)
    assert level_1.shape[0] == 0
    assert level_2.shape[0] == 0

    pst_file = os.path.join("pst","zoned_nz_64.pst")
    shutil.copy(pst_file, tmp_path)
    pst = pyemu.Pst(os.path.join(tmp_path, "zoned_nz_64.pst"))
    en = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=20)
    level_1, level_2 = pyemu.helpers.get_maha_obs_summary(en)
    level_1.sort_values(inplace=True)
    level_2.sort_values(by="sq_distance",inplace=True)
    print(level_1)
    print(level_2)
    assert level_1.shape[0] == 0
    assert level_2.shape[0] == 0


def gsf_reader_test():
    import pyemu
    gsffilename = os.path.join('utils','freyberg.usg.gsf')

    gsf = pyemu.gw_utils.GsfReader(gsffilename)
    nnodes = 4497

    assert len(gsf.get_node_data()) == nnodes

def conditional_prior_test():
    import os
    import numpy as np
    import pyemu

    prior_var = 1.0
    cond_var = 0.5
    v = pyemu.geostats.ExpVario(contribution=prior_var,a=5.0)
    gs = pyemu.geostats.GeoStruct(variograms=v,transform="none")
    x = np.arange(100)
    y = np.zeros_like(x)
    names = ["n{0}".format(i) for i in range(x.shape[0])]
    cov = gs.covariance_matrix(x,y,names)
    know_dict = {n:cond_var for n in [cov.col_names[int(x.shape[0]/2)],cov.col_names[int(3*x.shape[0]/4)]]}
    #know_dict = {cov.col_names[]:0.0001,cov.col_names[3]:0.0025}
    cond_cov = pyemu.helpers._condition_on_par_knowledge(cov,know_dict)
    i = int(x.shape[0] / 2)
    print(cov.x[i,i],cond_cov.x[i,i])
    print(np.diag(cov.x).min(), np.diag(cond_cov.x).min())
    print(prior_var,cond_var)


    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,figsize=(15,5))
    # thres = 0.001
    # x1 = cov.to_pearson().x.copy()
    # x1[np.abs(x1) < thres] = np.nan
    # x2 = cond_cov.to_pearson().x.copy()
    # x2[np.abs(x2) < thres] = np.nan
    #
    # axes[0].imshow(x1,vmin=0.0,vmax=1.0)
    # axes[1].imshow(x2,vmin=0.0,vmax=1.0)
    # axes[2].plot(np.diag(cov.x),"0.5",label="prior diag")
    # axes[2].plot(np.diag(cond_cov.x), "b",label="post diag")
    # axes[0].set_title("prior CC matrix, variance {0}".format(prior_var),loc="left")
    # axes[1].set_title("post CC matrix, conditional variance {0}".format(cond_var), loc="left")
    # axes[2].set_title("prior vs posterior diagonals", loc="left")
    #
    # plt.show()

def geostat_prior_builder2_test(tmp_path):
    import os
    import numpy as np
    import pyemu
    for fname in [Path("pst", "pest.pst"),
                  Path("utils", "pp_locs.tpl")]:
        shutil.copy(fname, tmp_path)
    pst_file = os.path.join(tmp_path, "pest.pst")
    pst = pyemu.Pst(pst_file)
    tpl_file = os.path.join(tmp_path, "pp_locs.tpl")
    df = pyemu.pp_utils.pp_tpl_to_dataframe(tpl_file).iloc[:200,:]
    df.loc[:,"x"] = np.arange(df.shape[0])
    df.loc[:,"y"] = 0.0
    print(df)
    v = pyemu.geostats.ExpVario(1.0,10.0)
    gs = pyemu.geostats.GeoStruct(variograms=v)

    # get a cov here where all pars in the same group have the same bounds so same variance
    cov1 = pyemu.helpers.geostatistical_prior_builder(pst,{gs:df})
    
    par = pst.parameter_data
    #give some pars narrower bounds to induce a lower variance 
    #par.loc[pst.par_names[10:40], "parubnd"] = par.loc[pst.par_names[10:40], "parval1"] * 1.5
    #par.loc[pst.par_names[10:40], "parlbnd"] = par.loc[pst.par_names[10:40], "parval1"] * 0.5
    par.loc[pst.par_names[10:100], "parubnd"] *= np.random.random(90) * 5
    par.loc[pst.par_names[10:100], "parlbnd"] *= np.random.random(90) * 0.5
    
    
    # get a diagonal bounds-based cov
    cov = pyemu.Cov.from_parameter_data(pst=pst)
    d = cov.x
    var_dict = {n:v for n,v in zip(cov.row_names,d)}
    know_dict = {n:var_dict[n] for n in pst.par_names[10:100]}
    #calc the conditional cov just for testing
    cov3 = pyemu.helpers._condition_on_par_knowledge(cov1,know_dict)    
    
    #this one should include the variance scaling
    cov2 = pyemu.helpers.geostatistical_prior_builder(pst, {gs: df})

    pe = pyemu.helpers.geostatistical_draws(pst, {gs: df}, 100000)
    pe = pe.loc[:,pst.par_names]
    ecov2 = pe.covariance_matrix()

    x1 = cov1.x.copy()
    x1[np.abs(cov1.to_pearson().x)<0.001] = np.nan
    x2 = cov2.x.copy()
    x2[np.abs(cov2.to_pearson().x) < 0.001] = np.nan
    ex2 = ecov2.x.copy()
    ex2[np.abs(ecov2.to_pearson().x) < 0.001] = np.nan
    x3 = cov3.x.copy()
    x3[np.abs(cov3.to_pearson().x) < 0.001] = np.nan

    # even tho we scaled cov2, the resulting corr coef matrix should be the same as cov1
    d = np.abs(cov1.to_pearson().x - cov2.to_pearson().x)
    print(d.max())
    assert d.max() < 1.0e-6
    
    #check that variances in cov2 match the diagonal bounds-based cov
    dd = np.diag(cov2.x)
    d = np.abs(cov.x.flatten() - dd)
    print(d.max())
    assert d.max() < 1.0e-6

    # check that empirical variances in cov2 match the diagonal bounds-based cov
    edd = np.diag(ecov2.x)
    ed = np.abs(cov.x.flatten()[10:100] - edd[10:100])
    print(ed.max())
    assert ed.max() < 1.0e-1

    #import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(1,1)
    # ax.plot(ed)
    # axt = plt.twinx(ax)
    # axt.plot(dd[10:100],"0.5")
    # axt.plot(edd[10:100], "m")
    # axt.plot(cov.x.flatten()[10:100],"b--")
    # ax.set_xticks(np.arange(ed.shape[0]))
    # ax.set_xticklabels(pst.par_names[10:100],rotation=90)
    # plt.show()
    #
    #fig,axes = plt.subplots(1,3,figsize=(15,5))
    #axes[0].imshow(x1[:200,:200],cmap="jet",vmax=np.nanmax(x2[:200,:200]),vmin=np.nanmin(x2[:200,:200]))
    #axes[1].imshow(x2[:200, :200], cmap="jet",vmax=np.nanmax(x2[:200,:200]),vmin=np.nanmin(x2[:200,:200]))
    #axes[2].imshow(ex2[:200, :200], cmap="jet",vmax=np.nanmax(x2[:200,:200]),vmin=np.nanmin(x2[:200,:200]))
    #plt.show()


def temporal_draw_invest():
    import numpy as np
    import pandas as pd
    import pyemu
    import matplotlib.pyplot as plt
    from datetime import datetime
    v = pyemu.geostats.ExpVario(contribution=1.0,a=500)
    gs = pyemu.geostats.GeoStruct(variograms=v)

    t = np.arange(0,1000)
    y = np.zeros_like(t)
    names = ["p{0}".format(i) for i in range(t.shape[0])]
    df = pd.DataFrame({"parnme":names,"x":t,"y":y})

    pst = pyemu.Pst.from_par_obs_names(names,names)
    #pst.parameter_data.loc[:,"parlbnd"] = 0.5
    #pst.parameter_data.loc[:, "parubnd"] = 1.5

    cov = gs.covariance_matrix(x=t,y=y,names=names)
    #plt.imshow(cov.x)
    #plt.show()
    s = datetime.now()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=10)

    e = datetime.now()
    print("took",(e-s).total_seconds())
    ecov = pe.loc[:,names].covariance_matrix()
    #plt.imshow(ecov.x)
    #plt.show()
    #plt.plot(pe.loc[pe.index[0]])
    #plt.show()


def maha_pdc_test(tmp_path):
    import pyemu
    # pst = pyemu.Pst(os.path.join("temp_files","freyberg_mf6.pst"))
    # obs = pst.observation_data
    # oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=os.path.join("temp_files","freyberg_mf6.0.obs.csv"))
    # z_scores, dmxs = pyemu.utils.maha_based_pdc(oe)
    # print(z_scores)
    # return
    pst_file = os.path.join("utils","freyberg6.pst")
    shutil.copy(pst_file, tmp_path)
    oe_file = os.path.join("utils", "freyberg6.0.obs.csv")
    shutil.copy(oe_file, tmp_path)
    oe_file = os.path.join(tmp_path, "freyberg6.0.obs.csv")
    pst = pyemu.Pst(os.path.join(tmp_path,"freyberg6.pst"))

    obs = pst.observation_data
    obs.loc[obs.weight>0,"obsval"] -= 5
    oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=oe_file)
    df, dmxs = pyemu.utils.maha_based_pdc(oe)
    print(df.z_scores)
    print(df.p_vals)

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # obs.loc[:, "datetime"] = pd.to_datetime(obs.obsnme.apply(lambda x: x.split("_")[-1]))
    # for group in pst.nnz_obs_groups:
    #     oobs = obs.loc[obs.obgnme==group,:].copy()
    #     oobs = oobs.loc[oobs.weight > 0]
    #     oobs.sort_values(by="datetime")
    #     fig,ax = plt.subplots(1,1,figsize=(10,4))
    #     for real in oe.index:
    #         ax.plot(oobs.datetime,oe.loc[real,oobs.obsnme].values,"0.5",lw=0.1)
    #     ax.set_title("group:{0}, zscore:{1}, pval:{2}".\
    #                  format(group,df.z_scores.loc[group],df.p_vals.loc[group]))
    #     ax.plot(oobs.datetime,oobs.obsval,"r",lw=2)
    # plt.show()

def rmr_parse_test():
    import pyemu
    df = pyemu.helpers.parse_rmr_file(os.path.join("utils","pest_local_pdc.rmr"))


def ac_draw_test(tmp_path):
    import pyemu
    import numpy as np
    #import matplotlib.pyplot as plt

    obs_per_group = 20
    avals = [1,180,365,3650]
    ngrp = len(avals)

    onames = []
    ogrps = []
    distance = []
    obsval = []
    struct_dict = {}
    for igrp,aval in enumerate(avals):
        x = np.linspace(0,np.pi*10,obs_per_group)
        trend = 0
        y = 3**(np.sin(x) + 10 + trend)
        obsval.extend(list(y))
        onamess = ["obs{0:04d}_grp{1:03d}_a{2}".format(i,igrp,aval) for i in range(obs_per_group)]
        onames.extend(onamess)
        ogrps.extend([igrp for _ in range(obs_per_group)])
        distance.extend(list(np.arange(obs_per_group)))
        v = pyemu.geostats.ExpVario(contribution=1.0, a=aval)
        gs = pyemu.geostats.GeoStruct(variograms=v)
        struct_dict[gs] = onamess
    onames.append("rand1")
    onames.append("rand2")
    obsval.append(1.0)
    obsval.append(2.0)
    ogrps.append("less_")
    ogrps.append("greater_")
    onames.append("zero1")
    obsval.append(1.0)
    ogrps.append("zero")

    onames.append("zero2")
    obsval.append(1.0)
    ogrps.append("less_zero")
    onames.append("zero3")
    obsval.append(1.0)
    ogrps.append("greater_zero")

    pst = pyemu.Pst.from_par_obs_names(obs_names=onames)
    pst.observation_data.loc[onames,"obgnme"] = ogrps
    pst.observation_data.loc[onames[:-5],"distance"] = distance
    pst.observation_data.loc[onames,"obsval"] = obsval
    pst.observation_data.loc[onames, "weight"] = 0.000001#1/(np.array(obsval))
    pst.observation_data.loc[["zero1","zero2","zero3"],"weight"] = 0.0
    print(obsval)
    pst.observation_data.loc[onames, "standard_deviation"] = np.array(obsval) * 0.1
    pst.observation_data.loc[onames, "lower_bound"] = np.array(obsval).min()
    #pst.observation_data.loc[onames, "upper_bound"] = np.array(obsval) + (
    #            pst.observation_data.loc[onames, "standard_deviation"] * 2)
    pst.observation_data.loc[onames, "upper_bound"] = np.array(obsval).max()
    print(pst.observation_data.standard_deviation.describe())
    pst.write(os.path.join(tmp_path, "test.pst"))
    print(pst.observation_data.distance)

    np.random.seed(pyemu.en.SEED)
    oe = pyemu.helpers.autocorrelated_draw(pst, struct_dict, num_reals=100, enforce_bounds=True)
    np.random.seed(pyemu.en.SEED)
    oe2 = pyemu.helpers.autocorrelated_draw(pst, struct_dict, num_reals=100, enforce_bounds=True)
    diff = oe - oe2
    print(diff.max())
    assert diff.max().max() == 0.0
    
    obs = pst.observation_data
    assert oe.max().max() <= obs.upper_bound.min()
    assert oe.min().min() >= obs.lower_bound.max()
    assert np.all(oe.loc[:,"rand1"].values==1.0)
    assert np.all(oe.loc[:, "rand2"].values == 2.0)

    oe = pyemu.helpers.autocorrelated_draw(pst,struct_dict,num_reals=8000)

    obs = pst.observation_data
    obs["emp_std"] = oe.std().loc[obs.obsnme]
    obs["std_diff"] = 100 * np.abs(obs.emp_std-obs.standard_deviation)/obs.emp_std
    sel = obs.emp_std != 0
    print(obs[sel].std_diff.min(), obs[sel].std_diff.max())
    assert obs[sel].std_diff.max() < 5.0


    # pst.observation_data.loc[:,"upper_bound"] = np.nan
    # oe = pyemu.helpers.autocorrelated_draw(pst, struct_dict, num_reals=100,enforce_bounds=True)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(len(avals), 1, figsize=(10, 10))
    # for gs, ax in zip(struct_dict, axes):
    #     onames = struct_dict[gs]
    #     dvals = pst.observation_data.loc[onames, "distance"].values
    #
    #     for real in oe.index:
    #         ax.plot(dvals, oe.loc[real, onames].values, "0.5", alpha=0.5, lw=0.1)
    #     ax.plot(dvals, pst.observation_data.loc[onames, "obsval"], "r")
    #     ax.set_title("correlation length: {0} time units".format(gs.variograms[0].a), loc="left")
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("something autocorrelated")
    #     ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig("test.pdf")
    #for o,s in std.items():
    #    d = 100 * np.abs(s-obs.loc[o,"standard_deviation"])/s
    #    print(o,d,s,obs.loc[o,"standard_deviation"])
    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(1,1)
    # ax.scatter(obs.standard_deviation,obs.emp_std)
    # mn = min(ax.get_ylim()[0],ax.get_xlim()[0])
    # mx = max(ax.get_ylim()[1], ax.get_xlim()[1])
    # ax.set_xlim(mn,mx)
    # ax.set_ylim(mn,mx)
    # ax.plot([mn,mx],[mn,mx],"k--",lw=2.5)
    # plt.show()


def test_fake_frun(tmp_path):
    from pst_from_tests import ies_exe_path, setup_freyberg_mf6
    pf, sim = setup_freyberg_mf6(tmp_path)
    v = pyemu.geostats.ExpVario(contribution=1.0, a=500)
    gs = pyemu.geostats.GeoStruct(variograms=v, transform='log')
    pf.add_parameters(
        "freyberg6.npf_k_layer1.txt",
        par_type="grid",
        geostruct=gs,
        pargp=f"hk_k:{0}"
    )
    pf.add_observations(
        "heads.csv",
        index_cols=['time'],
        obsgp="head"
    )
    pst = pf.build_pst()
    pst = pyemu.utils.setup_fake_forward_run(pst, "fake.pst", pf.new_d,
                                             new_cwd=pf.new_d)
    pyemu.os_utils.run(f"{ies_exe_path} fake.pst", cwd=pf.new_d)
    bd = Path.cwd()
    os.chdir(pf.new_d)
    try:
        pyemu.utils.calc_array_par_summary_stats("mult2model_info.csv")
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)
    pyemu.os_utils.run(f"{ies_exe_path} fake.pst", cwd=pf.new_d, use_sp=True)
    os.chdir(pf.new_d)
    try:
        pyemu.utils.calc_array_par_summary_stats("mult2model_info.csv")
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)

def obs_ensemble_quantile_test():
    import os
    import numpy as np
    import pyemu
    pst_file = os.path.join("pst","pest.pst")
    pst = pyemu.Pst(pst_file)

    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst,cov=pyemu.Cov.from_observation_data(pst),num_reals=100,fill=True)

    quans = [0.25,0.5,0.75]
    qt,d = pyemu.helpers.calc_observation_ensemble_quantiles(oe,pst,quans)


def thresh_pars_test():
    import os
    import shutil
    import numpy as np
    import pyemu
    test_d = "thresh_test"
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    os.makedirs(test_d)
    dim = 500
    arr = np.ones((dim,dim))
    gs = pyemu.geostats.GeoStruct(variograms=[pyemu.geostats.ExpVario(1.0,30.0)])
    ss = pyemu.geostats.SpecSim2d(np.ones(dim),np.ones(dim),gs)
    #seed = np.random.randint(100000)
    np.random.seed(9371)
    #print("seed",seed)
    arr = 10**(ss.draw_arrays()[0])
    print(arr)

    inact_arr = np.ones_like(arr,dtype=int)
    inact_arr[:50,:] = 0
    orgarr_file = os.path.join(test_d,"org_arr.dat")
    np.savetxt(orgarr_file,arr,fmt="%15.6E")
    p1 = np.percentile(arr,5)
    p2 = np.percentile(arr,95)
    cat_dict = {1:[0.4,p1],2:[0.6,p2]}
    pyemu.helpers.setup_threshold_pars(
        orgarr_file, cat_dict,
        testing_workspace=test_d, inact_arr=inact_arr)


    newarr = np.loadtxt(orgarr_file)
    print(newarr)
    newarr[inact_arr==0] = 0.0
    print(np.unique(newarr))

    tarr = np.zeros_like(newarr)
    tarr[np.isclose(newarr,cat_dict[1][1],rtol=1e-5,atol=1e-5)] = 1.0
    #tarr[inact_arr==0] = np.nan
    tot = inact_arr.sum()
    prop = np.nansum(tarr) / tot
    print(prop,cat_dict[1])
    print(np.nansum(tarr),tot)
    if not np.isclose(prop,cat_dict[1][0],0.01):
        print("cat_dict 1,{0} vs {1}, tot:{2}, prop:{3}".format(prop,cat_dict[1],tot,np.nansum(tarr)))

    tarr = np.zeros_like(newarr)
    tarr[np.isclose(newarr, cat_dict[2][1])] = 1.0
    prop = tarr.sum() / tot
    print(prop, cat_dict[2])
    assert np.isclose(prop, cat_dict[2][0],0.01),"cat_dict 2,{0} vs {1}".format(prop,cat_dict[2])

    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,2,figsize=(10,5))
    # arr = np.log10(arr)
    # arr[inact_arr==0] = np.nan
    # newarr = np.log10(newarr)
    # cb = axes[0].imshow(arr)
    # plt.colorbar(cb,ax=axes[0])
    # cb = axes[1].imshow(newarr,vmax=np.nanmax(arr),vmin=np.nanmin(arr))
    # plt.colorbar(cb,ax=axes[1])

    # plt.show()


def test_ppu_import():
    import pypestutils as ppu
    pass


@pytest.mark.timeout(method="thread")
def test_ppu_geostats(tmp_path):
    import sys
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pyemu
    
    import flopy

    # don't need on CI and can cause issues if wanting to use
    # env version of ppu not a local one.
    # sys.path.insert(0,os.path.join("..","..","pypestutils"))

    # quick test of ppu import
    import pypestutils as ppu

    o_model_ws = os.path.join("..","examples","Freyberg","extra_crispy")
    model_ws = os.path.join(tmp_path, "extra_crispy")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    shutil.copytree(o_model_ws, model_ws)
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,check=False)
    pp_dir = os.path.join(tmp_path)
    #ml.export(os.path.join("temp","test_unrot_grid.shp"))
    sr = pyemu.helpers.SpatialReference().from_namfile(
        os.path.join(ml.model_ws, ml.namefile),
        delc=ml.dis.delc, delr=ml.dis.delr)
    sr.rotation = 0.
    par_info_unrot = pyemu.pp_utils.setup_pilotpoints_grid(
        sr=sr, prefix_dict={0: "hk1",1:"hk2"},
        every_n_cell=6, pp_dir=pp_dir, tpl_dir=pp_dir,
        shapename=os.path.join(tmp_path, "test_unrot.shp"),
    )
    #print(par_info_unrot.parnme.value_counts())
    par_info_unrot.loc[:,"parval1"] = np.random.uniform(10,100,par_info_unrot.shape[0])
    gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(a=1000,contribution=1.0,anisotropy=3.0,bearing=45))
    ok = pyemu.geostats.OrdinaryKrige(gs,par_info_unrot)
    ppu_factor_filename = Path(tmp_path, "ppu_factors.dat")
    pyemu_factor_filename = Path(tmp_path, "pyemu_factors.dat")

    ok.calc_factors_grid(sr, try_use_ppu=False)
    ok.to_grid_factors_file(pyemu_factor_filename)
    ok.calc_factors_grid(sr, try_use_ppu=True,
                         ppu_factor_filename=ppu_factor_filename)
    out_file = Path(tmp_path, "pyemu_array.dat")
    pyemu.geostats.fac2real(par_info_unrot,
                            pyemu_factor_filename, out_file=out_file)
    out_file_ppu = Path(tmp_path, "ppu_array.dat")
    pyemu.geostats.fac2real(par_info_unrot,
                            ppu_factor_filename, out_file=out_file_ppu)
    arr_ppu = np.loadtxt(out_file_ppu)
    arr = np.loadtxt(out_file)
    diff = 100 * np.abs(arr - arr_ppu) / np.abs(arr)
    assert diff.max() < 1.0
    # fig,axes = plt.subplots(1,3,figsize=(10,10))
    # cb = axes[0].imshow(arr)
    # plt.colorbar(cb, ax=axes[0])
    #
    # cb = axes[1].imshow(arr_ppu,vmin=arr.min(),vmax=arr.max())
    # plt.colorbar(cb, ax=axes[1])
    #
    # cb = axes[2].imshow(diff)
    # plt.colorbar(cb,ax=axes[2])
    # plt.show()
    # exit()

def ppw_worker(id_num,case,t_d,host,port,frun):
    import numpy as np
    ppw = pyemu.os_utils.PyPestWorker(os.path.join(t_d,"{0}.pst".format(case)),
                                      host,port,verbose=False)
    
    obs = ppw._pst.observation_data
    count = 0
    par = ppw._pst.parameter_data
    dpar = par.loc[par.parnme.str.startswith("dv"),:].copy()
    assert len(dpar) > 0
    dpar["count"] = dpar.parnme.apply(lambda x: int(x.split('_')[1]))
    dpar.sort_values(by="count",inplace=True)
    pnames = dpar.parnme.tolist()
    print("starting worker",id_num)
    while True:
        parameters = ppw.get_parameters()
        if parameters is None:
            print("no more parameters...")
            break
        #print("parameters",parameters.loc[pnames])
        #print("got parameters:",parameters.values)
        objs,constr = frun(pvals=parameters.loc[pnames].values)
        #print("objs and constraints:",objs,constr)
        obs.loc[["obj_1","obj_2"],"obsval"] = np.array(objs)
        #print(obs)
        ppw.send_observations(obs.obsval.loc[ppw.obs_names].values)
        #input("press any key")
        #print("worker",id_num,"finished run",ppw.net_pack.runid)


@pytest.mark.timeout(method="thread")
def test_pypestworker(tmp_path):
    from datetime import datetime
    import numpy as np
    import subprocess as sp
    import multiprocessing as mp
    import sys
    import time

    host = "localhost"
    port = 4111
    case = "constr"
    org_d = os.path.join("utils","{0}_template".format(case))
    t_d = Path(tmp_path, "{0}_ppw_template".format(case))
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d,t_d)
    pst = pyemu.Pst(os.path.join(t_d,"{0}.pst".format(case)))
    pst.pestpp_options["mou_population_size"] = 20
    #need these options bc the py workers run so fast, even slight 
    #delays show up as timeouts...
    pst.pestpp_options["overdue_giveup_fac"] = 1e10
    pst.pestpp_options["overdue_resched_fac"] = 1e10
    
    pst.control_data.noptmax = 2
    pst.write(os.path.join(t_d,"{0}.pst".format(case)),version=2)
    sys.path.insert(1, t_d.as_posix())
    from forward_run import helper as frun

    m_d = tmp_path / "{0}_ppw_master".format(case)

    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d,m_d)

    # start the master
    start = datetime.now()
    b_d = os.getcwd()
    os.chdir(m_d)
    try:
        p = sp.Popen([mou_exe_path, "{0}.pst".format(case), "/h", ":{0}".format(port)], stderr=sp.PIPE)
    except Exception as e:
        print("failed to start master process")
        os.chdir(b_d)
        raise e
    os.chdir(b_d)
    #p.wait()
    #return

    num_workers=5

    # looper over and start the workers - in this
    # case they dont need unique dirs since they aren't writing
    # anything
    # little pause to let master get going (and possibly fail)
    time.sleep(5)
    procs = []
    for i in range(num_workers):
        # check master still running before deploying worker
        if p.poll() is not None:
            err = p.stderr.read()
            raise RuntimeError("master process failed before all workers started:\n\n"+
                               err.decode())
        try:  # make sure we kill the master if worker startup returns an error
            pp = mp.Process(target=ppw_worker,args=(i,case,t_d,host,port,frun))
            # procs.append(pp)
            pp.start()
            procs.append(pp)
        except Exception as e:
            print("failed to start worker {0}".format(i))
            p.terminate()
            raise e
    # if everything worked, the workers should receive the
    # shutdown signal from the master and exit gracefully...
    for i, pp in enumerate(procs):
        try:  # make sure we kill the master if worker startup returns an error
            pp.join()
        except Exception as e:
            print(f"exception thrown by worker {i}")
            p.terminate()
            raise e

    # wait for the master to finish...but should already be finished
    p.wait()

    finish = datetime.now()
    print("all done, took",(finish-start).total_seconds())
    # pop sys.path change (just in case it persists)
    sys.path.pop(1)
    start2 = datetime.now()
    #pyemu.os_utils.start_workers(t_d,mou_exe_path,"{0}.pst".format(case),num_workers=num_workers,worker_root='.',master_dir=m_d2)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=t_d)
    m_d2 = t_d
    finish2 = datetime.now()
    print("ppw took",(finish-start).total_seconds())
    print("org took",(finish2-start2).total_seconds())
    arc1 = pd.read_csv(os.path.join(m_d,"{0}.pareto.archive.summary.csv".format(case)))
    arc2 = pd.read_csv(os.path.join(m_d2,"{0}.pareto.archive.summary.csv".format(case)))
    diff1 = np.abs((arc1["obj_1"] - arc2["obj_1"]).values)
    diff2 = np.abs((arc1["obj_2"] - arc2["obj_2"]).values)
    print(diff1.max())
    print(diff2.max())
    assert diff1.max() < 1.0e-6
    assert diff2.max() < 1.0e-6


def gpr_compare_invest():
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "zdt1"
    use_chances = False
    m_d = os.path.join(case+"_gpr_baseline")
    org_d = os.path.join("utils",case+"_template")
    t_d = case+"_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d,t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5
   
    pop_size = 60
    num_workers = 60
    noptmax_full = 30
    noptmax_inner = 10
    noptmax_outer = 5
    port = 4554
    pst.control_data.noptmax = noptmax_full 
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case+".pst"))
    if not os.path.exists(m_d):
        pyemu.os_utils.start_workers(t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                    master_dir=m_d, verbose=True, port=port)
    #shutil.copytree(t_d,m_d)
    #pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=m_d)
    # use the initial population files for training
    dv_pops = [os.path.join(m_d,"{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_","obs_") for f in dv_pops]

    pst_fname = os.path.join(m_d,case+".pst")
    gpr_t_d = os.path.join(case+"_gpr_template")
    pyemu.helpers.prep_for_gpr(pst_fname,dv_pops,obs_pops,t_d=m_d,gpr_t_d=gpr_t_d,nverf=int(pop_size*.1),\
                               plot_fits=True,apply_standard_scalar=False,include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d,case+".pst"))
    shutil.copy2(os.path.join(m_d,case+".0.dv_pop.csv"),os.path.join(gpr_t_d,"initial_dv_pop.csv"))
    gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d,case+".pst"),version=2)
    gpr_m_d = gpr_t_d.replace("template","master")
    if os.path.exists(gpr_m_d):
         shutil.rmtree(gpr_m_d)
    pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                        master_dir=gpr_m_d, verbose=True, port=port)

    #o1 = pd.read_csv(os.path.join(m_d,case+".{0}.obs_pop.csv".format(max(0,pst.control_data.noptmax))))
    o1 = pd.read_csv(os.path.join(m_d,case+".pareto.archive.summary.csv"))
    o1 = o1.loc[o1.generation == o1.generation.max(), :]
    o1 = o1.loc[o1.is_feasible == True, :]
    o1 = o1.loc[o1.nsga2_front == 1, :]


    import matplotlib.pyplot as plt
    o2 = pd.read_csv(os.path.join(gpr_m_d, case + ".{0}.obs_pop.csv".format(max(0, gpst.control_data.noptmax))))
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(o1.obj_1,o1.obj_2,c="r",s=10)
    ax.scatter(o2.obj_1,o2.obj_2,c="0.5",s=10,alpha=0.5)
    plt.tight_layout()
    plt.savefig("gpr_{0}_compare_noiter.pdf".format(case))
    plt.close(fig)

    # now lets try an inner-outer scheme...
    
    gpst.control_data.noptmax = noptmax_inner
    gpst.write(os.path.join(gpr_t_d,case+".pst"),version=2)
    gpr_t_d_iter = gpr_t_d+"_outeriter{0}".format(0)
    if os.path.exists(gpr_t_d_iter):
        shutil.rmtree(gpr_t_d_iter)
    shutil.copytree(gpr_t_d,gpr_t_d_iter)
    for iouter in range(1,noptmax_outer+1):
        #run the gpr emulator
        gpr_m_d_iter = gpr_t_d_iter.replace("template","master")
        complex_m_d_iter = t_d.replace("template", "master_complex_retrain_outeriter{0}".format(iouter))
        if os.path.exists(gpr_m_d_iter):
            shutil.rmtree(gpr_m_d_iter)
        pyemu.os_utils.start_workers(gpr_t_d_iter, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                        master_dir=gpr_m_d_iter, verbose=True, port=port)
        o2 = pd.read_csv(os.path.join(gpr_m_d_iter,case+".{0}.obs_pop.csv".format(gpst.control_data.noptmax)))

        # now run the final dv pop thru the "complex" model
        final_gpr_dvpop_fname = os.path.join(gpr_m_d_iter,case+".archive.dv_pop.csv")
        assert os.path.exists(final_gpr_dvpop_fname)
        complex_model_dvpop_fname = os.path.join(t_d,"gpr_outeriter{0}_dvpop.csv".format(iouter))
        if os.path.exists(complex_model_dvpop_fname):
            os.remove(complex_model_dvpop_fname)
        # load the gpr archive and do something clever to pick new points to eval
        # with the complex model
        dvpop = pd.read_csv(final_gpr_dvpop_fname,index_col=0)
        if dvpop.shape[0] > pop_size:
            arc_sum = pd.read_csv(os.path.join(gpr_m_d_iter,case+".pareto.archive.summary.csv"))
            as_front_map = {member:front for member,front in zip(arc_sum.member,arc_sum.nsga2_front)}
            as_crowd_map = {member: crowd for member, crowd in zip(arc_sum.member, arc_sum.nsga2_crowding_distance)}
            as_feas_map = {member: feas for member, feas in zip(arc_sum.member, arc_sum.feasible_distance)}
            as_gen_map = {member: gen for member, gen in zip(arc_sum.member, arc_sum.generation)}

            dvpop.loc[:,"front"] = dvpop.index.map(lambda x: as_front_map.get(x,np.nan))
            dvpop.loc[:, "crowd"] = dvpop.index.map(lambda x: as_crowd_map.get(x, np.nan))
            dvpop.loc[:,"feas"] = dvpop.index.map(lambda x: as_feas_map.get(x,np.nan))
            dvpop.loc[:, "gen"] = dvpop.index.map(lambda x: as_gen_map.get(x, np.nan))
            #drop members that have missing archive info
            dvpop = dvpop.dropna()
            if dvpop.shape[0] > pop_size:
                dvpop.sort_values(by=["gen","feas","front","crowd"],ascending=[False,True,True,False],inplace=True)
                dvpop = dvpop.iloc[:pop_size,:]
            dvpop.drop(["gen","feas","front","crowd"],axis=1,inplace=True)

        #shutil.copy2(final_gpr_dvpop_fname,complex_model_dvpop_fname)
        dvpop.to_csv(complex_model_dvpop_fname)
        pst.pestpp_options["mou_dv_population_file"] = os.path.split(complex_model_dvpop_fname)[1]
        pst.control_data.noptmax = -1
        pst.write(os.path.join(t_d,case+".pst"),version=2)

        pyemu.os_utils.start_workers(t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                    master_dir=complex_m_d_iter, verbose=True, port=port)

        # plot the complex model results...
        o2 = pd.read_csv(os.path.join(complex_m_d_iter, case + ".pareto.archive.summary.csv"))
        o2 = o2.loc[o2.generation == o2.generation.max(), :]
        #o2 = o2.loc[o2.is_feasible==True,:]
        o2 = o2.loc[o2.nsga2_front == 1, :]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(o1.obj_1, o1.obj_2,c="r",s=10,label="full complex")
        ax.scatter(o2.obj_1, o2.obj_2,c="0.5",s=10,alpha=0.5,label="mixed emulated-complex")
        ax.legend(loc="upper right")
        ax.set_xlim(0,10)
        ax.set_ylim(0,20)
        plt.tight_layout()
        plt.savefig("gpr_{0}_compare_iterscheme_{1}.pdf".format(case,iouter))
        plt.close(fig)

        # now add those complex model input-output pop files to the list and retrain
        # the gpr
        dv_pops.append(os.path.join(complex_m_d_iter,case+".0.dv_pop.csv"))
        obs_pops.append(os.path.join(complex_m_d_iter,case+".0.obs_pop.csv"))
        gpr_t_d_iter = gpr_t_d+"_outeriter{0}".format(iouter)
        pyemu.helpers.prep_for_gpr(pst_fname,dv_pops,obs_pops,t_d=gpr_t_d,gpr_t_d=gpr_t_d_iter,nverf=int(pop_size*.1),
                                   plot_fits=True,apply_standard_scalar=False,include_emulated_std_obs=True)
        gpst_iter = pyemu.Pst(os.path.join(gpr_t_d_iter,case+".pst"))
        #aggdf = pd.read_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"),index_col=0)
        #aggdf.index = ["outeriter{0}_member{1}".format(iouter,i) for i in range(aggdf.shape[0])]
        restart_gpr_dvpop_fname = "gpr_restart_dvpop_outeriter{0}.csv".format(iouter)
        #aggdf.to_csv(os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        shutil.copy2(os.path.join(complex_m_d_iter,case+".0.dv_pop.csv"),os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        gpst_iter.pestpp_options["mou_dv_population_file"] = restart_gpr_dvpop_fname
        gpst_iter.control_data.noptmax = gpst.control_data.noptmax
        gpst_iter.write(os.path.join(gpr_t_d_iter,case+".pst"),version=2)


def gpr_constr_invest():
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "constr"
    use_chances = False
    m_d = os.path.join(case + "_gpr_baseline")
    org_d = os.path.join("utils", case + "_template")
    t_d = case + "_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d, t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5

    pop_size = 15
    num_workers = 5
    noptmax_full = 3
    noptmax_inner = 2
    noptmax_outer = 2
    port = 4554
    pst.control_data.noptmax = -1
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case + ".pst"))
    #if not os.path.exists(m_d):
    #    pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                                 master_dir=m_d, verbose=True, port=port)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d,m_d)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=m_d)
    # use the initial population files for training
    dv_pops = [os.path.join(m_d, "{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_", "obs_") for f in dv_pops]

    pst_fname = os.path.join(m_d, case + ".pst")
    gpr_t_d = os.path.join(case + "_gpr_template")
    pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops,t_d=m_d, gpr_t_d=gpr_t_d, nverf=int(pop_size * .1), \
                               plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d, case + ".pst"))
    #shutil.copy2(os.path.join(m_d, case + ".0.dv_pop.csv"), os.path.join(gpr_t_d, "initial_dv_pop.csv"))
    #gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.pestpp_options.pop("mou_dv_population_file",None) #= "initial_dv_pop.csv"
    
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_m_d = gpr_t_d.replace("template", "master")
    if os.path.exists(gpr_m_d):
        shutil.rmtree(gpr_m_d)
    #pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                             master_dir=gpr_m_d, verbose=True, port=port)
    shutil.copytree(gpr_t_d,gpr_m_d)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_m_d)
    
    # o1 = pd.read_csv(os.path.join(m_d,case+".{0}.obs_pop.csv".format(max(0,pst.control_data.noptmax))))
    o1 = pd.read_csv(os.path.join(m_d, case + ".pareto.archive.summary.csv"))
    o1 = o1.loc[o1.generation == o1.generation.max(), :]
    o1 = o1.loc[o1.is_feasible == True, :]
    o1 = o1.loc[o1.nsga2_front == 1, :]

    # import matplotlib.pyplot as plt
    # o2 = pd.read_csv(os.path.join(gpr_m_d, case + ".{0}.obs_pop.csv".format(max(0, gpst.control_data.noptmax))))
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.scatter(o1.obj_1, o1.obj_2, c="r", s=10)
    # ax.scatter(o2.obj_1, o2.obj_2, c="0.5", s=10, alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("gpr_{0}_compare_noiter.pdf".format(case))
    # plt.close(fig)

    # now lets try an inner-outer scheme...

    gpst.control_data.noptmax = noptmax_inner
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_t_d_iter = gpr_t_d + "_outeriter{0}".format(0)
    if os.path.exists(gpr_t_d_iter):
        shutil.rmtree(gpr_t_d_iter)
    shutil.copytree(gpr_t_d, gpr_t_d_iter)
    for iouter in range(1, noptmax_outer + 1):
        # run the gpr emulator
        gpr_m_d_iter = gpr_t_d_iter.replace("template", "master")
        complex_m_d_iter = t_d.replace("template", "master_complex_retrain_outeriter{0}".format(iouter))
        if os.path.exists(gpr_m_d_iter):
            shutil.rmtree(gpr_m_d_iter)
        shutil.copytree(gpr_t_d_iter,gpr_m_d_iter)

        pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_m_d_iter)
    
        #pyemu.os_utils.start_workers(gpr_t_d_iter, mou_exe_path, case + ".pst", num_workers, worker_root=".",
        #                             master_dir=gpr_m_d_iter, verbose=True, port=port)
        
        o2 = pd.read_csv(os.path.join(gpr_m_d_iter, case + ".{0}.obs_pop.csv".format(gpst.control_data.noptmax)))

        # now run the final dv pop thru the "complex" model
        final_gpr_dvpop_fname = os.path.join(gpr_m_d_iter, case + ".archive.dv_pop.csv")
        assert os.path.exists(final_gpr_dvpop_fname)
        complex_model_dvpop_fname = os.path.join(t_d, "gpr_outeriter{0}_dvpop.csv".format(iouter))
        if os.path.exists(complex_model_dvpop_fname):
            os.remove(complex_model_dvpop_fname)
        # load the gpr archive and do something clever to pick new points to eval
        # with the complex model
        dvpop = pd.read_csv(final_gpr_dvpop_fname, index_col=0)
        if dvpop.shape[0] > pop_size:
            arc_sum = pd.read_csv(os.path.join(gpr_m_d_iter, case + ".pareto.archive.summary.csv"))
            as_front_map = {member: front for member, front in zip(arc_sum.member, arc_sum.nsga2_front)}
            as_crowd_map = {member: crowd for member, crowd in zip(arc_sum.member, arc_sum.nsga2_crowding_distance)}
            as_feas_map = {member: feas for member, feas in zip(arc_sum.member, arc_sum.feasible_distance)}
            as_gen_map = {member: gen for member, gen in zip(arc_sum.member, arc_sum.generation)}

            dvpop.loc[:, "front"] = dvpop.index.map(lambda x: as_front_map.get(x, np.nan))
            dvpop.loc[:, "crowd"] = dvpop.index.map(lambda x: as_crowd_map.get(x, np.nan))
            dvpop.loc[:, "feas"] = dvpop.index.map(lambda x: as_feas_map.get(x, np.nan))
            dvpop.loc[:, "gen"] = dvpop.index.map(lambda x: as_gen_map.get(x, np.nan))
            # drop members that have missing archive info
            dvpop = dvpop.dropna()
            if dvpop.shape[0] > pop_size:
                dvpop.sort_values(by=["gen", "feas", "front", "crowd"], ascending=[False, True, True, False],
                                  inplace=True)
                dvpop = dvpop.iloc[:pop_size, :]
            dvpop.drop(["gen", "feas", "front", "crowd"], axis=1, inplace=True)

        # shutil.copy2(final_gpr_dvpop_fname,complex_model_dvpop_fname)
        dvpop.to_csv(complex_model_dvpop_fname)
        pst.pestpp_options["mou_dv_population_file"] = os.path.split(complex_model_dvpop_fname)[1]
        pst.control_data.noptmax = -1
        pst.write(os.path.join(t_d, case + ".pst"), version=2)
        if os.path.exists(complex_m_d_iter):
            shutil.rmtree(complex_m_d_iter)
        shutil.copytree(t_d,complex_m_d_iter)
        #pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
        #                             master_dir=complex_m_d_iter, verbose=True, port=port)
        pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=complex_m_d_iter)
    
        # plot the complex model results...
        o2 = pd.read_csv(os.path.join(complex_m_d_iter, case + ".pareto.archive.summary.csv"))
        o2 = o2.loc[o2.generation == o2.generation.max(), :]
        # o2 = o2.loc[o2.is_feasible==True,:]
        o2 = o2.loc[o2.nsga2_front == 1, :]
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.scatter(o1.obj_1, o1.obj_2, c="r", s=10, label="full complex")
        # ax.scatter(o2.obj_1, o2.obj_2, c="0.5", s=10, alpha=0.5, label="mixed emulated-complex")
        # ax.legend(loc="upper right")
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 20)
        # plt.tight_layout()
        # plt.savefig("gpr_{0}_compare_iterscheme_{1}.pdf".format(case, iouter))
        # plt.close(fig)

        # now add those complex model input-output pop files to the list and retrain
        # the gpr
        dv_pops.append(os.path.join(complex_m_d_iter, case + ".0.dv_pop.csv"))
        obs_pops.append(os.path.join(complex_m_d_iter, case + ".0.obs_pop.csv"))
        gpr_t_d_iter = gpr_t_d + "_outeriter{0}".format(iouter)
        pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops, t_d=gpr_t_d,gpr_t_d=gpr_t_d_iter, nverf=int(pop_size * .1),
                                   plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
        gpst_iter = pyemu.Pst(os.path.join(gpr_t_d_iter, case + ".pst"))
        # aggdf = pd.read_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"),index_col=0)
        # aggdf.index = ["outeriter{0}_member{1}".format(iouter,i) for i in range(aggdf.shape[0])]
        #restart_gpr_dvpop_fname = "gpr_restart_dvpop_outeriter{0}.csv".format(iouter)
        # aggdf.to_csv(os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        #shutil.copy2(os.path.join(complex_m_d_iter, case + ".0.dv_pop.csv"),
        #             os.path.join(gpr_t_d_iter, restart_gpr_dvpop_fname))
        gpst_iter.pestpp_options.pop("mou_dv_population_file",None)# = restart_gpr_dvpop_fname
        gpst_iter.control_data.noptmax = gpst.control_data.noptmax
        gpst_iter.write(os.path.join(gpr_t_d_iter, case + ".pst"), version=2)

    psum_fname = os.path.join(complex_m_d_iter,case+".pareto.archive.summary.csv")
    assert os.path.exists(psum_fname)
    psum = pd.read_csv(psum_fname)
    #assert 1.0 in psum.obj_1.values
    #assert 1.0 in psum.obj_2.values
    

def gpr_zdt1_invest():
    import numpy as np
    import subprocess as sp
    import multiprocessing as mp
    from datetime import datetime
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "zdt1"
    use_chances = False
    m_d = os.path.join(case + "_gpr_baseline")
    org_d = os.path.join("utils", case + "_template")
    t_d = case + "_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d, t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["overdue_giveup_fac"] = 1e10
    pst.pestpp_options["overdue_resched_fac"] = 1e10
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5

    pop_size = 20
    num_workers = 10
    noptmax_full = 1
    
    port = 4569
    pst.control_data.noptmax = -1
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case + ".pst"))
    #if not os.path.exists(m_d):
    #    pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                                 master_dir=m_d, verbose=True, port=port)
    
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=t_d)
    

    m_d = t_d
    dv_pops = [os.path.join(m_d, "{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_", "obs_") for f in dv_pops]
 
    pst_fname = os.path.join(m_d, case + ".pst")
    gpr_t_d = os.path.join(case + "_gpr_template")
    pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops,t_d=m_d,gpr_t_d=gpr_t_d, nverf=int(pop_size * .1), \
                               plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d, case + ".pst"))
    shutil.copy2(os.path.join(m_d, case + ".0.dv_pop.csv"), os.path.join(gpr_t_d, "initial_dv_pop.csv"))
    gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_m_d = gpr_t_d.replace("template", "master")
    if os.path.exists(gpr_m_d):
        shutil.rmtree(gpr_m_d)
    start = datetime.now()
    #pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                             master_dir=gpr_m_d, verbose=True, port=port)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_t_d)
    gpr_m_d = gpr_t_d
 
    finish = datetime.now()
    duration1 = (finish - start).total_seconds()
    arcorg = pd.read_csv(os.path.join(gpr_m_d,"zdt1.archive.obs_pop.csv"),index_col=0)
    
 
    psum_fname = os.path.join(gpr_m_d,case+".pareto.archive.summary.csv")
    assert os.path.exists(psum_fname)
    psum = pd.read_csv(psum_fname)
    print(psum.obj_1.min())
    print(psum.obj_2.min())
    assert psum.obj_1.min() < 0.05 
    gpr_t_d2 = gpr_t_d + "_ppw"
    if os.path.exists(gpr_t_d2):
        shutil.rmtree(gpr_t_d2)
    shutil.copytree(gpr_t_d,gpr_t_d2)
 
    gpr_m_d2 = gpr_t_d2.replace("template","master")
 
    input_df = pd.read_csv(os.path.join(gpr_t_d2,"gpr_input.csv"),index_col=0)
    mdf = pd.read_csv(os.path.join(gpr_t_d2,"gprmodel_info.csv"),index_col=0)
    mdf["model_fname"] = mdf.model_fname.apply(lambda x: os.path.join(gpr_t_d2,x))
    pyemu.os_utils.start_workers(gpr_t_d2, mou_exe_path, case + ".pst", num_workers, worker_root=".",
                                 master_dir=gpr_m_d2, verbose=True, port=port,
                                 ppw_function=pyemu.helpers.gpr_pyworker,
                                 ppw_kwargs={"input_df":input_df,
                                            "mdf":mdf})
    
    
    arcppw = pd.read_csv(os.path.join(gpr_m_d2,"zdt1.archive.obs_pop.csv"),index_col=0)
    diff = np.abs(arcppw.values - arcorg.values)
    print(diff.max())
    assert diff.max() < 1e-6
         
    start = datetime.now()
    b_d = os.getcwd()
    os.chdir(gpr_t_d2)
    p = sp.Popen([mou_exe_path,"{0}.pst".format(case),"/h",":{0}".format(port)])
    os.chdir(b_d)
    #p.wait()
    #return
    
    # looper over and start the workers - in this
    # case they dont need unique dirs since they aren't writing
    # anything
    procs = []
    # try this test with 1 worker as an edge case
    num_workers = 1
    for i in range(num_workers):
        pp = mp.Process(target=gpr_zdt1_ppw)
        pp.start()
        procs.append(pp)
    # if everything worked, the the workers should receive the 
    # shutdown signal from the master and exit gracefully...
    for pp in procs:
        pp.join()
 
    # wait for the master to finish...but should already be finished
    p.wait()
    finish = datetime.now()
    print("ppw` took",(finish-start).total_seconds())
    print("org took",duration1)
 
    arcppw = pd.read_csv(os.path.join(gpr_t_d2,"zdt1.archive.obs_pop.csv"),index_col=0)
    diff = np.abs(arcppw.values - arcorg.values)
    print(diff.max())
    assert diff.max() < 1e-6
        

def gpr_zdt1_ppw():
    t_d = "zdt1_gpr_template"
    os.chdir(t_d)
    pst_name = "zdt1.pst"
    ppw = pyemu.helpers.gpr_pyworker(pst_name,"localhost",4569)
    os.chdir("..")


def pestpp_runstorage_file_test(tmp_path):
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    org_rns_file = os.path.join("utils","runstor.rns")
    rns_file = os.path.join(tmp_path,"runstor.rns")
    if os.path.exists(rns_file):
        os.remove(rns_file)
    shutil.copy2(org_rns_file,rns_file)
    rs = pyemu.helpers.RunStor(rns_file)
    header,par_names,obs_names = rs.file_info(rns_file)
    cols = ["n_runs","run_size","p_name_size","o_name_size","run_start"]
    for col in cols:
        assert col in header
        assert header[col] > 0
    df = rs.get_data()
    assert "run_status_label" in df.columns
    assert np.all(df.run_pos.values>0)

    for entry in df.info_txt:
        assert len(str(entry)) > 0
        assert "realization:" in entry
        assert "da_cycle:-9999" in entry
    assert df.run_status.sum() == 0
    org_df = df.copy()
    df["run_status"] = -100
    df.loc[:,par_names[0]] = -111
    df.loc[:,obs_names[-1]] = -222
    df["buffer_status"] = 1
    #df.loc[0,par_names] = -1111
    rs.update(df)
    rs2 = pyemu.helpers.RunStor(rns_file)
    header,par_names,obs_names = rs.file_info(rns_file)
    #print(header)
    df2 = rs2.get_data()
    #print(df2.shape)
    assert df.shape == df2.shape
    #print(df2.run_status_label)
    assert np.all(df2.run_status.values == -100)

    print(df2.loc[:,par_names[0]])
    assert np.all(df2.loc[:,par_names[0]].values == -111)
    print(df2.loc[:, obs_names[-1]])
    assert np.all(df2.loc[:,obs_names[-1]].values == -222)
    print(df2.buffer_status)
    #buffer status should always be 0 no matter what values are put in the dataframe
    assert df2.buffer_status.sum() == 0
    rs2.update(org_df)
    p1,o1,meta = pyemu.helpers.read_pestpp_runstorage(rns_file,irun="all", with_metadata=True)

    p2 = pd.read_csv(os.path.join("utils","runstor.0.par.csv"),index_col=0)

    diff = np.abs(p1.loc[:,p2.columns].values - p2.values)
    print(diff.max())
    assert diff.max() < 1.0e-7


if __name__ == "__main__":
    pestpp_runstorage_file_test(".")
    #geostat_draws_test('.')
    #fac2real_wrapped_test('.')
    #maha_pdc_test('.')
    #ppu_geostats_test(".")
    # test_pypestworker()
    #test_ppu_geostats(".")
    #test_pypestworker()
    #gpr_zdt1_test()
    #gpr_compare_invest()
    #gpr_constr_test()
    # import sys
    # t_d = "constr_ppw_template"
    # case = "constr"
    # sys.path.insert(0,t_d)
    # from forward_run import helper as frun
    # ppw_worker(0,case,t_d,"localhost",4004,frun)
    #test_pypestworker()
    # gpr_constr_test()
    #gpr_zdt1_test()
    #ac_draw_test(".")
    #while True:
    #    thresh_pars_test()
    #obs_ensemble_quantile_test()
    #geostat_draws_test("temp")
    # ac_draw_test("temp")
    # maha_pdc_test()
    # rmr_parse_test()
    # temporal_draw_invest()
    # run_test()
    # specsim_test()
    # aniso_invest()
    # fieldgen_dev()
    # smp_test()
    # smp_dateparser_test()
    # smp_to_ins_test()
    #read_runstor_test()
    # # long_names()
    # master_and_workers()
    # plot_id_bar_test()
    # pst_from_parnames_obsnames_test()
    # write_jactest_test()
    # sfr_obs_test()
    # sfr_reach_obs_test()
    #gage_obs_test('.')
    # setup_pp_test()
    # sfr_helper_test()
    #gw_sft_ins_test('.')
    # par_knowledge_test()
    # grid_obs_test()
    # hds_timeseries_test()
    # postprocess_inactive_conc_test()
    # plot_summary_test()
    # load_sgems_expvar_test()
    # read_hydmod_test()
    # make_hydmod_insfile_test()
    # gslib_2_dataframe_test()
    # sgems_to_geostruct_test()
    # #linearuniversal_krige_test()
    # conditional_prior_invest()
    # geostat_prior_builder_test2()
    # geostat_draws_test()
    # jco_from_pestpp_runstorage_test()
    # mflist_budget_test()
    # mtlist_budget_test()
    # tpl_to_dataframe_test()
    # kl_test()
    # hfb_test()
    # hfb_zn_mult_test()
    # more_kl_test()
    #zero_order_regul_test('.')
    #first_order_pearson_regul_test('.')
    # master_and_workers()
    # smp_to_ins_test()
    #read_runstor_test(".")
    #read_pestpp_runstorage_file_test()
    #jco_from_pestpp_runstorage_test(".")
    # write_tpl_test()
    #pp_to_shapefile_test(".")
    # read_pval_test()
    # read_hob_test()
    # setup_pp_test(".")
    # pp_to_tpl_test()
    # setup_ppcov_complex()
    # ppcov_complex_test()
    # setup_ppcov_simple()
    # ppcov_simple_sparse_test()
    # ppcov_complex_sparse_test()
    # fac2real_test()
    # vario_test()
    # geostruct_test()
    # aniso_test()
    # struct_file_test()
    # covariance_matrix_test()
    # add_pi_obj_func_test()
    # ok_test('.')
    # ok_grid_test()
    #ok_grid_zone_test()
    # ppk2fac_verf_test()
    # ok_grid_invest()
    # ok_grid_test()
    # ok_grid_zone_test()
    #maha_pdc_summary_test("temp")
    # gsf_reader_test()
    #kl_test()
