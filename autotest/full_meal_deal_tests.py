import os
import platform

ext = ''
bin_path = os.path.join("..","..","bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")
    ext = '.exe'
    

mf_exe_name = os.path.join(bin_path,"mfnwt")
pp_exe_name = os.path.join(bin_path, "pestpp")
ies_exe_name = os.path.join(bin_path, "pestpp-ies")

# for f in [mf_exe_name,pp_exe_name,ies_exe_name]:
#     if not os.path.exists(f):
#         raise Exception("{0} not found",f)

def freyberg_test():
    import shutil

    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu

    ext = ''
    bin_path = os.path.join("..", "..", "bin")
    if "linux" in platform.platform().lower():
        bin_path = os.path.join(bin_path, "linux")
    elif "darwin" in platform.platform().lower():
        bin_path = os.path.join(bin_path, "mac")
    else:
        bin_path = os.path.join(bin_path, "win")
        ext = '.exe'

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False,forgive=False,
                                   exe_name=mf_exe_name)
    org_model_ws = "temp"

    m.change_model_ws(org_model_ws)
    m.write_input()
    print("{0} {1}".format(mf_exe_name,m.name+".nam"),org_model_ws)
    pyemu.os_utils.run("{0} {1}".format(mf_exe_name,m.name+".nam"),cwd=org_model_ws)
    hds_file = "freyberg.hds"
    list_file = "freyberg.list"
    for f in [hds_file, list_file]:
        assert os.path.exists(os.path.join(org_model_ws, f))

    new_model_ws = "template1"

    props = [["upw.hk",None],["upw.vka",None],["upw.ss",None],["rch.rech",None]]

    hds_kperk = [[kper,0] for kper in range(m.nper)]

    temp_bc_props = [["wel.flux",kper] for kper in range(m.nper)]
    spat_bc_props= [["wel.flux",2]]

    ph = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws,org_model_ws,
                                         const_props=props,
                                         zone_props=props,
                                         kl_props=props,
                                         pp_props=props,
                                         grid_props=props,
                                         hds_kperk=hds_kperk,
                                         sfr_pars=True,sfr_obs=True,
                                         spatial_bc_props=spat_bc_props,
                                         temporal_bc_props=temp_bc_props,
                                         remove_existing=True,
                                         model_exe_name="mfnwt")
    tmp = mf_exe_name.split(os.sep)
    tmp = os.path.join(*tmp[1:])+ext
    assert os.path.exists(tmp),tmp
    shutil.copy2(tmp,os.path.join(new_model_ws,"mfnwt"+ext))
    ph.pst.control_data.noptmax = 0
    ph.pst.write(os.path.join(new_model_ws,"test.pst"))
    print("{0} {1}".format(pp_exe_name,"test.pst"), new_model_ws)
    pyemu.os_utils.run("{0} {1}".format(pp_exe_name,"test.pst"),cwd=new_model_ws)
    for ext in ["rec",'rei',"par"]:
        assert os.path.exists(os.path.join(new_model_ws,"test.{0}".format(ext))),ext
    ph.pst.parrep(os.path.join(new_model_ws,"test.par"))
    res = pyemu.pst_utils.read_resfile(os.path.join(new_model_ws,"test.rei"))
    ph.pst.observation_data.loc[res.name,"obsval"] = res.modelled
    ph.pst.write(os.path.join(new_model_ws,"test.pst"))
    print("{0} {1}".format(pp_exe_name, "test.pst"), new_model_ws)

    pyemu.os_utils.run("{0} {1}".format(pp_exe_name,"test.pst"),cwd=new_model_ws)
    for ext in ["rec",'rei',"par","iobj"]:
        assert os.path.exists(os.path.join(new_model_ws,"test.{0}".format(ext))),ext
    df = pd.read_csv(os.path.join(new_model_ws,"test.iobj"))
    assert df.total_phi.iloc[0] < 1.0e-10
    pe = ph.draw(30)
    pe.to_csv(os.path.join(new_model_ws,"par_en.csv"))
    ph.pst.pestpp_options["ies_par_en"] = "par_en.csv"
    ph.pst.control_data.noptmax = 1
    ph.pst.write(os.path.join(new_model_ws, "test.pst"))
    master_dir = "test_master"
    pyemu.os_utils.start_slaves(new_model_ws,ies_exe_name,"test.pst",
                                num_slaves=10,slave_root='.',
                                master_dir=master_dir,silent_master=True)
    df = pd.read_csv(os.path.join(master_dir,"test.phi.meas.csv"),index_col=0)
    init_phi = df.loc[0,"mean"]
    final_phi = df.loc[1,"mean"]
    assert final_phi < init_phi


if __name__ == "__main__":
    freyberg_test()
