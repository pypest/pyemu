import os
import sys
import platform

#sys.path.append(os.path.join("..","pyemu"))
import pyemu
from pyemu import os_utils
from pyemu.prototypes.pst_from import PstFrom
import shutil


ext = ''
bin_path = os.path.join("..", "..", "bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "mac")
else:
    bin_path = os.path.join(bin_path, "win")
    ext = '.exe'

mf_exe_name = os.path.join(bin_path, "mfnwt")
mt_exe_name = os.path.join(bin_path, "mt3dusgs")
pp_exe_name = os.path.join(bin_path, "pestpp")
ies_exe_name = os.path.join(bin_path, "pestpp-ies")
swp_exe_name = os.path.join(bin_path, "pestpp-swp")


def freyberg_test():

    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

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
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, 
                                   check=False, forgive=False,
                                   exe_name=mf_exe_name)
    flopy.modflow.ModflowRiv(m, stress_period_data={
        0: [[0, 0, 0, m.dis.top.array[0, 0], 1.0, m.dis.botm.array[0, 0, 0]],
            [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]],
            [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]]]})

    org_model_ws = "temp_pst_from"
    if os.path.exists(org_model_ws):
        shutil.rmtree(org_model_ws)
    m.external_path = "."
    m.change_model_ws(org_model_ws)
    m.write_input()
    print("{0} {1}".format(mf_exe_name, m.name + ".nam"), org_model_ws)
    os_utils.run("{0} {1}".format(mf_exe_name, m.name + ".nam"), 
                 cwd=org_model_ws)
    hds_kperk = []
    for k in range(m.nlay):
        for kper in range(m.nper):
            hds_kperk.append([kper, k])
    hds_runline, df = pyemu.gw_utils.setup_hds_obs(
        os.path.join(m.model_ws, f"{m.name}.hds"), kperk_pairs=None, skip=None,
        prefix="hds", include_path=False)
    template_ws = "new_temp"
    # sr0 = m.sr
    sr = pyemu.helpers.SpatialReference.from_namfile(
        os.path.join(m.model_ws, m.namefile),
        delr=m.dis.delr, delc=m.dis.delc)
    # set up PstFrom object
    pf = PstFrom(original_d=org_model_ws, new_d=template_ws, 
                 remove_existing=True,
                 longnames=True, spatial_reference=sr,
                 zero_based=False)
    pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
                        index_cols='obsnme', use_cols='obsval', prefix='hds')
    pf.add_observations_from_ins(ins_file='freyberg.hds.dat.ins')
    pf.post_py_cmds.append(hds_runline)
    pf.tmp_files.append(f"{m.name}.hds")
    pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                      index_cols=[0, 1, 2], use_cols=[3, 4],
                      par_name_base=["rivbot_grid", "rivstage_grid"],
                      mfile_fmt='%10d%10d%10d %15.8F %15.8F %15.8F')
    pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                      index_cols=[0, 1, 2], use_cols=5)
    pf.add_parameters(filenames=["WEL_0000.dat", "WEL_0001.dat"], 
                      par_type="grid", index_cols=[0, 1, 2], use_cols=3, 
                      par_name_base="welflux_grid",
                      zone_array=m.bas6.ibound.array)
    pf.add_parameters(filenames=["WEL_0000.dat"], par_type="constant", 
                      index_cols=[0, 1, 2], use_cols=3, 
                      par_name_base=["flux_const"])
    pf.add_parameters(filenames="rech_1.ref", par_type="grid",
                      zone_array=m.bas6.ibound[0].array,
                      par_name_base="rch_datetime:1-1-1970")
    pf.add_parameters(filenames=["rech_1.ref", "rech_2.ref"],
                      par_type="zone", zone_array=m.bas6.ibound[0].array)
    pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
                      zone_array=m.bas6.ibound[0].array,
                      par_name_base="rch_datetime:1-1-1970", pp_space=4)
    pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
                      zone_array=m.bas6.ibound[0].array,
                      par_name_base="rch_datetime:1-1-1970", pp_space=1,
                      ult_ubound=100, ult_lbound=0.0)
    pf.mod_sys_cmds.append("{0} {1}".format(mf_exe_name, m.name + ".nam"))
    print(pf.mult_files)
    print(pf.org_files)
    pst = pf.build_pst('freyberg.pst')

    pst.write_input_files(pst_path=pf.new_d)
    # test par mults are working
    b_d = os.getcwd()
    os.chdir(pf.new_d)
    try:
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv")
    except Exception as e:
        os.chdir(b_d)
        raise Exception(str(e))
    os.chdir(b_d)

    pst.control_data.noptmax = 0
    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(
        os.path.join(bin_path, "pestpp-ies")), cwd=pf.new_d)

    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)
    assert pst.phi < 1.0e-5, pst.phi
# TO#DO: add test for model file with headers
# TO#DO add test for formatted file type


if __name__ == "__main__":
    freyberg_test()
