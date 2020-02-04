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
    flopy.modflow.ModflowRiv(m,
                             stress_period_data={0: [0, 0, 0, 1.0, 1.0, 1.0]})
    org_model_ws = "temp2"
    if os.path.exists(org_model_ws):
        shutil.rmtree(org_model_ws)
    m.external_path = "."
    m.change_model_ws(org_model_ws)
    m.write_input()
    print("{0} {1}".format(mf_exe_name, m.name + ".nam"), org_model_ws)
    os_utils.run("{0} {1}".format(mf_exe_name, m.name + ".nam"), 
                 cwd=org_model_ws)

    # set up pest control file with PstFrom() method
    pf = PstFrom(original_d=org_model_ws, new_d="new_temp", 
                 remove_existing=True,
                 longnames=True, spatial_reference=m.modelgrid, 
                 zero_based=False)

    # pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
    #                   zone_array=m.bas6.ibound[0].array,
    #                   par_name_base="pprch_datetime:1-1-1970")
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

    print(pf.mult_files)
    print(pf.org_files)
    pf.build_pst('freyberg.pst')
    os.chdir(pf.new_d)
    pyemu.helpers.apply_list_and_array_pars(arr_par_file="mult2model_info.csv")
    os.chdir("..")

# TO#DO: add test for model file with headers
# TODO add test for formatted file type


if __name__ == "__main__":
    freyberg_test()
