import os
import sys
import platform

#sys.path.append(os.path.join("..","pyemu"))
from pyemu import os_utils
from pyemu.prototypes.pst_from import PstFrom


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
pp_exe_name = os.path.join(bin_path, "pestpp")
ies_exe_name = os.path.join(bin_path, "pestpp-ies")
swp_exe_name = os.path.join(bin_path, "pestpp-swp")


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
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False, forgive=False,
                                   exe_name=mf_exe_name)
    flopy.modflow.ModflowRiv(m,stress_period_data={0:[0,0,0,1.0,1.0,1.0]})
    org_model_ws = "temp2"
    m.external_path="."
    m.change_model_ws(org_model_ws)
    m.write_input()
    print("{0} {1}".format(mf_exe_name, m.name + ".nam"), org_model_ws)
    os_utils.run("{0} {1}".format(mf_exe_name, m.name + ".nam"), cwd=org_model_ws)

    pf = PstFrom(original_d=org_model_ws,new_d="new_temp",remove_existing=True,
                 longnames=True,spatial_reference=m.sr,zero_based=False)
    pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                      index_cols=[0, 1, 2], use_cols=[3,4], par_name_base=["rivbot_grid","rivstage_grid"])
    pf.add_parameters(filenames=["WEL_0000.dat","WEL_0001.dat"], par_type="grid",
                      index_cols=[0, 1, 2], use_cols=3,par_name_base="welflux_grid",zone_array=m.bas6.ibound.array)
    pf.add_parameters(filenames=["WEL_0000.dat"], par_type="constant", index_cols=[0, 1, 2],
                      use_cols=3,par_name_base=["flux_const"])
    pf.add_parameters(filenames="rech_1.ref",par_type="grid",zone_array=m.bas6.ibound[0].array,par_name_base="rch_datetime:1-1-1970")
    pf.add_parameters(filenames=["rech_1.ref","rech_2.ref"],par_type="zone",zone_array=m.bas6.ibound[0].array)

if __name__ == "__main__":
    freyberg_test()