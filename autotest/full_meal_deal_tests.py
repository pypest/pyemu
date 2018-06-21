import os




def freyberg_test():
    import shutil
    import platform
    import numpy as np
    import pandas as pd
    try:
        import flopy
    except:
        return
    import pyemu

    if "linux" in platform.platform().lower():
        exe_name = os.path.join("..","travis_bin","mfnwt")
    else:
        raise Exception("unrecognized platform:{0}".format(platform.platform()))


    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False,forgive=False,
                                   exe_name=exe_name)
    org_model_ws = "temp"

    m.change_model_ws(org_model_ws)
    m.write_input()
    pyemu.os_utils.run("{0} {1}".format(exe_name,m.name+".nam"),cwd=org_model_ws)
    hds_file = "freyberg.hds"
    list_file = "freyberg.list"
    for f in [hds_file, list_file]:
        assert os.path.exists(os.path.join(org_model_ws, f))
    
    new_model_ws = "template"

    props = [["upw.hk",None],["upw.vka",None],["upw.ss",None],["rch.rech",None]]


if __name__ == "__main__":
    freyberg_test()
