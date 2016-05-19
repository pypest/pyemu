import os
import pyemu

pst = pyemu.Pst(os.path.join("freyberg.pst"))

par_dir = "proj_par_draws"
par_files = [os.path.join(par_dir,f) for f in os.listdir(par_dir) if f.endswith(".par")]

en = pyemu.ParameterEnsemble(pst)
en.read_parfiles(par_files)

sys.exit()