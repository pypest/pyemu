import os
import pyemu

exe = os.path.join("exe","pest++.exe")
slave_dir = "."
pst = "freyberg.pst"
num_slaves = 15

pyemu.pst_utils.start_slaves(slave_dir,exe,pst,num_slaves=num_slaves)