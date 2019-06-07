import os
import pyemu

exe = os.path.join("exe","pest++.exe")
worker_dir = "."
pst = "freyberg.pst"
num_workers = 15

pyemu.pst_utils.start_workers(worker_dir,exe,pst,num_workers=num_workers)