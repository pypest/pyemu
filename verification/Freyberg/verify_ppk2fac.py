
# coding: utf-8

# In[8]:

import os
import subprocess as sp
import numpy as np
import pandas as pd
import pyemu
import flopy


# In[9]:

ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws="extra_crispy",load_only=["UPW"])
ml.sr.write_gridSpec("grid.spc")
pp_df = pyemu.utils.gw_utils.setup_pilotpoints_grid(ml)


# In[10]:

np.savetxt("zone.dat",np.ones((ml.nrow,ml.ncol),dtype=np.int),fmt="%2d")


# In[11]:

args = ["grid.spc","pp_00_pp.dat","0.0","zone.dat","f","structure.complex.dat",        "struct1","o","1.0e+10","1","10","ppk2fac_fac.dat","f",        "ppk2fac_stdev.ref","reg.dat"]
with open("ppk2fac.in",'w') as f:
    f.write('\n'.join(args))
os.system("ppk2fac.exe < ppk2fac.in")


# In[12]:

ok = pyemu.utils.OrdinaryKrige("structure.complex.dat","pp_00_pp.dat")


# In[13]:

df_interp = ok.calc_factors_grid(ml.sr,maxpts_interp=10,verbose=0)


# In[14]:

ok.to_grid_factors_file("pyemu_factors.dat")


# In[ ]:




# In[ ]:



