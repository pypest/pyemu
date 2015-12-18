
# coding: utf-8

# #verify pyEMU null space projection with the freyberg problem

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyemu


# instaniate ```pyemu``` object and drop prior info.  Then reorder the jacobian and save as binary.  This is needed because the pest utilities require strict order between the control file and jacobian

# In[2]:

mc = pyemu.MonteCarlo(jco="freyberg.jcb",verbose=False)
mc.drop_prior_information()
jco_ord = mc.jco.get(mc.pst.obs_names,mc.pst.par_names)
ord_base = "freyberg_ord"
jco_ord.to_binary(ord_base + ".jco")  
mc.pst.control_data.parsaverun = ' '
mc.pst.write(ord_base+".pst")


# Draw some vectors from the prior and write the vectors to par files

# In[3]:

# setup the dirs to hold all this stuff
par_dir = "prior_par_draws"
proj_dir = "proj_par_draws"
parfile_base = os.path.join(par_dir,"draw_")
projparfile_base = os.path.join(proj_dir,"draw_")
if os.path.exists(par_dir):
   shutil.rmtree(par_dir)
os.mkdir(par_dir)
if os.path.exists(proj_dir):
   shutil.rmtree(proj_dir)
os.mkdir(proj_dir)

# make some draws
mc.draw(10)

#write them to files
mc.parensemble.to_parfiles(parfile_base)


# Run pnulpar

# In[4]:

exe = os.path.join("exe","pnulpar.exe")
args = [ord_base+".pst","y","5","y","pnulpar_qhalfx.mat",parfile_base,projparfile_base]
in_file = os.path.join("misc","pnulpar.in")
with open(in_file,'w') as f:
    f.write('\n'.join(args)+'\n') 
os.system(exe + ' <'+in_file)


# In[5]:

pnul_en = pyemu.ParameterEnsemble(mc.pst)
parfiles =[os.path.join(proj_dir,f) for f in os.listdir(proj_dir) if f.endswith(".par")]
pnul_en.read_parfiles(parfiles)


# In[6]:

pnul_en.loc[:,"fname"] = pnul_en.index
pnul_en.index = pnul_en.fname.apply(lambda x:str(int(x.split('.')[0].split('_')[-1])))
f = pnul_en.pop("fname")


# In[7]:

pnul_en.sort(axis=1)


# Now for pyemu

# In[11]:


en = mc.project_parensemble(nsing=5,inplace=False)


# In[ ]:

en.sort(axis=1)


# In[22]:

pnul_en.sort(inplace=True)
en.sort(inplace=True)
diff = 100.0 * np.abs(pnul_en - en) / en
dmax = diff.max(axis=0)
dmax.sort(ascending=False,inplace=True)
dmax.plot()
plt.show()
