import os,sys
import pandas as pd
import numpy as np
import flopy
from scipy.spatial.distance import cdist
import pyemu

print("**** Start Simulation ****")
# read input file
par_df = pd.read_csv("input_file.csv",  header=None)
kh = par_df.values.reshape(50,50)
kh = np.power(10,kh)

# run the model
mf_name = r"pp_model.nam"
mf = flopy.modflow.Modflow.load(mf_name, load_only=['DIS', 'BAS6', 'UPW'])
mf.upw.hk = kh
mf.change_model_ws(".")
mf.exe_name = r".\mfnwt.exe"
mf.upw.write_file()
mf.run_model()

#read output
obs_df_ = np.loadtxt('.\pp_model.hob.out', dtype = np.str, skiprows = 1)
obs_df = pd.DataFrame(columns=['obsname', 'sim'])
obs_df['sim'] =obs_df_[:,0].astype('float')
obs_df['obsname'] =obs_df_[:,2]

# write output
#obs_df.to_csv('output_file.csv')
#obs_df['sim'].to_csv('output_file.csv', index = False)
fid = open('output_file.csv', 'w')
obs = obs_df['sim'].values
for i, val in enumerate(obs):
    if i == 0:
        fid.write(str(val))
    else:
        fid.write("\n")
        fid.write(str(val))

fid.close()
print("**** End Simulation ****")



