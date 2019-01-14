import os,sys
import pandas as pd
import numpy as np
import flopy
from scipy.spatial.distance import cdist
import pyemu

print("**** Start Simulation ****")
# read input file
par_df = pd.read_csv("input_file.csv")
kh = par_df['parval1'].values.reshape(50,50)
kh = np.power(10,kh)

# run the model
mf_name = r".\model_data\pp_model.nam"
mf = flopy.modflow.Modflow.load(mf_name, load_only=['DIS', 'BAS6', 'UPW'])
mf.upw.hk = kh
mf.change_model_ws(".\model_data")
mf.exe_name = r".\model_data\mfnwt.exe"
mf.run_model()

#read output
obs_df_ = np.loadtxt('.\model_data\pp_model.hob.out', dtype = np.str, skiprows = 1)
obs_df = pd.DataFrame(columns=['obsname', 'sim'])
obs_df['sim'] =obs_df_[:,0].astype('float')
obs_df['obsname'] =obs_df_[:,2]

# write output
obs_df.to_csv('output_file.csv')

print("**** End Simulation ****")



