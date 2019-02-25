import pyemu
import pandas as pd
from pyemu.prototypes.da import Assimilator
import numpy as np
import os, sys
#TODO: Check if the tranformation works in sweep in?????
# -----------------------------------------------------------
## Prepare input for DA_EnKF. All these inputs will be part of the control file
#------------------------------------------------------------
#        *********************************
# 1) Input: General information
#        *********************************
num_real = 100
num_slaves = 5

#        *********************************
# 2) Input: Dataframe that hold observation at all times (or for all cycles)
#        *********************************
# Traditional obs information
obs_info = pd.read_csv(r"obs_values3.csv") # read file of all observation names
obsnames = [item.lower() for item in obs_info['obsname'].values] # obtain obs names
observation_data = pd.DataFrame(columns=['obsnme', 'obsval', 'weight', 'obgnme', 'cycle'])
observation_data['obsval'] = obs_info['sim'].values
obsnames2 = []
for i, ob in enumerate(obsnames):
    cycle = obs_info['cycle'][i]
    obsnames2.append(ob+"_{}".format(i))
obsnames = obsnames2
observation_data['obsnme'] = obsnames
observation_data['weight'] = 1000
observation_data['obgnme'] = 'HOB'

# We need to assign the number of the time cycle for each obs
observation_data['cycle'] = obs_info['cycle'].values

#        *********************************
# 3) Input: Dataframe that holds dynamic states information
#        *********************************
# head at each cell has a name that is derived from row/column numbers
# these are the dynamic states that will be updated by EnKF
nrows = 50
ncols = 50
state_names = []
for i in range(nrows):
    nm_row = "h_r{}".format(i+1)
    for j in range(ncols):
        nm = nm_row + "_c{}".format(j+1)
        state_names.append(nm)

# since states are similiar to obs, we can use observation dataframe to hold its info
state_data = pd.DataFrame(columns=['obsnme', 'obsval', 'weight', 'obgnme', 'cycle'])
state_data['obsnme'] = state_names
state_data['obsval'] = 45.0
state_data['weight'] = 100
state_data['obgnme'] = 'heads'
# Below: The negative value for the cycle number indicates that this information is used
# for all time cycles
state_data['cycle'] = -1

#        *********************************
# 4) Optional Input: Dataframe that holds initial ensemble of dynamic states
#        *********************************
# In case this is not provided then it will be sampled from state_data
# In this example, we assume that the initial head is constant
init_heads = 45.0 * np.ones((num_real, len(state_data['obsval'])))
stat_ens = pd.DataFrame(init_heads, columns=state_names)

#        *********************************
# 5) Optional Input: Dataframe that holds initial ensemble of static parameters
#        *********************************
# In case this is not provided then it will be sampled from par_data
par_df = pd.read_csv(r"input_file_init.csv")
par_df['parval1'] = 10.0
par_names = par_df['parnme'].values
par_ens = pd.read_csv('kh_ensemble0.csv')
par_ens = np.power(10.0,par_ens)
par_df['cycle'] = -1 # k parameters should be updated at each cycle
#par_df['partrans'] = 'log'

# since dynamic state will be updated like the static parameter, they should be augmented with par_ens
parM = par_ens.values
staM = stat_ens.values
cols = par_ens.columns.values.tolist() + stat_ens.columns.values.tolist()
par_stat_ens = pd.DataFrame(np.hstack((parM, staM)), columns= cols)

#        *********************************
# 6) Input: Dataframe that holds miscellaneous parameters (misc_par) that are not used in DA,
#           but are necessary for model restart
#        *********************************
# In our example, we need to change dis file and hob file.  Misc. information are read from a file
msc_df = pd.read_csv("misc_data.csv")


#        *********************************
# 6) Input: Dictionary to hold all in/out files
#        *********************************
# In our example, we use a csv file that holds in/out info
in_out_files = pd.read_csv("in_out_files.csv")


#        *********************************
# 6) Input: Generate template files and instruction files
#        *********************************
pyemu.utils.simple_tpl_from_pars(state_names, tplfilename = r".\template\stats_head.tpl")
pyemu.utils.simple_ins_from_obs2(state_names, insfilename = r".\template\stat_heads.ins")
cycles = observation_data['cycle'].unique()
for icyc in cycles:
    obsnm_cycle = observation_data[observation_data['cycle'] == icyc]['obsnme'].values
    fn_ins = os.path.join(r".\template", "obs_hob_{}.ins".format(icyc))
    pyemu.utils.simple_ins_from_obs2(obsnm_cycle, insfilename = fn_ins)
pyemu.utils.simple_tpl_from_pars(par_names, tplfilename = r".\template\par_k.tpl")

dis1_names = msc_df[(msc_df['cycle'] == 0) & (msc_df['pargp'] == 'dis_pkg')]['parnme'].values
pyemu.utils.simple_tpl_from_pars(dis1_names, tplfilename = r".\template\misc_dis1.tpl")
dis2_names = msc_df[(msc_df['cycle'] == 1) & (msc_df['pargp'] == 'dis_pkg')]['parnme'].values
pyemu.utils.simple_tpl_from_pars(dis2_names, tplfilename = r".\template\misc_dis2.tpl")
hob1_names = msc_df[(msc_df['cycle'] == 0) & (msc_df['pargp'] == 'hob_pkg')]['parnme'].values
pyemu.utils.simple_tpl_from_pars(hob1_names, tplfilename = r".\template\misc_hob1.tpl")
hob2_names = msc_df[(msc_df['cycle'] == 1) & (msc_df['pargp'] == 'hob_pkg')]['parnme'].values
pyemu.utils.simple_tpl_from_pars(hob2_names, tplfilename = r".\template\misc_hob2.tpl")

#        *********************************
# 7) Generate global pst object
#        *********************************
# the global parameters include static parameters, misc parameters
global_parnames = par_names.tolist() + msc_df['parnme'].values.tolist()

# the global obs include obs, stat, and forc
global_obsname = obsnames + state_names

pst = pyemu.Pst.from_par_obs_names(par_names=  global_parnames,
                                   obs_names= global_obsname)
#pst.parameter_data['cycle'] = np.nan
#pst.parameter_data['partrans'] = 'none'
gparameter_data = pd.concat([par_df, msc_df])
gobservation_data = pd.concat([observation_data, state_data])

pst.parameter_data= gparameter_data
pst.parameter_data.set_index(['parnme'], inplace=True)
pst.parameter_data['parnme'] = pst.parameter_data.index
pst.observation_data = gobservation_data
pst.observation_data.set_index(['obsnme'], inplace=True)
pst.observation_data['obsnme'] = pst.observation_data.index

pst.filename = r'global_control.pst'
pst.control_data.noptmax = 0
pst.pestpp_options["sweep_parameter_csv_file"] = "sweep_in.csv"
pst.model_command = sys.executable + " " + "forward_run.py"
pst.io_files =  in_out_files


sm = Assimilator(type= 'Kalman Filter', iterate=False, mode='stochastic', pst= pst, parens=par_stat_ens, obs_ens = None,
                 num_slaves= 5, num_real= 100)
sm.analysis()
xxx = 1
