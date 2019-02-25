import pyemu
import pandas as pd
from pyemu.prototypes.da import Assimilator
import numpy as np
import os, sys

# get par and obs names to build pst object
par_df = pd.read_csv(r"input_file_init.csv")
obs_df = pd.read_csv(r"obs_values.csv")

# generate pst object from parnames and obsnames
pst = pyemu.Pst.from_par_obs_names(par_names= par_df['parnme'].values,
                                   obs_names= obs_df['obsname'].values)
pyemu.utils.simple_tpl_from_pars(par_df['parnme'].values, tplfilename = r"tplfile.tpl")
pyemu.utils.simple_ins_from_obs2(obs_df['obsname'].values, insfilename = r"insfile.ins")


pst.control_data.noptmax = 0
# assign obs values and weights
pst.observation_data['obsval'] = obs_df['sim'].values
pst.observation_data['weight'] = 10
pst.pestpp_options["sweep_parameter_csv_file"] = "sweep_in.csv"
pst.model_command = sys.executable + " " + "forward_run.py"

#file names
pst.input_files = [r"input_file.csv"]
pst.output_files = [r"output_file.csv"]
pst.template_files = [r'tplfile.tpl']
pst.instruction_files =[r'insfile.ins']
#pst.parameter_data['partrans'] = 'none'
pst.write(new_filename = r'pmp_tomog.pst')
pst.filename = r'pmp_tomog.pst'

# use pr-generated parameter ensemble
par_ens = pd.read_csv('kh_ensemble0.csv')
obs_ens = 0.1 * np.random.randn(100, pst.nnz_obs)
obs_ens = pst.observation_data['obsval'].values + obs_ens
obs_ens = pd.DataFrame(obs_ens, columns=pst.obs_names)
sm = Assimilator(type='Smoother', iterate=False, mode='stochastic', pst= pst, parens=par_ens, obs_ens = None,
                 num_slaves= 3, num_real= 3)
sm.analysis()

pass