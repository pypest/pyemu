
"""
This examples uses DA- smoother to krige data
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import EnsKF_Tools as KF
import pandas as pd
import pyemu
from pyemu.prototypes.da import Assimilator
import os, sys

#
os.chdir(os.path.dirname(__file__))
#import pyemu


## Generate random fields
n = 50
N = 5000
m = 40
ref_real = 0
x = np.linspace(0,1,n)
x = np.stack((x, np.zeros_like(x)))
distMat = cdist(x.T, x.T, 'euclidean')
covMat = KF.cov(correlation_scale =0.3, stand_dev= 1.0,distMat = distMat)

K = KF.svd_random_generator(covM = covMat, seed = 6543, nreal = 100)
np.savetxt(r".\template\input.dat", K[:,0])

plt.plot(K, color = [0.7,0.7,0.7])
plt.plot(K[:,ref_real])
obs_location = np.linspace(1, n-1, 10).astype(int)
np.savetxt( r".\template\obs_locations.dat", obs_location)
obs = K[obs_location, ref_real]

H = K[obs_location, :] # model predictions
Ka = KF.EnsKF_evenson(H = H, K = K, d = obs, err_perc = 0.0, thr_perc = 0.1)

#plt.figure()
if 0:
    plt.plot(Ka, color = [0,0.7,0.7])
    plt.plot(Ka.mean(axis=1), label = "Estimated Field")
    plt.plot(K[:,ref_real], 'r', label = "Reference Field")
    plt.legend()
    plt.show()






# dimensions
n, N = K.shape
m = len(obs)

# generate parnames and obs names
parnames = ["par_"+str(i) for i in range(n)]
obsnames = ["obs_"+str(i) for i in range(m)]


# get par and obs names to build pst object
par_df = pd.DataFrame(K.T, columns= parnames)
obs_df = pd.DataFrame(obs.reshape(1,m), columns= obsnames)

# generate pst object from parnames and obsnames
pst = pyemu.Pst.from_par_obs_names(par_names= parnames,
                                   obs_names= obsnames)

pyemu.utils.simple_tpl_from_pars(parnames, tplfilename = r".\template\tplfile.tpl")
pyemu.utils.simple_ins_from_obs2(obsnames, insfilename = r".\template\insfile.ins")


pst.control_data.noptmax = 0
# assign obs values and weights
pst.observation_data['obsval'] = obs
pst.observation_data['obsnme'] = obsnames
pst.observation_data['weight'] = 10
pst.pestpp_options["sweep_parameter_csv_file"] = "sweep_in.csv"
pst.model_command = sys.executable + " " + "forward_run.py"

#file names
pst.input_files = [r"input.dat"]
pst.output_files = [r"output.dat"]
pst.template_files = [r'tplfile.tpl']
pst.instruction_files =[r'insfile.ins']
#pst.parameter_data['partrans'] = 'none'
pst.write(new_filename = r'.\template\krige_control.pst')
pst.filename = r'krige_control.pst'

# use pr-generated parameter ensemble
obs_ens = 0.1 * np.random.randn(100, pst.nnz_obs)
obs_ens = pst.observation_data['obsval'].values + obs_ens
obs_ens = pd.DataFrame(obs_ens, columns=pst.obs_names)
sm = Assimilator(type='Smoother', iterate=False, mode='stochastic', pst= pst, parens=par_df, obs_ens = None,
                 num_slaves= 5, num_real= 100)
sm.analysis()

KKa = sm.parensemble_a.values.T
plt.plot(KKa, color = [0,0.7,0.7])
plt.plot(KKa.mean(axis=1), label = "Estimated Field", color = 'k')
plt.plot(K[:,ref_real], 'r', label = "Reference Field")
plt.legend()
plt.show()
pass




