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
plt.plot(Ka, color = [0,0.7,0.7])
plt.plot(Ka.mean(axis=1), label = "Estimated Field")
plt.plot(Ka[:,ref_real], 'r', label = "Reference Field")
plt.show()

pass