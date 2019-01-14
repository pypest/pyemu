import os
import numpy as np

par = np.loadtxt("input.dat")
loc = np.loadtxt("obs_locations.dat").astype(int)
m = len(loc)

obs = par[loc].reshape(m,1)
np.savetxt("output.dat", obs, fmt = "%8.6f")

pass
