import os

ndim = 100

with open(os.path.join("par.dat"),'r') as f:
	x = f.readline().strip().split()
	x = [float(x) for x in x]

# see: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.rosen.html
result = 0
for i in range(ndim - 1):
    x0 = float(x[i])
    x1 = float(x[i+1])
    result_i = 100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0
    result += result_i

with open(os.path.join("obs.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(result))
