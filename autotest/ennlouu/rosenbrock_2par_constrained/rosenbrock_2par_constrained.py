import os
with open(os.path.join("par.dat"),'r') as f:
	x = f.readline().strip().split()
	x1, x2 = float(x[0]), float(x[1])

result = 100.0*(x2 - x1**2.0)**2.0 + (1 - x1)**2.0
# see: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.rosen.html

#constraint = x1**2 + x2**2  # (==/<=/>= 1) eq or ineq constraint based on dec vars
constraint = -6*x1 - x2

with open(os.path.join("obs.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(result))

with open(os.path.join("constraint.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(constraint))