import os
with open(os.path.join("par.dat"),'r') as f:
	par = float(f.readline().strip())

result = ((7.0/12.0) * par**3) - ((7.0/2.0) * par**2) + (8.0 * par)

with open(os.path.join("obs.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(result))