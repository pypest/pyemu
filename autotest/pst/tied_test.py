import os
import pyemu
with open("tied.log",'w') as f:
	f.write("pyemu is located:{0:s}\n".format(pyemu.__file__))

	pst = pyemu.Pst("br_opt_no_zero_weighted.pst")
	f.write("number of tied lines in control file:{0:d}\n".format(len(pst.tied_lines)))
	f.write("tied parameter lines in pst: \n" + ''.join(pst.tied_lines)+'\n')


