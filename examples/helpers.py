import os
import pyemu

def process_model_outputs():
	import numpy as np
	print("processing model outputs")
	arr = np.random.random(100)
	np.savetxt("special_outputs.dat",arr)
	return arr


def write_ins_file(d):
	cwd = os.getcwd()
	os.chdir(d)
	arr = process_model_outputs()
	os.chdir(cwd)
	with open(os.path.join(d,"special_outputs.dat.ins"),'w') as f:
		f.write("pif ~\n")
		for i in range(arr.shape[0]):
			f.write("l1 !sobs_{0}!\n".format(i))

	
	i = pyemu.pst_utils.InstructionFile(os.path.join(d,"special_outputs.dat.ins"))
	df = i.read_output_file("special_outputs.dat")
	
	return df


if __name__ == "__main__":
	#process_model_outputs()
	write_ins_file(".")
	