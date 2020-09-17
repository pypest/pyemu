
def process_model_outputs():
	import numpy as np
	print("processing model outputs")
	arr = np.random.random(100)
	np.savetxt("test.dat",arr)
	