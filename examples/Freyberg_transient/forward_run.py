import os
from datetime import datetime
import numpy as np
import pandas as pd
import pyemu

bak_dir = "bak"
out_dir = "ref"
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
def prepare():

	# apply drain conductance parameters
	drn_df = pd.read_csv("drain_mlt.dat",sep=r"\s+",header=None,names=["name","cond"])
	drn_df.index = drn_df.name.apply(lambda x: (int(x[-5:-3])+1,int(x[-2:])+1))
	drn_files = [f for f in os.listdir(bak_dir) if "drn" in f.lower()]
	for drn_file in drn_files:
		df = pd.read_csv(os.path.join(bak_dir,drn_file),header=None,
			names=["l","r","c","stage","cond"],sep=r"\s+")
		df.index = df.apply(lambda x: (x.r,x.c),axis=1)

		df.loc[:,"cond"] = drn_df.cond
		df.loc[:,["l","r","c","stage","cond"]].to_csv(os.path.join(out_dir,drn_file),
			sep=' ',index=False,header=False)

	# apply pilot point parameters as multipliers
	pp_files = ["hk1pp.dat","sy1pp.dat","rech1pp.dat"]
	prefixes = ["hk","sy","rech"]
	for pp_file,prefix in zip(pp_files,prefixes):
		arr = pyemu.gw_utils.fac2real(pp_file=pp_file,factors_file="pp.fac",out_file=None)
		base_arr_files = [f for f in os.listdir(bak_dir) if prefix in f.lower()]
		base_arrs = [np.loadtxt(os.path.join(bak_dir,f)) for f in base_arr_files]
		for fname in base_arr_files:
			base_arr = np.loadtxt(os.path.join(bak_dir,fname))
			base_arr *= arr
			np.savetxt(os.path.join(out_dir,fname),base_arr)


def run():
	os.system("mfnwt freyberg.nam")

def post_process():
	pass




if __name__ == "__main__":
	prepare()
	run()
	post_process()