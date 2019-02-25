import os, sys
import pandas as pd
from pyemu.pst.pst_utils import write_to_template, parse_tpl_file, parse_ins_file
import numpy as np
import matplotlib.pyplot as plt
# read io files
io_files = pd.read_csv("in_out_files.csv")
msc_df = pd.read_csv("misc_data.csv")
initial_head = pd.read_csv(r".\template\stat_heads_in.csv")
all_obs = []

# use realization (1)
par_ens = pd.read_csv('kh_ensemble0.csv')
kfield = np.power(10.0,par_ens.values[4,:])
np.savetxt(r'.\template\par_k.csv', kfield)
for icycle in range(0, 4, 1):

    base_folder = os.getcwd()

    # get current files
    curr_files = io_files[(io_files['cycle'] < 0) | (io_files["cycle"] == icycle)]
    misc_io_files = curr_files[curr_files['Type'] == 'misc']
    unique_iofiles = misc_io_files['in/out'].unique()
    for par_file in unique_iofiles:
        tpl_file = misc_io_files.loc[misc_io_files['in/out'] == par_file, 'tpl/ins'].values[0]
        tpl_file = os.path.join(r".\template", tpl_file)
        parnames = parse_tpl_file(tpl_file)

        parvals = {}
        for par in parnames:
            val = msc_df[msc_df['parnme'] == par]['parval1'].values[0]
            parvals[par] = val

        write_to_template(parvals, tpl_file, os.path.join(r".\template", par_file))
    # intial head
    if icycle == 0:
        initial_head = pd.read_csv(r".\template\stat_heads_in.csv", header=None)
        initial_head = initial_head.values * 0.0 + 45.0
        np.savetxt(r".\template\stat_heads_in.csv", initial_head)

    # run
    os.chdir(r".\template")
    runc = sys.executable + " " + "forward_run.py"
    os.system(runc)

    # read
    hob_out = pd.read_csv('obs_hob.csv', header=None)
    head_out = pd.read_csv('stat_heads_out.csv', header=None)
    np.savetxt('stat_heads_in.csv', head_out)

    obs_df_ = np.loadtxt('.\pp_model.hob.out', dtype=np.str, skiprows=1)
    obs_df = pd.DataFrame(columns=['obsname', 'sim'])
    obs_df['sim'] = obs_df_[:, 0].astype('float')
    obs_df['sim'] = (obs_df['sim'].values * 1000).astype('int') / 1000.0
    obs_df['obsname'] = obs_df_[:, 2]
    obs_df['cycle'] = icycle
    all_obs.append(obs_df)
    os.chdir(base_folder)
    h2d = head_out.values.reshape(50,50)
    plt.plot(h2d[24,:])
plt.show()
xx = pd.concat(all_obs)
xx.to_csv('obs_values3.csv')
pass
