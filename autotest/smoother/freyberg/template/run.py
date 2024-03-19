import os
import platform
from datetime import datetime
import pandas as pd

endpoint_file = os.path.join('freyberg.mpenpt')
list_file = os.path.join("freyberg.list")
out_file = os.path.join("other.obs")
ins_file = out_file + ".ins"
hds_file = os.path.join("freyberg.hds")
smp_file = os.path.join("freyberg_heads.smp")

obs_names = """h00_02_15
h00_02_09
h00_03_08
h00_09_01 
h00_13_10
h00_15_16
h00_21_10
h00_22_15
h00_24_04
h00_26-06 
h00_29_15
h00_33_07 
h00_34_10
""".split('\n')

def prep():
    to_remove = [endpoint_file, list_file,out_file,smp_file,
                 os.path.join(hds_file)]
    for f in to_remove:
        try:
            os.remove(f)
        except Exception as e:
            print("error removing file: {0}\n{1}".format(f,str(e)))


def run():
    if platform.system().lower() == "windows":
        mf_exe = os.path.join("mfnwt.exe")
        mp_exe = os.path.join("mp6.exe")
        m2s_exe = os.path.join("exe","mod2smp.exe")
    else:
        mf_exe = "mfnwt"
        mp_exe = "mp6"
        m2s_exe = None

    os.system(mf_exe + ' freyberg.nam')
    #os.system(mp_exe + ' <mpath.in')


def process():
    # particle travel time
    #lines = open(endpoint_file, 'r').readlines()

    #items = lines[-1].strip().split()

    #travel_time = float(items[4]) - float(items[3])
    #print(travel_time)

    # sw-gw exchange

    key_line = "RIVER LEAKAGE"
    vals = []
    with open(list_file, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            if key_line in line:
                vals.append(float(line.strip().split()[-1]))


    #print(vals)
    diffs = [i - o for i, o in zip(vals[:-1:2], vals[1::2])]
    #print(diffs)

    ovals = []
    onames = []
    with open(out_file, 'w') as f:
        #f.write("{0:20s} {1:20.7E}\n".format("travel_time", travel_time))
        for i, diff in enumerate(diffs):
            oname = "sw_gw_{0:d}".format(i)
            f.write("{0:20s} {1:20.7E}\n".format(oname, diff))
            onames.append(oname)

    return onames, ovals


def write_other_obs_ins():
    onames, ovals = process()
    with open(ins_file, 'w') as f:
        f.write("pif ~\n")
        for oname in onames:
            f.write("l1 w !{0:s}!\n".format(oname))



def write_hds_ins():
    with open("freyberg.hds.ins",'w') as f:
        f.write("pif ~\n")
        for iper in range(3):
            for i in range(40):
                f.write("l1 ")
                for j in range(20):
                    oname = "h{0:02d}_{1:02d}_{2:02d}".format(iper,i,j)
                    f.write(" w !{0}!".format(oname))
                f.write('\n')


def plot_list():
    import flopy
    import matplotlib.pyplot as plt

    mlf = flopy.utils.MfListBudget(list_file)
    df_flx, df_vol = mlf.get_dataframes()
    df_diff = df_flx.loc[:,"in"] - df_flx.loc[:,"out"]

    ax = plt.subplot(111)

    df_diff.iloc[-1,:].plot(kind="bar",ax=ax,grid=True)
    ylim = (-8000,8000)
    ax.set_ylim(ylim)
    plt.show()

def make_pst():
    import pyemu
    # build pst instance
    misc_files = os.listdir(".")
    ins_files = [f for f in misc_files if f.endswith(".ins")]
    out_files = [f.replace(".ins",'') for f in ins_files]
    tpl_files = [f for f in misc_files if f.endswith(".tpl")]
    in_files = [f.replace(".tpl",'') for f in tpl_files]
    
    pst = pyemu.pst_utils.pst_from_io_files(tpl_files,in_files,ins_files,out_files)
    # apply par values and bounds and groups
    pdata = pst.parameter_data
    grps = pdata.groupby(pdata.parnme.apply(lambda x:'hk' in x)).groups
    hk_mean = 5
    lb = hk_mean * 0.1
    ub = hk_mean * 10.0
    pdata.loc[grps[True],"parval1"] = hk_mean
    pdata.loc[grps[True],"parubnd"] = ub
    pdata.loc[grps[True],"parlbnd"] = lb
    pdata.loc[grps[True],"pargp"] = "hk"


    # constant mults
    grps = pdata.groupby(pdata.parnme.apply(lambda x:'rch' in x)).groups
    pdata.loc[grps[True],"parval1"] = 1.0
    pdata.loc[grps[True],"parubnd"] = 1.5
    pdata.loc[grps[True],"parlbnd"] = 0.5
    pdata.loc[grps[True],"pargp"] = "rch"
    pdata.loc["rch_1","parval1"] = 1.0
    pdata.loc["rch_1","parubnd"] = 1.1
    pdata.loc["rch_1","parlbnd"] = 0.9


    rcond_mean = 4500.0
    rcond_lb = rcond_mean * 0.1
    rcond_ub = rcond_mean * 10.0
    grps = pdata.groupby(pdata.parnme.apply(lambda x:'rcond' in x)).groups
    pdata.loc[grps[True],"parval1"] = rcond_mean
    pdata.loc[grps[True],"parubnd"] = rcond_ub
    pdata.loc[grps[True],"parlbnd"] = rcond_lb
    pdata.loc[grps[True],"pargp"] = "rcond"


    wf_base = 100.0
    grps = pdata.groupby(pdata.parnme.apply(lambda x:'wf' in x and x.endswith("_1"))).groups
    pdata.loc[grps[True],"parval1"] = -1.0 * wf_base
    pdata.loc[grps[True],"parubnd"] = -1.0 * wf_base * 1.1
    pdata.loc[grps[True],"parlbnd"] = -1.0 * wf_base * 0.9
    pdata.loc[grps[True],"scale"] = -1.0
    pdata.loc[grps[True],"pargp"] = "welflux"

    wf_fore = 100.0
    grps = pdata.groupby(pdata.parnme.apply(lambda x:'wf' in x and x.endswith("_2"))).groups
    pdata.loc[grps[True],"parval1"] = -1.0 * wf_fore
    pdata.loc[grps[True],"parubnd"] = -1.0 * wf_fore * 1.5
    pdata.loc[grps[True],"parlbnd"] = -1.0 * wf_fore * 0.5
    pdata.loc[grps[True],"scale"] = -1.0
    pdata.loc[grps[True],"pargp"] = "welflux"

    pdata.loc["ss","parval1"] = 0.001
    pdata.loc["ss","parubnd"] = 0.01
    pdata.loc["ss","parlbnd"] = 0.0001
    pdata.loc["ss","pargp"] = "storage"

    pdata.loc["sy","parval1"] = 0.1
    pdata.loc["sy","parubnd"] = 0.25
    pdata.loc["sy","parlbnd"] = 0.01
    pdata.loc["sy","pargp"] = "storage"

    hds = pd.read_csv("freyberg.hds.truth.obf",header=None,names=["obsnme","obsval"],index_col=0,
                      sep=r"\s+")
    pst.observation_data.loc[:,"weight"] = 0.0
    pst.observation_data.loc[:,"obgnme"] = "forecast"
    groups = pst.observation_data.groupby(pst.observation_data.obsnme.apply(lambda x:x in obs_names)).groups
    pst.observation_data.loc[groups[True],"weight"] = 100.0
    pst.observation_data.loc[groups[True],"obgnme"] = "head_cal"
    groups = pst.observation_data.groupby(pst.observation_data.obsnme.apply(lambda x:x.startswith('h'))).groups
    pst.observation_data.loc[groups[True],"obsval"] = hds.obsval
    pst.observation_data.index = pst.observation_data.obsnme
    with open(os.path.join("other.obs.truth"),'r') as f:
        for line in f:
            raw = line.strip().split()
            pst.observation_data.loc[raw[0],"obsval"] = float(raw[1])

    pst.model_command[0] = "python run.py"
    pst.control_data.noptmax = 0

    pst.pestpp_options['forecasts'] = "sw_gw_0,sw_gw_1,sw_gw_2"
    pst.pestpp_options["n_iter_base"] = 1
    pst.pestpp_options["n_iter_super"] = 4
    pst.prior_information = pst.null_prior
    pst.control_data.pestmode = "estimation"
    pst.write("freyberg.pst")


if __name__ == "__main__":
    start = datetime.now()
    prep()
    run()
    process()
    end = datetime.now()
    print("took: {0}".format(str(end-start)))
    #write_hds_ins()
    #make_pst()