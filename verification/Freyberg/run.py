import os
import platform
from datetime import datetime

endpoint_file = os.path.join('extra_crispy', 'freyberg.mpenpt')
list_file = os.path.join("extra_crispy", "freyberg.list")
out_file = os.path.join("misc", "other.obs")
ins_file = out_file + ".ins"
hds_file = os.path.join("extra_crispy","freyberg.hds")
smp_file = os.path.join("misc","freyberg_heads.smp")

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
        mf_exe = os.path.join("MF_NWT.exe")
        mp_exe = os.path.join("mp6.exe")
        m2s_exe = os.path.join("mod2smp.exe")
    else:
        mf_exe = "mfnwt"
        mp_exe = "mp6"
        m2s_exe = None

    os.chdir("extra_crispy")
    os.system(mf_exe + ' freyberg.nam')
    os.system(mp_exe + ' <mpath.in')
    os.chdir("..")


def process():
    if platform.system().lower() == "windows":
        m2s_exe = os.path.join("mod2smp.exe")
    else:
        m2s_exe = None
    os.system(m2s_exe + "< "+os.path.join("misc","mod2smp.in"))
    # particle travel time
    lines = open(endpoint_file, 'r').readlines()

    items = lines[-1].strip().split()

    travel_time = float(items[4]) - float(items[3])
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

    onames = ["travel_time"]
    ovals = [travel_time]
    ovals.extend(diffs)
    with open(out_file, 'w') as f:
        f.write("{0:20s} {1:20.7E}\n".format("travel_time", travel_time))
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

if __name__ == "__main__":
    start = datetime.now()
    prep()
    run()
    process()
    end = datetime.now()
    print("took: {0}".format(str(end-start)))