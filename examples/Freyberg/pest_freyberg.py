"""Note - this script is only kept for legacy reasons as a reference.

It no longer runs successfully on the latest versions of dependencies.
"""

import os
import shutil
import platform
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu

model_ws = os.path.join("extra_crispy")
nam_file = "freyberg.nam"

ml = flopy.modflow.Modflow.load(nam_file,exe_name="mf2005",model_ws=model_ws,verbose=True)
ml.dis.sr.xul = 619653
ml.dis.sr.yul = 3353277
ml.dis.sr.rotation = 0
ml.dis.epsg_str = "EPSG:32614"
ml.dis.start_datetime = "11-5-1955"

#write a grid spec file
ml.dis.sr.write_gridSpec(os.path.join("misc","freyberg.spc"))

# write the bore coords file
obs_rowcol = pd.read_csv(os.path.join("misc","obs_rowcol.dat"),sep=r"\s+")
obs_rowcol.loc[:,'x'] = ml.dis.sr.xcentergrid[obs_rowcol.row-1,obs_rowcol.col-1]
obs_rowcol.loc[:,'y'] = ml.dis.sr.ycentergrid[obs_rowcol.row-1,obs_rowcol.col-1]
obs_rowcol.loc[:,"top"] = ml.dis.top[obs_rowcol.row-1,obs_rowcol.col-1]
obs_rowcol.loc[:,"layer"] = 1
# use this later to set weights
obs_names = ["or{0:02d}c{1:02d}_0".format(r-1,c-1) for r,c in zip(obs_rowcol.row,obs_rowcol.col)]

# get the truth time series
h = flopy.utils.HeadFile(os.path.join(model_ws,"freyberg.hds"),model=ml)
data = h.get_alldata()

#write all those terrible mod2obs files
ibound = ml.bas6.ibound.array
#well_data = ml.wel.stress_period_data[0]
#ibound[0,well_data["i"],well_data['j']] = 0
#drn_data = ml.riv.stress_period_data[0]
#ibound[0,drn_data["i"],drn_data['j']] = 0
f_crd = open(os.path.join("misc","bore.crds"),'w')
for i in range(ml.nrow):
    for j in range(ml.ncol):
        if ibound[0,i,j] == 0:
            continue
        on = "or{0:02d}c{1:02d}".format(i,j)
        ox = ml.dis.sr.xcentergrid[i,j]
        oy = ml.dis.sr.ycentergrid[i,j]
        ol = 1
        f_crd.write("{0:20s} {1:15.6E} {2:15.6E} {3:d}\n".format(on,ox,oy,ol))
f_crd.close()

# run mod2smp to get the truth values
with open(os.path.join("settings.fig"),'w') as f:
    f.write("date=dd/mm/yyyy\ncolrow=no")

with open(os.path.join("misc","mod2smp.in"),'w') as f:
    f.write(os.path.join("misc","freyberg.spc")+'\n')
    f.write(os.path.join("misc","bore.crds")+'\n')
    f.write(os.path.join("misc","bore.crds")+'\n')
    f.write(os.path.join("extra_crispy","freyberg.hds")+'\n')
    f.write("f\n5\n1.0e+30\nd\n")
    f.write("01/01/2015\n00:00:00\n")
    f.write(os.path.join("misc","freyberg_heads.smp")+'\n')

os.system(os.path.join("exe","mod2smp.exe") + " <"+os.path.join("misc","mod2smp.in"))

# write the ins file for the head smp

pyemu.pst_utils.smp_to_ins(os.path.join("misc","freyberg_heads.smp"))

shutil.copy2(os.path.join("misc","freyberg_heads.smp"),os.path.join("misc","freyberg_heads_truth.smp"))
# write the hk template
pnames = []
with open(os.path.join("misc","hk_Layer_1.ref.tpl"),'w') as f:
    f.write("ptf ~\n")
    for i in range(ml.nrow):
        for j in range(ml.ncol):
            #print(i,j,ibound[0,i,j])
            if ibound[0,i,j] == 0:
                tpl_str = "  0.000000E+00"
            else:
                pn = "hkr{0:02d}c{1:02d}".format(i,j)
                tpl_str = "~  {0:8s}  ~".format(pn)
            f.write("{0:14s} ".format(tpl_str))
            pnames.append(pn)
        f.write('\n')


# build pst instance
misc_files = os.listdir(os.path.join("misc"))
ins_files = [os.path.join("misc",f) for f in misc_files if f.endswith(".ins")]
out_files = [f.replace(".ins",'') for f in ins_files]
tpl_files = [os.path.join("misc",f) for f in misc_files if f.endswith(".tpl")]
in_files = [os.path.join(ml.model_ws,os.path.split(f)[-1]).replace(".tpl",'') for f in tpl_files]
in_files = [os.path.join(ml.model_ws,"ref",os.path.split(f)[-1]) if "layer" in f.lower() else f for f in in_files]

pst = pyemu.pst_utils.pst_from_io_files(tpl_files,in_files,ins_files,out_files)
# apply par values and bounds and groups
pdata = pst.parameter_data
grps = pdata.groupby(pdata.parnme.apply(lambda x:'hk' in x)).groups
hk_mean = ml.upw.hk.array.mean()
hk_stdev = ml.upw.hk.array.std()
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


rcond_mean = ml.riv.stress_period_data[0]["cond"].mean()
rcond_std = ml.riv.stress_period_data[0]["cond"].std()
rcond_lb = rcond_mean * 0.1
rcond_ub = rcond_mean * 10.0
grps = pdata.groupby(pdata.parnme.apply(lambda x:'rcond' in x)).groups
pdata.loc[grps[True],"parval1"] = rcond_mean
pdata.loc[grps[True],"parubnd"] = rcond_ub
pdata.loc[grps[True],"parlbnd"] = rcond_lb
pdata.loc[grps[True],"pargp"] = "rcond"


wf_base = ml.wel.stress_period_data[0]["flux"]
wf_fore = ml.wel.stress_period_data[1]["flux"]
# grps = pdata.groupby(pdata.parnme.apply(lambda x:'wf' in x)).groups
# pdata.loc[grps[True],"parval1"] = 1.0
# pdata.loc[grps[True],"parubnd"] = 1.5
# pdata.loc[grps[True],"parlbnd"] = 0.5
# pdata.loc[grps[True],"pargp"] = "welflux"
grps = pdata.groupby(pdata.parnme.apply(lambda x:'wf' in x and x.endswith("_1"))).groups
pdata.loc[grps[True],"parval1"] = -1.0 * wf_base
pdata.loc[grps[True],"parubnd"] = -1.0 * wf_base * 1.1
pdata.loc[grps[True],"parlbnd"] = -1.0 * wf_base * 0.9
pdata.loc[grps[True],"scale"] = -1.0
pdata.loc[grps[True],"pargp"] = "welflux"

grps = pdata.groupby(pdata.parnme.apply(lambda x:'wf' in x and x.endswith("_2"))).groups
pdata.loc[grps[True],"parval1"] = -1.0 * wf_fore
pdata.loc[grps[True],"parubnd"] = -1.0 * wf_fore * 1.5
pdata.loc[grps[True],"parlbnd"] = -1.0 * wf_fore * 0.5
pdata.loc[grps[True],"scale"] = -1.0
pdata.loc[grps[True],"pargp"] = "welflux"




pdata.loc["ss","parval1"] = ml.upw.ss.array.mean()
pdata.loc["ss","parubnd"] = ml.upw.ss.array.mean() * 10.0
pdata.loc["ss","parlbnd"] = ml.upw.ss.array.mean() * 0.1
pdata.loc["ss","pargp"] = "storage"

pdata.loc["sy","parval1"] = ml.upw.sy.array.mean()
pdata.loc["sy","parubnd"] = ml.upw.sy.array.mean() * 10.0
pdata.loc["sy","parlbnd"] = ml.upw.sy.array.mean() * 0.1
pdata.loc["sy","pargp"] = "storage"

#apply obs weights and groups and values
import run
run.process()
run.write_other_obs_ins()
shutil.copy2(os.path.join("misc","other.obs"),os.path.join("misc","other.obs.truth"))

smp = pyemu.pst_utils.smp_to_dataframe(os.path.join("misc","freyberg_heads_truth.smp"))
values = list(smp.loc[:,"value"])
pst.observation_data.loc[:,"weight"] = 0.0
pst.observation_data.loc[:,"obgnme"] = "forecast"
groups = pst.observation_data.groupby(pst.observation_data.obsnme.apply(lambda x:x in obs_names)).groups
pst.observation_data.loc[groups[True],"weight"] = 100.0
pst.observation_data.loc[groups[True],"obgnme"] = "head_cal"
groups = pst.observation_data.groupby(pst.observation_data.obsnme.apply(lambda x:x.startswith('o'))).groups
pst.observation_data.loc[groups[True],"obsval"] = values
pst.observation_data.index = pst.observation_data.obsnme
with open(os.path.join("misc","other.obs.truth"),'r') as f:
    for line in f:
        raw = line.strip().split()
        pst.observation_data.loc[raw[0],"obsval"] = float(raw[1])

pst.model_command[0] = "python run.py"
pst.zero_order_tikhonov()
pst.control_data.noptmax = 20

pst.pestpp_lines.append('++forecasts(travel_time,sw_gw_0,sw_gw_1,sw_gw_2)')
pst.pestpp_lines.append('++n_iter_base(1)')
pst.pestpp_lines.append('++n_iter_super(4)')
pst.pestpp_lines.append('++max_reg_iter(5)')
pst.write("freyberg.pst",update_regul=True)

if platform.system().lower() == "windows":
    pest_exe = os.path.join("exe","pest++.exe")
else:
    pest_exe = None

os.system(pest_exe + ' freyberg.pst /h :4004')


# dt_deltas = pd.to_timedelta(h.get_times(),unit="d")
# idx = pd.to_datetime(ml.dis.start_datetime) + dt_deltas
# obs_data = pd.DataFrame(data[:,0,obs_rowcol.row-1,obs_rowcol.col-1],columns=obs_rowcol.name,
#                         index=idx)
#
# print(obs_data.shape)
# obs_rowcol.index = obs_rowcol.name
# for name in obs_data.columns:
#     top = obs_rowcol.loc[name,"top"]
#     if obs_data.loc[:,name].max() > top:
#         print(name,"flooded")
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     obs_data.loc[:,name].plot(ax=ax,legend=False,marker='.')
#
#     ax.plot(ax.get_xlim(),[top,top],"k--")
#     ax.set_title(name)
# plt.show()

# fig = plt.figure()
# ax = plt.subplot(111)
# ax = ml.wel.stress_period_data.plot(ax=ax)
# ax = ml.riv.stress_period_data.plot(ax=ax)
# ax.scatter(obs_rowcol.x,obs_rowcol.y)
# [ax.text(x,y,name) for x,y,name in zip(obs_rowcol.x,obs_rowcol.y,obs_rowcol.name)]


# ax = ml.wel.plot()[0]
# ax.scatter(obs_rowcol.x,obs_rowcol.y)
