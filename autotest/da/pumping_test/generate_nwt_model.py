import flopy
import datetime
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
workspace = r".\model_data"
model_name = 'pp_model'

# inialize
mf = flopy.modflow.Modflow(model_name, model_ws=workspace, version='mfnwt')
mf.exe_name = r"D:\Workspace\projects\mississippi\pyemu\autotest\da\pumping_test\model_data\mfnwt.exe"
#dis
# Simulation period & time steps -- -- Temporal discretization
nlay = 1
nrow = 50
ncol = 50
delr = 100.0
top = 50.0
botm = 0.0
start_date = datetime.date(year = 2000,month = 1,day = 1)
end_date = datetime.date(year = 2000,month = 2,day = 10)
time_span = end_date - start_date
time_span = time_span.days
nper = 2                                            # Number of stress periods
perlen = [1.0, time_span]                           # length of each stress period
nstp =   [1, time_span]                           # time steps in each stress period
is_steady =[ True, False]                           # First stress period is Steady-state, the second is Transient
tim_unit = 4                                        # days
len_unit = 2

dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=ncol, delc=ncol,top=top, botm=botm,
                                       nper=nper, perlen=perlen, nstp=nstp, steady=is_steady,  itmuni= tim_unit,
                                       lenuni = len_unit)

#bas
ibound = np.ones((50,50))
initial_head = 45.0 * np.ones((50,50))
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=initial_head)


#
laytyp = np.ones(nlay)  # convertable layer
avg_typ = 0               # 0 is harmonic mean 1 is logarithmic mean 2 is arithmetic mean
h_anis = 1.0              # a flag or the horizontal anisotropy
layvka = 0                # 0—indicates VKA is vertical hydraulic conductivity
laywet = np.zeros(nlay) #contains a flag for each layer that indicates if wetting is active.
                          #laywet should always be zero for the UPW Package


# get matrix of dist
x = mf.dis.delc.array
x = x.cumsum()
xx, yy = np.meshgrid(x, x)
xy = np.stack((xx.flatten(), yy.flatten()))
distMat = cdist(xy.T, xy.T, 'euclidean')
corr_len = delr * 4.0
cov = np.exp(-distMat/corr_len)
u,s,v = np.linalg.svd(cov)
us = np.dot(u, np.power(np.diag(s), 0.5))
usut = np.dot(us,v)
## 5236 is the seed used generate observation
np.random.seed(6543)
z = np.random.randn(len(np.diag(s)), 100)
ff = np.dot(usut,z)
# save the ensemble
par_names = []
for i in range(nrow):
    for j in range(ncol):
        nm = 'kh_l{}_r{}_c{}'.format(0, i, j)
        par_names.append(nm)
par_ens = pd.DataFrame(ff.T, columns=par_names)
par_ens.to_csv('kh_ensemble0.csv', index_label= False)
kh = ff[:,1].reshape(50,50)
plt.imshow(kh)
kh = np.power(10,kh)


# Hydraulic conductivity


# Specific Storage
ss = np.zeros((nlay, nrow, ncol)) + 1e-6

# Specific Yield
sy = np.zeros((nlay, nrow, ncol)) + 0.018


upw = flopy.modflow.mfupw.ModflowUpw(mf, laytyp=laytyp, layavg=avg_typ, chani=h_anis, layvka=layvka, laywet=laywet,
                                             hdry=-1e+30, iphdry=0, hk=kh, hani=1.0, vka=(kh * 0.1), ss=ss, sy=sy,
                                             vkcb=0.0, noparcheck=False, ipakcb = 55 , extension='upw')
# Add OC package to the MODFLOW model
options = ['PRINT HEAD', 'PRINT DRAWDOWN', 'PRINT BUDGET',
           'SAVE HEAD', 'SAVE DRAWDOWN', 'SAVE BUDGET',
           'SAVE IBOUND', 'DDREFERENCE']
idx = 0
spd = dict()
for sp in mf.dis.nstp.array:
    stress_period = idx
    step = sp - 1
    ke = (stress_period, step)
    idx = idx + 1
    spd[ke] = [options[3], options[2], options[5]]

oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, cboufm='(20i5)')

stress_period_data = {1: [0,24,24, -5000]}
flopy.modflow.mfwel.ModflowWel(mf,  stress_period_data=stress_period_data)

chd = np.zeros((nrow, ncol))
chd[:,0] = 1
chd[:,-1] = 1
loc = np.where(chd==1)
hd = len(loc[1])*[45.0]
cond = len(loc[1])*[50]
zz = len(loc[1])*[0]
all_cc = [zz, loc[0], loc[1], hd, cond]
stress_period = {0: np.array(all_cc).T}
stress_period[1] = np.array(all_cc).T
flopy.modflow.mfghb.ModflowGhb(mf, stress_period_data= stress_period)




flopy.modflow.mfnwt.ModflowNwt.load(r"D:\training3\gsflowID2447_classrepo\exercises\models_data\misc\solver_options"
                                    r".nwt", mf)

#hobs
x1 =  np.arange(5,50,7)
xv, yv = np.meshgrid(x1, x1)
xv = xv.flatten()
yv = yv.flatten()
obs_list = []
for ii, xvv in enumerate(xv):
    tim = np.arange(1, 365, 1)
    hds = np.zeros_like(tim)
    obsname = 'HO_' + str(ii)
    tim_ser_data = np.array([tim, hds]).T
    obs1 = flopy.modflow.HeadObservation(mf, obsname=obsname, layer=0, row=xvv,
                                         column=yv[ii], roff=0, coff=0, itt=1,
                                         time_series_data=tim_ser_data)
    obs_list.append(obs1)

hob = flopy.modflow.ModflowHob(mf, hobdry=-9999.,iuhobsv=52, obs_data=obs_list )
pass


mf.write_input()
mf.run_model()
aa = flopy.utils.HeadFile(r"D:\Workspace\projects\mississippi\pyemu\autotest\da\pumping_test\model_data\pp_model.hds")
import matplotlib.pyplot as plt
plt.imshow(aa.get_data(kstpkper = aa.get_kstpkper()[-1])[0,:,:])
plt.show()


## generate input file template
columns = ['parnme', 'partrans', 'parval1', 'pargp']
par_names = []
for i in range(nrow):
    for j in range(ncol):
        nm = 'kh_l{}_r{}_c{}'.format(0, i, j)
        par_names.append(nm)

par_df = pd.DataFrame(columns = columns)
par_df['parnme'] = par_names
par_df['parval1'] = np.log10(mf.upw.hk.array.flatten())
par_df['pargp'] = 'kh'
par_df['partrans'] = 'log'
par_df.to_csv('input_file.csv')
pass