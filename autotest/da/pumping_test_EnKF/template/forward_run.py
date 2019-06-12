import os,sys
import pandas as pd
import numpy as np
import flopy


def run_model():
    print("**** Start Simulation ****")

    # make sure to delete any output...
    files_to_remove = [r"pp_model.hds", r"stat_heads_out.csv", r"obs_hob.csv"]
    for file in files_to_remove:
        try:
            os.remove(file)
        except:
            pass

    # load the existing model
    mf_name = r"pp_model.nam"
    mf = flopy.modflow.Modflow.load(mf_name, load_only=['DIS', 'BAS6', 'UPW', 'HOB', 'OC', 'WEL'])

    # read input files
    dis_df = pd.read_csv(r"misc_dis.csv", header=None)
    dis_par = iter(dis_df.values)
    npr =  int(next(dis_par)[0]) # number of stress periods
    ntsep = []
    steady = []
    for i in range(npr):
        ntsep.append(int(next(dis_par)[0]))
        if (npr > 1) & (i== 0) :
            steady.append(True)
        else:
            steady.append(False)
    perlen = []
    for i in range(npr):
        perlen.append(next(dis_par)[0])

    nlay = mf.dis.nlay
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    top = mf.dis.top.array
    botm = mf.dis.botm.array
    nper = npr
    nstp = ntsep
    is_steady = steady
    tim_unit = mf.dis.itmuni
    len_unit = mf.dis.lenuni
    mf.remove_package('DIS')
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=20, delc=20,top=top, botm=botm,
                                           nper=nper, perlen=perlen, nstp=nstp, steady=is_steady,  itmuni= tim_unit,
                                           lenuni = len_unit)


    mf.change_model_ws(".")
    mf.dis.write_file()

    # wells
    mf.remove_package('wel')
    flow =  -2000
    if len(steady) == 2:
        stress_period_data = {0: [0,24,24, 0.0], 1: [0,24,24,flow]}
    else:
        stress_period_data = {0: [0, 24, 24, flow]}
    wel = flopy.modflow.mfwel.ModflowWel(mf,  stress_period_data=stress_period_data)
    wel.write_file()

    # read initial heads
    head_df = pd.read_csv(r"stat_heads_in.csv", header = None)
    h2d = head_df.values.reshape(mf.nrow, mf.ncol)
    mf.bas6.strt = h2d
    mf.bas6.write_file()

    # read hydrualic conductivity file
    par_df = pd.read_csv("par_k.csv",  header=None)
    kh = par_df.values.reshape(50,50) # 10 was add to improve k
    kh = np.power(10,kh)
    mf.upw.hk = kh
    mf.upw.vka = kh * 0.1
    mf.upw.ss = 1.0e-06
    mf.upw.sy = 0.18
    mf.upw.write_file()

    # control file
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
    mf.remove_package('OC')
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, cboufm='(20i5)')
    oc.write_file()

    # write hob files
    hob_df = pd.read_csv(r"misc_hob.csv", header=None)
    start_time = hob_df.values[0][0]
    end_time = hob_df.values[1][0]

    #hobs

    mf_name2 = r".\mf_EnS\pp_model.nam"
    mf2 = flopy.modflow.Modflow.load(mf_name2, load_only=['DIS', 'BAS6','HOB'])

    obs_data = mf2.hob.obs_data
    if len(steady) > 1:
        time_index = np.arange(start_time+1,end_time+1)
    else:
        time_index = np.arange(start_time+1, end_time + 1)

    nt_ons = len(time_index)
    new_obs_list = []
    for obs in obs_data:
        ts = obs.time_series_data.copy()
        ts = pd.DataFrame(ts)
        if len(steady) > 1:
            totim = np.arange(0,nt_ons, 1)
        else:
            totim = np.arange(1, nt_ons+1, 1)-0.001
        hds = np.zeros_like(totim)
        tim_ser_data = np.array([totim, hds]).T
        names = [obs.obsname+"."+ str(tt) for tt in time_index]

        obs1 = flopy.modflow.HeadObservation(mf, obsname=obs.obsname, layer=0, row=obs.row,
                                             column= obs.column, roff=0, coff=0, itt=1,
                                             time_series_data= tim_ser_data)
        #obs1.time_series_data['obsname'] = names
        #obs1.time_series_data['totim'] = obs1.time_series_data['totim']
        new_obs_list.append(obs1)


    hob = flopy.modflow.ModflowHob(mf, hobdry=-9999.,iuhobsv=52, obs_data=new_obs_list )
    mf.hob.write_file()

    # run the model

    mf.exe_name = r".\mfnwt.exe"

    (success, buff) = mf.run_model()



    #read output
    if success:
        # read head field in case we need it to replace head in next
        hds = flopy.utils.HeadFile("pp_model.hds")
        last_sp = hds.get_kstpkper()[-1]
        final_head = hds.get_data(kstpkper=last_sp)
        final_head = final_head.flatten()
        final_head = (final_head * 1000).astype(int) / 1000.0
        np.savetxt('stat_heads_out.csv', final_head)

        obs_df_ = np.loadtxt('.\pp_model.hob.out', dtype = np.str, skiprows = 1)
        obs_df = pd.DataFrame(columns=['obsname', 'sim'])
        obs_df['sim'] =obs_df_[:,0].astype('float')
        obs_df['sim'] = (obs_df['sim'].values * 1000).astype('int')/1000.0
        obs_df['obsname'] =obs_df_[:,2]

        # write output
        #obs_df.to_csv('output_file.csv')
        #obs_df['sim'].to_csv('output_file.csv', index = False)

        fid = open('obs_hob.csv', 'w')
        obs = obs_df['sim'].values
        for i, val in enumerate(obs):
            if i == 0:
                fid.write(str(val))
            else:
                fid.write("\n")
                fid.write(str(val))

        fid.close()
    print("**** End Simulation ****")

if __name__ == "__main__":
    run_model()


