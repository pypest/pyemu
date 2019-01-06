import os
import copy
import shutil
import numpy as np
import pandas as pd
import flopy
import pyemu

def setup_freyberg_transient_model():


    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    mo = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws, check=False,forgive=False)

    perlen = np.ones((365))
    m = flopy.modflow.Modflow("freyberg_transient",model_ws=os.path.join("da","freyberg","truth"),version="mfnwt",
                              external_path=".")

    flopy.modflow.ModflowDis(m,nrow=mo.nrow,ncol=mo.ncol,nlay=1,delr=mo.dis.delr,
                             delc=mo.dis.delc,top=mo.dis.top,botm=mo.dis.botm[-1],nper=len(perlen),perlen=perlen)
    flopy.modflow.ModflowBas(m,ibound=mo.bas6.ibound[0],strt=mo.bas6.strt[0])
    flopy.modflow.ModflowUpw(m,laytyp=mo.upw.laytyp,hk=mo.upw.hk[0],vka=mo.upw.vka[0],ss=0.00001,sy=0.01)
    flopy.modflow.ModflowNwt(m)
    oc_data = {}
    for iper in range(m.nper):
        oc_data[iper,0] = ["save head","save budget"]
    flopy.modflow.ModflowOc(m,stress_period_data=oc_data)
    flopy.modflow.ModflowRch(m,rech=mo.rch.rech.array[0])
    wel_data = mo.wel.stress_period_data[0]
    wel_data["k"][:] = 0
    flopy.modflow.ModflowWel(m,stress_period_data={0:wel_data})
    flopy.modflow.ModflowSfr2(m,nstrm=mo.sfr.nstrm,nss=mo.sfr.nss,istcb2=90,segment_data=mo.sfr.segment_data,reach_data=mo.sfr.reach_data)
    m.write_input()
    pyemu.os_utils.run("mfnwt {0}.nam".format(m.name),cwd=m.model_ws)
    return m


def setup_truth():
    #model_ws = os.path.join("da","freyberg","truth")
    #if not os.path.exists(model_ws)
    m = setup_freyberg_transient_model()
    os.chdir(os.path.join("da","freyberg"))
    grid_props = [["upw.hk",0],["upw.vka",0],["upw.ss",0],["upw.sy",0],["rch.rech",None]]
    rch_temporal_pars = []
    temporal_bc_props = []
    hds_kperk = []
    for iper in range(m.nper):
       rch_temporal_pars.append(["rch.rech",iper])
       temporal_bc_props.append(["wel.flux",iper])
       hds_kperk.append([iper,0])

    sfr_obs_dict = {iss+1:iss+1 for iss in range(m.sfr.nss)}
    sfr_obs_dict["hhwt"] = [iss+1 for iss in range(0,20)]


    ph = pyemu.helpers.PstFromFlopyModel("freyberg_transient.nam",org_model_ws="truth",new_model_ws="truth_template",
                                    grid_props=grid_props,const_props=rch_temporal_pars,
                                    temporal_bc_props=temporal_bc_props,build_prior=False,
                                         remove_existing=True,model_exe_name="mfnwt",
                                         sfr_obs=sfr_obs_dict,hds_kperk=hds_kperk)
    par = ph.pst.parameter_data
    wf_pars = par.parnme.apply(lambda x: x.startswith("wel"))
    par.loc[wf_pars,"parubnd"] = 10
    par.loc[wf_pars,"parlbnd"] = 0.1
    rch_pars = par.parnme.apply(lambda x: x.startswith("rech") and x.endswith("_cn"))
    par.loc[rch_pars, "parubnd"] = 1.2
    par.loc[rch_pars, "parlbnd"] = 0.8

    ph.pst.control_data.noptmax = 0
    ph.pst.write(os.path.join("truth_template","freyberg_truth.pst"))
    pyemu.os_utils.run("pestpp freyberg_truth.pst",cwd="truth_template")
    pe = ph.draw(20)
    pe.enforce()
    pe.to_csv(os.path.join("truth_template","sweep_in.csv"))
    os.chdir(os.path.join("..",".."))



def run_truth_sweep():
    pyemu.os_utils.start_slaves(os.path.join("da","freyberg","truth_template"),"pestpp-swp","freyberg_truth.pst",
                                20,slave_root=os.path.join("da","freyberg"),master_dir=os.path.join("da","freyberg","truth_sweep"))



def setup_daily_da():

    m = flopy.modflow.Modflow.load("freyberg_transient.nam",model_ws=os.path.join("da","freyberg","truth_template"),
                                   check=False)

    m.change_model_ws(os.path.join("da","freyberg","temp"),reset_external=True)
    m.dis.nper = 1
    m.external_path = '.'
    m.write_input()
    pyemu.os_utils.run("mfnwt freyberg_transient.nam",cwd=m.model_ws)
    grid_props = [["upw.hk",0],["upw.vka",0],["upw.ss",0],["upw.sy",0],["rch.rech",0],["bas6.strt",0]]
    os.chdir(os.path.join("da","freyberg"))
    sfr_obs_dict = {iss + 1: iss + 1 for iss in range(m.sfr.nss)}
    sfr_obs_dict["hhwt"] = [iss + 1 for iss in range(0, 20)]
    ph = pyemu.helpers.PstFromFlopyModel("freyberg_transient.nam",org_model_ws="temp",
                                         new_model_ws="daily_template",grid_props=grid_props,
                                         spatial_bc_props=[["wel.flux",0]],hds_kperk=[[0,0]],
                                         remove_existing=True,model_exe_name="mfnwt",build_prior=True,
                                         sfr_obs=sfr_obs_dict)
    ph.pst.control_data.noptmax = 0
    ph.pst.write(os.path.join("daily_template","freyberg_transient.pst"))
    pyemu.os_utils.run("pestpp freyberg_transient.pst",cwd="daily_template")

    #now for the super hack - change the names in the hds ins file to line up with the strt par names...
    # not doing this for now - waiting for Ayman's input RE if it is nessecary
    # name_map = {}
    # lines = []
    # pset = set(ph.pst.par_names)
    # ins_file = os.path.join("daily_template","freyberg_transient.hds.dat.ins")
    # with open(ins_file,'r') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if '!' in line:
    #             raw = line.strip().split()
    #             oname = raw[-1].replace('!','')
    #             if oname in name_map:
    #                 raise Exception(oname)
    #             oraw = oname.split('_')
    #             k,i,j = [int(x) for x in oraw[-4:-1]]
    #             pname = "strt{0:01d}{1:03d}{2:03d}".format(k,i,j)
    #             if pname not in pset:
    #                 print(pname)
    #             raw[-1] = "!{0}!".format(pname)
    #             line = ' '.join(raw)
    #             name_map[oname] = pname
    #         lines.append(line)
    # with open(ins_file,'w') as f:
    #     for line in lines:
    #         f.write(line+'\n')
    # obs = ph.pst.observation_data
    # obs.loc[:,"obsnme"] = obs.obsnme.apply(lambda x: name_map[x] if x in name_map else x)
    # obs.index = obs.obsnme
    #
    # ph.pst.write(os.path.join("daily_template", "freyberg_daily.pst"))
    # pyemu.os_utils.run("pestpp freyberg_daily.pst", cwd="daily_template")

    os.chdir(os.path.join("..",".."))


def process_truth_for_obs_states():
    # just use the first realization as truth
    truth_df = pd.read_csv(os.path.join("da", "freyberg", "truth_sweep", "sweep_out.csv"), nrows=1)
    truth_df.columns = truth_df.columns.str.lower()
    truth_obs = truth_df.iloc[[0], :].T
    truth_hds = truth_obs.loc[truth_obs.index.map(lambda x: x.startswith("hds") and '_' in x)]
    print(truth_hds)
    truth_hds.loc[:, "i"] = truth_hds.index.map(lambda x: int(x.split('_')[2]))
    truth_hds.loc[:, "j"] = truth_hds.index.map(lambda x: int(x.split('_')[3]))
    truth_hds.loc[:, "kper"] = truth_hds.index.map(lambda x: int(x.split('_')[4]))
    truth_hds.loc[:, "ij"] = truth_hds.apply(lambda x: (x.i,x.j),axis=1)
    # process the obs locations
    obs_locs = pd.read_csv(os.path.join("..", "examples", "freyberg_sfr_update", "obs_rowcol.dat"),
                           delim_whitespace=True)
    obs_ij = set(obs_locs.apply(lambda x: (x.row - 1, x.col - 1), axis=1).values)
    truth_obs_states = truth_hds.loc[truth_hds.ij.apply(lambda x: x in obs_ij),:]
    print(truth_obs_states)
    truth_obs_states.pop("ij")
    truth_obs_states.loc[:,"obsnme"] = truth_obs_states.apply(lambda x: "hds_00_{0:03d}_{1:03d}_000".format(int(x.i),int(x.j)),axis=1)
    truth_obs_states.to_csv(os.path.join("da","freyberg","daily_template","truth_states.csv"))


def freyberg_test():

    t_d = "daily_template"
    m_d = "daily_master"
    bd = os.getcwd()

    os.chdir(os.path.join("da","freyberg"))
    #load the truth states
    truth_df = pd.read_csv(os.path.join(t_d,"truth_states.csv"),index_col=0)
    # for now,not adding any noise to the truth states

    truth_df = truth_df.loc[truth_df.kper==0,:]
    truth_df.index = truth_df.obsnme
    print(truth_df)

    t_d = "daily_template"
    m_d = "daily_master"
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d,m_d)
    shutil.copytree(t_d,os.path.join(m_d,t_d))
    os.chdir(m_d)
    pst = pyemu.Pst(os.path.join("freyberg_transient.pst"))
    obs = pst.observation_data

    # set all obs weights to zero
    obs.loc[:,"weight"] = 0.0
    # replace the obs vals in the pst with the truth states at the end of the first assimilation cycle
    obs.loc[truth_df.index,"obsval"] = truth_df.loc[:,"0"]
    obs.loc[truth_df.index, "weight"] = 0.001 # oh, who knows...

    enkf = pyemu.EnsembleKalmanFilter(pst=pst,num_slaves=5,slave_dir=t_d)
    enkf.initialize(num_reals=10)
    enkf.analysis()
    os.chdir(bd)


def draw_forcing_ensemble():
    t_d = os.path.join("da","freyberg","daily_template")
    pst = pyemu.Pst(os.path.join(t_d,"freyberg_transient.pst"))
    par = pst.parameter_data
    forcing_groups = ["grrech0","grstrt0","welflux_k00"]
    par.loc[par.pargp.apply(lambda x: x not in forcing_groups),"partrans"] = "fixed"
    cov = pyemu.Cov.from_ascii(os.path.join(t_d,"freyberg_transient.pst.prior.cov")).to_dataframe()
    cov = cov.loc[pst.adj_par_names,pst.adj_par_names]
    cov = pyemu.Cov.from_dataframe(cov)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov,num_reals=10000)
    pe.to_csv(os.path.join("forcing.csv"))



if __name__ == "__main__":
    #setup_freyberg_transient_model()
    #setup_truth()
    #run_truth_sweep()
    #setup_daily_da()
    #process_truth_for_obs_states()
    freyberg_test()
    #draw_forcing_ensemble()