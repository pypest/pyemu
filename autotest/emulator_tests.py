import os
# import sys
import shutil
import pytest
import numpy as np
import pandas as pd
# import platform
import pyemu
from pst_from_tests import setup_tmp, _get_port
from pyemu.emulators import DSI, LPFA, GPR, dsi

from conftest import get_exe_path

ies_exe_path = get_exe_path("pestpp-ies")
mou_exe_path = get_exe_path("pestpp-mou")# Check for TensorFlow availability for DSIAE tests

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

def generate_synth_data(num_realizations=100, num_observations=10):

    # generate synth data
    data = np.random.normal(size=(num_realizations,num_observations))
    data = pd.DataFrame(data,columns=[f"obs{i}" for i in range(10)])
    # dummy observation data
    obsdata = pd.DataFrame(index=data.columns, columns=["obsnme","obsval","weight","obgnme"])
    obsdata.obsnme = data.columns
    obsdata.obsval = data.mean().values
    obsdata.weight = 1.0
    obsdata.obgnme = "obgnme"
    return data, obsdata

def dsi_synth(tmp_d,transforms=None,tag=""):

    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)

    dsi = DSI(data=data,transforms=transforms)
    dsi.fit()

    if transforms is not None:
        if "quadratic_extrapolation" in transforms[0].keys():
            nzobs = obsdata.loc[obsdata.weight>0].obsnme.tolist()
            ovals = data.max(axis=0) * 1.1
            obsdata.loc[nzobs,"obsval"] = ovals.values

    td = tmp_d / "template_dsi"
    pstdsi = dsi.prepare_pestpp(td,observation_data=obsdata)
    pstdsi.control_data.noptmax = 1
    pstdsi.pestpp_options["ies_num_reals"] = 10
    pstdsi.pestpp_options["ies_num_reals"] = 10
    pstdsi.write(os.path.join(td, "dsi.pst"),version=2)

    pvals = pd.read_csv(os.path.join(td, "dsi_pars.csv"), index_col=0)
    md = tmp_d / f"master_dsi{tag}"
    num_workers = 1
    worker_root = tmp_d
    print("dsi_exe: ", ies_exe_path)
    pyemu.os_utils.start_workers(
        td,ies_exe_path,"dsi.pst", num_workers=num_workers,
        worker_root=worker_root, master_dir=md, port=_get_port(),
        ppw_function=pyemu.helpers.dsi_pyworker,
        ppw_kwargs={
            "dsi": dsi, "pvals": pvals,
        }
    )
    return

def test_dsi_basic(tmp_path):
    dsi_synth(tmp_path,transforms=None)
    return

def test_dsi_nst(tmp_path):
    transforms = [
        {"type": "normal_score", }
    ]
    dsi_synth(tmp_path,transforms=transforms)
    return

def test_dsi_nst_extrap(tmp_path):
    transforms = [
        {"type": "normal_score", "quadratic_extrapolation":True}
    ]
    dsi_synth(tmp_path,transforms=transforms)
    return


def test_dsi_mixed(tmp_path):
    transforms = [
        {"type": "log10", "columns": [f"obs{i}" for i in range(2)]},
        {"type": "normal_score", }
    ]
    dsi_synth(tmp_path,transforms=transforms)
    return


# @pytest.mark.timeout(method="thread", timeout=1000)
def test_dsivc(tmp_path):
    # basic quick as so can re-run here
    dsi_synth(tmp_path, transforms=None)
    # now test dsicv
    # master_dsi should now exist
    md_hm = tmp_path / "master_dsi"
    # print(os.listdir('.'))
    assert os.path.exists(md_hm), f"Master directory {md_hm} does not exist."
    td = tmp_path / "template_dsivc"
    if os.path.exists(td):
        shutil.rmtree(td)
    shutil.copytree(md_hm, td)

    dsi = DSI.load(os.path.join(td, "dsi.pickle"))

    pst = pyemu.Pst(os.path.join(td, "dsi.pst"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(td, "dsi.0.obs.jcb"))

    obsdata = dsi.observation_data
    decvars = obsdata.obsnme.tolist()[:-2]
    pstdsivc = dsi.prepare_dsivc(t_d=td,
                                oe=oe,
                                decvar_names=decvars,
                                track_stack=False,
                                percentiles=[0.05,0.5,0.95],
                                dsi_args={
                                    "noptmax":1, #just for testing
                                    "decvar_weight":10.0,
                                    "num_pyworkers":1,
                                },
                                ies_exe_path=ies_exe_path,
                                )

    obs = pstdsivc.observation_data
    obs.org_obsnme.unique()

    obsnme = obsdata.obsnme.tolist()[0]
    mou_objectives = obs.loc[(obs.org_obsnme==obsnme) & (obs.stat=="50%")].obsnme.tolist()

    pstdsivc.pestpp_options["mou_objectives"] = mou_objectives
    obs.loc[mou_objectives, "weight"] = 1.0
    obs.loc[mou_objectives, "obgnme"] = "less_than_obj"

    pstdsivc.control_data.noptmax = 1 #just for testing
    pstdsivc.pestpp_options["mou_population_size"] = 4 #just for testing 

    pstdsivc.write(os.path.join(td, "dsivc.pst"),version=2)

    md = tmp_path / "master_dsivc"
    num_workers =  pstdsivc.pestpp_options["mou_population_size"]
    worker_root = tmp_path

    pyemu.os_utils.start_workers(td,
                                 mou_exe_path,
                                    "dsivc.pst",
                                    num_workers=num_workers,
                                    worker_root=worker_root,
                                    master_dir=md,
                                    port=_get_port(),)


def plot_freyberg_dsi():
    import pandas as pd
    import pyemu
    import matplotlib.pyplot as plt

    test_d = "ends_master"
    case = "freyberg6_run_ies"
    pst_name = os.path.join(test_d, case + ".pst")
    pst = pyemu.Pst(pst_name)
    predictions = ["headwater_20171130", "tailwater_20161130", "trgw_0_9_1_20161130"]
    oe_name = pst_name.replace(".pst", ".0.obs.csv")
    pr_oe = pd.read_csv(os.path.join(test_d,"freyberg6_run_ies.0.obs.csv"),index_col=0)
    #pt_oe = pd.read_csv(os.path.join(test_d, "freyberg6_run_ies.3.obs.csv"), index_col=0)
    pt_oe = pr_oe.copy()


    m_d = os.path.join( "master_dsi")
    pst = pyemu.Pst(os.path.join(m_d,"dsi.pst"))
    pr_oe_dsi = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d,"dsi.0.obs.jcb"))._df
    pt_oe_dsi = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d,"dsi.1.obs.jcb"))._df

    pv = pyemu.ObservationEnsemble(pst=pst,df=pt_oe).phi_vector
    pv_dsi = pyemu.ObservationEnsemble(pst=pst, df=pt_oe_dsi).phi_vector
    #print(pt_oe.shape)
    pt_oe = pt_oe.loc[pv<25, :]
    pt_oe_dsi = pt_oe_dsi.loc[pv_dsi < 25, :]

    # print(pt_oe.shape)
    # fig,ax = plt.subplots(1,1,figsize=(5,5))
    # ax.hist(pv,bins=10,facecolor="b",alpha=0.5,density=True)
    # ax.hist(pv_dsi, bins=10, facecolor="m", alpha=0.5,density=True)
    # ax.set_yticks([])
    # plt.tight_layout()
    # plt.show()



    fig,axes = plt.subplots(len(predictions),1,figsize=(10,10))
    for p,ax in zip(predictions,axes):
        ax.hist(pr_oe.loc[:,p].values,bins=10,alpha=0.5,facecolor="0.5",density=True,label="prior")
        ax.hist(pt_oe.loc[:, p].values, bins=10, alpha=0.5, facecolor="b",density=True,label="posterior")
        ax.hist(pr_oe_dsi.loc[:, p].values, bins=10, facecolor="none",hatch="/",edgecolor="0.5",
                lw=2.5,density=True,label="dsi prior")
        ax.hist(pt_oe_dsi.loc[:, p].values, bins=10, facecolor="none",density=True,hatch="/",edgecolor="b",lw=2.5,
                label="dsi posterior")
        ax.set_title(p,loc="left")
        ax.legend(loc="upper right")
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("dsi_pred.pdf")


def lpfa_freyberg(tmp_d="temp",transforms=None):

    test_d = "ends_master"
    test_d = setup_tmp(test_d, tmp_d)

    case = "freyberg6_run_ies"
    pst_name = os.path.join(test_d, case + ".pst")
    pst = pyemu.Pst(pst_name)
    predictions = ["headwater_20171130", "tailwater_20161130", "trgw_0_9_1_20161130"]
    pst.pestpp_options["predictions"] = predictions

    oe_name = pst_name.replace(".pst", ".0.obs.csv")
    oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=oe_name).iloc[:100, :]
    data = oe._df.copy()

    obs = pst.observation_data.copy()
    #obs["date"] = pd.to_datetime(obs.obsnme.str.split("_")[-1])
    #obs.sort_values(by=["obgnme", "date"], inplace=True)

    fit_groups = {
        o: obs.loc[obs.obgnme == o, "obsnme"].tolist()[:12] for o in obs.obgnme.unique()
    }
    groups ={
        o: obs.loc[obs.obgnme == o, "obsnme"].tolist() for o in obs.obgnme.unique()
    }

    input_cols = obs.loc[obs.weight>0, "obsnme"].tolist()
    forecast_names = obs.obsnme.tolist()

    # Create LPFA emulator
    lpfa = LPFA(
        data=data,
        input_names=input_cols,
        groups=groups,
        fit_groups=fit_groups,
        output_names=forecast_names,
        energy_threshold=0.9999,  # Keep most variance in PCA
        seed=42,
        early_stop=True,
        #transforms=None,  # No additional transforms for this demo
        transforms = transforms,
        verbose=True
    )

    #training_data = lpfa.prepare_training_data(test_size=0.2)

    # Define model parameters
    model_params = {
        'activation': 'relu',
        'hidden_units': [128, 64],  # Two hidden layers
        'dropout_rate': 0.1,
        'learning_rate': 0.01
    }

    # Create the model
    lpfa.create_model(model_params)

    # Train the model
    lpfa.fit(epochs=200)

    # Add noise model to capture residuals
    noise_params = {
        'activation': 'relu',
        'hidden_units': [64, 32],  # Smaller network for residuals
        'dropout_rate': 0.05,
        'learning_rate': 0.005
    }

    lpfa.add_noise_model(noise_params)

    # Generate predictions
    predictions = lpfa.predict(obs[["obsval"]].T)


    ## Create scatter plot comparing predictions vs truth
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ## Get non-zero weight observations for comparison
    #comparison_obs = obs.loc[obs.weight > 0].obsnme.values

    ## Extract values for plotting
    #nzobsnmes = obs.loc[obs.weight>0].obsnme.tolist()
    #truth_values = obs.loc[nzobsnmes].obsval.values.flatten()
    #pred_values = predictions.loc[:,nzobsnmes].values.flatten()

    ## Create scatter plot
    #ax.scatter(truth_values, pred_values, alpha=0.6, s=20)
    #ax.set_xlabel('Truth Values')
    #ax.set_ylabel('Predicted Values')
    #ax.set_title('lpfa Emulator: Predicted vs Truth')

    ## Add 1:1 line
    #min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    #max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    #ax.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1, alpha=0.7)
    #ax.set_xlim(min_val, max_val)
    #ax.set_ylim(min_val, max_val)

    ## Calculate R²
    #correlation = np.corrcoef(truth_values, pred_values)[0, 1]
    #r_squared = correlation ** 2
    #assert r_squared >= 0.9, "R-squared should deccent"
    #ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #plt.tight_layout()
    ##plt.show()

    #print(f"Correlation coefficient: {correlation:.3f}")
    #print(f"R-squared: {r_squared:.3f}")

    return

def test_lpfa_basic(tmp_path):
    lpfa_freyberg(tmp_path,transforms=None)
    return

def test_lpfa_std(tmp_path):
    #NOTE: fit with standard scaler transform are worse than without
    lpfa_freyberg(tmp_path,transforms=[
        {"type": "standard_scaler"}
    ])
    return


def gpr_compare_invest():
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "zdt1"
    use_chances = False
    m_d = os.path.join(case+"_gpr_baseline")
    org_d = os.path.join("utils",case+"_template")
    t_d = case+"_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d,t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case+".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5
   
    pop_size = 60
    num_workers = 60
    noptmax_full = 30
    noptmax_inner = 10
    noptmax_outer = 5
    port = 4554
    pst.control_data.noptmax = noptmax_full 
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case+".pst"))
    if not os.path.exists(m_d):
        pyemu.os_utils.start_workers(t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                    master_dir=m_d, verbose=True, port=port)
    #shutil.copytree(t_d,m_d)
    #pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=m_d)
    # use the initial population files for training
    dv_pops = [os.path.join(m_d,"{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_","obs_") for f in dv_pops]

    pst_fname = os.path.join(m_d,case+".pst")
    gpr_t_d = os.path.join(case+"_gpr_template")
    pyemu.helpers.prep_for_gpr(pst_fname,dv_pops,obs_pops,t_d=m_d,gpr_t_d=gpr_t_d,nverf=int(pop_size*.1),\
                               plot_fits=True,apply_standard_scalar=False,include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d,case+".pst"))
    shutil.copy2(os.path.join(m_d,case+".0.dv_pop.csv"),os.path.join(gpr_t_d,"initial_dv_pop.csv"))
    gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d,case+".pst"),version=2)
    gpr_m_d = gpr_t_d.replace("template","master")
    if os.path.exists(gpr_m_d):
         shutil.rmtree(gpr_m_d)
    pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                        master_dir=gpr_m_d, verbose=True, port=port)

    #o1 = pd.read_csv(os.path.join(m_d,case+".{0}.obs_pop.csv".format(max(0,pst.control_data.noptmax))))
    o1 = pd.read_csv(os.path.join(m_d,case+".pareto.archive.summary.csv"))
    o1 = o1.loc[o1.generation == o1.generation.max(), :]
    o1 = o1.loc[o1.is_feasible == True, :]
    o1 = o1.loc[o1.nsga2_front == 1, :]


    import matplotlib.pyplot as plt
    o2 = pd.read_csv(os.path.join(gpr_m_d, case + ".{0}.obs_pop.csv".format(max(0, gpst.control_data.noptmax))))
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(o1.obj_1,o1.obj_2,c="r",s=10)
    ax.scatter(o2.obj_1,o2.obj_2,c="0.5",s=10,alpha=0.5)
    plt.tight_layout()
    plt.savefig("gpr_{0}_compare_noiter.pdf".format(case))
    plt.close(fig)

    # now lets try an inner-outer scheme...
    
    gpst.control_data.noptmax = noptmax_inner
    gpst.write(os.path.join(gpr_t_d,case+".pst"),version=2)
    gpr_t_d_iter = gpr_t_d+"_outeriter{0}".format(0)
    if os.path.exists(gpr_t_d_iter):
        shutil.rmtree(gpr_t_d_iter)
    shutil.copytree(gpr_t_d,gpr_t_d_iter)
    for iouter in range(1,noptmax_outer+1):
        #run the gpr emulator
        gpr_m_d_iter = gpr_t_d_iter.replace("template","master")
        complex_m_d_iter = t_d.replace("template", "master_complex_retrain_outeriter{0}".format(iouter))
        if os.path.exists(gpr_m_d_iter):
            shutil.rmtree(gpr_m_d_iter)
        pyemu.os_utils.start_workers(gpr_t_d_iter, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                        master_dir=gpr_m_d_iter, verbose=True, port=port)
        o2 = pd.read_csv(os.path.join(gpr_m_d_iter,case+".{0}.obs_pop.csv".format(gpst.control_data.noptmax)))

        # now run the final dv pop thru the "complex" model
        final_gpr_dvpop_fname = os.path.join(gpr_m_d_iter,case+".archive.dv_pop.csv")
        assert os.path.exists(final_gpr_dvpop_fname)
        complex_model_dvpop_fname = os.path.join(t_d,"gpr_outeriter{0}_dvpop.csv".format(iouter))
        if os.path.exists(complex_model_dvpop_fname):
            os.remove(complex_model_dvpop_fname)
        # load the gpr archive and do something clever to pick new points to eval
        # with the complex model
        dvpop = pd.read_csv(final_gpr_dvpop_fname,index_col=0)
        if dvpop.shape[0] > pop_size:
            arc_sum = pd.read_csv(os.path.join(gpr_m_d_iter,case+".pareto.archive.summary.csv"))
            as_front_map = {member:front for member,front in zip(arc_sum.member,arc_sum.nsga2_front)}
            as_crowd_map = {member: crowd for member, crowd in zip(arc_sum.member, arc_sum.nsga2_crowding_distance)}
            as_feas_map = {member: feas for member, feas in zip(arc_sum.member, arc_sum.feasible_distance)}
            as_gen_map = {member: gen for member, gen in zip(arc_sum.member, arc_sum.generation)}

            dvpop.loc[:,"front"] = dvpop.index.map(lambda x: as_front_map.get(x,np.nan))
            dvpop.loc[:, "crowd"] = dvpop.index.map(lambda x: as_crowd_map.get(x, np.nan))
            dvpop.loc[:,"feas"] = dvpop.index.map(lambda x: as_feas_map.get(x,np.nan))
            dvpop.loc[:, "gen"] = dvpop.index.map(lambda x: as_gen_map.get(x, np.nan))
            #drop members that have missing archive info
            dvpop = dvpop.dropna()
            if dvpop.shape[0] > pop_size:
                dvpop.sort_values(by=["gen","feas","front","crowd"],ascending=[False,True,True,False],inplace=True)
                dvpop = dvpop.iloc[:pop_size,:]
            dvpop.drop(["gen","feas","front","crowd"],axis=1,inplace=True)

        #shutil.copy2(final_gpr_dvpop_fname,complex_model_dvpop_fname)
        dvpop.to_csv(complex_model_dvpop_fname)
        pst.pestpp_options["mou_dv_population_file"] = os.path.split(complex_model_dvpop_fname)[1]
        pst.control_data.noptmax = -1
        pst.write(os.path.join(t_d,case+".pst"),version=2)

        pyemu.os_utils.start_workers(t_d, mou_exe_path,  case+".pst", num_workers, worker_root=".",
                                    master_dir=complex_m_d_iter, verbose=True, port=port)

        # plot the complex model results...
        o2 = pd.read_csv(os.path.join(complex_m_d_iter, case + ".pareto.archive.summary.csv"))
        o2 = o2.loc[o2.generation == o2.generation.max(), :]
        #o2 = o2.loc[o2.is_feasible==True,:]
        o2 = o2.loc[o2.nsga2_front == 1, :]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(o1.obj_1, o1.obj_2,c="r",s=10,label="full complex")
        ax.scatter(o2.obj_1, o2.obj_2,c="0.5",s=10,alpha=0.5,label="mixed emulated-complex")
        ax.legend(loc="upper right")
        ax.set_xlim(0,10)
        ax.set_ylim(0,20)
        plt.tight_layout()
        plt.savefig("gpr_{0}_compare_iterscheme_{1}.pdf".format(case,iouter))
        plt.close(fig)

        # now add those complex model input-output pop files to the list and retrain
        # the gpr
        dv_pops.append(os.path.join(complex_m_d_iter,case+".0.dv_pop.csv"))
        obs_pops.append(os.path.join(complex_m_d_iter,case+".0.obs_pop.csv"))
        gpr_t_d_iter = gpr_t_d+"_outeriter{0}".format(iouter)
        pyemu.helpers.prep_for_gpr(pst_fname,dv_pops,obs_pops,t_d=gpr_t_d,gpr_t_d=gpr_t_d_iter,nverf=int(pop_size*.1),
                                   plot_fits=True,apply_standard_scalar=False,include_emulated_std_obs=True)
        gpst_iter = pyemu.Pst(os.path.join(gpr_t_d_iter,case+".pst"))
        #aggdf = pd.read_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"),index_col=0)
        #aggdf.index = ["outeriter{0}_member{1}".format(iouter,i) for i in range(aggdf.shape[0])]
        restart_gpr_dvpop_fname = "gpr_restart_dvpop_outeriter{0}.csv".format(iouter)
        #aggdf.to_csv(os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        shutil.copy2(os.path.join(complex_m_d_iter,case+".0.dv_pop.csv"),os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        gpst_iter.pestpp_options["mou_dv_population_file"] = restart_gpr_dvpop_fname
        gpst_iter.control_data.noptmax = gpst.control_data.noptmax
        gpst_iter.write(os.path.join(gpr_t_d_iter,case+".pst"),version=2)


def gpr_constr_invest():
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "constr"
    use_chances = False
    m_d = os.path.join(case + "_gpr_baseline")
    org_d = os.path.join("utils", case + "_template")
    t_d = case + "_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d, t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5

    pop_size = 15
    num_workers = 5
    noptmax_full = 3
    noptmax_inner = 2
    noptmax_outer = 2
    port = 4554
    pst.control_data.noptmax = -1
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case + ".pst"))
    #if not os.path.exists(m_d):
    #    pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                                 master_dir=m_d, verbose=True, port=port)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d,m_d)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=m_d)
    # use the initial population files for training
    dv_pops = [os.path.join(m_d, "{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_", "obs_") for f in dv_pops]

    pst_fname = os.path.join(m_d, case + ".pst")
    gpr_t_d = os.path.join(case + "_gpr_template")
    pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops,t_d=m_d, gpr_t_d=gpr_t_d, nverf=int(pop_size * .1), \
                               plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d, case + ".pst"))
    #shutil.copy2(os.path.join(m_d, case + ".0.dv_pop.csv"), os.path.join(gpr_t_d, "initial_dv_pop.csv"))
    #gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.pestpp_options.pop("mou_dv_population_file",None) #= "initial_dv_pop.csv"
    
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_m_d = gpr_t_d.replace("template", "master")
    if os.path.exists(gpr_m_d):
        shutil.rmtree(gpr_m_d)
    #pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                             master_dir=gpr_m_d, verbose=True, port=port)
    shutil.copytree(gpr_t_d,gpr_m_d)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_m_d)
    
    # o1 = pd.read_csv(os.path.join(m_d,case+".{0}.obs_pop.csv".format(max(0,pst.control_data.noptmax))))
    o1 = pd.read_csv(os.path.join(m_d, case + ".pareto.archive.summary.csv"))
    o1 = o1.loc[o1.generation == o1.generation.max(), :]
    o1 = o1.loc[o1.is_feasible == True, :]
    o1 = o1.loc[o1.nsga2_front == 1, :]

    # import matplotlib.pyplot as plt
    # o2 = pd.read_csv(os.path.join(gpr_m_d, case + ".{0}.obs_pop.csv".format(max(0, gpst.control_data.noptmax))))
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.scatter(o1.obj_1, o1.obj_2, c="r", s=10)
    # ax.scatter(o2.obj_1, o2.obj_2, c="0.5", s=10, alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("gpr_{0}_compare_noiter.pdf".format(case))
    # plt.close(fig)

    # now lets try an inner-outer scheme...

    gpst.control_data.noptmax = noptmax_inner
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_t_d_iter = gpr_t_d + "_outeriter{0}".format(0)
    if os.path.exists(gpr_t_d_iter):
        shutil.rmtree(gpr_t_d_iter)
    shutil.copytree(gpr_t_d, gpr_t_d_iter)
    for iouter in range(1, noptmax_outer + 1):
        # run the gpr emulator
        gpr_m_d_iter = gpr_t_d_iter.replace("template", "master")
        complex_m_d_iter = t_d.replace("template", "master_complex_retrain_outeriter{0}".format(iouter))
        if os.path.exists(gpr_m_d_iter):
            shutil.rmtree(gpr_m_d_iter)
        shutil.copytree(gpr_t_d_iter,gpr_m_d_iter)

        pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_m_d_iter)
    
        #pyemu.os_utils.start_workers(gpr_t_d_iter, mou_exe_path, case + ".pst", num_workers, worker_root=".",
        #                             master_dir=gpr_m_d_iter, verbose=True, port=port)
        
        o2 = pd.read_csv(os.path.join(gpr_m_d_iter, case + ".{0}.obs_pop.csv".format(gpst.control_data.noptmax)))

        # now run the final dv pop thru the "complex" model
        final_gpr_dvpop_fname = os.path.join(gpr_m_d_iter, case + ".archive.dv_pop.csv")
        assert os.path.exists(final_gpr_dvpop_fname)
        complex_model_dvpop_fname = os.path.join(t_d, "gpr_outeriter{0}_dvpop.csv".format(iouter))
        if os.path.exists(complex_model_dvpop_fname):
            os.remove(complex_model_dvpop_fname)
        # load the gpr archive and do something clever to pick new points to eval
        # with the complex model
        dvpop = pd.read_csv(final_gpr_dvpop_fname, index_col=0)
        if dvpop.shape[0] > pop_size:
            arc_sum = pd.read_csv(os.path.join(gpr_m_d_iter, case + ".pareto.archive.summary.csv"))
            as_front_map = {member: front for member, front in zip(arc_sum.member, arc_sum.nsga2_front)}
            as_crowd_map = {member: crowd for member, crowd in zip(arc_sum.member, arc_sum.nsga2_crowding_distance)}
            as_feas_map = {member: feas for member, feas in zip(arc_sum.member, arc_sum.feasible_distance)}
            as_gen_map = {member: gen for member, gen in zip(arc_sum.member, arc_sum.generation)}

            dvpop.loc[:, "front"] = dvpop.index.map(lambda x: as_front_map.get(x, np.nan))
            dvpop.loc[:, "crowd"] = dvpop.index.map(lambda x: as_crowd_map.get(x, np.nan))
            dvpop.loc[:, "feas"] = dvpop.index.map(lambda x: as_feas_map.get(x, np.nan))
            dvpop.loc[:, "gen"] = dvpop.index.map(lambda x: as_gen_map.get(x, np.nan))
            # drop members that have missing archive info
            dvpop = dvpop.dropna()
            if dvpop.shape[0] > pop_size:
                dvpop.sort_values(by=["gen", "feas", "front", "crowd"], ascending=[False, True, True, False],
                                  inplace=True)
                dvpop = dvpop.iloc[:pop_size, :]
            dvpop.drop(["gen", "feas", "front", "crowd"], axis=1, inplace=True)

        # shutil.copy2(final_gpr_dvpop_fname,complex_model_dvpop_fname)
        dvpop.to_csv(complex_model_dvpop_fname)
        pst.pestpp_options["mou_dv_population_file"] = os.path.split(complex_model_dvpop_fname)[1]
        pst.control_data.noptmax = -1
        pst.write(os.path.join(t_d, case + ".pst"), version=2)
        if os.path.exists(complex_m_d_iter):
            shutil.rmtree(complex_m_d_iter)
        shutil.copytree(t_d,complex_m_d_iter)
        #pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
        #                             master_dir=complex_m_d_iter, verbose=True, port=port)
        pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=complex_m_d_iter)
    
        # plot the complex model results...
        o2 = pd.read_csv(os.path.join(complex_m_d_iter, case + ".pareto.archive.summary.csv"))
        o2 = o2.loc[o2.generation == o2.generation.max(), :]
        # o2 = o2.loc[o2.is_feasible==True,:]
        o2 = o2.loc[o2.nsga2_front == 1, :]
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.scatter(o1.obj_1, o1.obj_2, c="r", s=10, label="full complex")
        # ax.scatter(o2.obj_1, o2.obj_2, c="0.5", s=10, alpha=0.5, label="mixed emulated-complex")
        # ax.legend(loc="upper right")
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 20)
        # plt.tight_layout()
        # plt.savefig("gpr_{0}_compare_iterscheme_{1}.pdf".format(case, iouter))
        # plt.close(fig)

        # now add those complex model input-output pop files to the list and retrain
        # the gpr
        dv_pops.append(os.path.join(complex_m_d_iter, case + ".0.dv_pop.csv"))
        obs_pops.append(os.path.join(complex_m_d_iter, case + ".0.obs_pop.csv"))
        gpr_t_d_iter = gpr_t_d + "_outeriter{0}".format(iouter)
        pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops, t_d=gpr_t_d,gpr_t_d=gpr_t_d_iter, nverf=int(pop_size * .1),
                                   plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
        gpst_iter = pyemu.Pst(os.path.join(gpr_t_d_iter, case + ".pst"))
        # aggdf = pd.read_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"),index_col=0)
        # aggdf.index = ["outeriter{0}_member{1}".format(iouter,i) for i in range(aggdf.shape[0])]
        #restart_gpr_dvpop_fname = "gpr_restart_dvpop_outeriter{0}.csv".format(iouter)
        # aggdf.to_csv(os.path.join(gpr_t_d_iter,restart_gpr_dvpop_fname))
        #shutil.copy2(os.path.join(complex_m_d_iter, case + ".0.dv_pop.csv"),
        #             os.path.join(gpr_t_d_iter, restart_gpr_dvpop_fname))
        gpst_iter.pestpp_options.pop("mou_dv_population_file",None)# = restart_gpr_dvpop_fname
        gpst_iter.control_data.noptmax = gpst.control_data.noptmax
        gpst_iter.write(os.path.join(gpr_t_d_iter, case + ".pst"), version=2)

    psum_fname = os.path.join(complex_m_d_iter,case+".pareto.archive.summary.csv")
    assert os.path.exists(psum_fname)
    psum = pd.read_csv(psum_fname)
    #assert 1.0 in psum.obj_1.values
    #assert 1.0 in psum.obj_2.values


def collate_training_data(pst,m_d,case):

    input_fnames = [os.path.join(m_d,"{0}.0.dv_pop.csv".format(case))]
    output_fnames = [f.replace("dv_","obs_") for f in input_fnames]

    # work out input variable names
    input_groups = pst.pestpp_options.get("opt_dec_var_groups",None)
    par = pst.parameter_data
    if input_groups is None:
        print("using all adjustable parameters as inputs")
        input_names = pst.adj_par_names
    else:
        input_groups = set([i.strip() for i in input_groups.lower().strip().split(",")])
        print("input groups:",input_groups)
        adj_par = par.loc[pst.adj_par_names,:].copy()
        adj_par = adj_par.loc[adj_par.pargp.apply(lambda x: x in input_groups),:]
        input_names = adj_par.parnme.tolist()
    print("input names:",input_names)

    #work out constraints and objectives
    ineq_names = pst.less_than_obs_constraints.tolist()
    ineq_names.extend(pst.greater_than_obs_constraints.tolist())
    obs = pst.observation_data
    objs = pst.pestpp_options.get("mou_objectives",None)
    constraints = []

    if objs is None:
        print("'mou_objectives' not found in ++ options, using all ineq tagged non-zero weighted obs as objectives")
        objs = ineq_names
    else:
        objs = objs.lower().strip().split(',')
        constraints = [n for n in ineq_names if n not in objs]

    print("objectives:",objs)
    print("constraints:",constraints)
    output_names = objs
    output_names.extend(constraints)

    print("loading input and output files")
    if isinstance(input_fnames,str):
        input_fnames = [input_fnames]
    if isinstance(output_fnames,str):
        output_fnames = [output_fnames]
    if len(output_fnames) != len(input_fnames):
        raise Exception("len(input_fnames) != len(output_fnames)")


    dfs = []
    for input_fname,output_fname in zip(input_fnames,output_fnames):
        if input_fname.lower().endswith(".csv"):
            input_df = pd.read_csv(os.path.join(input_fname),index_col=0)
        elif input_fname.lower().endswith(".jcb"):
            input_df = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=input_fname)._df
        else:
            raise Exception("unrecognized input_fname extension:'{0}', looking for csv or jcb".\
                            format(input_fname.lower()))

        if output_fname.lower().endswith(".csv"):
            output_df = pd.read_csv(os.path.join(output_fname),index_col=0)
        elif output_fname.lower().endswith(".jcb"):
            output_df = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=output_fname)._df
        else:
            raise Exception("unrecognized output_fname extension:'{0}', looking for csv or jcb".\
                            format(output_fname.lower()))

        if input_df.shape[0] != output_df.shape[0]:
            raise Exception("input rows != output rows for {0} and {1}".\
                            format(input_fname,output_fname))
        input_df = input_df.loc[:,input_names]
        assert input_df.shape == input_df.dropna().shape

        output_df = output_df.loc[:, output_names]
        assert output_df.shape == output_df.dropna().shape

        input_df.loc[:,output_names] = output_df.values
        dfs.append(input_df)
        print("...loaded",input_fname,output_fname)

    data = pd.concat(dfs)
    assert data.shape == data.dropna().shape
    #df.to_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"))
    #print("aggregated training dataset shape",df.shape,"saved to",pst_fname + ".aggresults.csv")
    return data, input_names, output_names


@pytest.mark.skip(reason="seems like it still in dev")
def gpr_zdt1_test():
    import numpy as np
    import subprocess as sp
    import multiprocessing as mp
    from datetime import datetime
    from sklearn.gaussian_process import GaussianProcessRegressor
    case = "zdt1"
    use_chances = False
    m_d = os.path.join(case + "_gpr_baseline")
    org_d = os.path.join("utils", case + "_template")
    t_d = case + "_template"
    if os.path.exists(t_d):
         shutil.rmtree(t_d)
    shutil.copytree(org_d, t_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d, case + ".pst"))
    pst.pestpp_options["mou_generator"] = "pso"
    pst.pestpp_options["overdue_giveup_fac"] = 1e10
    pst.pestpp_options["overdue_resched_fac"] = 1e10
    if use_chances:
        pst.pestpp_options["opt_risk"] = 0.95
        pst.pestpp_options["opt_stack_size"] = 50
        pst.pestpp_options["opt_recalc_chance_every"] = 10000
        pst.pestpp_options["opt_chance_points"] = "single"
    else:
        pst.pestpp_options["opt_risk"] = 0.5

    pop_size = 20
    num_workers = 3
    noptmax_full = 1
    
    port = 4569
    pst.control_data.noptmax = -1
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.write(os.path.join(t_d, case + ".pst"))
    #if not os.path.exists(m_d):
    #    pyemu.os_utils.start_workers(t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                                 master_dir=m_d, verbose=True, port=port)
    
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=t_d)
    

    m_d = t_d
    dv_pops = [os.path.join(m_d, "{0}.0.dv_pop.csv".format(case))]
    obs_pops = [f.replace("dv_", "obs_") for f in dv_pops]

    pst_fname = os.path.join(m_d, case + ".pst")
    gpr_t_d = os.path.join(case + "_gpr_template")

    data, input_names, output_names = collate_training_data(pst,m_d,case)
    from pyemu.emulators.gpr import GPR
    gpr = GPR(data=data.copy(),
          input_names=input_names,
          output_names=output_names,
          #transforms=transforms,
          #kernel=gp_kernel,
          n_restarts_optimizer=20,
          );
    gpr.fit()
    gpr.prepare_pestpp(m_d,case,gpr_t_d=gpr_t_d)

    #pyemu.helpers.prep_for_gpr(pst_fname, dv_pops, obs_pops, t_d=m_d,gpr_t_d=gpr_t_d, nverf=int(pop_size * .1), \
    #                           plot_fits=True, apply_standard_scalar=False, include_emulated_std_obs=True)
    gpst = pyemu.Pst(os.path.join(gpr_t_d, case + "_gpr.pst"))
    shutil.copy2(os.path.join(m_d, case + ".0.dv_pop.csv"), os.path.join(gpr_t_d, "initial_dv_pop.csv"))
    gpst.pestpp_options["mou_dv_population_file"] = "initial_dv_pop.csv"
    gpst.control_data.noptmax = noptmax_full
    gpst.write(os.path.join(gpr_t_d, case + ".pst"), version=2)
    gpr_m_d = gpr_t_d.replace("template", "master")
    if os.path.exists(gpr_m_d):
        shutil.rmtree(gpr_m_d)
    start = datetime.now()
    #pyemu.os_utils.start_workers(gpr_t_d, mou_exe_path, case + ".pst", num_workers, worker_root=".",
    #                             master_dir=gpr_m_d, verbose=True, port=port)
    pyemu.os_utils.run("{0} {1}.pst".format(mou_exe_path,case),cwd=gpr_t_d)

    gpr_m_d = gpr_t_d

    finish = datetime.now()
    duration1 = (finish - start).total_seconds()
    arcorg = pd.read_csv(os.path.join(gpr_m_d,"zdt1.archive.obs_pop.csv"),index_col=0)
    

    psum_fname = os.path.join(gpr_m_d,case+".pareto.archive.summary.csv")
    assert os.path.exists(psum_fname)
    psum = pd.read_csv(psum_fname)
    print(psum.obj_1.min())
    print(psum.obj_2.min())
    assert psum.obj_1.min() < 0.05

    gpr_t_d2 = gpr_t_d + "_ppw"
    if os.path.exists(gpr_t_d2):
        shutil.rmtree(gpr_t_d2)
    shutil.copytree(gpr_t_d,gpr_t_d2)

    gpr_m_d2 = gpr_t_d2.replace("template","master")
    gpr_d2 = GPR.load(os.path.join(gpr_m_d2,"gpr_emulator.pkl"))
    input_df = pd.read_csv(os.path.join(gpr_t_d2,"gpr_input.csv"),index_col=0)
    #mdf = pd.read_csv(os.path.join(gpr_t_d2,"gprmodel_info.csv"),index_col=0)
    #mdf["model_fname"] = mdf.model_fname.apply(lambda x: os.path.join(gpr_t_d2,x))
    pyemu.os_utils.start_workers(gpr_t_d2, mou_exe_path, case + ".pst", num_workers, worker_root=".",
                                 master_dir=gpr_m_d2, verbose=True, port=port,
                                 ppw_function=pyemu.helpers.gpr_pyworker,
                                 ppw_kwargs={"input_df":input_df,
                                            #"mdf":mdf,
                                            "gpr":gpr_d2})
    
    
    arcppw = pd.read_csv(os.path.join(gpr_m_d2,"zdt1.archive.obs_pop.csv"),index_col=0)
    diff = np.abs(arcppw.values - arcorg.values)
    print(diff.max())
    assert diff.max() < 1e-6
        

    start = datetime.now()
    b_d = os.getcwd()
    os.chdir(gpr_t_d2)
    p = sp.Popen([mou_exe_path,"{0}.pst".format(case),"/h",":{0}".format(port)])
    os.chdir(b_d)
    #p.wait()
    #return
    
    # looper over and start the workers - in this
    # case they dont need unique dirs since they aren't writing
    # anything
    procs = []
    # try this test with 1 worker as an edge case
    num_workers = 1
    for i in range(num_workers):
        pp = mp.Process(target=gpr_zdt1_ppw)
        pp.start()
        procs.append(pp)
    # if everything worked, the the workers should receive the 
    # shutdown signal from the master and exit gracefully...
    for pp in procs:
        pp.join()

    # wait for the master to finish...but should already be finished
    p.wait()
    finish = datetime.now()
    print("ppw` took",(finish-start).total_seconds())
    print("org took",duration1)

    arcppw = pd.read_csv(os.path.join(gpr_t_d2,"zdt1.archive.obs_pop.csv"),index_col=0)
    diff = np.abs(arcppw.values - arcorg.values)
    print(diff.max())
    assert diff.max() < 1e-6
        


def gpr_zdt1_ppw():
    t_d = "zdt1_gpr_template"
    os.chdir(t_d)
    pst_name = "zdt1.pst"
    ppw = pyemu.helpers.gpr_pyworker(pst_name,"localhost",4569,gpr=True)
    os.chdir("..")


def dsiae_basic(transforms=None):
    """Basic DSIAE test using synth dataset - minimal compute"""
    
    if not HAS_TENSORFLOW:
        pytest.skip("TensorFlow not available, skipping DSIAE tests")
    
    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)


    # Test DSIAE initialization and basic functionality
    from pyemu.emulators import DSIAE
    dsiae = DSIAE(data=data, transforms=transforms, latent_dim=3, verbose=False)  # Fixed small latent dim
    
    # Test fit with minimal parameters for speed
    dsiae.fit(validation_split=0.2, epochs=5, batch_size=16, early_stopping=False)  # Very few epochs
    
    # Test encoding
    Z = dsiae.encode(data.iloc[:5])  # Test with just 5 samples
    assert Z.shape[0] == 5
    assert Z.shape[1] == 3  # latent_dim
    
    # Test prediction
    sim_vals = dsiae.predict(Z.iloc[0])
    assert len(sim_vals) == len(data.columns)
    
    return dsiae, obsdata



@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_dsiae_basic():
    """Test basic DSIAE functionality with transforms"""
    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)

    transforms = [
        {"type": "normal_score", }
    ]

    # Test DSIAE initialization and basic functionality
    from pyemu.emulators import DSIAE
    dsiae = DSIAE(data=data, transforms=transforms, latent_dim=3, verbose=False)  # Fixed small latent dim
    # Test fit with minimal parameters for speed
    dsiae.fit(validation_split=0.2, epochs=5, batch_size=16, early_stopping=False)  # Very few epochs
    assert dsiae.fitted

    # Test encoding
    Z = dsiae.encode(data.iloc[:5])  # Test with just 5 samples
    assert Z.shape[0] == 5
    assert Z.shape[1] == 3  # latent_dim
    
    # Test prediction
    sim_vals = dsiae.predict(Z.iloc[0])
    assert len(sim_vals) == len(data.columns)

    
    return



@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_dsiae_auto_latent_dim():
    """Test DSIAE with automatic latent dimension selection"""
    
    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)

    from pyemu.emulators import DSIAE
    dsiae = DSIAE(data=data, latent_dim=None, energy_threshold=0.8)  # Auto dimension
    dsiae.fit(epochs=3, batch_size=8)  # Minimal training
    
    assert dsiae.fitted
    assert dsiae.latent_dim > 0
    return

#@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
@pytest.mark.skip(reason="it is hanging in CI for some reason")
def test_dsiae_with_ies(tmp_path):

    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)

    from pyemu.emulators import DSIAE
    dsiae = DSIAE(data=data, latent_dim=3)  # Auto dimension
    dsiae.fit(epochs=3, batch_size=8)  # Minimal training

    td = tmp_path / "template_dsiae"
    pstdsi = dsiae.prepare_pestpp(td,observation_data=obsdata)
    pstdsi.control_data.noptmax = -1
    pstdsi.pestpp_options["ies_num_reals"] = 3
    pstdsi.write(os.path.join(td, "dsi.pst"),version=2)

    pvals = pd.read_csv(os.path.join(td, "dsi_pars.csv"), index_col=0)
    md = tmp_path / f"master_dsiae"
    num_workers = 1
    worker_root = tmp_path
    print("dsi_exe: ", ies_exe_path)
    pyemu.os_utils.start_workers(
        td,ies_exe_path,"dsi.pst", num_workers=num_workers,
        worker_root=worker_root, master_dir=md, port=_get_port(),
        ppw_function=pyemu.helpers.dsi_pyworker,
        ppw_kwargs={
            "dsi": dsiae, "pvals": pvals,
        }
    )
    return


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_autoencoder_basic():
    """Test standalone AutoEncoder functionality"""
    
    from pyemu.emulators.dsiae import AutoEncoder
    
    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)  # 50 samples, 10 features
    
    # Test initialization
    ae = AutoEncoder(input_dim=10, latent_dim=3, hidden_dims=(8, 4))
    
    # Test fit with minimal parameters
    history = ae.fit(X, epochs=3, batch_size=16, verbose=0)
    assert history is not None
    
    # Test encode/decode
    Z = ae.encode(X[:5])  # Test with 5 samples
    assert Z.shape == (5, 3)  # latent_dim = 3
    
    X_reconstructed = ae.decode(Z)
    assert X_reconstructed.shape == (5, 10)  # original input_dim = 10
    
    return


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_autoencoder_pandas_input():
    """Test AutoEncoder with pandas DataFrame input"""
    
    from pyemu.emulators.dsiae import AutoEncoder
    
    # Create pandas DataFrame
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(30, 8), 
                       columns=[f'feature_{i}' for i in range(8)],
                       index=[f'sample_{i}' for i in range(30)])
    
    ae = AutoEncoder(input_dim=8, latent_dim=2, hidden_dims=(6,))
    ae.fit(data.values, epochs=2, verbose=0)
    
    # Test with DataFrame input
    Z = ae.encode(data.iloc[:3])
    assert Z.shape == (3, 2)
    
    # Test with Series input  
    Z_series = ae.encode(data.iloc[0])
    assert Z_series.shape == (1, 2)
    
    return


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_dsiae_hyperparam_search():
    """Test DSIAE hyperparameter search"""
    
    dsiae, obsdata = dsiae_basic()
    
    # Test with minimal search space
    results = dsiae.hyperparam_search(
        latent_dims=[2, 3],
        hidden_dims_list=[(8,)],  # Single architecture
        lrs=[1e-2],  # Single learning rate
        epochs=2,  # Very few epochs
        batch_size=8
    )
    
    assert isinstance(results, dict)
    assert len(results) > 0
    
    return

if __name__ == "__main__":
    
    test_dsi_basic("temp")
    #test_dsi_nst("temp")
    #test_dsi_nst_extrap("temp")
    #test_dsi_mixed("temp")
    #test_dsivc_freyberg("temp")
    #plot_freyberg_dsi()
    #test_lpfa_std()
    #gpr_zdt1_test()

