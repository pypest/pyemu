import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import platform
import pyemu
from pst_from_tests import setup_tmp, ies_exe_path, _get_port
from pyemu.emulators import DSI, LPFA, GPR



def dsi_freyberg(tmp_d,transforms=None,tag=""):

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

    dsi = DSI(data=data,transforms=transforms)
    #dsi._fit_transformer_pipeline()
    dsi.fit()

    # history match
    obsdata = pst.observation_data.copy()
    if transforms is not None:
        if "quadratic_extrapolation" in transforms[0].keys():
            nzobs = obsdata.loc[obsdata.weight>0].obsnme.tolist()
            ovals = oe.loc[:,nzobs].max(axis=0) * 1.1
            obsdata.loc[nzobs,"obsval"] = ovals.values

    td = "template_dsi"
    pstdsi = dsi.prepare_pestpp(td,observation_data=obsdata)
    pstdsi.control_data.noptmax = 1
    pstdsi.pestpp_options["ies_num_reals"] = 100
    pstdsi.write(os.path.join(td, "dsi.pst"),version=2)

    pvals = pd.read_csv(os.path.join(td, "dsi_pars.csv"), index_col=0)
    md = f"master_dsi{tag}"
    num_workers = 1
    worker_root = "."
    pyemu.os_utils.start_workers(
        td,ies_exe_path,"dsi.pst", num_workers=num_workers,
        worker_root=worker_root, master_dir=md, port=_get_port(),
        ppw_function=pyemu.helpers.dsi_pyworker,
        ppw_kwargs={
            "dsi": dsi, "pvals": pvals,
        }
    )
    return

def test_dsi_basic(tmp_d="temp"):
    dsi_freyberg(tmp_d,transforms=None)
    return

def test_dsi_nst(tmp_d="temp"):
    transforms = [
        {"type": "normal_score", }
    ]
    dsi_freyberg(tmp_d,transforms=transforms)
    return

def test_dsi_nst_extrap(tmp_d="temp"):
    transforms = [
        {"type": "normal_score", "quadratic_extrapolation":True}
    ]
    dsi_freyberg(tmp_d,transforms=transforms)
    return

def test_dsi_mixed(tmp_d="temp"):
    transforms = [
        {"type": "log10", "columns": ["headwater_20171130", "tailwater_20161130"]},
        {"type": "normal_score", }
    ]
    dsi_freyberg(tmp_d,transforms=transforms)
    return

def test_dsivc_freyberg():

    md_hm = "master_dsi"
    assert os.path.exists(md_hm), f"Master directory {md_hm} does not exist."
    td = "template_dsivc"
    if os.path.exists(td):
        shutil.rmtree(td)
    shutil.copytree(md_hm, td)

    dsi = DSI.load(os.path.join(td, "dsi.pickle"))

    pst = pyemu.Pst(os.path.join(td, "dsi.pst"))
    oe = pyemu.ObservationEnsemble.from_binary(pst,os.path.join(td, "dsi.1.obs.jcb"))

    obsdata = dsi.observation_data
    decvars = obsdata.loc[obsdata.obgnme=="out_wel"].obsnme.tolist()
    pstdsivc = dsi.prepare_dsivc(t_d=td,
                                oe=oe,
                                decvar_names=decvars,
                                track_stack=False,
                                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                dsi_args={
                                    "noptmax":3,
                                    "decvar_weight":10.0,
                                    "num_pyworkers":1,
                                },
                                ies_exe_path=ies_exe_path,
                                )

    obs = pstdsivc.observation_data
    obs.org_obsnme.unique()

    obsnme = obsdata.loc[obsdata.obgnme=="tailwater"].obsnme.tolist()[-1]
    mou_objectives = obs.loc[(obs.org_obsnme==obsnme) & (obs.stat=="50%")].obsnme.tolist()

    pstdsivc.pestpp_options["mou_objectives"] = mou_objectives
    obs.loc[mou_objectives, "weight"] = 1.0
    obs.loc[mou_objectives, "obgnme"] = "less_than_obj"

    pstdsivc.control_data.noptmax = 1 #just for testing
    pstdsivc.pestpp_options["mou_population_size"] = 10 #just for testing 

    pstdsivc.write(os.path.join(td, "dsivc.pst"),version=2)

    md = "master_dsivc"
    num_workers = 1
    worker_root = "."

    pyemu.os_utils.start_workers(td,
                                 "pestpp-mou",
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


    # Create scatter plot comparing predictions vs truth
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get non-zero weight observations for comparison
    comparison_obs = obs.loc[obs.weight > 0].obsnme.values

    # Extract values for plotting
    nzobsnmes = obs.loc[obs.weight>0].obsnme.tolist()
    truth_values = obs.loc[nzobsnmes].obsval.values.flatten()
    pred_values = predictions.loc[:,nzobsnmes].values.flatten()

    # Create scatter plot
    ax.scatter(truth_values, pred_values, alpha=0.6, s=20)
    ax.set_xlabel('Truth Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('lpfa Emulator: Predicted vs Truth')

    # Add 1:1 line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1, alpha=0.7)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Calculate R²
    correlation = np.corrcoef(truth_values, pred_values)[0, 1]
    r_squared = correlation ** 2
    assert r_squared >= 0.9, "R-squared should deccent"
    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    #plt.show()

    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"R-squared: {r_squared:.3f}")

    return

def test_lpfa_basic():
    lpfa_freyberg(tmp_d="temp",transforms=None)
    return

def test_lpfa_std():
    #NOTE: fit with standard scaler transform are worse than without
    lpfa_freyberg(tmp_d="temp",transforms=[
        {"type": "standard_scaler"}
    ])
    return

if __name__ == "__main__":
    #test_dsi_basic()
    #test_dsi_nst()
    #test_dsi_nst_extrap()
    #test_dsi_mixed()
    #test_dsivc_freyberg()
    #plot_freyberg_dsi()
    test_lpfa_std()
