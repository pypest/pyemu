import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import platform
import pyemu
from pst_from_tests import setup_tmp, ies_exe_path, _get_port
from pyemu.emulators import DSI


#def test_dsi_feature_transforms():
#    """Test feature transforms in DSI emulator"""
#    # Create test data simulating an ensemble
#    np.random.seed(42)
#    n_reals = 10
#    n_obs = 5
#    sim_names = [f"obs{i}" for i in range(n_obs)]
#    sim_data = np.random.lognormal(mean=0, sigma=1, size=(n_reals, n_obs))
#    sim_ensemble = pd.DataFrame(sim_data, columns=sim_names)
#    
#    # Create DSI emulator
#    pst = pyemu.Pst.from_par_obs_names(["p1"], sim_names)
#    dsi = pyemu.emulators.DSI(
#        pst=pst,
#        sim_ensemble=sim_ensemble,
#        transforms = [{"type": "log10", "columns": sim_names},
#                        {"type": "normal_score", "columns": sim_names}],
#        
#    )
#    
#    # Test feature transforms
#    dsi.apply_feature_transforms()
#    
#    # Check that transformed data exists
#    assert dsi.data_transformed is not None
#    
#    # Check log transform was applied (values should be smaller than original lognormal data)
#    assert dsi.data_transformed.mean().mean() < sim_ensemble.mean().mean()
#    
#    # Check the feature transformer object exists
#    assert hasattr(dsi, "feature_transformer")
#    
#    # Test with specific columns for log transform
#    dsi2 = pyemu.emulators.DSI(
#        pst=pst,
#        sim_ensemble=sim_ensemble,
#        transforms = [{"type": "log10", "columns": sim_names[:2]}]
#                            )
#    dsi2.apply_feature_transforms()
#    
#    # Check only specified columns were log transformed
#    orig_means = sim_ensemble.mean()
#    transformed_means = dsi2.data_transformed.mean()
#    
#    for i, col in enumerate(sim_names):
#        if i < 2:  # Should be log transformed
#            assert transformed_means[col] < orig_means[col]
#        else:  # Should be unchanged
#            assert np.isclose(transformed_means[col], orig_means[col])

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

    dsi = DSI(sim_ensemble=data,transforms=transforms)
    dsi.apply_feature_transforms()
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




if __name__ == "__main__":
    #test_dsi_basic()
    #test_dsi_nst()
    #test_dsi_nst_extrap()
    #test_dsi_mixed()
    test_dsivc_freyberg()