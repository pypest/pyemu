import os
# import sys
import shutil
import pytest
import numpy as np

from pathlib import Path
# import platform
import pyemu
from pst_from_tests import setup_tmp, _get_port, exepath_dict
from pyemu.emulators import DSI, LPFA, GPR


ies_exe_path = exepath_dict["pestpp-ies"]
mou_exe_path = exepath_dict["pestpp-mou"]

# Check for TensorFlow availability for DSIAE tests
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

#import pandas after TF to avoid pyarrow hanging BS
import pandas as pd


def generate_synth_data(num_realizations=100, num_observations=10):

    # generate synth data
    data = pyemu.en.rng.normal(size=(num_realizations,num_observations))
    data = pd.DataFrame(data,columns=[f"obs{i}" for i in range(10)])
    # dummy observation data
    obsdata = pd.DataFrame(index=data.columns, columns=["obsnme","obsval","weight","obgnme"])
    obsdata.obsnme = data.columns
    obsdata.obsval = data.mean().values
    obsdata.weight = 1.0
    obsdata.obgnme = "obgnme"
    return data, obsdata


def _synth_data(n_real=100, n_obs=10, seed=42):
    """Generate synthetic ensemble data and dummy observation metadata (deterministic)."""
    np.random.seed(seed)
    data = pd.DataFrame(
        np.random.normal(size=(n_real, n_obs)),
        columns=[f"obs{i}" for i in range(n_obs)],
    )
    obsdata = pd.DataFrame(
        {
            "obsnme": data.columns,
            "obsval": data.mean().values,
            "weight": 1.0,
            "obgnme": "obgnme",
        },
        index=data.columns,
    )
    return data, obsdata


def dsi_synth(tmp_d,transforms=None,tag="",use_runstor=True,**kwargs):

    tmp_d = Path(tmp_d)

    data, obsdata = generate_synth_data(num_realizations=100,num_observations=10)

    dsi = DSI(data=data,transforms=transforms,pst=obsdata,**kwargs)
    dsi.fit()

    if transforms is not None:
        if "quadratic_extrapolation" in transforms[0].keys():
            nzobs = obsdata.loc[obsdata.weight>0].obsnme.tolist()
            ovals = data.max(axis=0) * 1.1
            obsdata.loc[nzobs,"obsval"] = ovals.values

    td = tmp_d / "template_dsi"
    pstdsi = dsi.prepare_pestpp(td,observation_data=obsdata, use_runstor=use_runstor)
    pstdsi.control_data.noptmax = 1
    pstdsi.pestpp_options["ies_num_reals"] = 10
    pstdsi.write(os.path.join(td, "dsi.pst"),version=2)

    pvals = pd.read_csv(os.path.join(td, "dsi_pars.csv"), index_col=0)
    md = tmp_d / f"master_dsi{tag}"
    num_workers = 1
    worker_root = tmp_d
    print("dsi_exe: ", ies_exe_path)

    if use_runstor:
        shutil.copytree(td, md)
        pyemu.os_utils.run(f'{ies_exe_path} dsi.pst /e', cwd=md, verbose=True)
    else:
        pyemu.os_utils.start_workers(
                                    td,ies_exe_path,"dsi.pst", num_workers=num_workers,
                                    worker_root=worker_root, master_dir=md, port=_get_port(),
                                    ppw_function=pyemu.helpers.dsi_pyworker,
                                    ppw_kwargs={
                                        "dsi": dsi, "pvals": pvals,
                                    }
                                    )
    # verify that phi reduced
    pst = pyemu.Pst(os.path.join(md, "dsi.pst"))
    phis = pst.ies.phimeas['mean'].values
    assert phis[-1] < phis[0]
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


def test_generic_transformer(tmp_path):
    """Test using a generic sklearn transformer."""
    try:
        from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
    except ImportError:
        pytest.skip("sklearn not installed")
        
    transforms = [
        {"type": MinMaxScaler, "feature_range": (0, 1)},
    ]
    dsi_synth(tmp_path, transforms=transforms, tag="_generic")
    
    # Verify the transformed data range
    # Load DSI object to check internal state
    td = Path(tmp_path) / "template_dsi"
    dsi_loaded = DSI.load(os.path.join(td, "dsi.pickle"))
    
    # Check that data was transformed to [0, 1]
    transformed_data = dsi_loaded.data_transformed
    assert transformed_data.min().min() >= 0.0 - 1e-6
    assert transformed_data.max().max() <= 1.0 + 1e-6
    
    # Check inverse transform
    original_data = dsi_loaded.data
    inversed_data = dsi_loaded.transformer_pipeline.inverse(transformed_data)
    # check columnsa re the same
    assert all(original_data.columns == inversed_data.columns)
    # check values are close
    assert np.allclose(original_data.values,
                       inversed_data.loc[original_data.index,original_data.columns].values, 
                        atol=1e-5)

    # Test again with QuantileTransformer (more complex)
    transforms = [
        {"type": QuantileTransformer, "output_distribution": "normal", "n_quantiles": 50, "random_state": 42},
    ]
    dsi_synth(tmp_path, transforms=transforms, tag="_quantile")
    return

@pytest.mark.skip(reason="still in dev")
#@pytest.mark.timeout(method="thread", timeout=1000)
def test_dsivc(tmp_path):
    tmp_path = Path(tmp_path)
    # basic quick as so can re-run here
    dsi_synth(tmp_path, transforms=None, use_runstor=True)
    # now test dsicv
    # master_dsi should now exist

    md_hm = tmp_path / "template_dsi"
    # print(os.listdir('.'))
    assert os.path.exists(md_hm), f"Master directory {md_hm} does not exist."
    td = tmp_path / "template_dsivc"
    if os.path.exists(td):
        shutil.rmtree(td)
    shutil.copytree(md_hm, td)

    dsi = DSI.load(os.path.join(td, "dsi.pickle"))

    pst = pyemu.Pst(os.path.join(td, "dsi.pst"))
    try:
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(td, "dsi.0.obs.jcb"))
    except:
        oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(td, "dsi.0.obs.csv"))

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
    pstdsivc.pestpp_options["mou_population_size"] = 20 #just for testing 

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

@pytest.mark.skip(reason="depreceated; replace with synth test")
def test_lpfa_basic(tmp_path):
    lpfa_freyberg(tmp_path,transforms=None)
    return

@pytest.mark.skip(reason="depreceated; replace with synth test")
def test_lpfa_std(tmp_path):
    #NOTE: fit with standard scaler transform are worse than without
    lpfa_freyberg(tmp_path,transforms=[
        {"type": "standard_scaler"}
    ])
    return

@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
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
@pytest.mark.skip(reason="it is hanging in CI for some reason;passes locally")
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
    rng = np.random.RandomState(42)
    X = rng.standard_normal((50, 10,)).astype(np.float32)  # 50 samples, 10 features

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
    rng = np.random.RandomState(42)
    data = pd.DataFrame(rng.standard_normal((30, 8,)),
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
@pytest.mark.skip(reason="it is hanging in CI for some reason;passes locally")
def test_dsiae_with_ies(tmp_path, use_runstor=True):

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
    if use_runstor:
        pyemu.os_utils.run("pestpp-ies dsi.pst /e", cwd=td, verbose=True)
    else:
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
    rng = np.random.RandomState(42)
    X = rng.standard_normal((50, 10,)).astype(np.float32)  # 50 samples, 10 features

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
    rng = np.random.RandomState(42)
    data = pd.DataFrame(rng.standard_normal((30, 8,)),
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

@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
def test_dsiae_save_load(tmp_path):
    if isinstance(tmp_path, str) and not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # 1. Generate synthetic data
    num_realizations = 50
    num_observations = 20
    data = pyemu.en.rng.normal(size=(num_realizations, num_observations))
    data_df = pd.DataFrame(data, columns=[f"obs{i}" for i in range(num_observations)])

    # 2. Initialize and fit DSIAE
    # Using a small latent dim and few epochs for speed
    latent_dim = 5
    from pyemu.emulators.dsiae import DSIAE
    dsiae = DSIAE(data=data_df, latent_dim=latent_dim, verbose=True)

    # Fit the model
    dsiae.fit(epochs=10, batch_size=10, validation_split=0.2)

    assert dsiae.fitted is True
    assert hasattr(dsiae, 'encoder')

    # 3. Generate predictions on new data (or the training data)
    # Let's use some random "parameter" values in latent space to generate observations
    # The predict method takes pvals which are latent space values

    # Generate random latent vectors
    new_pvals = pyemu.en.rng.normal(size=(5, latent_dim))
    new_pvals_df = pd.DataFrame(new_pvals, columns=[f"latent_{i}" for i in range(latent_dim)])

    # Predict with original model
    preds_original = dsiae.predict(new_pvals_df)

    # 4. Save the model
    save_path = os.path.join(tmp_path, "dsiae_model.zip")
    dsiae.save(save_path)

    assert os.path.exists(save_path)

    # 5. Load the model
    dsiae_loaded = DSIAE.load(save_path)

    assert dsiae_loaded.fitted is True
    assert hasattr(dsiae_loaded, 'encoder')

    # 6. Compare structure and weights
    # Check encoder weights
    for w_orig, w_load in zip(dsiae.encoder.encoder.get_weights(), dsiae_loaded.encoder.encoder.get_weights()):
        np.testing.assert_allclose(w_orig, w_load, rtol=1e-5, atol=1e-5, err_msg="Encoder weights do not match")

    # Check decoder weights
    for w_orig, w_load in zip(dsiae.encoder.decoder.get_weights(), dsiae_loaded.encoder.decoder.get_weights()):
        np.testing.assert_allclose(w_orig, w_load, rtol=1e-5, atol=1e-5, err_msg="Decoder weights do not match")

    # 7. Compare predictions
    preds_loaded = dsiae_loaded.predict(new_pvals_df)

    if isinstance(preds_original, (pd.Series, pd.DataFrame)):
        pd.testing.assert_frame_equal(pd.DataFrame(preds_original), pd.DataFrame(preds_loaded), check_dtype=False)
    else:
        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5, atol=1e-5)

    print("Save/Load test passed successfully!")


def test_dsi_rowwise(tmp_path):
    rowwise_groups = {
        "g1": ["obs0", "obs1", "obs2"],
        "g2": ["obs3", "obs4", "obs5"]
    }
    dsi_synth(tmp_path, rowwise_groups=rowwise_groups)
    return

def test_dsi_rowwise_mixed(tmp_path):
    rowwise_groups = {
        "g1": ["obs0", "obs1", "obs2"],
        "g2": ["obs3", "obs4", "obs5"]
    }
    transforms = [
        {"type": "log10", "columns": ["obs0", "obs3"]},
        {"type": "normal_score", }
    ]
    dsi_synth(tmp_path, rowwise_groups=rowwise_groups, transforms=transforms)
    return



def test_gpr_basic(tmp_path):
    import pyemu
    from pyemu.emulators import GPR
    
    # 1. Create Data
    # Simple y = 2*x + 1 relationship
    # Training data: x=0..10
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * x + 1.0
    # Add small noise (very small so interpolation is almost exact)
    # y += pyemu.en.rng.normal(0, 0.001, 20) 
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    # 2. Init and Fit
    gpr = GPR(data=df, input_names=['x'], output_names=['y'], verbose=False)
    gpr.fit()
    
    # 3. Predict
    # Predict on training data
    pred = gpr.predict(df[['x']])
    # assert close
    diff = np.abs(pred['y'].values - y)
    assert np.max(diff) < 0.1, f"Prediction error too high"

    # 4. Prepare PEST++ (file-based)
    t_d = str(tmp_path / "gpr_basic_template")
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    
    # Create a dummy PST to satisfy GPR requirement
    pst = pyemu.Pst("dummy.pst", load=False)
    # Add parameter 'x'
    # Manually constructing parameter_data (minimal columns)
    pst.parameter_data = pd.DataFrame(
        {'parnme':['x','x2'], 'parval1':[5.0,6], 'parlbnd':[0.0,0.0], 'parubnd':[10.0,10.0], 
         'pargp':['pargp','junkus'], 'scale':[1.0,2.0], 'offset':[0.0,0.0], 'partrans':['none','none']}, 
        index=['x','x2']
    )
    # Add observation 'y'
    pst.observation_data = pd.DataFrame(
        {'obsnme':['y'], 'obsval':[11.0], 'weight':[1.0], 'obgnme':['obgnme']}, 
        index=['y']
    )

    # Add some prior information
    pst.prior_information = pd.DataFrame({"pilbl": None, "obgnme": None}, index=[])
    pst.add_pi_equation(['x','x2'], 
                    pilbl="obj_well",  
                    obs_group="less_than_gigantor",
                    rhs=1e13) 
    
    # prepare_pestpp
    pst_gen = gpr.prepare_pestpp(t_d, pst=pst, use_runstor=False)
    
    # 5. Check generated files
    assert os.path.exists(os.path.join(t_d, "forward_run.py"))
    assert pst_gen.prior_information is not None
    
    # 6. Verify forward run script content
    with open(os.path.join(t_d, "forward_run.py"), 'r') as f:
        content = f.read()
    
    # print(content)
    assert "gpr_file_forward_run" in content
    assert "gpr_runstore_forward_run" not in content.split('if __name__')[1]
    
    # 7. Check if forward run works (it is run inside prepare_pestpp via 'subprocess' or 'run')
    # If prepare_pestpp didn't raise, it ran successfully.
    
    # Validate result of the forward run (which should have created emulator_output.csv)
    out_file = os.path.join(t_d, "emulator_output.csv")
    assert os.path.exists(out_file)
    res_df = pd.read_csv(out_file)
    # check columns
    assert 'y' in res_df.columns or 'y' in res_df.iloc[:,0].values


def test_gpr_runstor(tmp_path):
    import pyemu
    from pyemu.emulators import GPR
    
    # 1. Create Data
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * x + 1.0 
    df = pd.DataFrame({'x': x, 'y': y})
    
    # 2. Init
    gpr = GPR(data=df, input_names=['x'], output_names=['y'], verbose=False)
    gpr.fit()
    
    # 3. Pst
    pst = pyemu.Pst("dummy.pst", load=False)
    pst.parameter_data = pd.DataFrame(
        {'parnme':['x'], 'parval1':[5.0], 'parlbnd':[0.0], 'parubnd':[10.0], 
         'pargp':['pargp'], 'scale':[1.0], 'offset':[0.0], 'partrans':['none']}, 
        index=['x']
    )
    pst.observation_data = pd.DataFrame(
        {'obsnme':['y'], 'obsval':[11.0], 'weight':[1.0], 'obgnme':['obgnme']}, 
        index=['y']
    )
    
    # 4. Prepare PEST++ (RunStor)
    t_d = str(tmp_path / "gpr_runstor_template")
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    
    gpr.prepare_pestpp(t_d, pst=pst, use_runstor=True, pst_name="my_chk_pstname")
    
    # 5. Verify forward run script content
    with open(os.path.join(t_d, "forward_run.py"), 'r') as f:
        content = f.read()
    
    assert "gpr_runstore_forward_run" in content
    # It should be the one called
    assert "gpr_runstore_forward_run(emu_file=" in content
    assert "pst_name='my_chk_pstname'" in content


def test_row_wise_minmax_scaler():
    from pyemu.emulators.transformers import RowWiseMinMaxScaler
    
    # 1. Create synthetic data
    # Group 1: 3 columns, values approx 0-10
    # Group 2: 2 columns, values approx 100-200
    df = pd.DataFrame({
        'g1_1': [0, 5, 10],
        'g1_2': [2, 7, 12], # slightly shifted
        'g1_3': [1, 6, 11],
        'g2_1': [100, 150, 200],
        'g2_2': [110, 160, 210]
    })
    
    groups = {
        'g1': ['g1_1', 'g1_2', 'g1_3'],
        'g2': ['g2_1', 'g2_2']
    }
    
    # 2. Fit scaler (feature_range -1 to 1)
    scaler = RowWiseMinMaxScaler(feature_range=(-1, 1), groups=groups)
    scaler.fit(df)
    
    # Check if row params were correctly identified
    # Row 0: g1 min=0, max=2 -> range=2. g2 min=100, max=110 -> range=10
    row0_min_g1 = scaler.row_params['g1'][0][0]
    row0_max_g1 = scaler.row_params['g1'][1][0]
    assert row0_min_g1 == 0
    assert row0_max_g1 == 2
    
    # 3. Transform
    transformed = scaler.transform(df)
    
    # Check limits
    assert transformed.min().min() >= -1.0 - 1e-6
    assert transformed.max().max() <= 1.0 + 1e-6
    
    # Verify specific value
    # Row 0, g1_1 (val 0). min=0, max=2. Normalized=(0-0)/2=0. Scaled = 0*2 + (-1) = -1.
    assert np.abs(transformed.iloc[0]['g1_1'] - (-1.0)) < 1e-6
    # Row 0, g1_2 (val 2). min=0, max=2. Normalized=(2-0)/2=1. Scaled = 1*2 + (-1) = 1.
    assert np.abs(transformed.iloc[0]['g1_2'] - 1.0) < 1e-6
    
    # 4. Inverse Transform
    inversed = scaler.inverse_transform(transformed)
    
    # Check roundtrip
    assert np.allclose(df.values, inversed.values)


def test_lpfa_synth(tmp_path):
    from pyemu.emulators import LPFA
    import numpy as np
    import pandas as pd
    
    # 1. Generate synth data
    # 50 samples
    # Input: sin wave + noise
    # Output: cos wave (forecast)
    t = np.linspace(0, 10, 50)
    data = []
    n_real = 30
    rng = np.random.RandomState(42)
    for i in range(n_real):
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.8, 1.2)
        # Inputs (history)
        hist = amp * np.sin(t[:10] + phase)
        # Outputs (forecast)
        fore = amp * np.cos(t[10:] + phase)
        row = np.concatenate([hist, fore])
        data.append(row)
        
    cols = [f"h_{i}" for i in range(10)] + [f"f_{i}" for i in range(40)]
    df = pd.DataFrame(data, columns=cols)
    
    # Use a single group for time series to allow scaling forecast based on history
    all_cols = cols
    history_cols = [f"h_{i}" for i in range(10)]
    forecast_cols = [f"f_{i}" for i in range(40)]
    
    groups = {
        'timeseries': all_cols
    }
    fit_groups = {
        'timeseries': history_cols
    }
    
    input_names = history_cols
    output_names = forecast_cols

    transforms = [
        {"type": "standard_scaler", "columns": all_cols}
    ]
    
    # 2. Init LPFA
    lpfa = LPFA(
        data=df,
        input_names=input_names,
        output_names=output_names,
        groups=groups,
        fit_groups=fit_groups,
        transforms=transforms,
        verbose=False
    )
    
    # 3. Create Model
    lpfa.create_model() # defaults
    
    # 4. Fit
    lpfa.fit(epochs=10) # fast fit
    
    # 5. Predict
    # Predict on training data - needs full structure for LPFA
    pred_input = df[input_names].copy()
    for col in output_names:
        pred_input[col] = np.nan
        
    preds = lpfa.predict(pred_input)
    
    # Check shape
    # preds includes inputs and outputs? logic in predict returns 'predictions' which is copy of input
    # AND assigns output cols.
    assert preds.shape == (n_real, 50) # 10 input + 40 output
    assert not preds[output_names].isnull().all().all() # Should be filled

    
    # 6. Basic noise model check
    lpfa.add_noise_model()
    # Should not crash


# ===========================================================================
# Unit tests — no PEST++ binaries required
# ===========================================================================


class TestDSIFitPredict:
    """Unit-level tests for DSI core logic (no PEST++ binaries)."""

    def test_fit_predict_basic(self):
        """DSI fit+predict with no transforms returns correct shape/type."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        assert dsi.fitted
        assert dsi.pmat is not None
        assert dsi.ovals is not None

        # Predict with zero pvals — should recover the mean (ovals)
        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert isinstance(result, pd.Series)
        assert len(result) == data.shape[1]
        np.testing.assert_allclose(result.values, dsi.ovals.values, atol=1e-10)

    def test_predict_multi_realization(self):
        """DSI.predict with DataFrame input returns DataFrame with correct shape."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        n_reals = 5
        pvals_df = pd.DataFrame(
            np.random.normal(size=(n_reals, dsi.pmat.shape[1])),
            columns=[f"p_{i}" for i in range(dsi.pmat.shape[1])],
        )
        result = dsi.predict(pvals_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (n_reals, data.shape[1])

    def test_predict_series_vs_array(self):
        """Predictions from Series, 1-D array, and 2-D array are identical."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        n_pars = dsi.pmat.shape[1]
        vals = np.random.normal(size=n_pars)

        result_array_1d = dsi.predict(vals)
        result_array_2d = dsi.predict(vals.reshape(1, -1))
        result_series = dsi.predict(pd.Series(vals, index=[f"p_{i}" for i in range(n_pars)]))

        np.testing.assert_allclose(result_array_1d.values, result_array_2d.values.flatten(), atol=1e-12)
        np.testing.assert_allclose(result_array_1d.values, result_series.values, atol=1e-12)

    def test_energy_truncation(self):
        """SVD truncation at energy_threshold < 1.0 reduces dimensionality."""
        data, obsdata = _synth_data(n_real=100, n_obs=10)
        dsi_full = DSI(data=data, pst=obsdata, energy_threshold=1.0, verbose=False)
        dsi_full.fit()

        dsi_trunc = DSI(data=data, pst=obsdata, energy_threshold=0.8, verbose=False)
        dsi_trunc.fit()

        assert dsi_trunc.pmat.shape[1] <= dsi_full.pmat.shape[1]
        s = dsi_trunc.s
        energy = np.sum(s ** 2) / np.sum(dsi_full.s ** 2)
        assert energy >= 0.8

    def test_predict_rowwise_inverse(self):
        """Row-wise scaling + inverse returns values in physical space."""
        data, obsdata = _synth_data()
        rowwise_groups = {
            "g1": [f"obs{i}" for i in range(5)],
            "g2": [f"obs{i}" for i in range(5, 10)],
        }
        dsi = DSI(data=data, pst=obsdata, rowwise_groups=rowwise_groups, verbose=False)
        dsi.fit()

        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert isinstance(result, pd.Series)
        assert len(result) == data.shape[1]
        assert np.all(np.isfinite(result.values))

    def test_save_load(self, tmp_path):
        """Round-trip pickle preserves predictions."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        pvals = np.random.normal(size=dsi.pmat.shape[1])
        pred_before = dsi.predict(pvals)

        path = str(tmp_path / "dsi_test.pkl")
        dsi.save(path)
        dsi_loaded = DSI.load(path)

        pred_after = dsi_loaded.predict(pvals)
        np.testing.assert_allclose(pred_before.values, pred_after.values, atol=1e-12)

    def test_predict_before_fit_raises(self):
        """Predicting before fit raises ValueError."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        with pytest.raises(ValueError, match="fitted"):
            dsi.predict(np.zeros(5))

    def test_predict_dimension_mismatch_raises(self):
        """Wrong pvals dimension raises ValueError."""
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        wrong_size = dsi.pmat.shape[1] + 3
        with pytest.raises(ValueError, match="parameters"):
            dsi.predict(np.zeros(wrong_size))

    def test_fit_with_transforms(self):
        """DSI with normal_score transform fits and round-trips."""
        data, obsdata = _synth_data()
        transforms = [{"type": "normal_score"}]
        dsi = DSI(data=data, pst=obsdata, transforms=transforms, verbose=False)
        dsi.fit()

        assert dsi.fitted
        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert len(result) == data.shape[1]
        assert np.all(np.isfinite(result.values))

    def test_fit_with_mixed_transforms(self):
        """DSI with log10 + normal_score fits and round-trips."""
        data, _ = _synth_data()
        data["obs0"] = np.abs(data["obs0"]) + 1.0
        data["obs1"] = np.abs(data["obs1"]) + 1.0
        obsdata = pd.DataFrame(
            {"obsnme": data.columns, "obsval": data.mean().values, "weight": 1.0, "obgnme": "obgnme"},
            index=data.columns,
        )

        transforms = [
            {"type": "log10", "columns": ["obs0", "obs1"]},
            {"type": "normal_score"},
        ]
        dsi = DSI(data=data, pst=obsdata, transforms=transforms, verbose=False)
        dsi.fit()

        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert len(result) == data.shape[1]
        assert np.all(np.isfinite(result.values))


class TestGPRFitPredict:
    """Unit-level tests for GPR core logic (no PEST++ binaries)."""

    @staticmethod
    def _simple_gpr_data():
        """y = 2*x + 1, perfect linear."""
        x = np.linspace(0.0, 10.0, 20)
        y = 2.0 * x + 1.0
        df = pd.DataFrame({"x": x, "y": y})
        return df

    def test_fit_predict_basic(self):
        """GPR fits and predicts a simple linear relationship."""
        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        assert gpr.fitted
        pred = gpr.predict(df[["x"]])
        diff = np.abs(pred["y"].values - df["y"].values)
        assert np.max(diff) < 0.1

    def test_predict_return_std(self):
        """return_std=True returns tuple of (predictions, std)."""
        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        result = gpr.predict(df[["x"]], return_std=True)
        assert isinstance(result, tuple)
        preds, stds = result
        assert isinstance(preds, pd.DataFrame)
        assert isinstance(stds, pd.DataFrame)
        assert stds.shape == preds.shape
        assert (stds.values >= 0).all()

    def test_custom_kernel(self):
        """GPR with explicit RBF kernel fits successfully."""
        from sklearn.gaussian_process.kernels import RBF

        df = self._simple_gpr_data()
        gpr = GPR(
            data=df,
            input_names=["x"],
            output_names=["y"],
            kernel=RBF(length_scale=1.0),
            verbose=False,
        )
        gpr.fit()
        assert gpr.fitted
        pred = gpr.predict(df[["x"]])
        assert pred.shape == (20, 1)

    def test_multi_output(self):
        """GPR with multiple outputs creates per-output models."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        df = pd.DataFrame({"x": x, "y1": 2 * x + 1, "y2": -x + 5})
        gpr = GPR(data=df, input_names=["x"], output_names=["y1", "y2"], verbose=False)
        gpr.fit()

        assert "y1" in gpr.gpr_models
        assert "y2" in gpr.gpr_models

        pred = gpr.predict(df[["x"]])
        assert pred.shape == (30, 2)
        assert np.max(np.abs(pred["y1"].values - df["y1"].values)) < 0.2
        assert np.max(np.abs(pred["y2"].values - df["y2"].values)) < 0.2

    def test_predict_before_fit_raises(self):
        """Predicting before fit raises ValueError."""
        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        with pytest.raises(ValueError, match="fitted"):
            gpr.predict(df[["x"]])

    def test_no_transforms(self):
        """GPR with transforms=None works correctly."""
        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], transforms=None, verbose=False)
        gpr.fit()
        pred = gpr.predict(df[["x"]])
        assert np.max(np.abs(pred["y"].values - df["y"].values)) < 0.1

    def test_save_load(self, tmp_path):
        """GPR round-trip pickle preserves predictions."""
        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        pred_before = gpr.predict(df[["x"]])

        path = str(tmp_path / "gpr_test.pkl")
        gpr.save(path)
        gpr_loaded = GPR.load(path)

        pred_after = gpr_loaded.predict(df[["x"]])
        np.testing.assert_allclose(
            pred_before["y"].values, pred_after["y"].values, atol=1e-10
        )


# ===========================================================================
# Transformer unit tests
# ===========================================================================


class TestNormalScoreTransformer:
    """Tests for NormalScoreTransformer."""

    def test_fit_transform_roundtrip(self):
        """Transform -> inverse_transform recovers original values."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.exponential(2, 100), "b": np.random.normal(5, 2, 100)})

        nst = NormalScoreTransformer(columns=["a", "b"])
        transformed = nst.fit_transform(df)

        for col in ["a", "b"]:
            vals = transformed[col].values
            assert np.abs(np.mean(vals)) < 0.5

        inversed = nst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=0.1)
        np.testing.assert_allclose(inversed["b"].values, df["b"].values, atol=0.1)

    def test_extrapolation(self):
        """Quadratic extrapolation handles values outside training range."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        train = pd.DataFrame({"a": np.sort(np.random.normal(0, 1, 50))})
        nst = NormalScoreTransformer(columns=["a"], quadratic_extrapolation=True)
        nst.fit(train)

        test = pd.DataFrame({"a": [train["a"].min() - 2.0, train["a"].max() + 2.0]})
        transformed = nst.transform(test)
        params = nst.column_parameters["a"]
        min_z, max_z = params["z_scores"].min(), params["z_scores"].max()
        assert transformed["a"].iloc[0] < min_z
        assert transformed["a"].iloc[1] > max_z

        inversed = nst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, test["a"].values, atol=0.5)

    def test_no_extrapolation_clamps(self):
        """Without extrapolation, out-of-range values are clamped."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        train = pd.DataFrame({"a": np.sort(np.random.normal(0, 1, 50))})
        nst = NormalScoreTransformer(columns=["a"], quadratic_extrapolation=False)
        nst.fit(train)

        test = pd.DataFrame({"a": [train["a"].min() - 5.0, train["a"].max() + 5.0]})
        transformed = nst.transform(test)
        params = nst.column_parameters["a"]
        min_z, max_z = params["z_scores"].min(), params["z_scores"].max()
        assert np.isclose(transformed["a"].iloc[0], min_z)
        assert np.isclose(transformed["a"].iloc[1], max_z)

    def test_selective_columns(self):
        """Only specified columns are transformed."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(0, 1, 50), "b": np.random.normal(10, 1, 50)})
        nst = NormalScoreTransformer(columns=["a"])
        transformed = nst.fit_transform(df)
        np.testing.assert_array_equal(transformed["b"].values, df["b"].values)


class TestStandardScalerTransformer:
    """Tests for StandardScalerTransformer."""

    def test_fit_transform_stats(self):
        """After transform, mean ~= 0 and std ~= 1."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 200), "b": np.random.normal(-5, 0.5, 200)})
        sst = StandardScalerTransformer(columns=["a", "b"])
        transformed = sst.fit_transform(df)

        for col in ["a", "b"]:
            assert np.abs(transformed[col].mean()) < 0.05
            assert np.abs(transformed[col].std() - 1.0) < 0.1

    def test_roundtrip(self):
        """Transform -> inverse recovers original."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 100), "b": np.random.normal(-5, 0.5, 100)})
        sst = StandardScalerTransformer(columns=["a", "b"])
        transformed = sst.fit_transform(df)
        inversed = sst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-5)

    def test_selective_columns(self):
        """Only specified columns are transformed."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 50), "b": [42.0] * 50})
        sst = StandardScalerTransformer(columns=["a"])
        transformed = sst.fit_transform(df)
        np.testing.assert_array_equal(transformed["b"].values, df["b"].values)


class TestMinMaxScalerTransformer:
    """Tests for MinMaxScaler."""

    def test_fit_transform_range(self):
        """Scaled values lie within feature_range."""
        from pyemu.emulators.transformers import MinMaxScaler

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.uniform(0, 100, 50), "b": np.random.uniform(-10, 10, 50)})
        scaler = MinMaxScaler(feature_range=(0, 1), columns=["a", "b"])
        transformed = scaler.fit_transform(df)
        assert transformed["a"].min() >= -1e-6
        assert transformed["a"].max() <= 1.0 + 1e-6
        assert transformed["b"].min() >= -1e-6
        assert transformed["b"].max() <= 1.0 + 1e-6

    def test_roundtrip(self):
        """Transform -> inverse recovers original."""
        from pyemu.emulators.transformers import MinMaxScaler

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.uniform(0, 100, 50), "b": np.random.uniform(-10, 10, 50)})
        scaler = MinMaxScaler(feature_range=(-1, 1), columns=["a", "b"])
        transformed = scaler.fit_transform(df)
        inversed = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-5)

    def test_constant_column_skip(self):
        """Constant columns are skipped when skip_constant=True."""
        from pyemu.emulators.transformers import MinMaxScaler

        df = pd.DataFrame({"a": [5.0] * 10, "b": np.arange(10, dtype=float)})
        scaler = MinMaxScaler(feature_range=(0, 1), skip_constant=True)
        transformed = scaler.fit_transform(df)
        np.testing.assert_array_equal(transformed["a"].values, df["a"].values)
        assert transformed["b"].min() >= -1e-6
        assert transformed["b"].max() <= 1.0 + 1e-6

    def test_near_constant_column(self):
        """Near-constant columns (range < 1e-10) are handled gracefully."""
        from pyemu.emulators.transformers import MinMaxScaler

        df = pd.DataFrame({"a": [1.0, 1.0 + 1e-15, 1.0 - 1e-15]})
        scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = scaler.fit_transform(df)
        assert np.all(np.isfinite(transformed.values))


class TestGenericTransformer:
    """Tests for GenericTransformer with sklearn wrappers."""

    def test_power_transformer_roundtrip(self):
        """GenericTransformer wrapping PowerTransformer round-trips."""
        from sklearn.preprocessing import PowerTransformer
        from pyemu.emulators.transformers import GenericTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 100)), "b": np.abs(np.random.normal(3, 1, 100))})
        gt = GenericTransformer(PowerTransformer, method="yeo-johnson")
        transformed = gt.fit_transform(df)
        inversed = gt.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-4)

    def test_missing_inverse_raises(self):
        """Transformer without inverse_transform raises on init."""
        from pyemu.emulators.transformers import GenericTransformer

        class BadTransformer:
            def fit(self, X):
                return self
            def transform(self, X):
                return X

        with pytest.raises(ValueError, match="inverse_transform"):
            GenericTransformer(BadTransformer)


class TestTransformerPipeline:
    """Tests for TransformerPipeline ordering via AutobotsAssemble."""

    def test_chained_inverse_reverses_order(self):
        """Chained log10 -> normal_score via AutobotsAssemble inverse round-trips."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 80)) + 1})

        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])
        ab.apply("normal_score", columns=["a"])

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=0.5)

    def test_multi_column_pipeline(self):
        """Pipeline with different transforms on different columns."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.abs(np.random.normal(5, 2, 50)) + 1,
            "b": np.random.normal(0, 1, 50),
        })

        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])
        ab.apply("standard_scaler", columns=["b"])

        np.testing.assert_allclose(ab.df["a"].values, np.log10(df["a"].values), atol=1e-10)
        assert np.abs(ab.df["b"].mean()) < 0.1

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)
        np.testing.assert_allclose(inversed["b"].values, df["b"].values, atol=1e-5)


class TestAutobotsAssemble:
    """Tests for AutobotsAssemble high-level API."""

    def test_apply_and_inverse(self):
        """apply() builds pipeline; inverse() reverses it."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 60)) + 1, "b": np.random.normal(0, 1, 60)})
        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])

        np.testing.assert_allclose(ab.df["a"].values, np.log10(df["a"].values), atol=1e-10)
        np.testing.assert_allclose(ab.df["b"].values, df["b"].values, atol=1e-10)

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)

    def test_transform_external(self):
        """transform() applies fitted pipeline to new data."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        train = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        ab = AutobotsAssemble(train.copy())
        ab.apply("normal_score", columns=["a"])

        test = pd.DataFrame({"a": np.random.normal(0, 1, 10)})
        transformed = ab.transform(test)
        assert transformed.shape == test.shape
        assert not np.allclose(transformed["a"].values, test["a"].values)

    def test_inverse_on_external_df(self):
        """inverse_on_external_df applies inverse to data not seen during fit."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 60)) + 1})
        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])

        external = ab.transform(df)
        inversed = ab.inverse_on_external_df(external)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)


class TestRowWiseMinMaxScaler:
    """Additional RowWiseMinMaxScaler tests (fit_groups, edge cases)."""

    def test_fit_groups_subset(self):
        """fit_groups controls which columns determine row-wise min/max."""
        from pyemu.emulators.transformers import RowWiseMinMaxScaler

        df = pd.DataFrame({
            "history_1": [0.0, 5.0],
            "history_2": [10.0, 15.0],
            "forecast_1": [100.0, 200.0],
        })
        groups = {"ts": ["history_1", "history_2", "forecast_1"]}
        fit_groups = {"ts": ["history_1", "history_2"]}

        scaler = RowWiseMinMaxScaler(feature_range=(-1, 1), groups=groups, fit_groups=fit_groups)
        scaler.fit(df)

        row_min, row_max = scaler.row_params["ts"]
        assert row_min.iloc[0] == 0.0
        assert row_max.iloc[0] == 10.0

    def test_zero_variance_row(self):
        """Row where all values are identical does not cause division by zero."""
        from pyemu.emulators.transformers import RowWiseMinMaxScaler

        df = pd.DataFrame({
            "a": [5.0, 1.0],
            "b": [5.0, 2.0],
        })
        groups = {"g": ["a", "b"]}
        scaler = RowWiseMinMaxScaler(feature_range=(-1, 1), groups=groups)
        transformed = scaler.fit_transform(df)
        assert np.all(np.isfinite(transformed.values))

        inversed = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.iloc[1].values, df.iloc[1].values, atol=1e-10)


class TestLog10Transformer:
    """Tests for Log10Transformer."""

    def test_positive_values_no_shift(self):
        """Positive values produce no shift."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [1.0, 10.0, 100.0]})
        t = Log10Transformer(columns=["a"])
        result = t.fit_transform(df)
        np.testing.assert_allclose(result["a"].values, [0.0, 1.0, 2.0], atol=1e-10)
        assert t.shifts["a"] == 0

    def test_negative_shift_roundtrip(self):
        """Columns with negatives get shifted and inverse round-trips."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [-5.0, 0.0, 10.0]})
        t = Log10Transformer(columns=["a"])
        transformed = t.fit_transform(df)
        inversed = t.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)

    def test_untouched_columns(self):
        """Columns not in 'columns' list remain unchanged."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [1.0, 10.0], "b": [42.0, 99.0]})
        t = Log10Transformer(columns=["a"])
        result = t.fit_transform(df)
        np.testing.assert_array_equal(result["b"].values, df["b"].values)


# ===========================================================================
# Base class / file generation tests
# ===========================================================================


class TestBaseWriteTemplateFile:
    """Tests for Emulator._write_template_file."""

    def test_template_file_format(self, tmp_path):
        """Generated .tpl file has ptf header and correct parameter markers."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [0.0, 1.0]},
            index=["p_0", "p_1"],
        )
        path = str(tmp_path / "test.tpl")
        emu._write_template_file(par_df, path)

        with open(path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "ptf ~"
        assert lines[1].strip() == "parnme,parval1"
        for i, pname in enumerate(["p_0", "p_1"]):
            assert f"~   {pname}   ~" in lines[i + 2]


class TestBaseWriteInstructionFile:
    """Tests for Emulator._write_instruction_file."""

    def test_instruction_file_format(self, tmp_path):
        """Generated .ins file has pif header and valid instruction grammar."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        obs_df = pd.DataFrame(
            {"obsnme": ["obs0", "obs1"], "obsval": [1.0, 2.0], "weight": [1.0, 1.0], "obgnme": ["g", "g"]},
            index=["obs0", "obs1"],
        )
        path = str(tmp_path / "test.ins")
        emu._write_instruction_file(obs_df, path)

        with open(path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "pif ~"
        assert lines[1].strip() == "l1"
        for i, oname in enumerate(["obs0", "obs1"]):
            assert f"!{oname}!" in lines[i + 2]


class TestBaseWriteInputFile:
    """Tests for Emulator._write_input_file."""

    def test_input_file_readable(self, tmp_path):
        """Written input file is readable by pd.read_csv and values match."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [3.14, -2.71]},
            index=["p_0", "p_1"],
        )
        path = str(tmp_path / "input.csv")
        emu._write_input_file(par_df, path)

        result = pd.read_csv(path, index_col=0)
        np.testing.assert_allclose(result["parval1"].values, par_df["parval1"].values, atol=1e-10)


class TestBaseWriteOutputFile:
    """Tests for Emulator._write_output_file."""

    def test_output_file_format(self, tmp_path):
        """Written output file has header and correct values."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        obs_df = pd.DataFrame(
            {"obsnme": ["obs0", "obs1"], "obsval": [1.5, -0.5], "weight": [1.0, 0.0], "obgnme": ["g", "g"]},
            index=["obs0", "obs1"],
        )
        path = str(tmp_path / "output.csv")
        emu._write_output_file(obs_df, path)

        result = pd.read_csv(path)
        assert list(result.columns) == ["obsnme", "simval"]
        assert list(result["obsnme"]) == ["obs0", "obs1"]
        np.testing.assert_allclose(result["simval"].values, [1.5, -0.5], atol=1e-10)


class TestBaseUpdateParameterData:
    """Tests for Emulator._update_parameter_data."""

    def test_merge_values(self):
        """Emulator par_df values are merged into pst par_df."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        pst_par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [0.0, 0.0], "parlbnd": [0.0, 0.0],
             "parubnd": [0.0, 0.0], "pargp": ["x", "x"]},
            index=["p_0", "p_1"],
        )
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [1.0, 2.0], "parlbnd": [-10.0, -10.0],
             "parubnd": [10.0, 10.0], "pargp": ["dsi", "dsi"]},
            index=["p_0", "p_1"],
        )
        result = emu._update_parameter_data(pst_par_df, par_df)
        assert result.loc["p_0", "parval1"] == 1.0
        assert result.loc["p_1", "pargp"] == "dsi"
        assert result.loc["p_0", "parlbnd"] == -10.0


class TestBaseUpdateObservationData:
    """Tests for Emulator._update_observation_data."""

    def test_merge_values(self):
        """Emulator obs_df values are merged into pst obs_df."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        pst_obs_df = pd.DataFrame(
            {"obsnme": ["o0", "o1"], "obsval": [0.0, 0.0], "weight": [0.0, 0.0], "obgnme": ["x", "x"]},
            index=["o0", "o1"],
        )
        obs_df = pd.DataFrame(
            {"obsnme": ["o0", "o1"], "obsval": [5.0, 6.0], "weight": [1.0, 2.0], "obgnme": ["grp", "grp"]},
            index=["o0", "o1"],
        )
        result = emu._update_observation_data(pst_obs_df, obs_df)
        assert result.loc["o0", "obsval"] == 5.0
        assert result.loc["o1", "weight"] == 2.0
        assert result.loc["o0", "obgnme"] == "grp"


class TestBaseValidateTransforms:
    """Tests for Emulator._validate_transforms error handling."""

    def test_not_list_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="list"):
            emu._validate_transforms("not a list")

    def test_not_dict_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="dict"):
            emu._validate_transforms(["not a dict"])

    def test_missing_type_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="type"):
            emu._validate_transforms([{"columns": ["a"]}])

    def test_columns_not_list_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="columns"):
            emu._validate_transforms([{"type": "log10", "columns": "not_a_list"}])

    def test_valid_transforms_pass(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        emu._validate_transforms([
            {"type": "log10", "columns": ["a", "b"]},
            {"type": "normal_score"},
        ])


# ---------------------------------------------------------------------------
# helpers.py emulator-function tests
# ---------------------------------------------------------------------------

class TestDsiForwardRun:
    """Tests for pyemu.utils.helpers.dsi_forward_run"""

    def _fitted_dsi(self):
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata)
        dsi.fit()
        return dsi

    def test_basic_series_input(self):
        from pyemu.utils.helpers import dsi_forward_run
        dsi = self._fitted_dsi()
        pvals = pd.Series(np.zeros(dsi.s.shape[0]),
                          index=[f"dsi_par_{i}" for i in range(dsi.s.shape[0])])
        result = dsi_forward_run(pvals, dsi)
        assert isinstance(result, (pd.Series, pd.DataFrame))
        assert result.shape[0] > 0

    def test_dataframe_input_extracts_parval1(self):
        from pyemu.utils.helpers import dsi_forward_run
        dsi = self._fitted_dsi()
        n = dsi.s.shape[0]
        pvals = pd.DataFrame({"parval1": np.zeros(n)},
                             index=[f"dsi_par_{i}" for i in range(n)])
        result = dsi_forward_run(pvals, dsi)
        assert result.shape[0] > 0

    def test_write_csv(self, tmp_path):
        from pyemu.utils.helpers import dsi_forward_run
        dsi = self._fitted_dsi()
        pvals = pd.Series(np.zeros(dsi.s.shape[0]),
                          index=[f"dsi_par_{i}" for i in range(dsi.s.shape[0])])
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = dsi_forward_run(pvals, dsi, write_csv=True)
            assert os.path.exists("dsi_sim_vals.csv")
        finally:
            os.chdir(orig_dir)

    def test_wrong_type_raises(self):
        from pyemu.utils.helpers import dsi_forward_run
        with pytest.raises(Exception, match="pyemu.emulators.DSI"):
            dsi_forward_run(pd.Series([1.0]), "not_a_dsi")


class TestDsiFileForwardRun:
    """Tests for pyemu.utils.helpers.dsi_file_forward_run"""

    def test_roundtrip(self, tmp_path):
        from pyemu.utils.helpers import dsi_file_forward_run
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata)
        dsi.fit()

        emu_file = str(tmp_path / "dsi.pickle")
        dsi.save(emu_file)

        # Build an input CSV with parval1 column matching expected DSI pars
        n = dsi.s.shape[0]
        inp = pd.DataFrame({"parval1": np.zeros(n)},
                           index=[f"dsi_par_{i}" for i in range(n)])
        input_file = str(tmp_path / "dsi_pars.csv")
        inp.to_csv(input_file)

        output_file = str(tmp_path / "dsi_sim_vals.csv")
        dsi_file_forward_run(emu_file, input_file, output_file)

        assert os.path.exists(output_file)
        out = pd.read_csv(output_file, index_col=0)
        assert out.shape[0] > 0

    def test_missing_input_raises(self, tmp_path):
        from pyemu.utils.helpers import dsi_file_forward_run
        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata)
        dsi.fit()
        emu_file = str(tmp_path / "dsi.pickle")
        dsi.save(emu_file)

        with pytest.raises(FileNotFoundError):
            dsi_file_forward_run(emu_file, str(tmp_path / "no_such.csv"),
                                 str(tmp_path / "out.csv"))


class TestGprFileForwardRun:
    """Tests for pyemu.utils.helpers.gpr_file_forward_run"""

    @staticmethod
    def _make_gpr():
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = np.sin(x)
        df = pd.DataFrame({"x": x, "y": y})
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()
        return gpr

    @pytest.fixture()
    def gpr_artefacts(self, tmp_path):
        """Create a fitted GPR, save it, and build an input CSV."""
        gpr = self._make_gpr()

        emu_file = str(tmp_path / "gpr_emulator.pkl")
        gpr.save(emu_file)

        # GPR input: one row with input columns; parval1 format
        inp = pd.DataFrame({"parval1": [1.0]}, index=["x"])
        input_file = str(tmp_path / "gpr_input.csv")
        inp.to_csv(input_file)

        output_file = str(tmp_path / "gpr_output.csv")
        return emu_file, input_file, output_file

    def test_roundtrip(self, gpr_artefacts):
        from pyemu.utils.helpers import gpr_file_forward_run
        emu_file, input_file, output_file = gpr_artefacts
        gpr_file_forward_run(emu_file, input_file, output_file)
        assert os.path.exists(output_file)
        out = pd.read_csv(output_file)
        assert out.shape[0] > 0

    def test_missing_input_raises(self, tmp_path):
        from pyemu.utils.helpers import gpr_file_forward_run
        gpr = self._make_gpr()
        emu_file = str(tmp_path / "gpr_emulator.pkl")
        gpr.save(emu_file)

        with pytest.raises(FileNotFoundError):
            gpr_file_forward_run(emu_file, str(tmp_path / "missing.csv"),
                                 str(tmp_path / "out.csv"))


class TestRunStorForwardRuns:
    """Tests for dsi_runstore_forward_run and gpr_runstore_forward_run
    using a programmatically-created .rns file."""

    @staticmethod
    def _create_rns(filename, par_names, obs_names, par_vals, obs_vals):
        """Create a minimal .rns binary file for one run."""
        import struct
        n_runs = par_vals.shape[0] if par_vals.ndim > 1 else 1
        if par_vals.ndim == 1:
            par_vals = par_vals.reshape(1, -1)
            obs_vals = obs_vals.reshape(1, -1)

        p_name_bytes = "\0".join(par_names).encode() + b"\0"
        o_name_bytes = "\0".join(obs_names).encode() + b"\0"

        info_txt_size = 1001
        # run_size = 1 (status) + info_txt_size + 8 (info_val) + 8*npar + 8*nobs + 1 (buf_status)
        run_size = 1 + info_txt_size + 8 + 8 * len(par_names) + 8 * len(obs_names) + 1

        header = np.array(
            [(n_runs, run_size, len(p_name_bytes), len(o_name_bytes))],
            dtype=np.dtype([("n_runs", np.int64), ("run_size", np.int64),
                            ("p_name_size", np.int64), ("o_name_size", np.int64)]),
        )
        with open(filename, "wb") as f:
            header.tofile(f)
            f.write(p_name_bytes)
            f.write(o_name_bytes)
            for i in range(n_runs):
                np.array([1], dtype=np.int8).tofile(f)  # run_status=completed
                f.write(struct.pack(f"{info_txt_size}s", b""))  # info_txt
                np.array([0.0], dtype=np.float64).tofile(f)  # info_val
                par_vals[i].astype(np.float64).tofile(f)
                obs_vals[i].astype(np.float64).tofile(f)
                np.array([0], dtype=np.int8).tofile(f)  # buf_status

    def test_runstor_update_shared_par_obs_names(self, tmp_path):
        """pest allows a parameter and an observation to share a name.  The
        file format, file_info() and get_data() are all order-based with the
        par block first, so update() must be too -- selecting by label returns
        both columns for each shared name, writing a double-width par block
        that shifts every run after the first and corrupts the file."""
        from pyemu.utils.helpers import RunStor

        par_names = ["o0", "o1", "o2"]              # also observation names
        obs_names = ["o0", "o1", "o2", "o3"]
        n_runs = 4
        pv = np.arange(n_runs * len(par_names), dtype=float).reshape(n_runs, -1)
        ov = np.arange(n_runs * len(obs_names), dtype=float).reshape(n_runs, -1) + 1000.

        f = os.path.join(str(tmp_path), "shared.rns")
        self._create_rns(f, par_names, obs_names, pv, ov)

        df = RunStor(f).get_data()
        assert len(df.columns) != len(set(df.columns)), "expected duplicate labels"
        # get_data is positional, so both blocks survive intact
        npar, nobs = len(par_names), len(obs_names)
        assert np.allclose(df.iloc[:, -(npar + nobs):-nobs].values, pv)
        assert np.allclose(df.iloc[:, -nobs:].values, ov)

        # writing back unchanged must be a no-op, not a corruption
        RunStor(f).update(df)

        df2 = RunStor(f).get_data()
        assert np.allclose(df2.iloc[:, -(npar + nobs):-nobs].values, pv), "par block"
        assert np.allclose(df2.iloc[:, -nobs:].values, ov), "obs block"

        # and an actual edit lands in the obs block only
        df2.iloc[:, -nobs:] = ov + 7.0
        RunStor(f).update(df2)
        df3 = RunStor(f).get_data()
        assert np.allclose(df3.iloc[:, -nobs:].values, ov + 7.0)
        assert np.allclose(df3.iloc[:, -(npar + nobs):-nobs].values, pv)

    def test_dsi_runstore_forward_run(self, tmp_path):
        from pyemu.utils.helpers import dsi_runstore_forward_run

        data, obsdata = _synth_data(n_real=50, n_obs=5)
        dsi = DSI(data=data, pst=obsdata)
        dsi.fit()

        ws = str(tmp_path)
        dsi.save(os.path.join(ws, "dsi.pickle"))

        n_latent = dsi.s.shape[0]
        par_names = [f"dsi_par_{i}" for i in range(n_latent)]
        obs_names = data.columns.tolist()
        n_runs = 3
        par_vals = np.random.normal(size=(n_runs, n_latent))
        obs_vals = np.zeros((n_runs, len(obs_names)))

        rns_file = os.path.join(ws, "dsi.rns")
        self._create_rns(rns_file, par_names, obs_names, par_vals, obs_vals)

        dsi_runstore_forward_run(ws=ws, pst_name="dsi")

        # Verify obs_vals were updated (no longer all zeros)
        from pyemu.utils.helpers import RunStor
        rs = RunStor(rns_file)
        df = rs.get_data()
        updated_obs = df.loc[:, obs_names].values
        assert not np.allclose(updated_obs, 0.0), "obs values should have been updated"

    def test_gpr_runstore_forward_run(self, tmp_path):
        from pyemu.utils.helpers import gpr_runstore_forward_run

        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = np.sin(x)
        df = pd.DataFrame({"x": x, "y": y})
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        ws = str(tmp_path)
        emu_file = "gpr_emulator.pkl"
        gpr.save(os.path.join(ws, emu_file))

        par_names = ["x"]
        obs_names = ["y"]
        n_runs = 3
        par_vals = np.random.uniform(0, 5, size=(n_runs, 1))
        obs_vals = np.zeros((n_runs, 1))

        rns_file = os.path.join(ws, "gpr.rns")
        self._create_rns(rns_file, par_names, obs_names, par_vals, obs_vals)

        gpr_runstore_forward_run(ws=ws, emu_file=emu_file, pst_name="gpr")

        from pyemu.utils.helpers import RunStor
        rs = RunStor(rns_file)
        rdf = rs.get_data()
        updated_obs = rdf.loc[:, obs_names].values
        assert not np.allclose(updated_obs, 0.0), "obs values should have been updated"


# ---------------------------------------------------------------------------
# PLS emulator tests (Phase 1)
# ---------------------------------------------------------------------------


def _synth_pls_data(n_real=80, n_par=15, n_obs=10, seed=42):
    """Linear-from-pars synthetic data for PLS sanity tests."""
    rng = np.random.RandomState(seed)
    pars = pd.DataFrame(
        rng.normal(size=(n_real, n_par)),
        columns=[f"par{i}" for i in range(n_par)],
    )
    W = rng.normal(size=(n_par, n_obs)) * 0.3
    obs = pars.values @ W + 0.05 * rng.normal(size=(n_real, n_obs))
    obs = pd.DataFrame(obs, columns=[f"obs{i}" for i in range(n_obs)])
    data = pd.concat([pars, obs], axis=1)
    return data, list(pars.columns), list(obs.columns)


def test_pls_basic():
    """PLS with explicit n_components fits and predicts the right shape."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols, n_components=3)
    pls.fit()
    assert pls.fitted
    assert pls.n_components == 3

    Y_hat = pls.predict(data.iloc[:5][par_cols])
    assert Y_hat.shape == (5, len(obs_cols))
    assert list(Y_hat.columns) == obs_cols

    # Single-row input returns a Series (matches DSI/DSIAE convention)
    y_one = pls.predict(data.iloc[0][par_cols])
    assert isinstance(y_one, pd.Series)
    assert len(y_one) == len(obs_cols)

    # encode returns latent scores of width n_components
    Z = pls.encode(data.iloc[:5][par_cols])
    assert Z.shape == (5, 3)


def test_pls_auto_components():
    """n_components=None triggers k-fold CV selection within valid bounds."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=None, cv_folds=3)
    pls.fit()
    assert pls.fitted
    max_k = min(len(data) - 1, len(par_cols), len(obs_cols))
    assert 1 <= pls.n_components <= max_k


def test_pls_with_transforms():
    """Transforms pipeline integrates with PLS the same way it does with DSIAE."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    transforms = [{"type": "normal_score"}]
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              transforms=transforms, n_components=3)
    pls.fit()
    Y_hat = pls.predict(data.iloc[:3][par_cols])
    assert Y_hat.shape == (3, len(obs_cols))


def test_pls_high_d_warning():
    """PLS warns when the input dim exceeds the high-d threshold and no reducer is given."""
    from pyemu.emulators import PLS
    from pyemu.emulators.pls import HIGH_D_WARN_THRESHOLD
    rng = np.random.RandomState(0)
    d = HIGH_D_WARN_THRESHOLD + 50
    n_real = 30
    pars = pd.DataFrame(
        rng.normal(size=(n_real, d)),
        columns=[f"par{i}" for i in range(d)],
    )
    obs = pd.DataFrame(
        rng.normal(size=(n_real, 3)),
        columns=[f"obs{i}" for i in range(3)],
    )
    data = pd.concat([pars, obs], axis=1)
    pls = PLS(data=data, input_names=list(pars.columns),
              output_names=list(obs.columns), n_components=2)
    with pytest.warns(UserWarning, match="input dimension"):
        pls.fit()


def test_pls_prepare_pestpp(tmp_path):
    """A fitted PLS round-trips through prepare_pestpp + pickle and predicts after reload."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data()
    obsdata = pd.DataFrame(
        {
            "obsnme": obs_cols,
            "obsval": data[obs_cols].mean().values,
            "weight": 1.0,
            "obgnme": "obgnme",
        },
        index=obs_cols,
    )

    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols, n_components=3)
    pls.fit()

    td = Path(tmp_path) / "template_pls"
    pst_obj = pls.prepare_pestpp(td, observation_data=obsdata)

    # Inherited prepare_pestpp pickles the emulator and lays down the I/O files
    # for the PEST++ forward-run script.
    assert (td / "emulator.pkl").exists()
    assert (td / "emulator_input.csv.tpl").exists()
    assert (td / "emulator_output.csv.ins").exists()
    assert (td / "emulator_output.csv").exists()
    assert (td / "forward_run.py").exists()

    # Reload and predict — verifies the pickle is self-contained.
    pls_loaded = PLS.load(td / "emulator.pkl")
    y_hat = pls_loaded.predict(data.iloc[:3][par_cols])
    assert y_hat.shape == (3, len(obs_cols))

    # The base class validated forward_run.py by running it once; the output
    # file should now contain the obsnme,simval table.
    out_df = pd.read_csv(td / "emulator_output.csv", index_col=0)
    assert "simval" in out_df.columns
    assert set(out_df.index) == set(obs_cols)

    # And the generated Pst object should have the right parameter and obs names.
    assert sorted(pst_obj.par_names) == sorted(par_cols)
    assert sorted(pst_obj.obs_names) == sorted(obs_cols)


def _pst_for_pls(par_names, obs_names, nonzero=None):
    """Build a shell Pst whose parameter_data/observation_data covers the
    supplied names. ``nonzero`` is a list of obs names that should get
    weight=1.0; everything else gets weight=0.0. Used by the PLS inference
    tests below."""
    import pyemu
    pst = pyemu.Pst.from_par_obs_names(par_names=list(par_names) or ["dummy_p"],
                                       obs_names=list(obs_names) or ["dummy_o"])
    if nonzero is not None:
        wts = pd.Series(0.0, index=pst.observation_data.index)
        for o in nonzero:
            wts.loc[o] = 1.0
        pst.observation_data["weight"] = wts.values
    return pst


def test_pls_infer_mixed_pars_and_obs():
    """When data has both pst pars and pst obs, infer pars->inputs, obs->outputs."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pst = _pst_for_pls(par_cols, obs_cols, nonzero=obs_cols)
    pls = PLS(pst=pst, data=data, n_components=2)
    assert sorted(pls.input_names) == sorted(par_cols)
    assert sorted(pls.output_names) == sorted(obs_cols)
    pls.fit()
    assert pls.fitted


def test_pls_infer_obs_only_by_weight():
    """When data has only pst obs, split by weight: nz->inputs, zw->outputs."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    # Treat the synthetic 'par*' cols as nonzero-weight obs (inputs) and
    # the 'obs*' cols as zero-weight obs (outputs). All columns are obs from
    # the pst's perspective.
    all_obs = par_cols + obs_cols
    pst = _pst_for_pls(par_names=["dummy_p"], obs_names=all_obs, nonzero=par_cols)
    pls = PLS(pst=pst, data=data, n_components=2)
    assert sorted(pls.input_names) == sorted(par_cols)
    assert sorted(pls.output_names) == sorted(obs_cols)
    pls.fit()
    assert pls.fitted


def test_pls_infer_obs_only_requires_both_weight_classes():
    """Obs-only data with all-nonzero or all-zero weights should fail to infer."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    all_obs = par_cols + obs_cols

    pst_all_nz = _pst_for_pls(par_names=["dummy_p"], obs_names=all_obs,
                              nonzero=all_obs)
    with pytest.raises(AssertionError, match="zero-weight"):
        PLS(pst=pst_all_nz, data=data, n_components=2)

    pst_all_zw = _pst_for_pls(par_names=["dummy_p"], obs_names=all_obs,
                              nonzero=[])
    with pytest.raises(AssertionError, match="nonzero-weight"):
        PLS(pst=pst_all_zw, data=data, n_components=2)


def test_pls_infer_requires_pst_when_names_omitted():
    """Omitting both names without a pst is a clear error."""
    from pyemu.emulators import PLS
    data, _, _ = _synth_pls_data()
    with pytest.raises(ValueError, match="no pst"):
        PLS(data=data, n_components=2)


def test_pls_infer_requires_overlap_with_pst():
    """A pst whose names don't match any data column is rejected."""
    from pyemu.emulators import PLS
    data, _, _ = _synth_pls_data()
    pst = _pst_for_pls(par_names=["other_p1"], obs_names=["other_o1"],
                       nonzero=["other_o1"])
    with pytest.raises(ValueError, match="no data columns matched"):
        PLS(pst=pst, data=data, n_components=2)


def test_pls_infer_outputs_from_inputs():
    """Passing only input_names should derive output_names from the data set diff."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, n_components=2)
    assert sorted(pls.input_names) == sorted(par_cols)
    assert sorted(pls.output_names) == sorted(obs_cols)
    pls.fit()
    assert pls.fitted


def test_pls_infer_inputs_from_outputs():
    """Passing only output_names should derive input_names from the data set diff."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, output_names=obs_cols, n_components=2)
    assert sorted(pls.input_names) == sorted(par_cols)
    assert sorted(pls.output_names) == sorted(obs_cols)
    pls.fit()
    assert pls.fitted


def test_pls_infer_one_side_with_extra_cols():
    """When data has unrelated extra cols and only output_names is passed, the
    extras are folded into the inferred input_names. The user can then pass
    both lists explicitly to opt out of the wide inference."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    wide = data.copy()
    wide["junk_a"] = 0.0
    wide["junk_b"] = 0.0
    pls = PLS(data=wide, output_names=obs_cols, n_components=2)
    assert "junk_a" in pls.input_names and "junk_b" in pls.input_names
    assert set(pls.input_names) == set(par_cols + ["junk_a", "junk_b"])


def test_pls_infer_one_side_empty_other_raises():
    """If the inferred side ends up empty, the explicit non-empty check still fires."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    # Pass every column as output — leaves zero inputs.
    all_cols = list(data.columns)
    with pytest.raises(ValueError, match="non-empty 'input_names'"):
        PLS(data=data, output_names=all_cols, n_components=2)


def test_pls_predict_ignores_extra_columns():
    """predict() should silently drop columns outside input_names."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=3)
    pls.fit()

    # DataFrame with the full par+obs columns plus a junk column — predict
    # should ignore the obs cols and 'junk' and use only par_cols.
    wide = data.iloc[:4].copy()
    wide["junk"] = 999.0
    Y_hat = pls.predict(wide)
    assert Y_hat.shape == (4, len(obs_cols))

    # Series with extra entries should also be tolerated.
    s = data.iloc[0].copy()
    s["extra_thing"] = 12.34
    y_one = pls.predict(s)
    assert isinstance(y_one, pd.Series)
    assert len(y_one) == len(obs_cols)


def test_pls_predict_missing_input_raises_clear_error():
    """Missing required input cols should raise a KeyError that names them."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=3)
    pls.fit()

    # Drop one required input column.
    short = data.iloc[:2][par_cols[1:]]
    with pytest.raises(KeyError, match="missing"):
        pls.predict(short)


def test_pls_cv_log_anchors_shape():
    """The coarse anchor grid is decade-spaced and always includes max_k."""
    from pyemu.emulators.pls import PLS
    # Small max_k — anchors should be subset of the decade pattern.
    a = PLS._log_anchors(8)
    assert a == [1, 2, 5, 8]                          # max_k appended
    a = PLS._log_anchors(50)
    assert a == [1, 2, 5, 10, 20, 50]
    a = PLS._log_anchors(150)
    assert a == [1, 2, 5, 10, 20, 50, 100, 150]
    a = PLS._log_anchors(1000)
    assert a == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    a = PLS._log_anchors(1)
    assert a == [1]


def test_pls_cv_coarse_evaluates_only_anchors():
    """The CV search should evaluate exactly the anchor set — no extras."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data(n_real=80, n_par=15, n_obs=10)
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=None, cv_folds=5)
    pls.fit()

    # 5-fold on 80 reals -> smallest train ~63, capped by min(15, 10) = 10.
    # Anchors in [1, 10] = [1, 2, 5, 10] -> exactly 4 evals.
    expected_anchors = PLS._log_anchors(10)
    assert sorted(pls._cv_scores.keys()) == expected_anchors
    assert pls.n_components == min(pls._cv_scores, key=lambda k: pls._cv_scores[k])


def test_pls_cv_coarse_big_max_k_skips_tail():
    """On a problem where max_k is large, the coarse pass should evaluate
    O(log max_k) anchors, not all max_k integers."""
    from pyemu.emulators import PLS
    rng = np.random.RandomState(0)
    n_real, n_par, n_obs = 50, 200, 60
    pars = pd.DataFrame(rng.normal(size=(n_real, n_par)),
                        columns=[f"par{i}" for i in range(n_par)])
    W = rng.normal(size=(n_par, n_obs)) * 0.1
    obs = pd.DataFrame(pars.values @ W + 0.5 * rng.normal(size=(n_real, n_obs)),
                       columns=[f"obs{i}" for i in range(n_obs)])
    data = pd.concat([pars, obs], axis=1)

    pls = PLS(data=data, input_names=list(pars.columns),
              output_names=list(obs.columns),
              n_components=None, cv_folds=5)
    pls.fit()

    # max_k = min(40, 200, 60) = 40. Anchors in [1, 40] = [1, 2, 5, 10, 20, 40].
    expected_anchors = PLS._log_anchors(40)
    assert sorted(pls._cv_scores.keys()) == expected_anchors
    # ~7x cheaper than brute force (6 anchors vs 40 k-values).
    assert len(pls._cv_scores) < 40


def test_pls_cv_coarse_finds_near_optimum():
    """On a problem with known latent rank, the coarse search should pick a k
    whose CV RMSE is within (loose) tolerance of the brute-force argmin's RMSE.
    The anchor grid is log-spaced so we don't constrain ``|pick - argmin|`` —
    only that the resulting model's CV RMSE is competitive."""
    from pyemu.emulators import PLS
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    rng = np.random.RandomState(0)
    n_real, n_par, n_obs = 60, 30, 20
    Z = rng.normal(size=(n_real, 3))
    A = rng.normal(size=(3, n_par))
    B = rng.normal(size=(3, n_obs))
    pars = pd.DataFrame(Z @ A + 0.05 * rng.normal(size=(n_real, n_par)),
                        columns=[f"p{i}" for i in range(n_par)])
    obs = pd.DataFrame(Z @ B + 0.05 * rng.normal(size=(n_real, n_obs)),
                       columns=[f"o{i}" for i in range(n_obs)])
    data = pd.concat([pars, obs], axis=1)

    pls = PLS(data=data, input_names=list(pars.columns),
              output_names=list(obs.columns),
              n_components=None, cv_folds=5)
    pls.fit()

    X = data[list(pars.columns)].values
    Y = data[list(obs.columns)].values
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(kf.split(X))
    max_k = min(min(len(t) for t, _ in splits), n_par, n_obs)
    brute = {}
    for k in range(1, max_k + 1):
        errs = []
        for ti, vi in splits:
            m = PLSRegression(n_components=k).fit(X[ti], Y[ti])
            errs.append(np.sqrt(np.mean((m.predict(X[vi]) - Y[vi]) ** 2)))
        brute[k] = float(np.mean(errs))
    brute_best_rmse = min(brute.values())

    # Coarse pick's CV RMSE should be within 20% of brute argmin RMSE — a
    # generous bound because the anchor grid won't necessarily land exactly
    # on the true argmin and we explicitly traded that off for speed.
    rmse_pls = pls._cv_scores[pls.n_components]
    assert rmse_pls <= brute_best_rmse * 1.20, (
        "coarse pick RMSE={0:.4g} vs brute best={1:.4g}".format(
            rmse_pls, brute_best_rmse))


def test_pls_predict_ndarray_wrong_width_raises():
    """Numpy arrays with the wrong number of columns can't be name-matched."""
    from pyemu.emulators import PLS
    data, par_cols, obs_cols = _synth_pls_data()
    pls = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=3)
    pls.fit()
    bad = np.zeros((2, len(par_cols) + 5))
    with pytest.raises(ValueError, match="ndarray"):
        pls.predict(bad)


# ---------------------------------------------------------------------------
# PLS emulator tests (Phase 2): runstore block selection, prepare_pestpp
# table fallback, and CV robustness
# ---------------------------------------------------------------------------


def _pls_rns_setup(tmp_path, par_names, obs_names, n_runs=4, seed=0):
    """Fit a PLS and lay down a matching .rns with zeroed obs columns."""
    from pyemu.emulators import PLS

    rng = np.random.RandomState(seed)
    all_names = list(dict.fromkeys(list(par_names) + list(obs_names)))
    data = pd.DataFrame(rng.normal(size=(60, len(all_names))), columns=all_names)
    # make the outputs a genuine linear function of the inputs
    W = rng.normal(size=(len(par_names), len(obs_names))) * 0.4
    data.loc[:, obs_names] = (data.loc[:, par_names].values @ W
                              + 0.01 * rng.normal(size=(60, len(obs_names))))

    emu = PLS(data=data, input_names=list(par_names),
              output_names=list(obs_names), n_components=3).fit()

    ws = str(tmp_path)
    emu.save(os.path.join(ws, "emulator.pkl"))

    par_vals = rng.normal(size=(n_runs, len(par_names)))
    obs_vals = np.zeros((n_runs, len(obs_names)))
    rns_file = os.path.join(ws, "pls.rns")
    TestRunStorForwardRuns._create_rns(rns_file, list(par_names), list(obs_names),
                                       par_vals, obs_vals)
    return emu, ws, rns_file, par_vals


def test_pls_prepare_pestpp_no_name_in_both_namespaces(tmp_path):
    """input_names are parameters as far as the emulator is concerned, so where
    output_names overlaps them the obs block gives way.  The emulator's own
    control file must never carry a name as both a parameter and an
    observation -- the run store addresses both blocks by label and cannot
    then tell them apart."""
    from pyemu.emulators import PLS

    obs_names = ["o{0}".format(i) for i in range(8)]
    par_names = obs_names[:4]
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.normal(size=(50, 8)), columns=obs_names)

    emu = PLS(data=data, input_names=par_names, output_names=obs_names,
              n_components=3).fit()
    pst = emu.prepare_pestpp(str(tmp_path / "td"), use_runstor=True)

    assert set(pst.par_names) & set(pst.obs_names) == set(), \
        "control file carries names as both par and obs"
    assert len(pst.par_names) == len(par_names)
    assert set(pst.obs_names) == set(obs_names[4:])

    # the emulator itself still predicts the full output_names
    assert list(emu.predict(data.loc[:, par_names]).columns) == obs_names


def test_pls_obs_block_all_inputs_raises():
    """If every output is also an input there is no obs block left to write."""
    from pyemu.emulators import PLS

    names = ["o{0}".format(i) for i in range(5)]
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.normal(size=(40, 5)), columns=names)
    emu = PLS(data=data, input_names=names, output_names=names,
              n_components=2).fit()
    with pytest.raises(ValueError, match="no observations"):
        emu._get_emulator_observations()


def test_pls_runstore_values_land_in_right_columns(tmp_path):
    """Positional writes are exactly where an off-by-one hides, so compare the
    run store against a manual predict() column by column."""
    from pyemu.emulators.pls import pls_runstore_forward_run
    from pyemu.utils.helpers import RunStor

    par_names = ["p{0}".format(i) for i in range(4)]
    obs_names = ["o{0}".format(i) for i in range(8)]
    emu, ws, rns_file, par_vals = _pls_rns_setup(tmp_path, par_names, obs_names)

    pls_runstore_forward_run(ws=ws, pst_name="pls")

    expected = emu.predict(pd.DataFrame(par_vals, columns=par_names))
    df = RunStor(rns_file).get_data()
    got = df.iloc[:, -len(obs_names):].values
    assert np.allclose(got, expected.loc[:, obs_names].values), \
        "obs values did not land in the expected columns"


def test_pls_runstore_par_order_permuted(tmp_path):
    """predict() selects by name, so a run store whose parameter block is
    ordered differently from the emulator's input_names must give the same
    answer once the positional slice is relabelled."""
    from pyemu.emulators.pls import pls_runstore_forward_run
    from pyemu.utils.helpers import RunStor
    from pyemu.emulators import PLS

    par_names = ["p{0}".format(i) for i in range(3)]
    obs_names = ["o{0}".format(i) for i in range(6)]
    emu, ws, rns_file, par_vals = _pls_rns_setup(tmp_path, par_names, obs_names)
    pls_runstore_forward_run(ws=ws, pst_name="pls")
    ref = RunStor(rns_file).get_data().iloc[:, -len(obs_names):].values

    # same emulator, run store with the parameter block reversed
    ws2 = os.path.join(str(tmp_path), "permuted")
    os.makedirs(ws2)
    emu.save(os.path.join(ws2, "emulator.pkl"))
    perm = list(reversed(par_names))
    perm_vals = pd.DataFrame(par_vals, columns=par_names).loc[:, perm].values
    rns2 = os.path.join(ws2, "pls.rns")
    TestRunStorForwardRuns._create_rns(rns2, perm, obs_names, perm_vals,
                                       np.zeros((par_vals.shape[0], len(obs_names))))
    pls_runstore_forward_run(ws=ws2, pst_name="pls")
    got = RunStor(rns2).get_data().iloc[:, -len(obs_names):].values
    assert np.allclose(ref, got), "permuting the parameter block changed the result"


def test_pls_runstore_derives_case_name(tmp_path):
    """A case not named 'pls' must still find its run store."""
    from pyemu.emulators.pls import pls_runstore_forward_run
    from pyemu.utils.helpers import RunStor

    par_names = ["p{0}".format(i) for i in range(3)]
    obs_names = ["o{0}".format(i) for i in range(6)]
    emu, ws, rns_file, par_vals = _pls_rns_setup(tmp_path, par_names, obs_names)
    renamed = os.path.join(ws, "mycase.rns")
    shutil.move(rns_file, renamed)

    # no pst_name given -- the case name is derived from what is on disk
    pls_runstore_forward_run(ws=ws)

    df = RunStor(renamed).get_data()
    assert not np.allclose(df.iloc[:, -len(obs_names):].values, 0.0)


def test_pls_runstore_case_name_with_glob_metachars(tmp_path):
    """The working directory is a path, not a pattern.  PEST++ worker dirs can
    contain [ ] * ?, which a glob-based search would read as pattern syntax and
    then report a present run store as missing."""
    from pyemu.emulators.pls import pls_runstore_forward_run
    from pyemu.utils.helpers import RunStor

    par_names = ["p{0}".format(i) for i in range(3)]
    obs_names = ["o{0}".format(i) for i in range(6)]
    odd = os.path.join(str(tmp_path), "worker[1]")
    os.makedirs(odd)
    emu, ws, rns_file, par_vals = _pls_rns_setup(Path(odd), par_names, obs_names)
    shutil.move(rns_file, os.path.join(odd, "mycase.rns"))

    pls_runstore_forward_run(ws=odd)

    df = RunStor(os.path.join(odd, "mycase.rns")).get_data()
    assert not np.allclose(df.iloc[:, -len(obs_names):].values, 0.0)


def test_pls_runstore_ambiguous_case_name_raises(tmp_path):
    """Two run stores and no pst_name is ambiguous -- say so rather than guess."""
    from pyemu.emulators.pls import pls_runstore_forward_run

    par_names = ["p{0}".format(i) for i in range(3)]
    obs_names = ["o{0}".format(i) for i in range(6)]
    emu, ws, rns_file, par_vals = _pls_rns_setup(tmp_path, par_names, obs_names)
    shutil.copy(rns_file, os.path.join(ws, "other.rns"))

    with pytest.raises(RuntimeError, match="multiple run stores"):
        pls_runstore_forward_run(ws=ws)


def test_pls_prepare_pestpp_par_table_falls_back(tmp_path):
    """With a pst whose parameter_data holds none of the input_names, the
    parameter table must still be populated AND the observation weights must
    still come from that pst.  Today exactly one of those two holds."""
    from pyemu.emulators import PLS

    obs_names = ["o{0}".format(i) for i in range(8)]
    par_names = obs_names[:4]
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.normal(size=(40, len(obs_names))), columns=obs_names)

    pst = _pst_for_pls(["some_other_par"], obs_names, nonzero=par_names)
    pst.observation_data.loc[:, "obsval"] = 3.14
    pst.observation_data.loc[obs_names[4:], "weight"] = 0.25

    emu = PLS(data=data, input_names=par_names, output_names=obs_names[4:],
              n_components=2).fit()

    par_df = emu._get_emulator_parameters(pst)
    obs_df = emu._get_emulator_observations(pst)

    assert len(par_df) == len(par_names), \
        "parameter table should fall back to synthesized rows, got {0}".format(
            len(par_df))
    assert list(par_df["parnme"]) == list(par_names)
    # and the obs side still comes from the supplied pst, not training means
    assert np.allclose(obs_df["obsval"].values, 3.14), obs_df["obsval"].values
    assert np.allclose(obs_df["weight"].values, 0.25), obs_df["weight"].values


def test_pls_prepare_pestpp_par_table_prefers_pst(tmp_path):
    """The fallback must not fire when the pst does describe the inputs."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=40, n_par=6, n_obs=4)
    pst = _pst_for_pls(par_cols, obs_cols)
    pst.parameter_data.loc[:, "parubnd"] = 12345.0

    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=2).fit()
    par_df = emu._get_emulator_parameters(pst)
    assert len(par_df) == len(par_cols)
    assert np.allclose(par_df["parubnd"].values, 12345.0), \
        "should have used the pst parameter_data, not synthesized bounds"


def test_pls_cv_skips_non_finite_anchor():
    """A degenerate anchor must be excluded, not merely outscored."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=40, n_par=8, n_obs=5)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols)

    real_pick = PLS._pick_components_cv

    import sklearn.cross_decomposition as scd
    orig = scd.PLSRegression

    class _Poisoned(orig):
        """PLSRegression that returns non-finite predictions at k >= 5."""
        def predict(self, X, *a, **kw):
            out = super().predict(X, *a, **kw)
            if self.n_components >= 5:
                out = np.asarray(out, dtype=float).copy()
                out[:] = np.inf
            return out

    scd.PLSRegression = _Poisoned
    try:
        emu.fit()
    finally:
        scd.PLSRegression = orig

    assert emu.n_components < 5, emu.n_components
    assert all(k < 5 for k in emu._cv_scores), sorted(emu._cv_scores)
    assert all(k >= 5 for k in emu._cv_skipped), sorted(emu._cv_skipped)
    assert all(np.isfinite(v) for v in emu._cv_scores.values())


def test_pls_cv_skips_nan_anchor():
    """NaN is the case min() gets wrong: min() over a dict containing NaN is
    order-dependent, so a NaN-scoring anchor can win outright."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=40, n_par=8, n_obs=5)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols)

    import sklearn.cross_decomposition as scd
    orig = scd.PLSRegression

    class _Poisoned(orig):
        """The lowest anchor scores NaN, so an unguarded min() picks it first."""
        def predict(self, X, *a, **kw):
            out = super().predict(X, *a, **kw)
            if self.n_components == 1:
                out = np.asarray(out, dtype=float).copy()
                out[:] = np.nan
            return out

    scd.PLSRegression = _Poisoned
    try:
        emu.fit()
    finally:
        scd.PLSRegression = orig

    assert emu.n_components != 1, "NaN-scoring anchor was selected"
    assert 1 in emu._cv_skipped
    assert 1 not in emu._cv_scores


def test_pls_cv_all_anchors_non_finite_raises():
    """No finite anchor at all should be a clear error, not an arbitrary pick."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=40, n_par=8, n_obs=5)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols)

    import sklearn.cross_decomposition as scd
    orig = scd.PLSRegression

    class _Poisoned(orig):
        def predict(self, X, *a, **kw):
            out = np.asarray(super().predict(X, *a, **kw), dtype=float).copy()
            out[:] = np.nan
            return out

    scd.PLSRegression = _Poisoned
    try:
        with pytest.raises(RuntimeError, match="no anchor with finite"):
            emu.fit()
    finally:
        scd.PLSRegression = orig


def test_pls_max_components_honoured():
    """max_components truncates the anchor ladder rather than clipping after."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=60, n_par=20, n_obs=12)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              max_components=4).fit()

    assert emu.n_components <= 4, emu.n_components
    assert max(emu._cv_scores) <= 4, sorted(emu._cv_scores)
    # the capped anchors were never fitted at all
    assert not any(k > 4 for k in emu._cv_scores), sorted(emu._cv_scores)


def test_pls_max_components_no_effect_when_ncomp_explicit():
    """An explicit n_components means explicit."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=60, n_par=20, n_obs=12)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols,
              n_components=7, max_components=4).fit()
    assert emu.n_components == 7


def test_pls_max_components_rejects_bad_value():
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=40, n_par=8, n_obs=5)
    with pytest.raises(ValueError, match="max_components"):
        PLS(data=data, input_names=par_cols, output_names=obs_cols,
            max_components=0)


def test_pls_cv_no_regression_on_well_conditioned_data():
    """Where every anchor is finite, the guard must not change the pick."""
    from pyemu.emulators import PLS

    data, par_cols, obs_cols = _synth_pls_data(n_real=60, n_par=12, n_obs=8)
    emu = PLS(data=data, input_names=par_cols, output_names=obs_cols).fit()
    assert emu._cv_skipped == {}, emu._cv_skipped
    # deterministic: KFold(random_state=0) fixes the split
    emu2 = PLS(data=data, input_names=par_cols, output_names=obs_cols).fit()
    assert emu2.n_components == emu.n_components

# ---------------------------------------------------------------------------
# PLS predictive spread (Group B): residual bootstrap
# ---------------------------------------------------------------------------


def _pls_noisy_data(n_real=90, n_par=12, n_obs=6, noise=0.3, seed=7):
    """Linear map plus known iid noise, so the residual has real width."""
    rng = np.random.RandomState(seed)
    pars = pd.DataFrame(rng.normal(size=(n_real, n_par)),
                        columns=["p{0}".format(i) for i in range(n_par)])
    W = rng.normal(size=(n_par, n_obs)) * 0.3
    obs = pd.DataFrame(pars.values @ W + noise * rng.normal(size=(n_real, n_obs)),
                       columns=["o{0}".format(i) for i in range(n_obs)])
    return pd.concat([pars, obs], axis=1), list(pars.columns), list(obs.columns)


def test_pls_residuals_zero_mean():
    """The map is fitted with an intercept, so the residual adds width without
    moving the centre.  Catches an implementation that recentres or rescales."""
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=4).fit()

    assert emu.residuals_ is not None
    assert emu.residuals_.shape == (len(data), len(oc))
    assert np.allclose(emu.residuals_.mean(axis=0), 0.0, atol=1e-8), \
        emu.residuals_.mean(axis=0)


def test_pls_ensemble_preserves_cross_column_correlation():
    """Whole residual rows are sampled, never individual entries.  Entry-wise
    sampling keeps the marginal variances correct while destroying the
    cross-observation structure, so a variance-only test passes and this one
    does not -- which is the point of it."""
    from pyemu.emulators import PLS

    # outputs share a strong common factor, so residuals are correlated
    rng = np.random.RandomState(3)
    n, npar, nobs = 90, 10, 6
    pars = pd.DataFrame(rng.normal(size=(n, npar)),
                        columns=["p{0}".format(i) for i in range(npar)])
    common = rng.normal(size=(n, 1))
    obs = pd.DataFrame(pars.values @ (rng.normal(size=(npar, nobs)) * 0.2)
                       + common * 0.9 + 0.1 * rng.normal(size=(n, nobs)),
                       columns=["o{0}".format(i) for i in range(nobs)])
    data = pd.concat([pars, obs], axis=1)
    pc, oc = list(pars.columns), list(obs.columns)

    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()
    ens = emu.predict_ensemble(data.loc[[0], pc], nreals=4000, seed=0)

    want = np.corrcoef(emu.residuals_, rowvar=False)
    got = np.corrcoef(ens.values, rowvar=False)
    assert np.allclose(got, want, atol=0.06), np.abs(got - want).max()

    # and the structure is real, not an identity matrix either side
    off = want[~np.eye(len(oc), dtype=bool)]
    assert np.abs(off).max() > 0.3, "fixture should have correlated residuals"


def test_pls_ensemble_samples_with_replacement():
    """nreals may exceed the training ensemble size."""
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data(n_real=40)
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()

    nreals = 300
    ens = emu.predict_ensemble(data.loc[[0], pc], nreals=nreals, seed=0)
    assert ens.shape == (nreals, len(oc))

    # distinct rows drawn should match the with-replacement expectation
    n = len(data)
    distinct = len(np.unique(ens.values.round(10), axis=0))
    expected = n * (1.0 - (1.0 - 1.0 / n) ** nreals)
    assert abs(distinct - expected) < 0.25 * expected, (distinct, expected)


def test_pls_ensemble_deterministic():
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=4).fit()
    x = data.loc[[0], pc]

    a = emu.predict_ensemble(x, nreals=200, seed=0)
    b = emu.predict_ensemble(x, nreals=200, seed=0)
    assert np.array_equal(a.values, b.values), "same seed must be bit-identical"
    c = emu.predict_ensemble(x, nreals=200, seed=1)
    assert not np.allclose(a.values, c.values), "different seeds must differ"


def test_pls_ensemble_constant_output_column():
    """A constant output gives a zero residual column and zero spread there,
    rather than NaN."""
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    data = data.copy()
    data.loc[:, oc[0]] = 5.0
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()

    assert np.all(np.isfinite(emu.residuals_))
    assert np.allclose(emu.residuals_[:, 0], 0.0), emu.residuals_[:, 0]

    ens = emu.predict_ensemble(data.loc[[0], pc], nreals=100, seed=0)
    assert np.all(np.isfinite(ens.values))
    assert np.isclose(ens.iloc[:, 0].std(), 0.0)
    assert np.allclose(ens.iloc[:, 0].values, 5.0)


def test_pls_ensemble_coverage_of_held_out_truth():
    """The only check that the spread is the right *size* rather than merely
    present: data from a known linear map plus known noise, so the ensemble's
    empirical coverage of held-out truth should approach nominal."""
    from pyemu.emulators import PLS

    rng = np.random.RandomState(11)
    npar, nobs, noise = 10, 5, 0.4
    W = rng.normal(size=(npar, nobs)) * 0.3

    def make(n):
        X = rng.normal(size=(n, npar))
        Y = X @ W + noise * rng.normal(size=(n, nobs))
        return (pd.DataFrame(X, columns=["p{0}".format(i) for i in range(npar)]),
                pd.DataFrame(Y, columns=["o{0}".format(i) for i in range(nobs)]))

    Xtr, Ytr = make(300)
    data = pd.concat([Xtr, Ytr], axis=1)
    pc, oc = list(Xtr.columns), list(Ytr.columns)
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=npar).fit()

    Xte, Yte = make(120)
    hits = total = 0
    for i in range(len(Xte)):
        ens = emu.predict_ensemble(Xte.iloc[[i]], nreals=400, seed=i)
        lo = np.percentile(ens.values, 5.0, axis=0)
        hi = np.percentile(ens.values, 95.0, axis=0)
        truth = Yte.iloc[i].values
        hits += int(((truth >= lo) & (truth <= hi)).sum())
        total += len(truth)
    coverage = hits / total
    assert 0.85 <= coverage <= 0.98, "90% interval covered {0:.3f}".format(coverage)


def test_pls_ensemble_residual_added_before_inverse_transform():
    """The zero-mean guarantee holds in the space the map was fitted in, so the
    residual is added before the output inverse-transform.  Under a log
    transform that shows up as a median tracking the point prediction with the
    mean above it; adding after the inverse would recentre the median."""
    from pyemu.emulators import PLS

    rng = np.random.RandomState(0)
    n, npar, nobs = 90, 10, 5
    pars = pd.DataFrame(rng.normal(size=(n, npar)),
                        columns=["p{0}".format(i) for i in range(npar)])
    obs = pd.DataFrame(
        np.exp(pars.values @ (rng.normal(size=(npar, nobs)) * 0.3)
               + 0.4 * rng.normal(size=(n, nobs))),
        columns=["o{0}".format(i) for i in range(nobs)])
    data = pd.concat([pars, obs], axis=1)
    pc, oc = list(pars.columns), list(obs.columns)

    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=4,
              transforms=[{"type": "log10", "columns": oc}]).fit()

    x = data.loc[[0], pc]
    point = np.asarray(emu.predict(x), dtype=float)
    ens = emu.predict_ensemble(x, nreals=4000, seed=0)

    # the inverse transform was applied to the perturbed values
    assert (ens.values > 0).all(), "log inverse should keep everything positive"
    med = np.median(ens.values, axis=0)
    assert np.allclose(med / point, 1.0, atol=0.1), med / point
    assert np.all(ens.mean(axis=0).values > point), \
        "log inverse of a symmetric residual should be right-skewed"


def test_pls_ensemble_shapes_and_index():
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()

    one = emu.predict_ensemble(data.loc[[0], pc], nreals=25, seed=0)
    assert one.shape == (25, len(oc))
    assert one.index.name == "realization"
    assert list(one.columns) == oc

    many = emu.predict_ensemble(data.loc[[0, 1, 2], pc], nreals=7, seed=0)
    assert many.shape == (3 * 7, len(oc))
    assert many.index.names[-1] == "realization"
    assert len(many.index.levels[0]) == 3


def test_pls_ensemble_errors():
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    unfit = PLS(data=data, input_names=pc, output_names=oc, n_components=3)
    with pytest.raises(ValueError, match="fitted"):
        unfit.predict_ensemble(data.loc[[0], pc])

    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()
    with pytest.raises(ValueError, match="nreals"):
        emu.predict_ensemble(data.loc[[0], pc], nreals=0)


def test_pls_parameter_reducer_not_aliased():
    """_fitted_reducer must be a copy: sharing one reducer instance between two
    emulators otherwise makes the second fit change the first's predictions."""
    from sklearn.decomposition import PCA
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data(n_real=80, n_par=12)
    shared = PCA(n_components=5)

    e1 = PLS(data=data, input_names=pc, output_names=oc, n_components=3,
             parameter_reducer=shared).fit()
    before = np.asarray(e1.predict(data.loc[:, pc]), dtype=float)

    PLS(data=data.iloc[:40], input_names=pc, output_names=oc, n_components=3,
        parameter_reducer=shared).fit()
    after = np.asarray(e1.predict(data.loc[:, pc]), dtype=float)

    assert e1._fitted_reducer is not shared, "reducer was aliased, not copied"
    assert np.allclose(before, after), \
        "fitting a second emulator changed the first's predictions"


def test_pls_residuals_survive_pickle_round_trip(tmp_path):
    """predict_ensemble must work after load(), so residuals_ has to persist."""
    from pyemu.emulators import PLS

    data, pc, oc = _pls_noisy_data()
    emu = PLS(data=data, input_names=pc, output_names=oc, n_components=3).fit()
    f = os.path.join(str(tmp_path), "emu.pkl")
    emu.save(f)

    loaded = PLS.load(f)
    assert loaded.residuals_ is not None
    assert np.allclose(loaded.residuals_, emu.residuals_)
    x = data.loc[[0], pc]
    assert np.allclose(loaded.predict_ensemble(x, nreals=50, seed=0).values,
                       emu.predict_ensemble(x, nreals=50, seed=0).values)

if __name__ == "__main__":
    tmp_path = Path("temp")
    test_gpr_basic(tmp_path=tmp_path)


