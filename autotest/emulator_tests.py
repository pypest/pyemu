import os
# import sys
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
# import platform
import pyemu
from pst_from_tests import setup_tmp, _get_port, exepath_dict
from pst_from_tests import setup_tmp, _get_port
from pyemu.emulators import DSI, LPFA, GPR, dsi


ies_exe_path = exepath_dict["pestpp-ies"]
mou_exe_path = exepath_dict["pestpp-mou"]# Check for TensorFlow availability for DSIAE tests

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
        pyemu.os_utils.run(f'{ies_exe_path} dsi.pst /e', cwd=td, verbose=True)
    else:
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

@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
def test_dsiae_save_load(tmp_path):
    if isinstance(tmp_path, str) and not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # 1. Generate synthetic data
    num_realizations = 50
    num_observations = 20
    data = np.random.normal(size=(num_realizations, num_observations))
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
    new_pvals = np.random.normal(size=(5, latent_dim))
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
    # y += np.random.normal(0, 0.001, 20) 
    
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


def test_log10_transformer():
    from pyemu.emulators.transformers import Log10Transformer
    
    df = pd.DataFrame({
        'a': [1, 10, 100],
        'b': [0, -1, 10] # Contains non-positives
    })
    
    # Test 1: Simple logs
    t = Log10Transformer(columns=['a'])
    res = t.fit_transform(df)
    assert np.allclose(res['a'].values, [0, 1, 2])
    assert np.all(res['b'] == df['b']) # Untouched
    
    inv = t.inverse_transform(res)
    assert np.allclose(inv['a'].values, df['a'].values)
    
    # Test 2: Shift handling
    t2 = Log10Transformer(columns=['b'])
    res2 = t2.fit_transform(df)
    # min is -1. shift should be -(-1) + 1e-6 = 1.000001
    # values become 0+shift, -1+shift, 10+shift
    # just check roundtrip
    inv2 = t2.inverse_transform(res2)
    assert np.allclose(inv2['b'].values, df['b'].values)


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
    np.random.seed(42)
    for i in range(n_real):
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.8, 1.2)
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


if __name__ == "__main__":
    #test_dsiae_save_load("temp")
    #test_dsi_basic("temp")
    #test_dsi_nst("temp")
    #test_dsi_nst_extrap("temp")
    #test_dsi_mixed("temp")
    #test_dsivc("temp")
    #plot_freyberg_dsi()
    #test_lpfa_std()
    #gpr_zdt1_test()
    tmp_path = Path("temp")
    test_gpr_basic(tmp_path=tmp_path)


