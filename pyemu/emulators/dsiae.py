"""
Data Space Inversion (DSI) Autoencoder (AE) emulator implementation.
"""
from __future__ import print_function, division
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import inspect
from pyemu.utils.helpers import dsi_forward_run,dsi_runstore_forward_run, series_to_insfile
import os
import shutil
from pyemu.pst.pst_handler import Pst
from pyemu.en import ObservationEnsemble,ParameterEnsemble
from .base import Emulator
import pickle
import tempfile
import zipfile

try:
    import tensorflow as tf
    from keras.saving import register_keras_serializable
except ImportError:
    tf = None
    # Dummy decorator to prevent NameError on class definitions
    def register_keras_serializable(package=None, name=None):
        def decorator(cls):
            return cls
        return decorator

from sklearn.model_selection import train_test_split



class DSIAE(Emulator):
    """
    Data Space Inversion Autoencoder (DSIAE) emulator.
    """

    def __init__(self, 
                pst: Optional['Pst'] = None,
                data: Optional[Union[pd.DataFrame, 'ObservationEnsemble']] = None,
                transforms: Optional[List[Dict[str, Any]]] = None,
                latent_dim: Optional[int] = None,
                energy_threshold: float = 1.0,
                verbose: bool = False) -> None:
        """
        Initialize the DSIAE emulator.

        Args:
            pst: PEST control file object.
            data: Training data (DataFrame or ObservationEnsemble).
            transforms: List of dicts defining preprocessing transformations.
            latent_dim: Latent space dimension. If None, determined from energy_threshold.
            energy_threshold: Variance threshold for automatic latent dimension (0.0-1.0).
            verbose: Enable verbose logging.
        """
        super().__init__(verbose=verbose)

        self.observation_data = pst.observation_data.copy() if pst is not None else None
        
        if isinstance(data, ObservationEnsemble):
            data = data._df.copy()
        
        # Ensure float data
        self.data = data.astype(float).copy() if data is not None else None
        
        self.energy_threshold = energy_threshold
        
        if transforms is not None:
            if not isinstance(transforms, list):
                raise TypeError("transforms must be a list of dicts")
            for t in transforms:
                if not isinstance(t, dict) or 'type' not in t:
                    raise ValueError("Each transform must be a dict with a 'type' key")
                if 'columns' in t:
                    missing = [c for c in t['columns'] if c not in self.data.columns]
                    if missing:
                        raise ValueError(f"Transform columns not found in data: {missing}")

        self.transforms = transforms
        self.fitted = False
        self.data_transformed = self._prepare_training_data()
        self.decision_variable_names = None 
        self.latent_dim = latent_dim
        
        if self.latent_dim is None and self.data is not None:
            self.logger.statement("calculating latent dimension from energy threshold")
            self.latent_dim = self._calc_explained_variance()

        
    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare and transform training data for model fitting.
        
        This method applies the configured transformation pipeline to the raw training
        data, preparing it for use in autoencoder training. If no transformations are
        specified, the data is passed through unchanged but a dummy transformer is
        still created for consistency in the prediction pipeline.
        
        Returns
        -------
        pd.DataFrame
            Transformed training data ready for model fitting. All values will be
            numeric (float64) and any specified transformations will have been applied.
            
        Raises
        ------
        ValueError
            If no data is stored in the emulator instance.
            
        Notes
        -----
        This method is automatically called during emulator initialization and stores
        the transformed data in `self.data_transformed`. The transformation pipeline
        is preserved in `self.transformer_pipeline` for use during prediction to
        ensure consistent data preprocessing.
        
        The method always creates a transformer pipeline object, even when no 
        transformations are specified, to maintain consistency in the prediction
        workflow where inverse transformations may be needed.
        """
        data = self.data
        if data is None:
            raise ValueError("No data stored in the emulator")

        self.logger.statement("applying feature transforms")
        # Always use the base class transformation method for consistency
        if self.transforms is not None:
            self.data_transformed = self._fit_transformer_pipeline(data, self.transforms)
        else:
            # Still need to set up a dummy transformer for inverse operations
            from .transformers import AutobotsAssemble
            self.transformer_pipeline = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()
    
        return self.data_transformed
        
    def encode(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Encode input data into latent space representation.
        
        This method transforms input observation data into the lower-dimensional 
        latent space learned by the autoencoder. The encoding process applies any
        configured data transformations before passing the data through the encoder
        network.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input observation data to encode. Should have the same feature structure
            as the training data. If DataFrame, the index will be preserved in the
            output. Shape should be (n_samples, n_features) where n_features matches
            the original observation space dimension.
            
        Returns
        -------
        pd.DataFrame
            Encoded latent space representation with shape (n_samples, latent_dim).
            If input was a DataFrame, the original index is preserved. Column names
            will be generated automatically for the latent dimensions.
            
        Raises
        ------
        ValueError
            If the encoder has not been fitted (emulator not trained).
            If input data shape is incompatible with the trained model.
            
        Notes
        -----
        This method automatically applies the same data transformations that were
        used during training, ensuring consistent preprocessing. The transformations
        are applied via the stored `transformer_pipeline`.
        
        The latent space representation can be used for:
        - Dimensionality reduction and visualization
        - Parameter space exploration
        - Input to optimization routines
        - Analysis of model behavior in reduced space
        
        Examples
        --------
        >>> # Encode training data
        >>> latent_repr = emulator.encode(training_data)
        >>> 
        >>> # Encode new observations
        >>> new_latent = emulator.encode(new_observations)
        >>> print(f"Latent dimensions: {new_latent.shape[1]}")
        """
        # check encoder exists
        if not hasattr(self, 'encoder'):
            raise ValueError("Encoder not found. Fit the emulator before encoding.")

        if isinstance(X, pd.DataFrame):
            index = X.index

        if self.transforms is not None:
            X = self.transformer_pipeline.transform(X)
        Z = self.encoder.encode(X)
        Z = pd.DataFrame(Z, index=index if 'index' in locals() else None)
        return Z

    
    def _calc_explained_variance(self) -> int:
        """
        Calculate optimal latent dimension using PCA explained variance threshold.
        
        Returns
        -------
        int
            Minimum latent dimensions to capture `energy_threshold` variance.
            Falls back to full dimensionality if 99% variance threshold not reached.
            
        Notes
        -----
        Uses scikit-learn PCA on `self.data_transformed`. The energy_threshold 
        represents cumulative explained variance ratio (e.g., 0.95 = 95% variance).
        """
        from sklearn.decomposition import PCA  # light dependency; optional
        # PCA explained variance (optional)
        pca = PCA()
        pca.fit(self.data_transformed.values.astype(float))
        cum_explained = np.cumsum(pca.explained_variance_ratio_)
        latent_dim = int(np.searchsorted(cum_explained, self.energy_threshold) + 1) if cum_explained[-1] >= 0.99 else len(cum_explained)
        return latent_dim

    def fit(self, validation_split: float = 0.1, hidden_dims: tuple = (128, 64), 
            lr: float = 1e-3, epochs: int = 300, batch_size: int = 128, 
            early_stopping: bool = True, dropout_rate: float = 0.0, 
            random_state: int = 42, loss_type: str = 'energy', 
            loss_kwargs: Optional[Dict[str, Any]] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'DSIAE':
        """
        Fit the autoencoder emulator to training data.
        
        Parameters
        ----------
        validation_split : float, default 0.1
            Fraction of data to use for validation.
        hidden_dims : tuple, default (128, 64)
            Hidden layer dimensions for encoder/decoder.
        lr : float, default 1e-3
            Learning rate for Adam optimizer.
        epochs : int, default 300
            Maximum training epochs.
        batch_size : int, default 128
            Training batch size.
        early_stopping : bool, default True
            Whether to use early stopping on validation loss.
        dropout_rate : float, default 0.0
            Dropout rate for regularization during training.
        random_state : int, default 42
            Random seed for reproducibility.
        loss_type : str, default 'energy'
            Type of loss function to use. Options: 'energy', 'mmd', 'wasserstein', 
            'statistical', 'adaptive', 'mse', 'huber'.
        loss_kwargs : dict, optional
            Additional parameters for the loss function.
        sample_weight : np.ndarray, optional
            Sample weights for training. Shape should be (n_samples,).
            
        Returns
        -------
        DSIAE
            Self (fitted emulator instance).
        """
        
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()

        X = self.data_transformed.values.astype(float)
        if self.latent_dim is None:
            self.logger.statement("calculating latent dimension from energy threshold")
            self.latent_dim = self._calc_explained_variance()

        # Configure loss function
        if loss_kwargs is None:
            loss_kwargs = {}
        
        # Set default loss parameters if not specified
        if loss_type == 'energy' and 'lambda_energy' not in loss_kwargs:
            loss_kwargs['lambda_energy'] = 1e-3
        elif loss_type == 'mmd' and 'lambda_mmd' not in loss_kwargs:
            loss_kwargs['lambda_mmd'] = 1e-3
        elif loss_type == 'wasserstein' and 'lambda_w' not in loss_kwargs:
            loss_kwargs['lambda_w'] = 1e-3
        elif loss_type == 'statistical':
            if 'lambda_moments' not in loss_kwargs:
                loss_kwargs['lambda_moments'] = 1e-3
            if 'lambda_corr' not in loss_kwargs:
                loss_kwargs['lambda_corr'] = 5e-4
            if 'lambda_dist' not in loss_kwargs:
                loss_kwargs['lambda_dist'] = 1e-3
        
        loss_fn = create_distribution_loss(loss_type, **loss_kwargs)
        
        self.logger.statement(f"using {loss_type} loss function with parameters: {loss_kwargs}")
        # train autoencoder on transformed data
        ae = AutoEncoder(input_dim=X.shape[1], 
                        latent_dim=self.latent_dim,
                        hidden_dims=hidden_dims,
                        loss=loss_fn,
                        lr=lr,
                        dropout_rate=dropout_rate,
                        random_state=random_state,
                        )
        ae.fit(X,
               validation_split=validation_split,
                epochs=epochs, batch_size=batch_size,
                early_stopping=early_stopping,
                patience=10,
                sample_weight=sample_weight,
                )
        self.encoder = ae
        self.fitted = True
        return self
    
    def predict(self, pvals: Union[np.ndarray, pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Generate predictions from the emulator.
        
        Parameters
        ----------
        pvals : np.ndarray, pd.Series, or pd.DataFrame
            Parameter values for prediction in latent space.
            Shape should match latent_dim.
            
        Returns
        -------
        pd.Series
            Predicted observation values in original scale.
            
        Raises
        ------
        ValueError
            If emulator not fitted or input dimensions incorrect.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
            
        if self.transforms is not None and (not hasattr(self, 'transformer_pipeline') or self.transformer_pipeline is None):
            raise ValueError("Emulator must be fitted and have valid transformations before prediction")
        
        if isinstance(pvals, pd.Series):
            pvals = pvals.values.flatten().reshape(1,-1).astype(np.float32)
        elif isinstance(pvals, np.ndarray) and len(pvals.shape) == 2 and pvals.shape[0] == 1:
            pvals = pvals.flatten().reshape(1,-1)
            pvals = pvals.astype(np.float32)
        elif isinstance(pvals, pd.DataFrame):
            index = pvals.index
            pvals = pvals.values.astype(np.float32)
            
        #assert pvals.shape[0] == self.latent_dim , f"Input parameter dimension {pvals.shape[0]} does not match latent dimension {self.latent_dim}"
        sim_vals = self.encoder.decode(pvals)
        sim_vals = pd.DataFrame(sim_vals,
                                columns=self.data_transformed.columns,
                                index=index if 'index' in locals() else None)
        sim_vals = sim_vals.squeeze()
        #if isinstance(sim_vals, np.ndarray):
        #    sim_vals = pd.Series(sim_vals.flatten(), index=self.data_transformed.columns)
        if self.transforms is not None:
            pipeline = self.transformer_pipeline
            sim_vals = pipeline.inverse(sim_vals)
        sim_vals.index.name = 'obsnme'
        sim_vals.name = "obsval"
        self.sim_vals = sim_vals
        return sim_vals
    
    def check_for_pdc(self):
        """Check for Prior data conflict."""
        #TODO
        return
        
    def prepare_pestpp(self, t_d: Optional[str] = None, 
                       observation_data: Optional[pd.DataFrame] = None,
                       use_runstor: bool = False) -> 'Pst':
        """
        Prepare PEST++ control files for the emulator.
        
        Parameters
        ----------
        t_d : str, optional
            Template directory path. Must be provided.
        observation_data : pd.DataFrame, optional
            Observation data to use. If None, uses data from initialization.
        use_runstor : bool, default False
            Whether to use the Runstor batch file format for PEST++. Setting to True will setup the forward run script to work with PEST++ external run maganer.
        Returns
        -------
        Pst
            PEST++ control file object ready for inversion.
            
        Notes
        -----
        Creates template files, instruction files, and configures PEST++ options
        for running the emulator in parameter estimation mode.
        """
        
        assert t_d is not None, "template directory must be provided"
        self.template_dir = t_d

        if os.path.exists(t_d):
            shutil.rmtree(t_d)
        os.makedirs(t_d)
        self.logger.statement("creating template directory {0}".format(t_d))

        self.logger.log("creating tpl files")
        dsi_in_file = os.path.join(t_d, "dsi_pars.csv")
        dsi_tpl_file = dsi_in_file + ".tpl"
        ftpl = open(dsi_tpl_file, 'w')
        fin = open(dsi_in_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        npar = self.latent_dim
        assert npar>0, "no parameters found in the DSI emulator"
        dsi_pnames = []
        for i in range(npar):
            pname = "dsi_par{0:04d}".format(i)
            dsi_pnames.append(pname)
            fin.write("{0},0.0\n".format(pname))
            ftpl.write("{0},~   {0}   ~\n".format(pname, pname))
        fin.close()
        ftpl.close()
        self.logger.log("creating tpl files")

        # run once to get the dsi_pars.csv file
        Z = self.encode(self.data)
        if 'base' in Z.index:
            pvals = Z.loc['base',:]
        else:
            pvals = Z.mean(axis=0) #TODO: the mean of the latent prior...doesnt realy mean anything hrere
        sim_vals = self.predict(pvals)
        
        self.logger.log("creating ins file")
        out_file = os.path.join(t_d,"dsi_sim_vals.csv")
        sim_vals.to_csv(out_file,index=True)
              
        ins_file = out_file + ".ins"
        sdf = pd.read_csv(out_file,index_col=0)
        with open(ins_file,'w') as f:
            f.write("pif ~\n")
            f.write("l1\n")
            for oname in sdf.index.values:
                f.write("l1 ~,~ !{0}!\n".format(oname))
        self.logger.log("creating ins file")

        self.logger.log("creating Pst")
        pst = Pst.from_io_files([dsi_tpl_file],[dsi_in_file],[ins_file],[out_file],pst_path=".")

        par = pst.parameter_data
        dsi_pars = par.loc[par.parnme.str.startswith("dsi_par"),"parnme"]
        par.loc[dsi_pars,"parval1"] = pvals.values.flatten()
        par.loc[dsi_pars,"parubnd"] = Z.max(axis=0).values
        par.loc[dsi_pars,"parlbnd"] = Z.min(axis=0).values
        par.loc[dsi_pars,"partrans"] = "none"

        #with open(os.path.join(t_d,"dsi.unc"),'w') as f:
        #    f.write("START STANDARD_DEVIATION\n")
        #    for p in dsi_pars:
        #        f.write("{0} 1.0\n".format(p))
        #    f.write("END STANDARD_DEVIATION")
        #pst.pestpp_options['parcov'] = "dsi.unc"
        Z.columns = par.parnme.values
        pe = ParameterEnsemble(pst,Z)
        pe.to_binary(os.path.join(t_d,'latent_prior.jcb'))
        pst.pestpp_options['ies_parameter_ensemble'] = "latent_prior.jcb"

        obs = pst.observation_data

        if observation_data is not None:
            self.observation_data = observation_data
        else:
            observation_data = self.observation_data
        assert isinstance(observation_data, pd.DataFrame), "observation_data must be a pandas DataFrame"
        for col in observation_data.columns:
            obs.loc[sim_vals.index,col] = observation_data.loc[:,col]

        # check if any observations are missing
        missing_obs = list(set(obs.index) - set(observation_data.index))
        assert len(missing_obs) == 0, "missing observations: {0}".format(missing_obs)

        pst.control_data.noptmax = 0
        pst.model_command = "python forward_run.py"
        self.logger.log("creating Pst")

        if use_runstor:
            function_source = inspect.getsource(dsi_runstore_forward_run)
        else:
            function_source = inspect.getsource(dsi_forward_run)
        with open(os.path.join(t_d,"forward_run.py"),'w') as file:
            file.write(function_source)
            file.write("\n\n")
            file.write("if __name__ == \"__main__\":\n")
            file.write(f"    {function_source.split('(')[0].split('def ')[1]}()\n")
        self.logger.log("creating Pst")

        pst.pestpp_options["save_binary"] = True
        pst.pestpp_options["overdue_giveup_fac"] = 1e30
        pst.pestpp_options["overdue_giveup_minutes"] = 1e30
        pst.pestpp_options["panther_agent_freeze_on_fail"] = True
        pst.pestpp_options["ies_no_noise"] = False
        pst.pestpp_options["ies_subset_size"] = -10 # the more the merrier
        #pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
        #pst.pestpp_options["save_binary"] = True

        pst.write(os.path.join(t_d,"dsi.pst"),version=2)
        self.logger.statement("saved pst to {0}".format(os.path.join(t_d,"dsi.pst")))
        
        self.logger.statement("pickling dsi object to {0}".format(os.path.join(t_d,"dsi.pickle")))
        self.save(os.path.join(t_d,"dsi.pickle"))
        return pst
        
    def prepare_dsivc(self, decvar_names: Union[List[str], str], t_d: Optional[str] = None, 
                      pst: Optional['Pst'] = None, oe: Optional['ObservationEnsemble'] = None, 
                      track_stack: bool = False, dsi_args: Optional[Dict[str, Any]] = None, 
                      percentiles: List[float] = [0.25, 0.75, 0.5], 
                      mou_population_size: Optional[int] = None, 
                      ies_exe_path: str = "pestpp-ies") -> 'Pst':
        """
        Prepare Data Space Inversion Variable Control (DSIVC) control files.
        
        Parameters
        ----------
        decvar_names : list or str
            Names of decision variables for optimization.
        t_d : str, optional
            Template directory path. Uses existing if None.
        pst : Pst, optional
            PST control file object. Uses existing if None.
        oe : ObservationEnsemble, optional
            Observation ensemble. Uses existing if None.
        track_stack : bool, default False
            Whether to include individual ensemble realizations as observations.
        dsi_args : dict, optional
            DSI configuration arguments.
        percentiles : list, default [0.25, 0.75, 0.5]
            Percentiles to calculate from ensemble statistics.
        mou_population_size : int, optional
            Population size for multi-objective optimization.
        ies_exe_path : str, default "pestpp-ies"
            Path to PEST++ IES executable.
            
        Returns
        -------
        Pst
            PEST++ control file object for DSIVC optimization.
            
        Notes
        -----
        Sets up multi-objective optimization with decision variables constrained
        to training data bounds. Creates stack statistics observations for ensemble
        matching and configures PEST++-MOU options.
        """
        # check that percentiles is a list or array of floats between 0 and 1.
        assert isinstance(percentiles, (list, np.ndarray)), "percentiles must be a list or array of floats"
        assert all([isinstance(i, (float, int)) for i in percentiles]), "percentiles must be a list or array of floats"
        assert all([0 <= i <= 1 for i in percentiles]), "percentiles must be between 0 and 1"
        # ensure that pecentiles are unique
        percentiles = np.unique(percentiles)


        #track dsivc args for forward run
        self.dsivc_args = {"percentiles":percentiles,
                            "decvar_names":decvar_names,
                            "track_stack":track_stack,
                        }

        if t_d is None:
            self.logger.statement("using existing DSI template dir...")
            t_d = self.template_dir
        self.logger.statement(f"using {t_d} as template directory...")
        assert os.path.exists(t_d), f"template directory {t_d} does not exist"

        if pst is None:
            self.logger.statement("no pst provided...")
            self.logger.statement("using dsi.pst in DSI template dir...")
            assert os.path.exists(os.path.join(t_d,"dsi.pst")), f"dsi.pst not found in {t_d}"
            pst = Pst(os.path.join(t_d,"dsi.pst"))
        if oe is None:
            self.logger.statement(f"no posterior DSI observation ensemble provided, using dsi.{dsi_args['noptmax']}.obs.jcb in DSI template dir...")
            assert os.path.exists(os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb")), f"dsi.{dsi_args['noptmax']}.obs.jcb not found in {t_d}"
            oe = ObservationEnsemble.from_binary(pst,os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb"))
        else:
            assert isinstance(oe, ObservationEnsemble), "oe must be an ObservationEnsemble"

        #check if decvar_names str
        if isinstance(decvar_names, str):
            decvar_names = [decvar_names]
        # chekc htat decvars are in the oe columns
        missing = [col for col in decvar_names if col not in oe.columns]
        assert len(missing) == 0, f"The following decvars are missing from the DSI obs ensemble: {missing}"
        # chekc htat decvars are in the pst observation data
        missing = [col for col in decvar_names if col not in pst.obs_names]
        assert len(missing) == 0, f"The following decvars are missing from the DSI pst control file: {missing}"


        # handle DSI args
        default_dsi_args =  {"noptmax":pst.control_data.noptmax,
                            "decvar_weight":1.0,
                            #"decvar_phi_factor":0.5,
                            "num_pyworkers":1,
                            }
        # ensure it's a dict
        if dsi_args is None:
            dsi_args = default_dsi_args
        elif not isinstance(dsi_args, dict):
            raise TypeError("Expected a dictionary for 'options'")
        # merge with defaults (user values override defaults)
        #dsi_args = {**default_dsi_args, **dsi_args}
        else:
            for key, value in default_dsi_args.items():
                if key not in dsi_args:
                    dsi_args[key] = value

        # check that dsi_args has the required keys
        required_keys = ["noptmax", "decvar_weight", "num_pyworkers"]
        for key in required_keys:
            if key not in dsi_args:
                raise KeyError(f"Missing required key '{key}' in 'dsi_args'")
        self.dsi_args = dsi_args
        out_files = []

        self.logger.statement(f"preparing stack stats observations...")
        assert isinstance(oe, ObservationEnsemble), "oe must be an ObservationEnsemble"
        if oe.index.name is None:
            id_vars="index"
        else:
            id_vars=oe.index.name
        stack_stats = oe._df.describe(percentiles=percentiles).reset_index().melt(id_vars=id_vars)
        stack_stats.rename(columns={"value":"obsval","index":"stat"},inplace=True)
        stack_stats['obsnme'] = stack_stats.apply(lambda x: x.variable+"_stat:"+x.stat,axis=1)
        stack_stats.set_index("obsnme",inplace=True)
        stack_stats = stack_stats.obsval
        self.logger.statement(f"stack osb recorded to dsi.stack_stats.csv...")
        out_file = os.path.join(t_d,"dsi.stack_stats.csv")
        out_files.append(out_file)
        stack_stats.to_csv(out_file,float_format="%.6e")
        series_to_insfile(out_file,ins_file=None)


        if track_stack:
            self.logger.statement(f"including {oe.values.flatten().shape[0]} stack observations...")

            stack = oe._df.reset_index().melt(id_vars=id_vars)
            stack.rename(columns={"value":"obsval"},inplace=True)
            stack['obsnme'] = stack.apply(lambda x: x.variable+"_real:"+x.index,axis=1)
            stack.set_index("obsnme",inplace=True)
            stack = stack.obsval
            out_file = os.path.join(t_d,"dsi.stack.csv")
            out_files.append(out_file)
            stack.to_csv(out_file,float_format="%.6e")
            series_to_insfile(out_file,ins_file=None)



        self.logger.statement(f"prepare DSIVC template files...")
        dsi_in_file = os.path.join(t_d, "dsivc_pars.csv")
        dsi_tpl_file = dsi_in_file + ".tpl"
        ftpl = open(dsi_tpl_file, 'w')
        fin = open(dsi_in_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        for pname in decvar_names:
            val = oe._df.loc[:,pname].mean()
            fin.write(f"{pname},{val:.6e}\n")
            ftpl.write(f"{pname},~   {pname}   ~\n")
        fin.close()
        ftpl.close()

        
        self.logger.statement(f"building DSIVC control file...")
        pst_dsivc = Pst.from_io_files([dsi_tpl_file],[dsi_in_file],[i+".ins" for i in out_files],out_files,pst_path=".")

        self.logger.statement(f"setting dec var bounds...")
        par = pst_dsivc.parameter_data
        # set all parameters fixed
        par.loc[:,"partrans"] = "fixed"
        # constrain decvar pars to training data bounds
        par.loc[decvar_names,"pargp"] = "decvars"
        par.loc[decvar_names,"partrans"] = "none"
        par.loc[decvar_names,"parubnd"] = self.data.loc[:,decvar_names].max()
        par.loc[decvar_names,"parlbnd"] = self.data.loc[:,decvar_names].min()
        par.loc[decvar_names,"parval1"] = self.data.loc[:,decvar_names].quantile(.5)
        
        self.logger.statement(f"zero-weighting observation data...")
        # prepemtpively set obs weights 0.0
        obs = pst_dsivc.observation_data
        obs.loc[:,"weight"] = 0.0

        self.logger.statement(f"getting obs metadata from DSI observation_data...")
        obsorg = pst.observation_data.copy()
        columns = [i for i in obsorg.columns if i !='obsnme']
        for o in obsorg.obsnme.values:
            obs.loc[obs.obsnme.str.startswith(o), columns] = obsorg.loc[obsorg.obsnme==o, columns].values

        obs.loc[stack_stats.index,"obgnme"] = "stack_stats"
        obs.loc[stack_stats.index,"org_obsnme"] = [i.split("_stat:")[0] for i in stack_stats.index.values]
        pst_dsivc.try_parse_name_metadata()

        #obs.loc[stack.index,"obgnme"] = "stack"

        self.logger.statement(f"building dsivc_forward_run.py...")
        pst_dsivc.model_command = "python dsivc_forward_run.py"
        from pyemu.utils.helpers import dsivc_forward_run
        function_source = inspect.getsource(dsivc_forward_run)
        with open(os.path.join(t_d,"dsivc_forward_run.py"),'w') as file:
            file.write(function_source)
            file.write("\n\n")
            file.write("if __name__ == \"__main__\":\n")
            file.write(f"    {function_source.split('(')[0].split('def ')[1]}(ies_exe_path='{ies_exe_path}')\n")

        self.logger.statement(f"preparing nominal initial population...")
        if mou_population_size is None:
            # set the population size to 2 * number of decision variables
            # this is a good rule of thumb for MOU
            mou_population_size = 2 * len(decvar_names)
        # these should generally be twice the number of decision variables
        if mou_population_size < 2 * len(decvar_names):
            self.logger.statement(f"mou population is less than 2x number of decision variables, this may be too small...")
        # sample 160 sets of decision variables from a unform distribution
        dvpop = ParameterEnsemble.from_uniform_draw(pst_dsivc,num_reals=mou_population_size)
        # record to external file for PESTPP-MOU
        dvpop.to_binary(os.path.join(t_d,"initial_dvpop.jcb"))
        # tell PESTPP-MOU about the new file
        pst_dsivc.pestpp_options["mou_dv_population_file"] = 'initial_dvpop.jcb'


        # some additional PESTPP-MOU options:
        pst_dsivc.pestpp_options["mou_population_size"] = mou_population_size #twice the number of decision variables
        pst_dsivc.pestpp_options["mou_save_population_every"] = 1 # save lots of files! 
        
        pst_dsivc.control_data.noptmax = 0 #just for a test run
        pst_dsivc.write(os.path.join(t_d,"dsivc.pst"),version=2)  

        # updating the DSI pst control file
        self.logger.statement(f"updating DSI pst control file...")
        self.logger.statement("overwriting dsi.pst file...")
        pst.observation_data.loc[decvar_names, "weight"] = dsi_args["decvar_weight"]
        pst.control_data.noptmax = dsi_args["noptmax"]

        #TODO: ensure no noise for dvars obs

        pst.write(os.path.join(t_d,"dsi.pst"), version=2)
        
        
        self.logger.statement("overwriting dsi.pickle file...")
        self.decision_variable_names = decvar_names
        # re-pickle dsi to track dsivc args
        self.save(os.path.join(t_d,"dsi.pickle"))
  
        self.logger.statement("DSIVC control files created...the user still needs to specify objectives and constraints...")
        return pst_dsivc
    
    def hyperparam_search(self, latent_dims: Optional[List[int]] = None,
                          latent_dim_mults: List[float] = [0.5, 1.0, 2.0],
                          hidden_dims_list: List[tuple] = [(64, 32), (128, 64)],
                          lrs: List[float] = [1e-2, 1e-3],
                          epochs: int = 50, batch_size: int = 32,
                          random_state: int = 0) -> Dict[tuple, float]:
        """
        Grid search over autoencoder hyperparameters.
        
        Parameters
        ----------
        latent_dims : list of int, optional
            Latent dimensions to test. If None, uses latent_dim_mults.
        latent_dim_mults : list of float, default [0.5, 1.0, 2.0]
            Multipliers for current latent_dim if latent_dims not provided.
        hidden_dims_list : list of tuple, default [(64, 32), (128, 64)]
            Hidden layer architectures to test.
        lrs : list of float, default [1e-2, 1e-3]
            Learning rates to test.
        epochs : int, default 50
            Training epochs for each configuration.
        batch_size : int, default 32
            Training batch size.
        random_state : int, default 0
            Random seed for reproducibility.
            
        Returns
        -------
        dict
            Mapping from (latent_dim, hidden_dims, lr) to validation loss.
        """
        if latent_dims is None:
            assert self.latent_dim is not None, "Either latent_dims or self.latent_dim must be set"
            latent_dims = [int(self.latent_dim * m) for m in latent_dim_mults]

        X = self.data_transformed.values.astype(float)
        results = AutoEncoder.hyperparam_search(
            X,
            latent_dims=latent_dims,
            hidden_dims_list=hidden_dims_list,
            lrs=lrs,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state
        )
        return results

    def save(self, filename: str) -> None:
        """
        Save the emulator to a file.
        
        Bundles the pickled object and the TensorFlow model into a zip archive.
        """
        # Create a temporary directory to save components
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Save TF model
            model_dir = os.path.join(tmp_dir, "tf_model")
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder.save(model_dir)
            
            # 2. Remove TF model from self to allow pickling
            encoder_ref = self.encoder
            self.encoder = None
            
            # 3. Pickle the rest of the object
            pkl_path = os.path.join(tmp_dir, "dsiae.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self, f)
            
            # Restore encoder
            self.encoder = encoder_ref
            
            # 4. Zip everything into the target filename
            with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add pickle
                zipf.write(pkl_path, arcname="dsiae.pkl")
                # Add TF model directory contents
                if os.path.exists(model_dir):
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, tmp_dir)
                            zipf.write(file_path, arcname=arcname)
        
        print(f"Saved emulator to {filename}")

    @classmethod
    def load(cls, filename: str) -> 'DSIAE':
        """
        Load the emulator from a file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(filename, 'r') as zipf:
                zipf.extractall(tmp_dir)
            
            # 1. Unpickle
            with open(os.path.join(tmp_dir, "dsiae.pkl"), "rb") as f:
                obj = pickle.load(f)
            
            # 2. Reload TF model if it exists
            model_dir = os.path.join(tmp_dir, "tf_model")
            if os.path.exists(model_dir):
                # We need to reconstruct the AutoEncoder wrapper
                # Since we don't have the init params easily available, we rely on the fact
                # that AutoEncoder.load loads the Keras models directly.
                # But we need an AutoEncoder instance first.
                
                # We can create a dummy AutoEncoder instance and then load the weights/models
                # However, AutoEncoder.__init__ builds the model.
                # We can bypass __init__ or use default params if we are just going to overwrite the models.
                
                # Better approach: The AutoEncoder class should have a classmethod to load from disk
                # or we instantiate it with dummy params and then load.
                
                # Let's assume we can instantiate it with minimal params.
                # We need input_dim and latent_dim.
                # obj.data_transformed should be available.
                input_dim = obj.data_transformed.shape[1] if obj.data_transformed is not None else 0
                latent_dim = obj.latent_dim if obj.latent_dim is not None else 2
                
                # Create a blank AutoEncoder instance
                # We use __new__ to bypass __init__ since we are loading the full model structure
                ae = AutoEncoder.__new__(AutoEncoder)
                ae.load(model_dir)
                obj.encoder = ae
            
            return obj

class AutoEncoder:
    def __init__(self, input_dim: int, latent_dim: int = 2, 
                 hidden_dims: tuple = (128, 64), lr: float = 1e-3,
                 activation: str = 'relu', loss: str = 'Huber', 
                 dropout_rate: float = 0.0, random_state: int = 0) -> None:
        """
        Initialize AutoEncoder.

        Args:
            input_dim: Input feature dimension.
            latent_dim: Latent space dimension.
            hidden_dims: Tuple of hidden layer sizes for encoder (reversed for decoder).
            lr: Learning rate.
            activation: Activation function name.
            loss: Loss function name.
            dropout_rate: Dropout rate (0.0-1.0).
            random_state: Random seed.
        """
        if tf is None:
            raise ImportError("TensorFlow is required for AutoEncoder but not installed.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.activation = activation
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self._build_model()

    # Build encoder/decoder
    def _build_model(self):
        tf.keras.backend.set_floatx('float32')
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(self.input_dim,))
        x = encoder_inputs
        for h in self.hidden_dims:
            x = tf.keras.layers.Dense(h, activation=self.activation)(x)
            if hasattr(self, 'dropout_rate') and self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        latent = tf.keras.layers.Dense(self.latent_dim, name='latent')(x)
        self.encoder = tf.keras.Model(encoder_inputs, latent, name='encoder')

        # Decoder
        decoder_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = decoder_inputs
        for h in reversed(self.hidden_dims):
            x = tf.keras.layers.Dense(h, activation=self.activation)(x)
            if hasattr(self, 'dropout_rate') and self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.input_dim, activation=None)(x)
        self.decoder = tf.keras.Model(decoder_inputs, outputs, name='decoder')

        # Autoencoder model
        ae_inputs = encoder_inputs
        ae_outputs = self.decoder(self.encoder(ae_inputs))
        self.model = tf.keras.Model(ae_inputs, ae_outputs, name='autoencoder')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=self.loss)


    def fit(self, X: np.ndarray, X_val: Optional[np.ndarray] = None, 
            epochs: int = 100, batch_size: int = 32, 
            validation_split: float = 0.1, early_stopping: bool = True,
            patience: int = 10, lr_schedule: Optional[Any] = None, 
            verbose: int = 2, sample_weight: Optional[np.ndarray] = None,
            validation_sample_weight: Optional[np.ndarray] = None) -> Any:
        """
        Train the autoencoder.

        Args:
            X: Training data.
            X_val: Validation data (optional).
            epochs: Max epochs.
            batch_size: Batch size.
            validation_split: Validation split fraction (if X_val is None).
            early_stopping: Enable early stopping.
            patience: Early stopping patience.
            lr_schedule: Learning rate scheduler callback.
            verbose: Verbosity level.
            sample_weight: Training sample weights.
            validation_sample_weight: Validation sample weights.

        Returns:
            Training history.
        """
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ))
        if lr_schedule is not None:
            callbacks.append(lr_schedule)


        # Train
        history = self.model.fit(
            X, X,
            sample_weight=sample_weight,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def encode(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Encode input data to latent representation.
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or pd.Series
            Input data to encode to latent space.
            
        Returns
        -------
        np.ndarray
            Latent representation with shape (n_samples, latent_dim).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.reshape(1,-1).astype(np.float32)
        return self.encoder(X, training=False)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation back to input space.
        
        Parameters
        ----------
        Z : np.ndarray
            Latent representation with shape (n_samples, latent_dim).
            
        Returns
        -------
        np.ndarray
            Reconstructed data with shape (n_samples, input_dim).
        """
        #X_hat = self.decoder.predict(Z, verbose=0,)
        X_hat = self.decoder(Z,training=False)
        return X_hat


    def save(self, folder: str) -> None:
        """
        Save trained models to disk.
        """
        os.makedirs(folder, exist_ok=True)
        self.encoder.save(os.path.join(folder, 'encoder.keras'))
        self.decoder.save(os.path.join(folder, 'decoder.keras'))
        self.model.save(os.path.join(folder, 'autoencoder.keras'))

    def load(self, folder: str) -> None:
        """
        Load trained models from disk.
        """
        self.encoder = tf.keras.models.load_model(os.path.join(folder, 'encoder.keras'))
        self.decoder = tf.keras.models.load_model(os.path.join(folder, 'decoder.keras'))
        self.model = tf.keras.models.load_model(os.path.join(folder, 'autoencoder.keras'))


    @staticmethod
    def hyperparam_search(X: np.ndarray, latent_dims: List[int] = [2, 3, 5],
                          hidden_dims_list: List[tuple] = [(64, 32), (128, 64)],
                          lrs: List[float] = [1e-2, 1e-3], epochs: int = 50,
                          batch_size: int = 32, random_state: int = 42) -> Dict[tuple, float]:
        """
        Perform grid search over autoencoder hyperparameters.
        
        Systematically evaluates different combinations of latent dimensions,
        network architectures, and learning rates to find optimal configurations
        based on validation loss performance.
        
        Parameters
        ----------
        X : np.ndarray
            Training data for hyperparameter optimization.
            
        latent_dims : list of int, default [2, 3, 5]
            Latent space dimensions to evaluate.
            
        hidden_dims_list : list of tuple, default [(64, 32), (128, 64)]
            Network architectures to test. Each tuple specifies hidden layer sizes.
            
        lrs : list of float, default [1e-2, 1e-3]
            Learning rates to evaluate.
            
        epochs : int, default 50
            Training epochs for each configuration.
            
        batch_size : int, default 32
            Batch size for training.
            
        random_state : int, default 42
            Random seed for reproducible train/validation splits.
            
        Returns
        -------
        dict
            Mapping from (latent_dim, hidden_dims, lr) tuples to validation loss values.
            Lower values indicate better performance.
            
        Notes
        -----
        Uses 10% of data for validation via train_test_split. Each configuration
        is trained independently with early stopping disabled to ensure fair
        comparison across hyperparameter combinations.
        
        Examples
        --------
        >>> results = AutoEncoder.hyperparam_search(X_train, epochs=100)
        >>> best_params = min(results.keys(), key=results.get)
        >>> print(f"Best configuration: {best_params}")
        """
        results = {}
        X_train, X_val = train_test_split(X, test_size=0.1, random_state=random_state)
        for ld in latent_dims:
            for hd in hidden_dims_list:
                for lr in lrs:
                    print(f"Training AE: latent_dim={ld}, hidden_dims={hd}, lr={lr}")
                    ae = AutoEncoder(input_dim=X.shape[1], latent_dim=ld, hidden_dims=hd, lr=lr)
                    history = ae.fit(X_train, X_val=X_val, epochs=epochs, batch_size=batch_size,verbose=0)
                    val_loss = history.history['val_loss'][-1]
                    results[(ld, hd, lr)] = val_loss
                    print(f"Validation loss: {val_loss:.4f}")
        return results
    



# Efficient pairwise L2 distances
def pairwise_distances(x, y, eps=1e-12):
    x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    dist_sq = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
    dist_sq = tf.maximum(dist_sq, eps)
    return tf.sqrt(dist_sq)



# Energy distance core function
def energy_distance_optimized(y_true, y_pred):
    d_xy = pairwise_distances(y_true, y_pred)
    cross = 2.0 * tf.reduce_mean(d_xy)

    d_xx = pairwise_distances(y_true, y_true)
    d_yy = pairwise_distances(y_pred, y_pred)

    return cross - tf.reduce_mean(d_xx) - tf.reduce_mean(d_yy)


# UTILITY FUNCTIONS FOR DISTRIBUTION-AWARE LOSSES
def maximum_mean_discrepancy(x, y, kernel='rbf', sigma=1.0):
    """Compute Maximum Mean Discrepancy between two distributions."""
    if kernel == 'rbf':
        # RBF kernel k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        
        # Pairwise distances
        xx = x_norm + tf.transpose(x_norm) - 2.0 * tf.matmul(x, x, transpose_b=True)
        yy = y_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(y, y, transpose_b=True)
        xy = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
        
        # Apply RBF kernel
        k_xx = tf.exp(-xx / (2 * sigma**2))
        k_yy = tf.exp(-yy / (2 * sigma**2))
        k_xy = tf.exp(-xy / (2 * sigma**2))
        
    elif kernel == 'linear':
        k_xx = tf.matmul(x, x, transpose_b=True)
        k_yy = tf.matmul(y, y, transpose_b=True)
        k_xy = tf.matmul(x, y, transpose_b=True)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    # MMD calculation
    mmd = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2.0 * tf.reduce_mean(k_xy)
    return tf.maximum(mmd, 0.0)  # Ensure non-negative


def wasserstein_distance_sliced(x, y, num_projections=50):
    """Approximate Wasserstein-1 distance using sliced Wasserstein distance."""
    # Generate random projections
    d = tf.shape(x)[1]
    theta = tf.random.normal([d, num_projections])
    theta = theta / tf.norm(theta, axis=0, keepdims=True)
    
    # Project data onto random directions
    x_proj = tf.matmul(x, theta)  # [batch_size, num_projections]
    y_proj = tf.matmul(y, theta)  # [batch_size, num_projections]
    
    # Sort projections
    x_sorted = tf.sort(x_proj, axis=0)
    y_sorted = tf.sort(y_proj, axis=0)
    
    # Compute L1 distance between sorted projections
    distances = tf.reduce_mean(tf.abs(x_sorted - y_sorted), axis=0)
    return tf.reduce_mean(distances)


def correlation_loss(x, y):
    """Penalize differences in correlation structure between datasets."""
    # Center the data
    x_centered = x - tf.reduce_mean(x, axis=0, keepdims=True)
    y_centered = y - tf.reduce_mean(y, axis=0, keepdims=True)
    
    # Compute correlation matrices
    x_cov = tf.matmul(x_centered, x_centered, transpose_a=True) / tf.cast(tf.shape(x)[0] - 1, tf.float32)
    y_cov = tf.matmul(y_centered, y_centered, transpose_a=True) / tf.cast(tf.shape(y)[0] - 1, tf.float32)
    
    # Normalize to get correlation
    x_std = tf.sqrt(tf.diag_part(x_cov))
    y_std = tf.sqrt(tf.diag_part(y_cov))
    
    x_corr = x_cov / (tf.expand_dims(x_std, 0) * tf.expand_dims(x_std, 1))
    y_corr = y_cov / (tf.expand_dims(y_std, 0) * tf.expand_dims(y_std, 1))
    
    # Frobenius norm of difference
    return tf.reduce_mean(tf.square(x_corr - y_corr))



if tf is not None:
    LossBase = tf.keras.losses.Loss
else:
    class LossBase:
        def __init__(self, name=None, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            pass

@register_keras_serializable(package="pyemu_emulators", name="EnergyLoss")
class EnergyLoss(LossBase):
    """
    Energy distance loss combining MSE reconstruction with energy distance.
    
    The energy distance measures dissimilarity between probability distributions
    and helps ensure the reconstructed samples preserve the overall data distribution.
    """

    def __init__(self, lambda_energy=1e-2, name="energy_loss"):
        super().__init__(name=name)
        self.lambda_energy = lambda_energy

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ed = energy_distance_optimized(y_true, y_pred)
        return mse + self.lambda_energy * ed

    def get_config(self):
        return {
            "lambda_energy": self.lambda_energy,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="pyemu_emulators", name="MMDLoss")
class MMDLoss(LossBase):
    """
    Maximum Mean Discrepancy loss for distribution matching.
    
    MMD measures the distance between distributions in a reproducing kernel
    Hilbert space. More computationally efficient than energy distance.
    """

    def __init__(self, lambda_mmd=1e-2, kernel='rbf', sigma=1.0, name="mmd_loss"):
        super().__init__(name=name)
        self.lambda_mmd = lambda_mmd
        self.kernel = kernel
        self.sigma = sigma

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mmd = maximum_mean_discrepancy(y_true, y_pred, kernel=self.kernel, sigma=self.sigma)
        return mse + self.lambda_mmd * mmd

    def get_config(self):
        return {
            "lambda_mmd": self.lambda_mmd,
            "kernel": self.kernel,
            "sigma": self.sigma,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="pyemu_emulators", name="WassersteinLoss")
class WassersteinLoss(LossBase):
    """
    Sliced Wasserstein distance loss for distribution matching.
    
    Uses random projections to approximate the Wasserstein-1 distance,
    which is particularly effective for high-dimensional distributions.
    """

    def __init__(self, lambda_w=1e-2, num_projections=50, name="wasserstein_loss"):
        super().__init__(name=name)
        self.lambda_w = lambda_w
        self.num_projections = num_projections

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        w_dist = wasserstein_distance_sliced(y_true, y_pred, self.num_projections)
        return mse + self.lambda_w * w_dist

    def get_config(self):
        return {
            "lambda_w": self.lambda_w,
            "num_projections": self.num_projections,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="pyemu_emulators", name="StatisticalLoss")
class StatisticalLoss(LossBase):
    """
    Multi-component statistical loss for comprehensive distribution matching.
    
    Combines reconstruction error with multiple statistical measures:
    - Moment matching (mean, variance, skewness, kurtosis)
    - Correlation structure preservation
    - Optional distribution distance (MMD or Energy)
    """

    def __init__(self, lambda_moments=1e-2, lambda_corr=1e-3, lambda_dist=1e-3,
                 dist_type='mmd', mmd_sigma=1.0, name="statistical_loss"):
        super().__init__(name=name)
        self.lambda_moments = lambda_moments
        self.lambda_corr = lambda_corr
        self.lambda_dist = lambda_dist
        self.dist_type = dist_type
        self.mmd_sigma = mmd_sigma

    def call(self, y_true, y_pred):
        # Reconstruction loss
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Moment matching
        moments_loss = 0.0
        for moment in range(1, 5):  # mean, variance, skewness, kurtosis
            true_moment = tf.reduce_mean(tf.pow(y_true - tf.reduce_mean(y_true, axis=0), moment), axis=0)
            pred_moment = tf.reduce_mean(tf.pow(y_pred - tf.reduce_mean(y_pred, axis=0), moment), axis=0)
            moments_loss += tf.reduce_mean(tf.square(true_moment - pred_moment))
        
        # Correlation structure loss
        corr_loss = correlation_loss(y_true, y_pred)
        
        # Distribution distance
        if self.dist_type == 'mmd':
            dist_loss = maximum_mean_discrepancy(y_true, y_pred, sigma=self.mmd_sigma)
        elif self.dist_type == 'energy':
            dist_loss = energy_distance_optimized(y_true, y_pred)
        else:
            dist_loss = 0.0
        
        total_loss = (mse + 
                     self.lambda_moments * moments_loss + 
                     self.lambda_corr * corr_loss + 
                     self.lambda_dist * dist_loss)
        
        return total_loss

    def get_config(self):
        return {
            "lambda_moments": self.lambda_moments,
            "lambda_corr": self.lambda_corr,
            "lambda_dist": self.lambda_dist,
            "dist_type": self.dist_type,
            "mmd_sigma": self.mmd_sigma,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="pyemu_emulators", name="AdaptiveLoss")
class AdaptiveLoss(LossBase):
    """
    Adaptive loss that balances reconstruction and distribution terms dynamically.
    
    Automatically adjusts the weighting between reconstruction and distribution
    preservation based on their relative magnitudes during training.
    """

    def __init__(self, base_lambda=1e-2, adaptation_rate=0.01, min_lambda=1e-5, 
                 max_lambda=1e-1, name="adaptive_loss"):
        super().__init__(name=name)
        self.base_lambda = base_lambda
        self.adaptation_rate = adaptation_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.current_lambda = tf.Variable(base_lambda, trainable=False, name="adaptive_lambda")

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ed = energy_distance_optimized(y_true, y_pred)
        
        # Adaptive weighting based on relative magnitudes
        mse_magnitude = tf.stop_gradient(mse)
        ed_magnitude = tf.stop_gradient(ed)
        
        # Update lambda to balance the terms
        ratio = ed_magnitude / (mse_magnitude + 1e-8)
        target_lambda = self.base_lambda * tf.clip_by_value(ratio, 0.1, 10.0)
        
        # Smooth update of lambda
        self.current_lambda.assign(
            self.current_lambda * (1 - self.adaptation_rate) + 
            target_lambda * self.adaptation_rate
        )
        
        # Clip lambda to reasonable bounds
        clipped_lambda = tf.clip_by_value(self.current_lambda, self.min_lambda, self.max_lambda)
        
        return mse + clipped_lambda * ed

    def get_config(self):
        return {
            "base_lambda": self.base_lambda,
            "adaptation_rate": self.adaptation_rate,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="custom_losses")
class PerSampleMSE(LossBase):
    def __init__(self, name="per_sample_mse"):
        super().__init__(reduction="none", name=name)

    def call(self, y_true, y_pred):
        # shape (batch,)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=1)


    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def create_observation_weights(data: Union[pd.DataFrame, np.ndarray], 
                             observed_values: List[float], 
                             critical_features: List[int],
                             weight_type: str = 'inverse_distance',
                             temperature: float = 1.0,
                             normalize: bool = True,
                             clip_range: tuple = (0.1, 10.0)) -> np.ndarray:
    """
    Create sample weights based on proximity to observed values.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Training data with shape (n_samples, n_features)
    observed_values : list of float
        Target observed values at critical features
    critical_features : list of int
        Column indices of critical observation features  
    weight_type : str, default 'inverse_distance'
        Type of weighting: 'inverse_distance', 'gaussian', 'exponential'
    temperature : float, default 1.0
        Temperature parameter for weight decay (lower = sharper weighting)
    normalize : bool, default True
        Whether to normalize weights to mean = 1.0
    clip_range : tuple, default (0.1, 10.0)
        Range to clip extreme weights (min, max)
        
    Returns
    -------
    np.ndarray
        Sample weights with shape (n_samples,)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    observed_values = np.array(observed_values)
    sample_weights = np.ones(len(data))
    
    for i in range(len(data)):
        sample_values = data[i][critical_features]
        
        if weight_type == 'inverse_distance':
            distance = np.sqrt(np.sum((sample_values - observed_values)**2))
            weight = 1.0 / (1.0 + distance / temperature)
            
        elif weight_type == 'gaussian':
            distance_sq = np.sum((sample_values - observed_values)**2)
            weight = np.exp(-distance_sq / (2.0 * temperature**2))
            
        elif weight_type == 'exponential':
            distance = np.sqrt(np.sum((sample_values - observed_values)**2))
            weight = np.exp(-distance / temperature)
            
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
            
        sample_weights[i] = weight
    
    if normalize:
        sample_weights = sample_weights / np.mean(sample_weights)
    
    if clip_range is not None:
        sample_weights = np.clip(sample_weights, clip_range[0], clip_range[1])
    
    return sample_weights


def create_pest_observation_weights(pst: 'Pst', 
                                   emulator_data: pd.DataFrame,
                                   weight_scaling: float = 1.0,
                                   **kwargs) -> np.ndarray:
    """
    Create sample weights using PEST observation data.
    
    Parameters
    ----------
    pst : Pst
        PEST control file object with observation data
    emulator_data : pd.DataFrame
        Training data for the emulator
    weight_scaling : float, default 1.0
        Overall scaling factor for weights
    **kwargs
        Additional arguments passed to create_observation_weights
        
    Returns
    -------
    np.ndarray
        Sample weights based on PEST observations
    """
    obs_data = pst.observation_data
    
    # Map observation names to column indices
    critical_features = []
    observed_values = []
    
    for obs_name in obs_data.index:
        if obs_name in emulator_data.columns:
            col_idx = emulator_data.columns.get_loc(obs_name)
            critical_features.append(col_idx)
            observed_values.append(obs_data.loc[obs_name, 'obsval'])
    
    if len(critical_features) == 0:
        raise ValueError("No matching observations found between PST and emulator data")
    
    weights = create_observation_weights(
        emulator_data, observed_values, critical_features, **kwargs
    )
    
    return weights * weight_scaling



def create_distribution_loss(loss_type='energy', **kwargs):
    """
    Factory function to create distribution-aware loss functions.
    
    Parameters
    ----------
    loss_type : str
        Type of loss function to create:
        - 'energy': EnergyLoss (default, robust but computationally expensive)
        - 'mmd': MMDLoss (efficient, good for high-dim data)
        - 'wasserstein': WassersteinLoss (good for smooth distributions)
        - 'statistical': StatisticalLoss (comprehensive statistical matching)
        - 'adaptive': AdaptiveLoss (automatically balances terms)
        - 'mse': Standard MSE (no distribution matching)
        - 'huber': Huber loss (robust to outliers, no distribution matching)
    **kwargs : dict
        Additional parameters specific to each loss type
        
    Returns
    -------
    tf.keras.losses.Loss
        Configured loss function
        
    Examples
    --------
    >>> # Energy loss with custom weighting
    >>> loss = create_distribution_loss('energy', lambda_energy=1e-3)
    >>> 
    >>> # MMD loss with RBF kernel
    >>> loss = create_distribution_loss('mmd', lambda_mmd=1e-2, sigma=2.0)
    >>> 
    >>> # Statistical loss with all components
    >>> loss = create_distribution_loss('statistical', 
    ...                               lambda_moments=1e-2, 
    ...                               lambda_corr=1e-3,
    ...                               lambda_dist=5e-3)
    """
    if loss_type == 'energy':
        return EnergyLoss(**kwargs)
    elif loss_type == 'mmd':
        return MMDLoss(**kwargs)
    elif loss_type == 'wasserstein':
        return WassersteinLoss(**kwargs)
    elif loss_type == 'statistical':
        return StatisticalLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveLoss(**kwargs)
    elif loss_type == 'per_sample_mse':
        return PerSampleMSE(**kwargs)
    elif loss_type == 'mse':
        return 'mse'
    elif loss_type == 'huber':
        return tf.keras.losses.Huber(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Supported types: energy, mmd, wasserstein, statistical, "
                        f"adaptive, per_sample_mse, mse, huber")