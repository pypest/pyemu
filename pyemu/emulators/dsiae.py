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
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib


class DSIAE(Emulator):
    """
    Data Space Inversion Autoencoder (DSIAE) emulator class. 
        
    """

    def __init__(self, 
                pst: Optional['Pst'] = None,
                data: Optional[Union[pd.DataFrame, 'ObservationEnsemble']] = None,
                transforms: Optional[List[Dict[str, Any]]] = None,
                latent_dim: Optional[int] = None,
                energy_threshold: float = 1.0,
                verbose: bool = False) -> None:
        """
        Initialize the Data Space Inversion Autoencoder (DSIAE) emulator.

        The DSIAE emulator combines dimensionality reduction with neural network 
        autoencoders for parameter estimation and uncertainty quantification in 
        environmental modeling applications.

        Parameters
        ----------
        pst : Pst, optional
            A PEST control file object containing parameter and observation metadata.
            If provided, observation data will be extracted for emulator configuration.
            Default is None.
            
        data : DataFrame or ObservationEnsemble, optional
            Training data containing simulated observations. Can be either:
            - pandas.DataFrame: Direct observation data
            - ObservationEnsemble: pyemu ensemble object (will be converted to DataFrame)
            If provided, all data will be converted to float64 for numerical stability.
            Default is None.
            
        transforms : list of dict, optional
            Preprocessing transformations to apply to the training data. Each dictionary
            should contain transformation specifications with the following structure:
            - 'type': str - Transformation type ('log10', 'normal_score', etc.)
            - 'columns': list of str, optional - Target columns. If omitted, applies to all columns
            - Additional transformation-specific parameters
            
            Example:
                transforms = [
                    {'type': 'log10', 'columns': ['obs1', 'obs2']},
                    {'type': 'normal_score', 'quadratic_extrapolation': True}
                ]
            Default is None (no transformations applied).
            
        latent_dim : int, optional
            Dimensionality of the latent space for the autoencoder. If None, will be
            automatically determined from energy_threshold using PCA analysis.
            Must be positive integer less than the number of observations.
            Default is None.
            
        energy_threshold : float, optional
            Energy threshold for automatic latent dimension selection via SVD/PCA.
            Represents the cumulative explained variance ratio threshold (0.0 to 1.0).
            Only used if latent_dim is None. Values closer to 1.0 retain more information
            but result in higher dimensional latent spaces.
            Default is 1.0 (no truncation).
            
        verbose : bool, optional
            Enable verbose logging output during emulator operations.
            Default is False.

        Raises
        ------
        AssertionError
            If transforms is not a list of dictionaries or None.
            If transform dictionaries are missing required keys.
            If specified columns don't exist in the data.
            If transformation parameters have invalid types.
            
        Notes
        -----
        The emulator must be fitted using the `fit()` method before making predictions.
        Data transformations are applied during the data preparation phase and can be
        inverted during prediction to return results in original scales.
        
        Examples
        --------
        >>> # Basic initialization with data
        >>> emulator = DSIAE(data=observation_data, latent_dim=5)
        >>> 
        >>> # With transformations and automatic dimension selection
        >>> transforms = [{'type': 'log10', 'columns': ['head_obs']}]
        >>> emulator = DSIAE(data=obs_data, transforms=transforms, 
        ...                  energy_threshold=0.95, verbose=True)
        """

        super().__init__(verbose=verbose)

        self.observation_data = pst.observation_data.copy() if pst is not None else None
        #self.__org_parameter_data = pst.parameter_data.copy() if pst is not None else None
        #self.__org_control_data = pst.control_data.copy() #breaks pickling
        if isinstance(data, ObservationEnsemble):
            data = data._df.copy()
        # set all data to be floats
        data = data.astype(float) if data is not None else None
        #self.__org_data = data.copy() if data is not None else None
        self.data = data.copy() if data is not None else None
        self.energy_threshold = energy_threshold
        assert isinstance(transforms, list) or transforms is None, "transforms must be a list of dicts or None"
        if transforms is not None:
            for t in transforms:
                assert isinstance(t, dict), "each transform must be a dict"
                assert 'type' in t, "each transform dict must have a 'type' key"
                if 'columns' in t:
                    assert isinstance(t['columns'], list), "'columns' must be a list of column names"
                    #all columns must be in the data
                    assert all([col in self.data.columns for col in t['columns']]), "some columns in 'columns' are not in the data"
                if t['type'] == 'normal_score':
                    # check for quadratic_extrapolation
                    if 'quadratic_extrapolation' in t:
                        assert isinstance(t['quadratic_extrapolation'], bool), "'quadratic_extrapolation' must be a boolean"
        self.transforms = transforms
        self.latent_dim = latent_dim
        self.fitted = False
        self.data_transformed = self._prepare_training_data()
        self.decision_variable_names = None #used for DSIVC
        
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
            early_stopping: bool = True, random_state: int = 42) -> 'DSIAE':
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
        random_state : int, default 42
            Random seed for reproducibility.
            
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


        # train autoencoder on transformed data
        ae = AutoEncoder(input_dim=X.shape[1], 
                        latent_dim=self.latent_dim,
                        hidden_dims=hidden_dims,
                        lr=lr,
                        random_state=random_state,
                        )
        ae.fit(X,
               validation_split=validation_split,
                epochs=epochs, batch_size=batch_size,
                early_stopping=early_stopping,
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
            pvals = pvals.values.flatten().reshape(1,-1)
        elif isinstance(pvals, np.ndarray) and len(pvals.shape) == 2 and pvals.shape[0] == 1:
            pvals = pvals.flatten().reshape(1,-1)
        elif isinstance(pvals, pd.DataFrame):
            index = pvals.index
            pvals = pvals.values.astype(float)
            
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


class AutoEncoder:
    def __init__(self, input_dim: int, latent_dim: int = 2, 
                 hidden_dims: tuple = (128, 64), lr: float = 1e-3,
                 activation: str = 'relu', loss: str = 'Huber', 
                 random_state: int = 0) -> None:
        """
        Initialize AutoEncoder with specified neural network architecture.
        
        Creates a symmetric encoder-decoder autoencoder architecture where the encoder
        compresses input data to a lower-dimensional latent representation, and the
        decoder reconstructs the original input from this compressed representation.
        The network is designed for dimensionality reduction and feature learning.
        
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature space. Must be positive integer
            representing the number of input features/observations.
            
        latent_dim : int, default 2
            Dimensionality of the latent (compressed) representation. Should be
            significantly smaller than input_dim for meaningful compression.
            Common values range from 2-50 depending on data complexity.
            
        hidden_dims : tuple, default (128, 64)
            Architecture specification for encoder hidden layers. Each integer
            represents the number of neurons in a hidden layer, specified from
            input to latent space. The decoder uses the reverse order automatically.
            Example: (128, 64) creates encoder: input -> 128 -> 64 -> latent
                                      decoder: latent -> 64 -> 128 -> output
                                      
        lr : float, default 1e-3
            Learning rate for the Adam optimizer. Controls the step size during
            gradient descent. Typical values: 1e-4 to 1e-2. Lower values provide
            more stable but slower training.
            
        activation : str, default 'relu'
            Activation function applied to hidden layers. Supported TensorFlow/Keras
            activations include 'relu', 'tanh', 'sigmoid', 'elu', 'swish', etc.
            Output layer uses linear activation for reconstruction tasks.
            
        loss : str, default 'Huber'
            Loss function for training the autoencoder. Options include:
            - 'Huber': Robust to outliers, good for noisy data
            - 'mse' or 'mean_squared_error': Standard for regression
            - 'mae' or 'mean_absolute_error': Less sensitive to outliers
            
        random_state : int, default 0
            Random seed for reproducible model initialization and training.
            Controls TensorFlow random operations for consistent results across runs.
            
        Attributes
        ----------
        encoder : tf.keras.Model
            Encoder network (input -> latent space)
        decoder : tf.keras.Model  
            Decoder network (latent space -> reconstruction)
        model : tf.keras.Model
            Complete autoencoder (input -> latent -> reconstruction)
            
        Notes
        -----
        The autoencoder is compiled with Adam optimizer and ready for training
        upon initialization. The architecture is symmetric - encoder layers are
        mirrored in the decoder for balanced compression/reconstruction capacity.
        
        Random seed affects:
        - Weight initialization
        - Dropout behavior (if added)
        - Training data shuffling
        - Any stochastic operations during training
        
        Examples
        --------
        >>> # Basic autoencoder for 100-dimensional data
        >>> ae = AutoEncoder(input_dim=100, latent_dim=10)
        >>> 
        >>> # Deep autoencoder with custom architecture
        >>> ae = AutoEncoder(input_dim=500, latent_dim=20, 
        ...                  hidden_dims=(256, 128, 64), lr=5e-4)
        """

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.activation = activation
        self.loss = loss
        self.random_state = random_state
        # Set random seeds properly
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self._build_model()

    # Build encoder/decoder
    def _build_model(self):
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(self.input_dim,))
        x = encoder_inputs
        for h in self.hidden_dims:
            x = tf.keras.layers.Dense(h, activation=self.activation)(x)
        latent = tf.keras.layers.Dense(self.latent_dim, name='latent')(x)
        self.encoder = tf.keras.Model(encoder_inputs, latent, name='encoder')

        # Decoder
        decoder_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = decoder_inputs
        for h in reversed(self.hidden_dims):
            x = tf.keras.layers.Dense(h, activation=self.activation)(x)
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
            verbose: int = 2) -> Any:
        """
        Train the autoencoder on provided data with comprehensive training control.
        
        Implements supervised training of the autoencoder using the reconstruction
        objective, where the network learns to minimize the difference between
        input and reconstructed output. Supports validation monitoring, early
        stopping, and learning rate scheduling for optimal training.
        
        Parameters
        ----------
        X : np.ndarray
            Training data with shape (n_samples, input_dim). Will be used as both
            input and target for reconstruction training.
            
        X_val : np.ndarray, optional
            Explicit validation data with same shape as X. If provided, takes
            precedence over validation_split for validation monitoring.
            
        epochs : int, default 100
            Maximum number of training epochs. Training may stop earlier if
            early stopping criteria are met.
            
        batch_size : int, default 32
            Number of samples per training batch. Larger values provide more
            stable gradients but require more memory. Typical range: 16-256.
            
        validation_split : float, default 0.1
            Fraction of training data to reserve for validation when X_val is None.
            Must be between 0.0 and 1.0. Ignored if X_val is provided.
            
        early_stopping : bool, default True
            Whether to monitor validation loss and stop training when it stops
            improving. Helps prevent overfitting and reduces training time.
            
        patience : int, default 10
            Number of epochs to wait for validation improvement before stopping.
            Only used when early_stopping=True. Higher values allow more time
            for recovery from local minima.
            
        lr_schedule : tf.keras.callbacks.Callback, optional
            Learning rate scheduler callback. Common options include ReduceLROnPlateau,
            ExponentialDecay, or CosineDecay for adaptive learning rate adjustment.
            
        verbose : int, default 2
            Training verbosity level:
            - 0: Silent training
            - 1: Progress bar per epoch
            - 2: One line per epoch (recommended)
            
        Returns
        -------
        tf.keras.callbacks.History
            Training history object containing loss and metrics for each epoch.
            Useful for analyzing training progression and convergence behavior.
            
        Notes
        -----
        The method automatically sets up callbacks based on the provided options:
        - EarlyStopping callback when early_stopping=True
        - Custom learning rate schedule when lr_schedule is provided
        
        For autoencoder training, both input and target are the same data (X),
        as the objective is to learn a compressed representation that can
        accurately reconstruct the original input.
        
        Examples
        --------
        >>> # Basic training
        >>> history = ae.fit(X_train, epochs=200)
        >>> 
        >>> # Training with explicit validation set
        >>> history = ae.fit(X_train, X_val=X_test, early_stopping=True, patience=15)
        >>> 
        >>> # Training with learning rate scheduling
        >>> lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        >>> history = ae.fit(X_train, lr_schedule=lr_scheduler, verbose=1)
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
            validation_data=(X_val, X_val) if X_val is not None else None,
            validation_split=validation_split if X_val is None else 0.0,
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
            X = X.values.astype(float)
        elif isinstance(X, pd.Series):
            X = X.values.reshape(1,-1).astype(float)
        return self.encoder.predict(X, verbose=0)

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
        X_hat = self.decoder.predict(Z, verbose=0)
        return X_hat


    def save(self, folder: str) -> None:
        """
        Save trained autoencoder models to disk.
        
        Parameters
        ----------
        folder : str
            Directory path to save the models. Will be created if it doesn't exist.
            
        Notes
        -----
        Saves three separate model files:
        - encoder: The encoder network
        - decoder: The decoder network  
        - autoencoder: The complete autoencoder
        """
        os.makedirs(folder, exist_ok=True)
        self.encoder.save(os.path.join(folder, 'encoder'))
        self.decoder.save(os.path.join(folder, 'decoder'))
        self.model.save(os.path.join(folder, 'autoencoder'))

    def load(self, folder: str) -> None:
        """
        Load trained autoencoder models from disk.
        
        Parameters
        ----------
        folder : str
            Directory path containing the saved models.
            
        Notes
        -----
        Loads the three model components saved by the save() method.
        The models must have been saved with compatible TensorFlow/Keras versions.
        """
        self.encoder = tf.keras.models.load_model(os.path.join(folder, 'encoder'))
        self.decoder = tf.keras.models.load_model(os.path.join(folder, 'decoder'))
        self.model = tf.keras.models.load_model(os.path.join(folder, 'autoencoder'))


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