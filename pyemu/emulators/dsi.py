"""
Data Space Inversion (DSI) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import inspect
from pyemu.utils.helpers import dsi_forward_run, series_to_insfile
import os
import shutil
from pyemu.pst.pst_handler import Pst
from pyemu.en import ObservationEnsemble,ParameterEnsemble
from .base import Emulator
from .transformers import AutobotsAssemble, RowWiseMinMaxScaler

class DSI(Emulator):
    """
    Data Space Inversion (DSI) emulator class. Based on DSI as described in Sun &
    Durlofsky (2017) and Sun et al (2017).
        
    """

    def __init__(self, 
                pst=None,
                data=None,
                transforms=None,
                energy_threshold=1.0,
                rowwise_groups=None,
                rowwise_fit_groups=None,
                feature_range=(-1, 1),
                verbose=False):
        """
        Initialize the DSI emulator.
        
        If rowwise_groups is provided, training data are row-wise scaled per-group
        before SVD. Predictions are returned in scaled space and then inverse-scaled
        using per-row parameters derived from truth values found in
        pst.observation_data.

        Parameters
        ----------
        pst : Pst, optional
            A Pst object. If provided, the emulator will be initialized with the
            information from the Pst object.
        data : DataFrame or ObservationEnsemble, optional
            An ensemble of simulated observations. If provided, the emulator will
            be initialized with the information from the ensemble.
        transforms : list of dict, optional
            List of transformation specifications. Each dict should have:
            - 'type': str - Type of transformation (e.g.,'log10', 'normal_score').
            - 'columns': list of str,optional - Columns to apply the transformation to. If not supplied, transformation is applied to all columns.
            - Additional kwargs for the transformation (e.g., 'quadratic_extrapolation' for normal score transform).
            Example:
            transforms = [
                {'type': 'log10', 'columns': ['obs1', 'obs2']},
                {'type': 'normal_score', 'quadratic_extrapolation': True}
            ]
            Default is None, which means no transformations will be applied.
        energy_threshold : float, optional 
            The energy threshold for the SVD. Default is 1.0, no truncation.
        rowwise_groups : dict, optional
            Dictionary mapping groups to column lists for row-wise scaling.
        rowwise_fit_groups : dict, optional
            Dictionary mapping groups to column lists for fitting row-wise scalers.
        feature_range : tuple, optional
            Feature range for row-wise scaling. Default is (-1, 1).
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
        """

        super().__init__(verbose=verbose)

        if isinstance(pst,Pst):
            self.observation_data = pst.observation_data.copy() if pst is not None else None
        elif isinstance(pst, pd.DataFrame):
            self.observation_data = pst.copy()
        else:
             self.observation_data = None

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

        # Row-wise scaling config (optional)
        self.rowwise_groups = rowwise_groups
        self.rowwise_fit_groups = rowwise_fit_groups if rowwise_fit_groups is not None else rowwise_groups
        self.feature_range = feature_range
        self._rowwise_train_scaler = None

        self.fitted = False

        self.data_transformed = self._prepare_training_data()

        # If row-wise scaling is enabled and truth is available, pre-fit truth scaler once
        self._truth_rowwise_scaler = None
        if self.rowwise_groups is not None and self.observation_data is not None:
            try:
                self._truth_rowwise_scaler = self._prefit_truth_rowwise_scaler()
                self._truth_row_index = 'truth'
                self.logger.statement("prefit truth row-wise scaler from self.observation_data")
            except Exception as ex:
                self.logger.statement(f"warning: could not prefit truth scaler: {ex}")
        else:
            self._truth_row_index = 'truth'
            
        self.decision_variable_names = None #used for DSIVC
        
    def _prepare_training_data(self):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        self : DSI
            The DSI emulator instance.
            
        Returns
        -------
        tuple
            Processed data ready for model fitting.
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
            self.transformer_pipeline = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()

        # 2) Optional row-wise scaling for training
        if self.rowwise_groups is not None:
            self.logger.statement("applying row-wise min-max scaling (training)")
            self._rowwise_train_scaler = RowWiseMinMaxScaler(
                feature_range=self.feature_range,
                groups=self.rowwise_groups,
                fit_groups=self.rowwise_fit_groups
            )
            self._rowwise_train_scaler.fit(self.data_transformed)
            self.data_transformed = self._rowwise_train_scaler.transform(self.data_transformed)
    
        return self.data_transformed
        
    def _build_truth_rowwise_scaler(self, truth_df_transformed):
        """Build a RowWiseMinMaxScaler fitted on provided truth values."""
        scaler = RowWiseMinMaxScaler(
            feature_range=self.feature_range,
            groups=self.rowwise_groups,
            fit_groups=self.rowwise_fit_groups,
        )
        scaler.fit(truth_df_transformed)
        return scaler

    def _prefit_truth_rowwise_scaler(self):
        """Fit a truth-based RowWiseMinMaxScaler once, using pst.observation_data.

        Uses only rowwise_fit_groups columns (intersected with availability) so that
        future/forecast columns are not required in truth.
        """
        if self.rowwise_groups is None:
            return None

        obsdf = self.observation_data
        #obsdf = obsdf.loc[obsdf.weight > 0]
        if obsdf is None or 'obsval' not in obsdf.columns:
            raise ValueError("observation_data with 'obsval' column is required to prefit truth scaler")

        # Determine which columns to use from truth: union of fit groups
        fit_cols_union = []
        if self.rowwise_fit_groups is not None:
            for _, cols in self.rowwise_fit_groups.items():
                fit_cols_union.extend(cols)
        # Intersect with available columns in training-transformed data and truth index
        available_cols = [c for c in fit_cols_union if c in self.data_transformed.columns and c in obsdf.index]
        if not available_cols:
            raise ValueError("No overlapping columns between fit_groups, training data, and truth index for prefit")

        # Build single-row truth DataFrame
        truth_df = obsdf.loc[available_cols, 'obsval'].to_frame().T
        truth_df.index = ['truth']
        # Apply feature transforms to truth
        truth_transformed = self.transformer_pipeline.transform(truth_df)

        # Trim fit groups per availability
        fit_groups = {}
        if self.rowwise_fit_groups is not None:
            for g, cols in self.rowwise_fit_groups.items():
                fit_groups[g] = [c for c in cols if c in truth_transformed.columns]
        empty = [g for g, cols in fit_groups.items() if len(cols) == 0]
        if empty:
            raise ValueError(f"No fit columns available for groups during prefit: {empty}")

        scaler = RowWiseMinMaxScaler(
            feature_range=self.feature_range,
            groups=self.rowwise_groups,
            fit_groups=fit_groups,
        )
        scaler.fit(truth_transformed)
        return scaler

    def compute_projection_matrix(self, energy_threshold=None):
        """
        Compute the projection matrix using SVD.
        
        Parameters
        ----------
        energy_threshold : float, optional
            Energy threshold for truncation. Default is None, which uses the threshold from initialization.
            
        Returns
        -------
        None
        """
        self.logger.statement("normalizing data")
        # normalize the data by subtracting the mean and dividing by the standard deviation
        X = self.data_transformed.copy()
        deviations = X - X.mean()
        z = deviations / np.sqrt(float(X.shape[0] - 1))
        if isinstance(z, pd.DataFrame):
            z = z.values

        self.logger.statement("undertaking SVD")
        u, s, v = np.linalg.svd(z, full_matrices=False)
        us = np.dot(v.T, np.diag(s))
        max_n_components = s.shape[0]
        if energy_threshold is None:
            energy_threshold = self.energy_threshold
        if energy_threshold<1.0:
            self.logger.statement("applying energy truncation")
            # compute the cumulative energy of the singular values
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            print(cumulative_energy)
            # find the number of components needed to reach the energy threshold
            num_components = np.argmax(cumulative_energy >= energy_threshold) + 1
            # keep only the first num_components singular values and vectors
            us = us[:, :num_components]
            s = s[:num_components]
            u = u[:, :num_components]
            print(f"Truncated from {max_n_components} to {num_components} components while retaining {energy_threshold*100:.1f}% of variance")
            if num_components<=1:
                print(f"Warning: only {num_components} component retained, you may need to check the data")
        
        self.logger.statement("calculating us matrix")
        
        # store components needed for forward run
        # store mean vector
        self.ovals = self.data_transformed.mean(axis=0)
        # store proj matrix and singular values
        self.pmat = us
        self.s = s
        return
    
    def fit(self):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        self : DSI
            The DSI emulator instance.
            
        Returns
        -------
        self : DSI
            The fitted emulator.
        """
        
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()

        # Compute projection matrix
        self.compute_projection_matrix()
        self.fitted = True
        return self
    
    def predict(self, pvals, pst: Pst = None, truth_selector=None):
        """
        Predict observations in original units. When row-wise scaling is enabled,
        use truth from pst.observation_data to invert scaling.

        Parameters
        ----------
        pvals : array-like
            DSI latent parameters (same length as retained singular values).
        pst : Pst, optional
            If provided (or if self.observation_data exists), used to obtain
            truth values for inverse row-wise scaling.
        truth_selector : callable or list-like, optional
            If provided, used to subset columns per group that are available as
            truth at prediction time (e.g., history window). Should return the
            same columns as rowwise_fit_groups per group, but limited to known
            values.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        if self.transforms is not None and (not hasattr(self, 'transformer_pipeline') or self.transformer_pipeline is None):
            raise ValueError("Emulator must be fitted and have valid transformations before prediction")

        if isinstance(pvals, pd.Series):
            pvals = pvals.values.flatten()
        assert pvals.shape[0] == self.s.shape[0], "pvals must match number of singular values"
        assert pvals.shape[0] == self.pmat.shape[1], "pvals must match number of singular values"

        # Simulate in working space (possibly row-wise scaled)
        sim_vals = self.ovals + np.dot(self.pmat, pvals)
        sim_series = pd.Series(sim_vals, index=self.data_transformed.columns, name="obsval")

        # If no row-wise scaling used, just inverse the feature pipeline and return
        if self.rowwise_groups is None:
            if self.transforms is not None:
                sim_vals = self.transformer_pipeline.inverse(sim_series)
            else:
                sim_series.index.name = 'obsnme'
                sim_vals = sim_series
            self.sim_vals = sim_vals
            return sim_vals

        # Row-wise scaling used: use pre-fitted truth scaler if available, else build once from provided pst
        truth_scaler = self._truth_rowwise_scaler
        if truth_scaler is None:
            if pst is None or getattr(pst, 'observation_data', None) is None:
                raise ValueError("Truth scaler not prefit; provide pst with observation_data to build it once.")
            # Temporarily set observation_data and prefit
            prev_obs = self.observation_data
            self.observation_data = pst.observation_data
            truth_scaler = self._prefit_truth_rowwise_scaler()
            self._truth_rowwise_scaler = truth_scaler
            self.observation_data = prev_obs if prev_obs is not None else self.observation_data

        sim_df_scaled = sim_series.to_frame().T
        # Align row index with the one used when truth scaler was fitted
        sim_df_scaled.index = [getattr(self, '_truth_row_index', 'truth')]
        pred_transformed = truth_scaler.inverse_transform(sim_df_scaled)
        pred_series = pred_transformed.iloc[0]

        # 5) Inverse feature transforms back to original units
        if self.transforms is not None:
            pred_series = self.transformer_pipeline.inverse(pred_series)
        pred_series.index.name = 'obsnme'
        pred_series.name = 'obsval'
        
        self.sim_vals = pred_series
        return pred_series
    
    def check_for_pdc(self):
        """Check for Prior data conflict."""
        #TODO
        return
        
    def prepare_pestpp(self, t_d=None, observation_data=None):
        """
        Prepare PEST++ control files for the emulator.
        
        Parameters
        ----------
        t_d : str, optional
            Template directory path. Must be provided.
        observation_data : pandas.DataFrame, optional
            Observation data to use. If None, uses the data from initialization.
            
        Returns
        -------
        Pst
            PEST++ control file object.
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
        npar = self.s.shape[0]
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
        pvals = np.zeros_like(self.s)

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
        par.loc[dsi_pars,"parval1"] = 0
        par.loc[dsi_pars,"parubnd"] = 10.0
        par.loc[dsi_pars,"parlbnd"] = -10.0
        par.loc[dsi_pars,"partrans"] = "none"
        with open(os.path.join(t_d,"dsi.unc"),'w') as f:
            f.write("START STANDARD_DEVIATION\n")
            for p in dsi_pars:
                f.write("{0} 1.0\n".format(p))
            f.write("END STANDARD_DEVIATION")
        pst.pestpp_options['parcov'] = "dsi.unc"

        obs = pst.observation_data

        if observation_data is not None:
            self.observation_data = observation_data
        else:
            observation_data = self.observation_data
        assert isinstance(observation_data, pd.DataFrame), "observation_data must be a pandas DataFrame"
        observation_data.obsval = observation_data.obsval.astype(float)
        observation_data.weight = observation_data.weight.astype(float)
        for col in observation_data.columns:
            obs.loc[sim_vals.index,col] = observation_data.loc[:,col]

        # check if any observations are missing
        missing_obs = list(set(obs.index) - set(observation_data.index))
        assert len(missing_obs) == 0, "missing observations: {0}".format(missing_obs)

        pst.control_data.noptmax = 0
        pst.model_command = "python forward_run.py"
        self.logger.log("creating Pst")


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
        
    def prepare_dsivc(self, decvar_names, t_d=None, pst=None, oe=None, track_stack=False, dsi_args=None, percentiles=[0.25,0.75,0.5], mou_population_size=None,ies_exe_path="pestpp-ies"):
        """
        Prepare Data Space Inversion Variable Control (DSIVC) control files.
        
        Parameters
        ----------
        decvar_names : list or str
            Names of decision variables.
        t_d : str, optional
            Template directory path.
        pst : Pst, optional
            PST control file object.
        oe : ObservationEnsemble, optional
            Observation ensemble.
        track_stack : bool, optional
            Whether to track the stack. Default is False.
        dsi_args : dict, optional
            Arguments for DSI.
        percentiles : list, optional
            Percentiles to calculate. Default is [0.25, 0.75, 0.5].
        mou_population_size : int, optional
            Population size for multi-objective optimization.
        ies_exe_path : str, optional
            Path to the PEST++ IES executable. Default is "pestpp-ies".
        Returns
        -------
        Pst
            PEST++ control file object for DSIVC.
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
        #function_source = inspect.getsource(dsivc_forward_run)
        with open(os.path.join(t_d,"dsivc_forward_run.py"),'w') as file:
            #file.write(function_source)
            file.write("from pyemu.utils.helpers import dsivc_forward_run\n")
            file.write("\n\n")
            file.write("if __name__ == \"__main__\":\n")
            #file.write(f"    {function_source.split('(')[0].split('def ')[1]}(ies_exe_path='{ies_exe_path}')\n")
            file.write(f"    dsivc_forward_run(ies_exe_path='{ies_exe_path}')\n")

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