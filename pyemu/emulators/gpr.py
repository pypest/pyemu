"""
Gaussian Process Regression (GPR) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import shutil
import inspect
from .base import Emulator
from .transformers import AutobotsAssemble
from sklearn.gaussian_process import GaussianProcessRegressor
from pyemu.utils import run

from pyemu.pst import Pst


class GPR(Emulator):
    """
    Gaussian Process Regression (GPR) emulator class.
    
    This class implements a GPR-based emulator that trains separate Gaussian Process
    models for each output variable. It supports various kernel types, feature
    transformations, and provides uncertainty quantification.
    
    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input and output features for training.
    input_names : list of str, optional
        Names of input features to use. If None, all columns in input_data are used.
    output_names : list of str, optional
        Names of output variables to emulate. If None, all columns in output_data are used.
    kernel : sklearn kernel object, optional
        Kernel to use for GP regression. If None, defaults to Matern kernel.
    transforms : list of dict, optional. Defaults to [{'type': 'standard_scaler'}]
    n_restarts_optimizer : int, optional
        Number of restarts for kernel hyperparameter optimization. Default is 10.
    return_std : bool, optional
        Whether to return prediction uncertainties. Default is True.
    verbose : bool, optional
        Enable verbose logging. Default is True.
    """
    
    def __init__(self, 
                 data,
                 input_names=None,
                 output_names=None,
                 kernel=None,
                 transforms=[{'type': 'standard_scaler'}],
                 n_restarts_optimizer=10,
                 return_std=True,
                 verbose=True):
        """Initialize the GPR emulator."""
        
        super().__init__(verbose=verbose)
        
        # Store initialization parameters
        # check data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        self.data = data.copy()

        # Check input and output names
        # check input_names and output_names are lists or None
        if input_names is not None and not isinstance(input_names, list):
            raise ValueError("input_names must be a list or None")
        if output_names is not None and not isinstance(output_names, list):
            raise ValueError("output_names must be a list or None")
        self.input_names = input_names
        self.output_names = output_names

        self.kernel = kernel
        self.transforms = transforms
        self.n_restarts_optimizer = n_restarts_optimizer
        self.return_std = return_std
        
        # Initialize data
        self.data = data
        
        # Model storage
        self.models = {}
        self.model_info = None
        self.verification_results = {}
        
        # PEST++ integration
        self.template_dir = None
        
        # Validate transforms parameter
        if transforms is not None:
            self._validate_transforms(transforms)
            self._validate_transforms_for_gpr()
         
    def _validate_transforms_for_gpr(self):
        """Validate transforms parameter for GPR. Make sure transforms are only applied to input data."""
        # Validate transforms parameter
        transforms = self.transforms
        if transforms is not None:
            # For the speicif case of GPR, we only transform input data    
            for t in transforms:
                if 'columns' in t:
                    # check if any columns are in output_names
                    if self.output_names is not None:
                        common_cols = set(t['columns']).intersection(self.output_names)
                        if common_cols:
                            self.logger.statement(f"Transform {t['type']} will not be applied to output columns: {common_cols}")
                            # remove these columns from transforms
                            t['columns'] = [col for col in t['columns'] if col not in common_cols]
                            if not t['columns']:
                                self.logger.statement(f"Transform {t['type']} has no columns left after removing output columns: {common_cols}")
                                # remove this transform
                                self.logger.statement(f"Removing transform {t['type']} as it has no columns left")
                                self.transforms.remove(t)
                else:
                    self.logger.statement(f"Transform {t['type']} has no specified columns, applying to all input columns")
                    t['columns'] = self.input_names if self.input_names is not None else []
        return transforms   

#    def _combine_input_output_data(self, input_data, output_data):
#        """Combine input and output data into a single DataFrame."""
#        if input_data.shape[0] != output_data.shape[0]:
#            raise ValueError("Input and output data must have the same number of rows")
#        
#        combined = input_data.copy()
#        for col in output_data.columns:
#            if col not in combined.columns:
#                combined[col] = output_data[col]
#            else:
#                self.logger.statement(f"Warning: column '{col}' exists in both input and output data, using output data")
#                combined[col] = output_data[col]
#        
#        return combined
    
    def _setup_kernel(self):
        """Set up the GP kernel if not provided."""
        if self.kernel is None:
            try:
                from sklearn.gaussian_process.kernels import Matern,ConstantKernel,RBF
                self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                                                                length_scale=np.ones(len(self.input_names)) * 2.0,
                                                                length_scale_bounds=(1e-4, 1e4),
                                                                nu=1.5)
                self.logger.statement("Using default Matern kernel")
            except ImportError:
                raise ImportError("scikit-learn is required for GPR emulator")

        # Log kernel hyperparameters
        self.logger.statement(f"Using kernel: {self.kernel}")

    
    def _prepare_training_data(self):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        self : GPR
            The GPR emulator instance containing the data and configuration.
            
        Returns
        -------
        pandas.DataFrame
            Processed data ready for model fitting.
        """

        if self.data is None:
            raise ValueError("No data provided and no data stored in the emulator")
        data = self.data
        
        # Apply feature transformations if specified
        if self.transforms is not None:
            self._validate_transforms_for_gpr()
            self.logger.statement("applying feature transforms")
            self.data_transformed = self._fit_transformer_pipeline(data, self.transforms)
        else:
            # Still need to set up a dummy transformer for consistency
            from .transformers import AutobotsAssemble
            self.transformer_pipeline = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()
        
        return self.data_transformed
    

    def fit(self):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        self: GPR
            The GPR emulator instance containing the data and configuration.
            
        Returns
        -------
        self : GPR
            Fitted GPR emulator instance.
        """
        
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()
        if self.kernel is None:
            self._setup_kernel()
        # transformed input data
        X_transformed = self.data_transformed.loc[:,self.input_names].copy()
        y_transformed = self.data_transformed.loc[:,self.output_names].copy() #Note that these are actualy not transformed

        assert X_transformed.shape[0] == y_transformed.shape[0], \
            "Input and output data must have the same number of rows"
        assert X_transformed.shape[1] > 0, "Input data must have at least one feature"
        assert y_transformed.shape[1] > 0, "Output data must have at least one variable"

        # Create and fit separate GPR model for each output
        self.gpr_models = {}
        for output_name in self.output_names:
            gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                #alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
                #random_state=self.random_state
            )
            
            # Fit the GPR model for this output
            gpr.fit(X_transformed.loc[:,self.input_names].values, y_transformed.loc[:,output_name].values)
            self.gpr_models[output_name] = gpr
        
        self.fitted = True
        return self

    def predict(self, X, return_std=False):
        """
        Make predictions using the fitted GPR emulators.

        Parameters
        ----------
        X : pandas.DataFrame 
            Input features for prediction
        return_std : bool, default False
            Whether to return prediction standard deviation

        Returns
        -------
        predictions : pandas.DataFrame
            Predicted values for each output
        std : pandas.DataFrame, optional
            Prediction standard deviations (if return_std=True)
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before making predictions")
        
        if not hasattr(self, 'transformer_pipeline') or self.transformer_pipeline is None:
            raise ValueError("Emulator must be fitted and have valid transformations before prediction")
        
        # Apply same transforms as training data
        X_transformed = self.transformer_pipeline.transform(X.copy())

        
        # Make predictions for each output
        predictions_dict = {}
        std_dict = {}
        
        for output_name in self.output_names:
            gpr = self.gpr_models[output_name]
            
            if return_std:
                pred, std = gpr.predict(X_transformed.values, return_std=True)
                predictions_dict[output_name] = pred
                std_dict[output_name] = std
            else:
                pred = gpr.predict(X_transformed.values)
                predictions_dict[output_name] = pred
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_dict, index=X.index)
        
        if return_std:
            std_df = pd.DataFrame(std_dict, index=X.index)
            return predictions_df, std_df
        else:
            return predictions_df



        


    
    def _get_emulator_parameters(self, pst=None):
        """
        Get the parameters (inputs) for the GPR emulator.
        For GPR, inputs are typically a subset of parameters from a process model's PST file.
        Returns a DataFrame with columns: parnme, parval1, parlbnd, parubnd, pargp
        """
        if pst is None:
             raise ValueError("GPR._get_emulator_parameters requires a valid 'pst' argument "
                              "(pyemu.Pst object) to identify decision variable definitions.")
        
        # We need to find the parameters in the PST that match self.input_names
        if self.input_names is None:
             raise ValueError("GPR instance has no input_names defined. Cannot determine parameters.")

        # Filter PST parameter data
        missing_inputs = set(self.input_names) - set(pst.parameter_data.index)
        if missing_inputs:
             self.logger.warning(f"The following GPR input names are missing from the provided PST: {missing_inputs}")
        
        # Valid inputs
        valid_inputs = sorted(list(set(self.input_names).intersection(set(pst.parameter_data.index))))
        
        # Return the subset of the parameter dataframe
        # Columns align with what base.py expects (parnme is index in pyemu Pst.parameter_data)
        # We copy to ensure we don't mutate original
        par_df = pst.parameter_data.loc[valid_inputs].copy()
        par_df["parnme"] = par_df.index
        
        return par_df

    def _get_emulator_observations(self, pst=None):
        """
        Get the observations (outputs) for the GPR emulator.
        For GPR, outputs are specific objectives/constraints.
        Returns a DataFrame with columns: obsnme, obsval, weight, obgnme
        """
        if pst is None:
             # If no PST provided, try to construct from self.output_names if they exist, assuming dummy values
             if self.output_names:
                 df = pd.DataFrame(index=self.output_names)
                 df["obsnme"] = self.output_names
                 df["obsval"] = 0.0
                 df["weight"] = 1.0
                 df["obgnme"] = "gpr_pred"
                 if self.return_std:
                     # Add std observations
                     std_names = [f"{n}_gprstd" for n in self.output_names]
                     df_std = pd.DataFrame(index=std_names, columns=["obsnme", "obsval", "weight", "obgnme"])
                     df_std["obsnme"] = std_names
                     df_std["obsval"] = 0.0
                     df_std["weight"] = 0.0 # Uncertainty estimates usually have zero weight
                     df_std["obgnme"] = "gpr_std"
                     df = pd.concat([df, df_std])
                 return df
             else:
                 raise ValueError("GPR._get_emulator_observations requires either a 'pst' argument or defined 'output_names'.")
        
        # If PST provided, look for output names there to get exact weights/groups
        
        if self.output_names is None:
             raise ValueError("GPR instance has no output_names defined.")

        missing_outputs = set(self.output_names) - set(pst.observation_data.index)
        if missing_outputs:
             # If they are missing from PST, create dummy entries (maybe they are new outputs?)
             self.logger.warning(f"Outputs {missing_outputs} not found in PST. Creating generic definitions.")
             
        # Create list of all needed outputs
        all_outputs = self.output_names
        
        # Build DataFrame
        obs_df = pd.DataFrame(index=all_outputs, columns=["obsnme", "obsval", "weight", "obgnme"])
        obs_df["obsnme"] = all_outputs
        
        # Fill from PST where possible
        common = list(set(all_outputs).intersection(set(pst.observation_data.index)))
        if common:
             obs_df.loc[common, ["obsval", "weight", "obgnme"]] = pst.observation_data.loc[common, ["obsval", "weight", "obgnme"]]
        
        # Fill missing with defaults
        obs_df["obsval"] = obs_df["obsval"].astype(float).fillna(0.0)
        obs_df["weight"] = obs_df["weight"].astype(float).fillna(1.0)
        obs_df["obgnme"] = obs_df["obgnme"].fillna("gpr_pred")
        
        if self.return_std:
             std_names = [f"{n}_gprstd" for n in all_outputs]
             df_std = pd.DataFrame(index=std_names, columns=["obsnme", "obsval", "weight", "obgnme"])
             df_std["obsnme"] = std_names
             
             # Check if they exist in PST
             common_std = list(set(std_names).intersection(set(pst.observation_data.index)))
             if common_std:
                 df_std.loc[common_std, ["obsval", "weight", "obgnme"]] = pst.observation_data.loc[common_std, ["obsval", "weight", "obgnme"]]
             
             df_std["obsval"] = df_std["obsval"].astype(float).fillna(0.0)
             df_std["weight"] = df_std["weight"].astype(float).fillna(0.0)
             df_std["obgnme"] = df_std["obgnme"].fillna("gpr_std")
             
             obs_df = pd.concat([obs_df, df_std])

        return obs_df

    def prepare_pestpp(self, t_d, pst=None, verbose=False, **kwargs):
        """
        Prepare PEST++ interface for GPR.
        Wraps base implementation with support for legacy signature.
        
        Legacy signature: prepare_pestpp(pst_dir, casename, gpr_t_d="gpr_template")
        """
        # Check for legacy signature
        # case: 2nd arg 'pst' is string (casename) AND 'gpr_t_d' in kwargs
        if isinstance(pst, str) and "gpr_t_d" in kwargs:
             pst_dir = t_d
             casename = pst
             target_dir = kwargs.pop("gpr_t_d")
             
             self.logger.statement(f"Detected legacy GPR.prepare_pestpp call. "
                                   f"pst_dir={pst_dir}, casename={casename}, target={target_dir}")
             
             pst_path = os.path.join(pst_dir, f"{casename}.pst")
             if not os.path.exists(pst_path):
                 raise FileNotFoundError(f"Legacy GPR setup: PST file not found at {pst_path}")
             
             pst_obj = Pst(pst_path)
             
             # Call standard method
             # We pass specific filenames to match legacy GPR expectation if needed?
             # Old GPR used "gpr_input.csv.tpl" etc.
             # Base defaults to "emulator_input.csv.tpl".
             # We should probably enforce legacy naming for GPR too.
             p_obj = super().prepare_pestpp(target_dir, pst=pst_obj, verbose=verbose, 
                                            tpl_filename="gpr_input.csv.tpl",
                                            input_filename="gpr_input.csv",
                                            ins_filename="gpr_output.csv.ins",
                                            output_filename="gpr_output.csv",
                                            emu_filename="gpr_emulator.pkl",
                                            **kwargs)
             
             # Legacy behavior: Write the PST file to disk
             pst_name = f"{casename}_gpr.pst"
             p_obj.write(os.path.join(target_dir, pst_name), version=2)
             self.logger.statement(f"Legacy GPR: Wrote control file to {pst_name}")
             
             return p_obj

        # Also support legacy named arg 'gpr_t_d' mapped to 't_d' if t_d is not the target
        # But in new signature t_d IS the target.
        
        return super().prepare_pestpp(t_d, pst=pst, verbose=verbose, **kwargs)
    
    def _write_output_file(self, obs_df, filename):
        """Writes GPR-specific output file (handling std dev)."""
        with open(filename, 'w') as f:
            f.write("obsnme,obsval\n") # header
            for output_name in self.output_names:
                if self.return_std:
                    # e.g. "obsnme, val, std"
                    f.write(f"{output_name},0.0\n")
                    f.write(f"{output_name}_gprstd,0.0\n")
                else:
                    f.write(f"{output_name},0.0\n")

    def _write_instruction_file(self, obs_df, filename):
        """Writes GPR-specific instruction file (handling std dev)."""
        with open(filename, 'w') as f:
            f.write("pif ~\n")
            f.write("l1\n") # header
            for output_name in self.output_names:
                if self.return_std:
                     # e.g. "obsnme, val, std"
                     # Instruction: Skip obsnme, read val, read std
                     f.write("l1 ~,~ !{0}! ~,~ !{0}_gprstd!\n".format(output_name))
                else:
                     f.write("l1 ~,~ !{0}!\n".format(output_name))

    def _write_forward_run_script(self, filename, emu_file, input_file, output_file, class_name, pst_name=None):
        """Generates the python script that PEST++ runs for GPR (handles tuple return)."""
        import inspect
        from pyemu.utils.helpers import gpr_file_forward_run, gpr_runstore_forward_run, gpr_forward_run

        use_runstor = getattr(self, "_use_runstor", False)
        
        target_func = "gpr_runstore_forward_run" if use_runstor else "gpr_file_forward_run"
        if use_runstor:
            call_args = f"emu_file='{emu_file}'"
            if pst_name is not None:
                call_args += f", pst_name='{pst_name}'"
        else:
            call_args = f"'{emu_file}', '{input_file}', '{output_file}'"

        lines = [
            "import sys",
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "import traceback",
            "import pickle",
            "",
            "sys.path.append(os.getcwd())",
            ""
        ]

        # Inject code for all use cases
        for func in [gpr_forward_run, gpr_file_forward_run, gpr_runstore_forward_run]:
             lines.append(f"# Source for {func.__name__}")
             lines.append(inspect.getsource(func))
             lines.append("")

        lines.append('if __name__ == "__main__":')
        lines.append(f'    {target_func}({call_args})')

        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")

