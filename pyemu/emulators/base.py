"""
Base class for emulators.
"""
from __future__ import print_function, division
import os
import pickle
import numpy as np
import pandas as pd
from ..logger import Logger
from pyemu.pst.pst_handler import Pst
from pyemu.utils.os_utils import run

class Emulator:
    """
    Base class for emulators.
    
    This class defines the common interface for all emulator implementations
    and provides shared functionality used by multiple emulator types.
    
    """

    def __init__(self,transforms=None, verbose=True):
        """
        Initialize the Emulator base class.

        Parameters
        ----------
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
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.fitted = False
        self.data = None
        self.data_transformed = None
        self.transforms = transforms
        self.transformer_pipeline = None
        self._use_runstor = False

    def fit(self, X, y=None):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features for training.
        y : pandas.DataFrame or None, optional
            Target values for training if separate from X.
            
        Returns
        -------
        self : Emulator
            Returns self for method chaining.
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        """
        Generate predictions using the fitted emulator.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data to generate predictions for.
            
        Returns
        -------
        pandas.DataFrame or pandas.Series
            Predictions for the input data.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict method")

    def _prepare_training_data(self):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        self : Emulator
            The emulator instance.
        Returns
        -------
        tuple
            Processed data ready for model fitting.
        """
        data = self.data
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
 
         # Common preprocessing logic could go here
        self.logger.statement("preparing training data")
        
        # apply feature transformations if they exist, etc..        
        # Always use the base class transformation method for consistency
        if self.transforms is not None:
            self.logger.statement("applying feature transforms")
            self.data_transformed = self._fit_transformer_pipeline(data, self.transforms)
        else:
            # Still need to set up a dummy transformer for inverse operations
            from .transformers import AutobotsAssemble
            self.feature_transformer = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()
    
        return self.data_transformed

        return 
        
    def _fit_transformer_pipeline(self, data=None, transforms=None):
        """
        Apply feature transformations to data with customizable transformer sequence.
        This function is not intended to be used directly by users.
        External data must be accepted to handle train/test spliting for certain emulators (e.g., LPFA).

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to transform. If None, uses self.data.
        transforms : list of dict, optional
            List of transformation specifications. Each dict should have:
            - 'type': str - Type of transformation (e.g., 'log10', 'normal_score')
            - 'columns': list - Columns to apply the transformation to (optional)
            - Additional kwargs specific to the transformer
            If None, no transformations are applied.
            
        Returns
        -------
        pandas.DataFrame
            Transformed data.
        
        Examples
        --------
        # Using the transforms parameter:
        emulator.apply_feature_transforms(
            transforms=[
                {'type': 'log10', 'columns': ['flow', 'heads']},
                {'type': 'normal_score', 'columns': None, 'quadratic_extrapolation': True}
            ]
        )
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
                
        self.logger.statement("applying feature transforms")
        # Import AutobotsAssemble here to avoid circular import
        from .transformers import AutobotsAssemble
        
        transformer_pipeline = AutobotsAssemble(data.copy())
        
        # Process the transforms parameter if provided
        if transforms is None:
            transforms = self.transforms
        if transforms:
            self._validate_transforms(transforms)
            for transform in transforms:
                transform_type = transform.get('type')
                columns = transform.get('columns')
                # Extract transformer-specific kwargs
                kwargs = {k: v for k, v in transform.items() 
                        if k not in ('type', 'columns')}
                
                self.logger.statement(f"applying {transform_type} transform")
                transformer_pipeline.apply(transform_type, columns=columns, **kwargs)
        
        self.transformer_pipeline = transformer_pipeline
        self.data_transformed = transformer_pipeline.df.copy()
            
        return self.data_transformed 

    def save(self, filename):
        """
        Save the fitted emulator to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the emulator.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """
        Load a fitted emulator from a file.
        
        Parameters
        ----------
        filename : str
            Path to the saved emulator file.
            
        Returns
        -------
        Emulator
            The loaded emulator instance.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
        

    def _validate_transforms(self, transforms):
        """Validate the transforms parameter."""
        if not isinstance(transforms, list):
            raise ValueError("transforms must be a list of dicts or None")
        
        for t in transforms:
            if not isinstance(t, dict):
                raise ValueError("each transform must be a dict")
            if 'type' not in t:
                raise ValueError("each transform dict must have a 'type' key")
            if 'columns' in t and not isinstance(t['columns'], list):
                raise ValueError("'columns' must be a list of column names")
    


    def prepare_pestpp(self, t_d, pst=None, verbose=False, **kwargs):
        """
        Generic method to prepare a PEST++ interface for the emulator.
        
        This method automates the creation of template files, instruction files,
        control files, and the forward run script needed to run the emulator
        within a PEST++ workflow (e.g. IES).
        
        Parameters
        ----------
        t_d : str
            Path to the template directory where files will be written.
        pst : Pst, optional
            A Pst object representing the original control file. 
            Useful for scraping constraint weights, observation lists, etc.
            Subclasses may use this to determine specific parameters or observations.
        verbose : bool
            Enable verbose logging.

        Returns
        -------
        Pst
            The generated Pst object for the emulator.
        """
        if verbose:
            self.logger.statement(f"Preparing PEST++ interface in directory: {t_d}")
        
        if os.path.exists(t_d):
            self.logger.statement(f"Removing existing template directory: {t_d}")
            import shutil
            shutil.rmtree(t_d)
        # Create the template directory
        os.makedirs(t_d)
        
        # 1. Save the emulator itself (using pickle)
        # We save it locally in the template dir so the forward run script can load it
        emu_filename = kwargs.get("emu_filename", "emulator.pkl")
        self.save(os.path.join(t_d, emu_filename))
        
        # 2. Get Input Parameters (DataFrame: Name, Value, Bounds, Groups)
        par_df = self._get_emulator_parameters(pst)
        # Expected cols: parnme, parval1, pargp, parlbnd, parubnd
        
        # 3. Get Output Observations (DataFrame: Name, Value, Weight)
        obs_df = self._get_emulator_observations(pst)
        # Expected cols: obsnme, obsval, weight, obgnme
        
        # 4. Generate Template File (parameters) and Initial Input File
        tpl_filename = kwargs.get("tpl_filename", "emulator_input.csv.tpl")
        input_filename = kwargs.get("input_filename", "emulator_input.csv")
        self._use_runstor = kwargs.get("use_runstor", False)
        self._write_template_file(par_df, os.path.join(t_d, tpl_filename))
        self._write_input_file(par_df, os.path.join(t_d, input_filename))
        
        # 5. Generate Instruction File (observations)
        ins_filename = kwargs.get("ins_filename", "emulator_output.csv.ins")
        output_filename = kwargs.get("output_filename", "emulator_output.csv")
        self._write_instruction_file(obs_df, os.path.join(t_d, ins_filename))
        
        # 6. Generate Forward Run Script
        # Pass the class name so the script knows what to import (e.g. "DSI", "GPR")
        class_name = self.__class__.__name__
        self._write_forward_run_script(os.path.join(t_d, "forward_run.py"), 
                                       emu_filename, 
                                       input_filename, 
                                       output_filename, 
                                       class_name,
                                       pst_name=kwargs.get("pst_name",None))
        
        # run the forward run once
        # to ensure the script works and the emulator is valid
        if not self._use_runstor:
            self.logger.statement("Validating forward_run.py script by executing once")
            try:
                run(f"python forward_run.py", cwd=os.path.abspath(t_d),)
            except Exception as e:
                self.logger.statement("Error running forward_run.py to validate emulator interface")
                raise
        else:
            self.logger.statement("Skipping forward_run.py validation (use_runstor=True)")
            self._write_output_file(obs_df, os.path.join(t_d, output_filename))  
        # 7. Generate Pst Control File
        # Use Pst.from_io_files to wire everything up
        pst_obj = Pst.from_io_files(
            tpl_files=[os.path.join(t_d, tpl_filename)],
            in_files=[os.path.join(t_d, input_filename)],
            ins_files=[os.path.join(t_d, ins_filename)],
            out_files=[os.path.join(t_d, output_filename)],
            pst_path="."
        )
        
        # Update Pst Control Data with correct parameter and observation data
        # (Bounds, Initial Values, Groups, Weights)
        # The 'from_io_files' might have defaults; overwritten here:
        pst_obj.parameter_data = self._update_parameter_data(pst_obj.parameter_data, par_df)
        pst_obj.observation_data = self._update_observation_data(pst_obj.observation_data, obs_df)
        
        # Set command line
        pst_obj.model_command = "python forward_run.py"

        # Allow subclasses to refine the PST object (e.g. transfer pestpp_options)
        self.logger.statement("Configuring final Pst object")
        print(kwargs.get("observation_data", None))
        self._configure_pst_object(
                                pst_obj, pst, 
                                observation_data=kwargs.get("observation_data", None),
                                t_d=t_d
                                )
        self.save(os.path.join(t_d, emu_filename))
        return pst_obj

    def _get_emulator_parameters(self, pst=None):
        """
        Get the parameters (inputs) for the emulator.
        Must return a DataFrame with columns: parnme, parval1, parlbnd, parubnd, pargp
        """
        raise NotImplementedError("Subclasses must implement _get_emulator_parameters")

    def _get_emulator_observations(self, pst=None):
        """
        Get the observations (outputs) for the emulator.
        Must return a DataFrame with columns: obsnme, obsval, weight, obgnme
        """
        raise NotImplementedError("Subclasses must implement _get_emulator_observations")
    
    def _configure_pst_object(self, pst_new, pst_old=None, observation_data=None, t_d=None):
        """
        Hook for subclasses to modify the final Pst object.
        For example, copying options from the original PST.
        """
        if pst_old is not None:
            if pst_old.pestpp_options is not None:
                # carry across pestpp options
                pst_new.pestpp_options = pst_old.pestpp_options.copy()

            if pst_old.prior_information is not None:
                # and prior info
                pst_new.prior_information = pst_old.prior_information.copy()

        if observation_data is None and pst_old is not None:
            self.logger.statement("No observation_data provided, copying from old pst")
            observation_data = pst_old.observation_data.copy()
        if observation_data is not None:
            observation_data.obsval = observation_data.obsval.astype(float)
            observation_data.weight = observation_data.weight.astype(float)
            self.logger.statement("Updating observation data in new pst from provided observation_data")
            assert  set(['obsnme','obsval','weight']).issubset(observation_data.columns.tolist()), 'observation_data must have obsnme, obsval, and weight columns'
            assert observation_data.obsnme.is_unique, "observation_data must have unique obsnmes"
            # update observation data values from old pst
            obsnmes = pst_new.observation_data.obsnme.tolist()
            # only update obsvals for obs in both psts
            common_obsnmes = [o for o in obsnmes if o in observation_data.obsnme.tolist()]
            obs_new = pst_new.observation_data
            old_cols = observation_data.columns.tolist()
            obs_new.loc[common_obsnmes,old_cols] = observation_data.loc[common_obsnmes,old_cols].copy()

        if pst_old is not None:
            # check if any obs are in parameter_data
            parnmes = pst_old.parameter_data.parnme.tolist()
            common_nmes = [n for n in obsnmes if n in parnmes]
            if len(common_nmes) > 0:
                obs_new.loc[common_nmes,'obsval'] = pst_old.parameter_data.loc[common_nmes,'parval1'].copy()
                obs_new.loc[common_nmes,'weight'] = 0.0 # Default safety
        pst_new.try_parse_name_metadata()
        pst_new.control_data.noptmax = 0 # Default safety
        return pst_new

    def _update_parameter_data(self, pst_par_df, par_df):
        """Helper to merge emulator parameter definitions into Pst DataFrame."""
        # Ensure alignment
        pst_par_df = pst_par_df.loc[par_df.index]
        for col in ['parval1', 'parlbnd', 'parubnd', 'pargp', 'partrans', 'scale', 'offset']:
            if col in par_df.columns:
                 pst_par_df[col] = par_df[col]
        return pst_par_df

    def _update_observation_data(self, pst_obs_df, obs_df):
        """Helper to merge emulator observation definitions into Pst DataFrame."""
        pst_obs_df = pst_obs_df.loc[obs_df.index]
        for col in ['obsval', 'weight', 'obgnme']:
             if col in obs_df.columns:
                 pst_obs_df[col] = obs_df[col]
        return pst_obs_df

    def _write_template_file(self, par_df, filename):
        """Writes a simple CSV template file."""
        # This implementation assumes parameters are rows in a single-column CSV or similar structure
        # Subclasses might want different input formats, but a standard vertical CSV is robust.
        # Format: parnme, parval1
        
        with open(filename, 'w') as f:
            f.write("ptf ~\n")
            f.write("parnme,parval1\n")
            for parnme in par_df.index:
                 f.write(f"{parnme},~   {parnme}   ~\n")

    def _write_input_file(self, par_df, filename):
        """Writes the initial input file corresponding to the template."""
        # We need to write a CSV with index (parnme) and parval1
        # This matches the read_csv in forward_run.py and the template structure
        par_df.loc[:, ["parval1"]].to_csv(filename)
    
    def _write_output_file(self, obs_df, filename):
        """Writes a simple CSV output file with base values."""
        # Assumes output format from forward_run.py is: obsnme,simval
        # Standard vertical CSV
        with open(filename, 'w') as f:
            f.write("obsnme,simval\n") # header
            for obsnme in obs_df.index:
                f.write(f"{obsnme},{obs_df.loc[obsnme, 'obsval']}\n") # base value

    def _write_instruction_file(self, obs_df, filename):
        """Writes a simple CSV instruction file."""
        # Assumes output format from forward_run.py is: obsnme,simval
        # Standard vertical CSV
        with open(filename, 'w') as f:
            f.write("pif ~\n")
            f.write("l1\n") # header
            for obsnme in obs_df.index:
                f.write(f"l1~,~ !{obsnme}!\n")

    def _write_forward_run_script(self, filename, emu_file, input_file, output_file, class_name, pst_name=None):
        """Generates the python script that PEST++ runs.
           Subclasses must implement this method to handle specific return types and behaviors."""
        raise NotImplementedError("Subclasses must implement _write_forward_run_script")

    #TODO: implment helper function that scrapes  directory and collates training data from Pst ensemble files + control file information.