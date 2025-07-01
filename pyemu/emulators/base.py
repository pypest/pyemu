"""
Base class for emulators.
"""
from __future__ import print_function, division
import pickle
import numpy as np
import pandas as pd
from ..logger import Logger

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
    


    #TODO: implment helper function that scrapes  directory and collates training data from Pst ensemble files + control file information.