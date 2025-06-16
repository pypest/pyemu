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
    
    Parameters
    ----------
    verbose : bool, optional
        If True, enable verbose logging. Default is True.
    """

    def __init__(self, verbose=True):
        """
        Initialize the Emulator base class.

        Parameters
        ----------
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.fitted = False
        self.data = None
        self.data_transformed = None
        self.feature_scaler = None
        self.energy_threshold = 1.0
        self.feature_transformer = None

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

    def prepare_training_data(self, data=None):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Raw training data. If None, uses self.data.
            
        Returns
        -------
        tuple
            Processed data ready for model fitting.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data stored in the emulator")
            data = self.data
        
        # Common preprocessing logic could go here
        return data
        
    def apply_feature_transforms(self, data=None, transforms=None):
        """
        Apply feature transformations to data with customizable transformer sequence.
        
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
        
        ft = AutobotsAssemble(data.copy())
        
        # Process the transforms parameter if provided
        if transforms:
            for transform in transforms:
                transform_type = transform.get('type')
                columns = transform.get('columns')
                # Extract transformer-specific kwargs
                kwargs = {k: v for k, v in transform.items() 
                        if k not in ('type', 'columns')}
                
                self.logger.statement(f"applying {transform_type} transform")
                ft.apply(transform_type, columns=columns, **kwargs)
        
        transformed_data = ft.df.copy()
        self.feature_transformer = ft
        self.data_transformed = transformed_data
            
        return transformed_data

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