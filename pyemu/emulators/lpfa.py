"""
Learning-based pattern-data-driven forecast approach (LPFA) emulator implementation.

"""
from __future__ import print_function, division
import importlib.util

# Check sklearn availability at module level
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

if HAS_SKLEARN:
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
else:
    # Create dummy classes or set to None
    train_test_split = None
    PCA = None
    MLPRegressor = None

import numpy as np
import pandas as pd

from .base import Emulator
from .transformers import RowWiseMinMaxScaler

# Define scikit-learn based model class
class LPFAModel:
    """
    Scikit-learn MLPRegressor wrapper for LPFA neural network model.
    """
    def __init__(self, input_dim, output_dim, hidden_units=None, activation='relu', 
                 dropout_rate=0.0, learning_rate=0.01, max_iter=200, early_stopping=True):
        
        if hidden_units is None:
            hidden_units = (2 * input_dim,)
        elif isinstance(hidden_units, list):
            hidden_units = tuple(hidden_units)
            
        # Map activation functions from PyTorch to scikit-learn
        activation_map = {
            'relu': 'relu',
            'tanh': 'tanh', 
            'sigmoid': 'logistic'
        }
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_units,
            activation=activation_map.get(activation, 'relu'),
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.2,
            n_iter_no_change=20,  # Patience for early stopping
            random_state=42,
            warm_start=False,
            alpha=dropout_rate if dropout_rate > 0 else 0.0001  # Use L2 regularization instead of dropout
        )
    
    def fit(self, X, y):
        """Fit the model"""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    @property
    def loss_curve_(self):
        """Get training loss curve"""
        return getattr(self.model, 'loss_curve_', [])
    

class LPFA(Emulator):
    """
    Class for the Learning-based pattern-data-driven forecast approach from Kim et al (2025).
    
    This emulator uses neural networks to learn the relationships between inputs 
    and forecast outputs, with dimensionality reduction via PCA.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The training data with input and forecast columns.
    input_names : list
        List of column names to use as inputs.
    groups : dict
        Dictionary mapping group names to lists of column names. Used for row-wise min-max scaling.
    fit_groups : dict
        Dictionary mapping group names to lists of column names used to fit the scaling.
    output_names : list, optional
        List of column names to forecast. If None, all columns not in input_names are used.
    energy_threshold : float, optional
        Energy threshold for the PCA. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    early_stop : bool, optional
        Whether to use early stopping during training. Default is True.
    transforms : list of dict, optional
        List of transformation specifications. Each dict should have:
        - 'type': str - Type of transformation (e.g., 'log10', 'normal_score').
        - 'columns': list of str, optional - Columns to apply the transformation to. If not supplied, transformation is applied to all columns.
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

    def __init__(self,
                 data,
                 input_names,
                 groups,
                 fit_groups,
                 output_names=None,
                 energy_threshold=1.0,
                 seed=None,
                 early_stop=True,
                 transforms=None,
                 test_size=0.2,
                 verbose=True):
        """
        Initialize the Learning-based pattern-data-driven NN emulator.

        Parameters
        ----------
        data : pandas.DataFrame
            The training data with input and forecast columns.
        input_names : list
            List of column names to use as inputs.
        groups : dict
            Dictionary mapping group names to lists of column names. Used for row-wise min-max scaling.
        fit_groups : dict
            Dictionary mapping group names to lists of column names used to fit the scaling.
        output_names : list, optional
            List of column names to forecast. If None, all columns in data will be used.
        energy_threshold : float, optional
            Energy threshold for the PCA. Default is 1.0.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        early_stop : bool, optional
            Whether to use early stopping during training. Default is True.
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
        test_size : float, optional
            Fraction of data to use for testing. Default is 0.2.
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """

        
        super().__init__(verbose=verbose)

        self.seed = seed
        self.data = data
        self.input_names = input_names
        self.groups = groups
        self.fit_groups = fit_groups
        
        if output_names is None:
            output_names = data.columns
        self.output_names = output_names
        
        self.energy_threshold = energy_threshold
        
        # Store early stopping preference
        self.use_early_stopping = early_stop
            
        self.transforms = transforms
        self.noise_model = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.test_size = test_size

        # Prepare the training data
        
        self._prepare_training_data()
        
    def _prepare_training_data(self):
        """
        Prepare the training data for model fitting.
        
        This method:
        1. Splits the data into training and test sets
        2. Applies transform pipelines if specified
        3. Applies row-wise min-max scaling
        4. Performs PCA dimensionality reduction
        
        Parameters
        ----------
        self: LPFA
            The emulator instance containing the data and configuration.
            
        Returns
        -------
        dict
            Dictionary containing prepared data components:
            - X_train: Input training data after transformation and PCA
            - y_train: Target training data after transformation and PCA
            - X_test: Input testing data after transformation and PCA
            - y_test: Target testing data after transformation and PCA
        """
        
        self.logger.statement("preparing training data")
        data = self.data
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
            
        # Split the data into training and test sets
        train, test = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=self.seed
        )
        
        self.logger.statement("train/test data split complete")
        
        # Store for later use
        self.train_data = train.copy()
        self.test_data = test.copy()
        
        self.logger.statement("applying feature transformation pipeline")
        # Apply feature transformations if specified
        # Always use the base class transformation method for consistency
        if self.transforms is None:
            from .transformers import AutobotsAssemble
            self.transformer_pipeline = AutobotsAssemble(train.copy())
            train_transformed = train
            test_transformed = test
        else:
            train_transformed = self._fit_transformer_pipeline(train, self.transforms)
            test_transformed = self.transformer_pipeline.transform(test)

        
        # Apply row-wise min-max scaling directly (not through the pipeline)
        # We need to keep train and test separate; there may be a more elgant solution to this....
        # training data
        self.logger.statement("applying row-wise min-max scaling")
        self.rowwise_mm_scalers ={
            "train": RowWiseMinMaxScaler(
                        feature_range=(-1, 1),
                        groups=self.groups,
                        fit_groups=self.fit_groups )
        }    
        self.rowwise_mm_scalers["train"].fit(train_transformed)
        train_scaled = self.rowwise_mm_scalers["train"].transform(train_transformed)

        # test data
        # We need to fit a new scaler on the test data
        self.rowwise_mm_scalers["test"] =  RowWiseMinMaxScaler(
                        feature_range=(-1, 1),
                        groups=self.groups,
                        fit_groups=self.fit_groups )
        self.rowwise_mm_scalers["train"].fit(test_transformed)
        test_scaled = self.rowwise_mm_scalers["test"].transform(test_transformed)

        self.logger.statement("row-wise min-max scaling complete")
        
        # Split datasets into input (X) and target (y) variables
        X_train = train_scaled.loc[:, self.input_names].copy()
        y_train = train_scaled.loc[:, self.output_names].copy()
        
        X_test = test_scaled.loc[:, self.input_names].copy()
        y_test = test_scaled.loc[:, self.output_names].copy()
        
        # Apply PCA to reduce the dimensionality of the data
        self.logger.statement("applying PCA dimensionality reduction")
        self.pcaX = PCA()
        self.pcay = PCA()

        X_transformed = self.pcaX.fit_transform(X_train)
        y_transformed = self.pcay.fit_transform(y_train)
        
        # Apply energy-based truncation
        if self.energy_threshold < 1.0:
            self.logger.statement("applying energy-based PCA truncation")
            # For input PCA
            explained_var_ratio_X = np.cumsum(self.pcaX.explained_variance_ratio_)
            n_components_X = np.argmax(explained_var_ratio_X >= self.energy_threshold) + 1
            self.pcaX = PCA(n_components=n_components_X)
            X_transformed = self.pcaX.fit_transform(X_train)
            
            # For output PCA
            explained_var_ratio_y = np.cumsum(self.pcay.explained_variance_ratio_)
            n_components_y = np.argmax(explained_var_ratio_y >= self.energy_threshold) + 1
            self.pcay = PCA(n_components=n_components_y)
            y_transformed = self.pcay.fit_transform(y_train)
            
            self.logger.statement(f"Reduced X from {X_train.shape[1]} to {n_components_X} components")
            self.logger.statement(f"Reduced y from {y_train.shape[1]} to {n_components_y} components")
        
        self.X = X_transformed
        self.y = y_transformed
        
        self.X_test = self.pcaX.transform(X_test)
        self.y_test = self.pcay.transform(y_test)
        
        self.logger.statement("PCA dimensionality reduction complete")
        
        return {
            'X_train': self.X,
            'y_train': self.y,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
    def _build_model(self, params=None, prob=False):
        """
        Build a neural network model with the specified parameters.
        
        Parameters
        ----------
        params : dict or pandas.Series, optional
            Dictionary with model parameters including:
            - activation: Activation function to use
            - hidden_units: List of units in each hidden layer
            - dropout_rate: Rate of dropout for regularization
            - learning_rate: Learning rate for optimizer
            If None, uses default parameters. Default is None.
        prob : bool, optional
            Whether to build a probabilistic model. Default is False.
            
        Returns
        -------
        LPFAModel
            The scikit-learn MLPRegressor wrapper instance.
        """
        if params is None:
            params = {
                'activation': 'relu', 
                'hidden_units': None, 
                'dropout_rate': 0.0,
                'learning_rate': 0.01
            }
        
        if isinstance(params, pd.Series):
            params = params.to_dict()

        input_dim = self.X.shape[1]
        output_dim = self.y.shape[1]
        
        # Create the model architecture
        model = LPFAModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=params['hidden_units'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            early_stopping=self.use_early_stopping
        )
        
        return model

    def create_model(self, params=None):
        """
        Create and store the main model.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of model parameters. Default is None.
            
        Returns
        -------
        self : LPFA
            The emulator instance with model created.
        """
        self.model = self._build_model(params)
        return self

    def add_noise_model(self, params=None):
        """
        Add a noise model to capture residuals.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of model parameters for the noise model. Default is None.
            
        Returns
        -------
        self : LPFA
            The emulator instance with noise model added.
        """
        # Create noise model
        self.noise_model = self._build_model(params)
        
        # Get residuals from main model
        self.logger.statement("calculating residuals for noise model")
        
        # Get predictions from main model
        pred_train = self.model.predict(self.X)
        residuals_train = self.y - pred_train
        
        # Train noise model on residuals
        self.logger.statement("training noise model on residuals")
        self.noise_model.fit(self.X, residuals_train)
        
        return self

    def fit(self, epochs=200):
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 200.
        Returns
        -------
        self : LPFA
            The fitted emulator.
        """
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()
            
        if self.model is None:
            self.create_model()
        
        # Update max_iter for the model
        self.model.model.max_iter = epochs
        
        # Simple fit - scikit-learn handles batching, early stopping, etc.
        self.logger.statement(f"fitting model with MLPRegressor: {epochs} epochs")
        
        X_train = self.X 
        y_train = self.y
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Store training history
        self.history = {
            'loss': self.model.loss_curve_,
            'val_loss': []  # MLPRegressor doesn't provide separate validation loss
        }
        
        # Log final training info
        n_iter = getattr(self.model.model, 'n_iter_', epochs)
        final_loss = self.model.loss_curve_[-1] if self.model.loss_curve_ else "N/A"
        self.logger.statement(f"Training completed in {n_iter} iterations, final loss: {final_loss}")
        
        self.fitted = True
        return self

    def predict(self, data):
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            New data to generate predictions for.
            
        Returns
        -------
        pandas.DataFrame
            Predictions for the input data.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        
        if self.model is None:
            raise ValueError("No model has been created. Call create_model() first")
            
        self.logger.statement("generating predictions from fitted model")
            
        # Make a copy of the input data to avoid modifying the original
        truth = data.copy()
        predictions = truth.copy()
        predictions[:] = np.nan
        
        # STEP 1: Apply the same sequence of transformations used during training
        self.logger.statement("applying transformations to input data")
        
        # Apply transfrom pipeline if it was used during training
        truth_transformed = self.transformer_pipeline.transform(truth)

        
        # Apply row-wise min-max scaling
        # We need to fit a new scaler on the truth data
        forecast_rowwise_mm_scaler = RowWiseMinMaxScaler(
            feature_range=(-1, 1),
            groups=self.groups,
            fit_groups=self.fit_groups
        )
        forecast_rowwise_mm_scaler.fit(truth_transformed)
        truth_scaled = forecast_rowwise_mm_scaler.transform(truth_transformed)
        
        # Extract input columns and apply PCA transformation
        X_truth = truth_scaled.loc[:, self.input_names].copy()
        y_truth = truth_scaled.loc[:, self.output_names].copy()
        
        # Apply PCA transform
        truth_pca = self.pcaX.transform(X_truth.values)
        
        # Run model prediction 
        self.logger.statement("running model prediction")
        
        # Get model prediction
        pred_pca = self.model.predict(truth_pca)
        
        # Add noise prediction if available
        if self.noise_model is not None:
            self.logger.statement("adding noise model prediction")
            noise_pred = self.noise_model.predict(truth_pca)
            pred_pca = pred_pca + noise_pred
        
        # Apply inverse transformations in REVERSE order of the original transformations
        self.logger.statement("performing inverse transformations")
        
        # First inverse the PCA transform (was the last transform applied)
        pred_scaled = pd.DataFrame(
            self.pcay.inverse_transform(pred_pca),
            columns=y_truth.columns, 
            index=y_truth.index
        )
        
        # Then inverse the row-wise min-max scaling (applied before PCA)
        pred_transformed = forecast_rowwise_mm_scaler.inverse_transform(pred_scaled)
        
        # Assign predictions to output
        predictions.loc[:, self.output_names] = pred_transformed.loc[:, self.output_names]
        
        # Finally, inverse the transform pipeline if it was applied (was the first transform)
        predictions = self.transformer_pipeline.inverse(predictions)
        
        return predictions