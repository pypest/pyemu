"""
Transformer classes for data transformations in emulators.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import importlib.util

# Check sklearn availability at module level
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

if HAS_SKLEARN:
    from sklearn.preprocessing import StandardScaler
else:
    # Create dummy classes or set to None
    StandardScaler = None


class BaseTransformer:
    """Base class for all transformers providing a consistent interface."""

    def fit(self, X):
        """Learn parameters from data if needed."""
        return self

    def transform(self, X):
        """Apply transformation to X."""
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transform X back to original space."""
        raise NotImplementedError

class Log10Transformer(BaseTransformer):
    """Apply log10 transformation.
    
    Parameters
    ----------
    columns : list, optional
        List of column names to be transformed. If None, all columns will be transformed.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.shifts = {}

    def transform(self, X):
        result = X.copy()
        columns = self.columns if self.columns is not None else X.columns
        columns = [col for col in columns if col in X.columns]
        
        for col in columns:
            min_val = X[col].min()
            shift = -min_val + 1e-6 if min_val <= 0 else 0
            self.shifts[col] = shift
            result[col] = np.log10(X[col] + shift)
        return result

    def inverse_transform(self, X):
        result = X.copy()
        for col in self.shifts.keys():
            if col in X.columns:
                shift = self.shifts.get(col, 0)
                result[col] = (10 ** X[col]) - shift
        return result

class RowWiseMinMaxScaler(BaseTransformer):
    """Scale each row of a DataFrame to a specified range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, 1)
        The range to scale features into.
    groups : dict or None, default=None
        Dict mapping group names to lists of column names to be scaled together (entire timeseries for that group).
        If None, all columns will be treated as a single group.
        Example: {'group1': ['col1', 'col2'], 'group2': ['col3', 'col4']}
    fit_groups : dict or None, default=None
        Dict mapping group names to lists of column names (subset of groups) used to compute row-wise min and max.
        If None, defaults to using the same columns as in groups.
    """

    def __init__(self, feature_range=(-1, 1), groups=None, fit_groups=None):
        self.feature_range = feature_range
        self.groups = groups
        self.fit_groups = fit_groups if fit_groups is not None else groups
        self.row_params = {}  # Will store per-row (min, max) for each group

    def fit(self, X):
        """Compute row-wise min and max for each group.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to fit the scaler on.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # If groups not specified, treat all columns as one group
        if self.groups is None:
            self.groups = {"all": X.columns.tolist()}
            
        if self.fit_groups is None:
            self.fit_groups = self.groups.copy()
        
        # Calculate and store row-wise min and max for each group
        self.row_params = {}
        for group_name, group_cols in self.groups.items():
            # Determine which columns to use for computing min/max for each row
            fit_cols = self.fit_groups.get(group_name, group_cols)
            # Keep only columns that exist in the DataFrame
            fit_cols = [col for col in fit_cols if col in X.columns]
            if not fit_cols:
                continue
                
            # Compute row-wise min and max using the fit columns
            row_min = X[fit_cols].min(axis=1)
            row_max = X[fit_cols].max(axis=1)
            self.row_params[group_name] = (row_min, row_max)
        
        return self

    def transform(self, X):
        """Scale each row of data to the specified range.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        result = X.copy()
        f_min, f_max = self.feature_range
        
        # Auto-fit if not already fitted or if groups weren't specified
        if not self.row_params or self.groups is None:
            self.fit(X)
        
        # Transform each group
        for group_name, group_cols in self.groups.items():
            # Keep only columns that exist in the DataFrame
            valid_cols = [col for col in group_cols if col in X.columns]
            if not valid_cols:
                continue
                
            # Get the min and max for each row in this group
            row_min, row_max = self.row_params[group_name]
            
            # Calculate the row range, avoiding division by zero
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0  # Set to 1 where range is 0
            
            # For all columns in the group, scale using the row-wise parameters
            group_data = X[valid_cols]
            # First scale to [0, 1]
            group_std = group_data.sub(row_min, axis=0).div(row_range, axis=0)
            # Then scale to the desired feature range
            result[valid_cols] = group_std * (f_max - f_min) + f_min
        
        return result

    def inverse_transform(self, X):
        """Inverse transform data back to the original scale.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        if not self.row_params:
            raise ValueError("This RowWiseMinMaxScaler instance is not fitted yet. "
                            "Call 'fit' before using this method.")
        
        result = X.copy()
        f_min, f_max = self.feature_range
        
        # Inverse transform each group
        for group_name, group_cols in self.groups.items():
            # Keep only columns that exist in the DataFrame
            valid_cols = [col for col in group_cols if col in X.columns]
            if not valid_cols:
                continue
                
            # Get the min and max for each row in this group
            row_min, row_max = self.row_params[group_name]
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0  # Avoid division by zero
            
            # Get the scaled data for this group
            group_data = X[valid_cols]
            
            # First convert from feature_range to [0, 1]
            group_std = (group_data - f_min) / (f_max - f_min)
            
            # Then recover original values
            result[valid_cols] = group_std.mul(row_range, axis=0).add(row_min, axis=0)
        
        return result

class MinMaxScaler(BaseTransformer):
    """Scale each column of a DataFrame to a specified range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, 1)
        The range to scale features into.
    columns : list, optional
        List of column names to be scaled. If None, all columns will be scaled.
    skip_constant : bool, optional
        If True, columns with constant values will be skipped. Default is True.
    """

    def __init__(self, feature_range=(-1, 1), columns=None, skip_constant=True):
        self.feature_range = feature_range
        self.columns = columns
        self.skip_constant = skip_constant
        self.min_ = {}
        self.scale_ = {}
        
    def fit(self, X):
        """Learn min and max values for scaling.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to fit the scaler on.
            
        Returns
        -------
        self : object
            Returns self.
        """
        columns = self.columns if self.columns is not None else X.columns
        
        # Ensure we only work with columns that exist in the DataFrame
        columns = [col for col in columns if col in X.columns]
        
        for col in columns:
            col_min = X[col].min()
            col_max = X[col].max()
            
            # If the column has constant values and skip_constant is True, store the values but don't transform
            if self.skip_constant and col_min == col_max:
                self.min_[col] = col_min
                self.scale_[col] = 0  # Flag for constant column
            else:
                # Store min and calculate scale factor for non-constant columns
                self.min_[col] = col_min
                # Avoid division by zero for nearly constant columns
                if col_max - col_min > 1e-10:
                    self.scale_[col] = (self.feature_range[1] - self.feature_range[0]) / (col_max - col_min)
                else:
                    # For nearly constant columns, set scale to 0 to keep original value
                    self.scale_[col] = 0
                    
        return self
        
    def transform(self, X):
        """Scale features according to feature_range.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        if not self.min_:
            self.fit(X)
            
        result = X.copy()
        
        f_min, f_max = self.feature_range
        
        for col in self.min_.keys():
            if col not in X.columns:
                continue
                
            # Skip columns marked as constant
            if self.scale_[col] == 0:
                continue
                
            # Apply scaling: X_std = (X - X.min) / (X.max - X.min) -> X_scaled = X_std * (max - min) + min
            result[col] = (X[col] - self.min_[col]) * self.scale_[col] + f_min
            
        return result
        
    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        if not self.min_:
            raise ValueError("This MinMaxScaler instance is not fitted yet. Call 'fit' before using this method.")
            
        result = X.copy()
        
        f_min, f_max = self.feature_range
        
        for col in self.min_.keys():
            if col not in X.columns:
                continue
                
            # Skip columns marked as constant
            if self.scale_[col] == 0:
                continue
                
            # Apply inverse scaling: X_original = (X_scaled - min) / (max - min) * (X.max - X.min) + X.min
            result[col] = (X[col] - f_min) / self.scale_[col] + self.min_[col]
            
        return result

class StandardScalerTransformer(BaseTransformer):
    """Wrapper around sklearn's StandardScaler for DataFrame compatibility.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    copy : bool, default=True
        If True, a copy of X will be created. If False, centering and scaling happen in-place.
    columns : list, optional
        List of column names to be transformed. If None, all columns will be transformed.
    """
    
    def __init__(self, with_mean=True, with_std=True, copy=True, columns=None):
        self.with_mean = with_mean
        self.with_std = with_std  
        self.copy = copy
        self.columns = columns
        self._sklearn_scaler = None
        self._fitted_columns = None
        
    def fit(self, X):
        # Determine which columns to fit
        columns = self.columns if self.columns is not None else X.columns
        columns = [col for col in columns if col in X.columns]
        self._fitted_columns = columns
        
        # Create sklearn StandardScaler
        self._sklearn_scaler = StandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=self.copy
        )
        
        # Fit on numpy array (sklearn expects this)
        if columns:
            self._sklearn_scaler.fit(X[columns].values)
        return self
        
    def transform(self, X):
        if self._sklearn_scaler is None:
            raise ValueError("Transformer must be fitted before transform")
        
        result = X.copy()
        
        if self._fitted_columns:
            # Transform using sklearn
            transformed_values = self._sklearn_scaler.transform(X[self._fitted_columns].values)
            
            # Update only the fitted columns in the result
            result[self._fitted_columns] = transformed_values
            
        return result
            
    def inverse_transform(self, X):
        if self._sklearn_scaler is None:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        result = X.copy()
        
        if self._fitted_columns:
            # Inverse transform using sklearn
            inverse_values = self._sklearn_scaler.inverse_transform(X[self._fitted_columns].values)
            
            # Update only the fitted columns in the result
            result[self._fitted_columns] = inverse_values
            
        return result

class NormalScoreTransformer(BaseTransformer):
    """A transformer for normal score transformation.
    
    Parameters
    ----------
    tol : float, default=1e-7
        Tolerance for convergence in random generation.
    max_samples : int, default=1000000
        Maximum number of samples for random generation.
    quadratic_extrapolation : bool, default=False
        Whether to use quadratic extrapolation for values outside the fitted range.
    columns : list, optional
        List of column names to be transformed. If None, all columns will be transformed.
    """

    def __init__(self, tol=1e-7, max_samples=1000000, quadratic_extrapolation=False, columns=None):
        self.tol = tol
        self.max_samples = max_samples
        self.quadratic_extrapolation = quadratic_extrapolation
        self.columns = columns
        self.column_parameters = {}
        self.shared_z_scores = {}

    def fit(self, X):
        """Fit the transformer to the data."""
        columns = self.columns if self.columns is not None else X.columns
        columns = [col for col in columns if col in X.columns]
        
        for col in columns:
            values = X[col].values
            sorted_vals = np.sort(values)
            smoothed_vals = self._moving_average_with_endpoints(sorted_vals)

            n_points = len(smoothed_vals)
            if n_points not in self.shared_z_scores:
                self.shared_z_scores[n_points] = self._randrealgen_optimized(n_points)

            z_scores = self.shared_z_scores[n_points]
            
            self.column_parameters[col] = {
                'z_scores': z_scores,
                'originals': smoothed_vals,
            }
        return self
        
    def transform(self, X):
        """Transform the data using normal score transformation.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with normal scores.
        """
        result = X.copy()
        for col in self.column_parameters.keys():
            if col not in X.columns:
                continue
                
            params = self.column_parameters.get(col, {})
            z_scores = params.get('z_scores', [])
            originals = params.get('originals', [])
            
            if len(z_scores) == 0 or len(originals) == 0:
                continue
                
            values = X[col].values
            
            # Handle values outside the original range
            min_orig, max_orig = np.min(originals), np.max(originals)
            min_z, max_z = np.min(z_scores), np.max(z_scores)
            
            # For values within range, use interpolation
            within_range = (values >= min_orig) & (values <= max_orig)
            if within_range.any():
                result.loc[within_range, col] = np.interp(
                    values[within_range], originals, z_scores
                )
                
            # For values outside range, use extrapolation if enabled or clamp to bounds
            below_min = values < min_orig
            above_max = values > max_orig
            
            if below_min.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation below minimum
                    slope = (z_scores[1] - z_scores[0]) / (originals[1] - originals[0])
                    result.loc[below_min, col] = min_z + slope * (values[below_min] - min_orig)
                else:
                    # Otherwise clamp to minimum z-score
                    result.loc[below_min, col] = min_z
                    
            if above_max.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation above maximum
                    slope = (z_scores[-1] - z_scores[-2]) / (originals[-1] - originals[-2])
                    result.loc[above_max, col] = max_z + slope * (values[above_max] - max_orig)
                else:
                    # Otherwise clamp to maximum z-score
                    result.loc[above_max, col] = max_z
            
        return result

    def inverse_transform(self, X):
        """Inverse transform data back to original space.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame with transformed data to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        result = X.astype(float).copy()
        for col in self.column_parameters.keys():
            if col not in X.columns:
                continue
                
            params = self.column_parameters.get(col, {})
            z_scores = params.get('z_scores', [])
            originals = params.get('originals', [])
            if len(z_scores) == 0 or len(originals) == 0:
                continue

            # Get values to inverse transform
            values = X[col].values
            min_z, max_z = np.min(z_scores), np.max(z_scores)
            min_orig, max_orig = np.min(originals), np.max(originals)

            # For values within the z-score range, use interpolation
            within_range = (values >= min_z) & (values <= max_z)
            if within_range.any():
                result.loc[within_range, col] = np.interp(values[within_range], z_scores, originals)
            
            # For values outside the z-score range, use extrapolation if enabled
            below_min = values < min_z
            above_max = values > max_z
            
            if below_min.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation below minimum z-score
                    slope = (originals[1] - originals[0]) / (z_scores[1] - z_scores[0])
                    intercept = originals[0] - slope * z_scores[0]
                    result.loc[below_min, col] = slope * values[below_min] + intercept
                else:
                    # Otherwise clamp to minimum original value
                    result.loc[below_min, col] = min_orig
                    
            if above_max.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation above maximum z-score
                    slope = (originals[-1] - originals[-2]) / (z_scores[-1] - z_scores[-2])
                    intercept = originals[-1] - slope * z_scores[-1]
                    result.loc[above_max, col] = slope * values[above_max] + intercept
                else:
                    # Otherwise clamp to maximum original value
                    result.loc[above_max, col] = max_orig

        return result

    def _randrealgen_optimized(self, nreal):
        rval = np.zeros(nreal)
        nsamp = 0
        numsort = (nreal + 1) // 2 if nreal % 2 == 0 else nreal // 2

        while nsamp < self.max_samples:
            nsamp += 1
            work1 = np.random.normal(size=nreal)
            work1.sort()

            if nsamp > 1:
                previous_mean = rval[:numsort] / (nsamp - 1)
                rval[:numsort] += work1[:numsort]
                current_mean = rval[:numsort] / nsamp
                max_diff = np.max(np.abs(current_mean - previous_mean))

                if max_diff <= self.tol:
                    break
            else:
                rval[:numsort] = work1[:numsort]

        rval[:numsort] /= nsamp
        rval[numsort:] = -rval[:numsort][::-1] if nreal % 2 == 0 else np.concatenate(([-rval[numsort]], -rval[:numsort][::-1]))
        return rval

    def _moving_average_with_endpoints(self, y_values):
        """Apply a moving average smoothing to an array while preserving endpoints."""
        window_size = 3
        if y_values.shape[0] > 40:
            window_size = 5
        if y_values.shape[0] > 90:
            window_size = 7
        if y_values.shape[0] > 200:
            window_size = 9

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        half_window = window_size // 2
        smoothed_y = np.zeros_like(y_values)

        # Handle start points correctly
        for i in range(0, half_window):
            smoothed_y[i] = np.mean(y_values[:i + half_window + 1])
        
        # Handle end points correctly 
        for i in range(1, half_window + 1):
            smoothed_y[-i] = np.mean(y_values[-(i + half_window):])
        
        # Middle points
        for i in range(half_window, len(y_values) - half_window):
            smoothed_y[i] = np.mean(y_values[i - half_window:i + half_window + 1])

        # Preserve original endpoints exactly
        smoothed_y[0] = y_values[0]
        smoothed_y[-1] = y_values[-1]
        
        # Ensure monotonicity
        for i in range(1, len(smoothed_y)):
            if smoothed_y[i] <= smoothed_y[i - 1]:
                smoothed_y[i] = smoothed_y[i - 1] + 1e-16

        return smoothed_y

class TransformerPipeline:
    """Apply a sequence of transformers in order."""

    def __init__(self):
        self.transformers = []
        self.fitted = False

    def add(self, transformer, columns=None):
        """Add a transformer to the pipeline, optionally for specific columns."""
        self.transformers.append((transformer, columns))
        return self

    def fit(self, X):
        """Fit all transformers in the pipeline."""
        for transformer, columns in self.transformers:
            cols_to_transform = columns if columns is not None else X.columns
            sub_X = X[cols_to_transform]
            transformer.fit(sub_X)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform data using all transformers in the pipeline.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        result = X.copy()
        for transformer, columns in self.transformers:
            cols_to_transform = columns if columns is not None else X.columns
            # Only use columns that exist in the input data
            valid_cols = [col for col in cols_to_transform if col in X.columns]
            if not valid_cols:
                continue
            sub_X = result[valid_cols]
            result[valid_cols] = transformer.transform(sub_X)
        return result

    def fit_transform(self, X):
        """Fit all transformers and transform data in one operation."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Apply inverse transformations in reverse order.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        
        if isinstance(X, pd.Series):
            result = X.copy().to_frame().T
        else:
            result = X.copy().astype(np.float32)
        # Need to reverse the order of transformers for inverse
        for transformer, columns in reversed(self.transformers):
            cols_to_transform = columns if columns is not None else result.columns
            # Only use columns that exist in the input data
            valid_cols = [col for col in cols_to_transform if col in result.columns]
            if not valid_cols:
                continue
            sub_X = result[valid_cols].copy()  # Create a copy to avoid reference issues
            inverted = transformer.inverse_transform(sub_X)
            result.loc[:, valid_cols] = np.array(inverted, dtype=np.float32).flatten().reshape(result.loc[:, valid_cols].shape)  # Use loc for proper assignment
        if isinstance(X, pd.Series):
            result = result.iloc[0]
        return result

class AutobotsAssemble:
    """Class for transforming features in a DataFrame using a pipeline approach."""

    def __init__(self, df=None):
        self.df = df.copy() if df is not None else None
        self.pipeline = TransformerPipeline()

    def apply(self, transform_type, columns=None, **kwargs):
        """Apply a transformation to specified columns."""
        transformer = self._create_transformer(transform_type, **kwargs)
        if columns is None:
            columns = list(self.df.columns)  # Convert to list to avoid pandas index issues
        
        # Fit transformer to data if needed
        if hasattr(transformer, 'fit') and callable(transformer.fit):
            if self.df is not None:
                df_subset = self.df[columns]
                transformer.fit(df_subset)
        
        # Add to pipeline
        self.pipeline.add(transformer, columns)
        
        # Apply transformation to current df if available
        if self.df is not None:
            # Use transform directly to ensure correct application
            df_subset = self.df[columns].copy()
            transformed = transformer.transform(df_subset)
            self.df[columns] = transformed
            
        return self

    def transform(self, df):
        """Transform an external DataFrame using the pipeline.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        if self.pipeline.transformers:
            return self.pipeline.transform(df)
        return df.copy()

    def inverse(self, df=None):
        """Apply inverse transformations in reverse order."""
        to_transform = df if df is not None else self.df
        result = self.pipeline.inverse_transform(to_transform)
        if df is None:
            self.df = result
        return result

    def inverse_on_external_df(self, df, columns=None):
        """Apply inverse transformations to an external DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to inverse transform.
        columns : list, optional
            Specific columns to inverse transform. If None, all columns are processed.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        to_transform = df.copy()
        if columns is not None:
            # Ensure we only process specified columns
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        return self.pipeline.inverse_transform(to_transform)

    def _create_transformer(self, transform_type, **kwargs):
        """Factory method to create appropriate transformer."""
        if transform_type == "log10":
            return Log10Transformer(**kwargs)
        elif transform_type == "normal_score":
            return NormalScoreTransformer(**kwargs)
        elif transform_type == "row_wise_minmax":
            return RowWiseMinMaxScaler(**kwargs)
        elif transform_type == "standard_scaler":
            return StandardScalerTransformer(**kwargs)
        elif transform_type == "minmax_scaler":
            return MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")