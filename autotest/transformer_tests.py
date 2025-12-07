import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import platform
sys.path.append("..")
import pyemu

def test_base_transformer():
    """Test the BaseTransformer abstract class functionality"""
    bt = pyemu.emulators.BaseTransformer()
    
    # fit should return self
    assert bt.fit(None) is bt
    
    # fit_transform should call fit and transform
    with pytest.raises(NotImplementedError):
        bt.fit_transform(None)
    
    # transform should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        bt.transform(None)
    
    # inverse_transform should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        bt.inverse_transform(None)

def test_log10_transformer():
    """Test the Log10Transformer functionality"""
    # Create test dataframe with positive and negative values
    df = pd.DataFrame({
        'pos': [1, 10, 100, 1000],
        'zero': [0, 0.1, 0.01, 0.001],
        'neg': [-1, -10, -100, -1000]
    })
    
    # Initialize and test transformer
    lt = pyemu.emulators.Log10Transformer()
    
    # Transform data
    transformed = lt.transform(df)
    
    # Check that positive values are properly transformed
    np.testing.assert_allclose(
        transformed['pos'].values,
        np.log10(df['pos'].values)
    )
    
    # Check that zeros/small values are handled correctly
    assert not np.any(np.isinf(transformed['zero'].values))
    
    # Check that negative values are handled correctly
    assert not np.any(np.isnan(transformed['neg'].values))
    
    # Test inverse transform
    back_transformed = lt.inverse_transform(transformed)
    
    # Check that we get back very close to original values
    np.testing.assert_allclose(
        back_transformed['pos'].values, 
        df['pos'].values
    )
    
    # For zero/very small values
    np.testing.assert_allclose(
        back_transformed['zero'].values,
        df['zero'].values ,
        rtol=1e-6
    )
    
    # For negative values
    np.testing.assert_allclose(
        back_transformed['neg'].values,
        df['neg'].values ,
        rtol=1e-6
    )

def test_row_wise_minmax_scaler():
    """Test the RowWiseMinMaxScaler functionality"""
    # Test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [100, 200, 300, 400]
    })
    
    # Initialize scaler
    scaler = pyemu.emulators.RowWiseMinMaxScaler()
    
    # Fit and transform
    transformed = scaler.fit_transform(df)
    
    # Check each row is scaled to [0, 1]
    for i in range(len(df)):
        row_min = transformed.iloc[i].min()
        row_max = transformed.iloc[i].max()
        assert np.isclose(row_min, -1.0)
        assert np.isclose(row_max, 1.0)
    
    # Test inverse transform
    back_transformed = scaler.inverse_transform(transformed)
    
    # Check we get back original values
    np.testing.assert_allclose(back_transformed.values, df.values)

def test_normal_score_transformer():
    """Test the NormalScoreTransformer functionality"""
    # Create test data with various distributions
    np.random.seed(42)
    n = 200
    
    # Uniform data
    uniform_data = np.random.uniform(0, 10, n)
    
    # Log-normal data
    lognormal_data = np.exp(np.random.normal(0, 1, n))
    
    # Bimodal data
    bimodal_data = np.concatenate([
        np.random.normal(-3, 1, n//2),
        np.random.normal(3, 1, n//2)
    ])
    
    df = pd.DataFrame({
        'uniform': uniform_data,
        'lognormal': lognormal_data,
        'bimodal': bimodal_data
    })
    
    # Initialize transformer
    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    
    # Transform data
    transformed = nst.fit_transform(df)
    
    # Check transformed distributions are more normal
    # For each column, check skewness and kurtosis are closer to normal
    for col in df.columns:
        # Calculate statistics of original and transformed data
        orig_skew = skewness(df[col].values)
        trans_skew = skewness(transformed[col].values)
        
        orig_kurt = kurtosis(df[col].values)
        trans_kurt = kurtosis(transformed[col].values)
        
        # Transformed data should have skewness closer to 0
        assert abs(trans_skew) < abs(orig_skew) or np.isclose(abs(trans_skew), 0, atol=0.5)
        
        # Transformed data should have kurtosis closer to 3 (normal distribution)
        assert abs(trans_kurt - 3) < abs(orig_kurt - 3) or np.isclose(trans_kurt, 3, atol=1.0)
    
    # Test inverse transform
    back_transformed = nst.inverse_transform(transformed)
    
    # Check we get back close to original values 
    # (not exact due to binning and smoothing)
    np.testing.assert_allclose(
        back_transformed.values, 
        df.values,
        rtol=0.1,
        atol=0.1
    )
    
    # Test with quadratic extrapolation
    nst_quad = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    transformed_quad = nst_quad.fit_transform(df)
    
    # Create data outside the original range for extrapolation test
    # Transform should not fail for out-of-range values when using quadratic extrapolation
    extreme_transformed = transformed_quad.copy()
    extreme_transformed.loc[0] = transformed_quad.min() - 1
    extreme_transformed.loc[1] = transformed_quad.max() + 1
    
    back_extreme = nst_quad.inverse_transform(extreme_transformed)
    assert not np.any(np.isnan(back_extreme.values))
    assert not np.any(np.isinf(back_extreme.values))

def test_transformer_pipeline():
    """Test the TransformerPipeline functionality"""
    # Create test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [100, 200, 300, 400]
    })
    
    # Create pipeline with multiple transformers
    pipeline = pyemu.emulators.TransformerPipeline()
    
    # Add log transformer for all columns
    log_trans = pyemu.emulators.Log10Transformer()
    pipeline.add(log_trans)
    
    # Add row-wise min-max scaler for specific columns
    minmax_trans = pyemu.emulators.RowWiseMinMaxScaler()
    pipeline.add(minmax_trans, columns=['a', 'b'])
    
    # Transform data
    transformed = pipeline.transform(df)
    
    # Check log was applied to all columns
    np.testing.assert_allclose(
        transformed['c'].values,
        np.log10(df['c'].values)
    )
    
    # Check minmax was applied only to a and b
    for i in range(len(df)):
        row_subset = transformed.iloc[i][['a', 'b']]
        assert np.isclose(row_subset.min(), 0.0) or np.isclose(row_subset.max(), 1.0)
    
    # Test inverse transform
    back_transformed = pipeline.inverse_transform(transformed)
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_transformed.values, df.values, rtol=1e-5)

def test_autobots_assemble():
    """Test the AutobotsAssemble class functionality"""
    # Create test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [-10, -20, -30, -40]
    })
    
    # Save original data for comparison
    original_df = df.copy()
    
    # Initialize with data
    aa = pyemu.emulators.AutobotsAssemble(df)
    
    # Apply log transform to positive columns
    aa.apply('log10', columns=['a', 'b'])
    
    # Check the transform was applied correctly
    np.testing.assert_allclose(
        aa.df[['a', 'b']].values,
        np.log10(original_df[['a', 'b']].values)
    )
    
    # Check that column c is unchanged
    np.testing.assert_array_equal(aa.df['c'].values, original_df['c'].values)
    
    # Save intermediate state after log transform
    log_transformed = aa.df.copy()
    
    # Apply normal score transform to all columns
    aa.apply('normal_score')
    
    # Save state after normal score transform
    normal_transformed = aa.df.copy()
    
    # Verify both transforms were applied (data should be different from log transform)
    assert not np.allclose(normal_transformed.values, log_transformed.values)
    
    # Apply the inverse transformation
    back_transformed = aa.inverse()
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_transformed.values, original_df.values, rtol=0.1)
    
    # Test with external already-transformed data
    external_transformed = pd.DataFrame({
        'a': [-0.5, 0.0, 0.5],  # Already transformed data in normal score space
        'b': [0.5, 0.0, -0.5],  # (approximately in the normal distribution range)
        'c': [1.0, 0.0, -1.0]
    })
    
    # Test inverse transform on external transformed data
    back_external = aa.inverse(external_transformed)
    
    # Check that shape is preserved
    assert back_external.shape == external_transformed.shape
    
    # Verify output has reasonable values (should be in the range of original data)
    for col in ['a', 'b']:
        # These columns had log transform applied, so should be positive
        assert np.all(back_external[col] > 0)
    
    # Column c should have values in the range of the original data
    assert np.min(back_external['c']) >= -40
    assert np.max(back_external['c']) <= -10
    
    # Apply transform again to verify roundtrip accuracy
    roundtrip = aa.transform(back_external)
    
    # Check roundtrip accuracy for values within standard normal range (-2 to 2)
    for col in external_transformed.columns:
        # Find values within the normal range
        mask = (external_transformed[col] >= -2) & (external_transformed[col] <= 2)
        if mask.any():
            # Get the values to compare
            expected = external_transformed.loc[mask, col].values
            actual = roundtrip.loc[mask, col].values
            
            # Handle zeros and near-zeros with absolute tolerance instead of relative
            zero_mask = np.isclose(expected, 0, atol=1e-10)
            if zero_mask.any():
                # For zeros, use absolute tolerance
                np.testing.assert_allclose(
                    actual[zero_mask],
                    expected[zero_mask],
                    atol=0.1  # Absolute tolerance for zeros
                )
                
                # For non-zeros, use relative tolerance
                if (~zero_mask).any():
                    np.testing.assert_allclose(
                        actual[~zero_mask],
                        expected[~zero_mask],
                        rtol=0.1  # Relative tolerance for non-zeros
                    )
            else:
                # No zeros, use normal comparison
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.1
                )
    
    # Additional test to verify pipeline order is maintained
    # Create a new pipeline with transforms in different order
    bb = pyemu.emulators.AutobotsAssemble(original_df.copy())
    
    # First normal score, then log10
    bb.apply('normal_score')
    bb.apply('log10', columns=['a', 'b'])
    
    # Apply inverse - should revert log10 first, then normal_score
    back_bb = bb.inverse()
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_bb.values, original_df.values, rtol=0.1)



def skewness(x):
    """Calculate skewness of a distribution"""
    n = len(x)
    x_mean = np.mean(x)
    return (np.sum((x - x_mean) ** 3) / n) / ((np.sum((x - x_mean) ** 2) / n) ** 1.5)

def kurtosis(x):
    """Calculate kurtosis of a distribution"""
    n = len(x)
    x_mean = np.mean(x)
    return (np.sum((x - x_mean) ** 4) / n) / ((np.sum((x - x_mean) ** 2) / n) ** 2)




def test_normal_score_with_external_data():
    """Test NormalScoreTransformer with external already-transformed data"""
    # Create training data with a specific distribution
    np.random.seed(42)
    n = 100
    training_data = pd.DataFrame({
        'normal': np.random.normal(5, 2, n),
        'lognormal': np.exp(np.random.normal(1, 0.5, n)),
        'uniform': np.random.uniform(0, 10, n)
    })
    
    # Create "external" data that we'll pretend is already transformed
    # For this test, we'll generate values in the typical normal score range (-3 to 3)
    external_transformed = pd.DataFrame({
        'normal': np.random.normal(0, 1, 1),  # Already in normal score space
        'lognormal': np.random.normal(0, 1, 1),
        'uniform': np.random.normal(0, 1, 1)
    })
    
    # Initialize and fit transformer on training data
    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(training_data)
    
    # Transform training data to verify transformation works
    transformed_training = nst.transform(training_data)
    
    # Check that transformed data has properties of normal distribution
    for col in training_data.columns:
        # Mean should be close to 0
        assert abs(transformed_training[col].mean()) < 0.3
        # Standard deviation should be close to 1
        assert abs(transformed_training[col].std() - 1.0) < 0.3
    
    # Store column parameters for inspection
    z_scores = {}
    originals = {}
    for col in training_data.columns:
        params = nst.column_parameters.get(col, {})
        z_scores[col] = params.get('z_scores', [])
        originals[col] = params.get('originals', [])
        
        # Verify column parameters were created
        assert len(z_scores[col]) > 0
        assert len(originals[col]) > 0
    
    # Apply inverse transform to external transformed data directly
    back_external = nst.inverse_transform(external_transformed)
    
    # Verify the shape matches
    assert back_external.shape == external_transformed.shape
    
    # Apply the transform to back_external to check if it recovers external_transformed
    re_transformed = nst.transform(back_external)
    
    # Check that re-transforming recovers values close to the external_transformed
    # Note: exact recovery isn't expected due to interpolation/extrapolation
    for col in external_transformed.columns:
        # Values inside the normal range (-2 to 2) should be very close
        inside_range = (external_transformed[col] >= -2) & (external_transformed[col] <= 2)
        if inside_range.any():
            np.testing.assert_allclose(
                re_transformed.loc[inside_range, col].values,
                external_transformed.loc[inside_range, col].values,
                rtol=0.2
            )
    
    # Test external values that are far outside the z-score range
    extreme_transformed = pd.DataFrame({
        'normal': np.array([-5, 0, 5],dtype=float),  # Includes extreme values
        'lognormal': np.array([-5, 0, 5],dtype=float),
        'uniform': np.array([-5, 0, 5],dtype=float)
    })
    
    # Test with extrapolation first
    nst_extrap = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst_extrap.fit(training_data)
    back_extreme_extrap = nst_extrap.inverse_transform(extreme_transformed)
    
    # Test without extrapolation
    nst_no_extrap = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    nst_no_extrap.fit(training_data)
    back_extreme_no_extrap = nst_no_extrap.inverse_transform(extreme_transformed)
    
    # With extrapolation, extreme values should be outside the original data range
    for col in training_data.columns:
        min_orig = training_data[col].min()
        max_orig = training_data[col].max()
        
        # Check extrapolation is working (values outside original range)
        assert back_extreme_extrap[col].min() < min_orig or back_extreme_extrap[col].max() > max_orig
        
        # Without extrapolation, values should be clamped to original range
        assert back_extreme_no_extrap[col].min() >= min_orig - 1e-10  # Allow for floating point error
        assert back_extreme_no_extrap[col].max() <= max_orig + 1e-10
    
    # Test with AutobotsAssemble to ensure the pipeline works with external transformed data
    aa = pyemu.emulators.AutobotsAssemble(training_data.copy())
    aa.apply('normal_score')
    
    # Test applying inverse transform to external data
    back_from_aa = aa.inverse(external_transformed.copy())
    
    # Verify results with direct inverse transform
    np.testing.assert_allclose(
        back_from_aa.values,
        nst.inverse_transform(external_transformed).values,
        rtol=1e-3
    )