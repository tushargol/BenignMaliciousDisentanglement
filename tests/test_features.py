"""Unit tests for feature extraction functions."""

import numpy as np
import pytest
from src.features.feature_engineering import basic_window_stats


def test_basic_window_stats_shape():
    """Test that basic_window_stats returns correct output shape."""
    # Create sample window: 90 time steps, 10 features
    x_win = np.random.randn(90, 10)
    
    result = basic_window_stats(x_win)
    
    # Expected output shape: 14 stats per feature * 10 features = 140
    expected_shape = 140
    assert result.shape == (expected_shape,), f"Expected shape {expected_shape}, got {result.shape}"


def test_basic_window_stats_with_single_timestep():
    """Test edge case with single timestep."""
    x_win = np.random.randn(1, 10)
    
    result = basic_window_stats(x_win)
    
    # Should still work, returning correct shape
    assert result.shape == (140,)


def test_basic_window_stats_statistics():
    """Test that basic_window_stats computes reasonable statistics."""
    # Create window with known properties
    x_win = np.ones((90, 10))  # All ones
    
    result = basic_window_stats(x_win)
    
    # Mean should be 1.0 for all features
    mean_values = result[:10]
    assert np.allclose(mean_values, 1.0, atol=1e-6), "Mean should be 1.0"
    
    # Std should be 0.0 for all features
    std_values = result[10:20]
    assert np.allclose(std_values, 0.0, atol=1e-6), "Std should be 0.0"


def test_basic_window_stats_handles_nan():
    """Test that basic_window_stats handles NaN values gracefully."""
    x_win = np.random.randn(90, 10)
    x_win[0, 0] = np.nan
    x_win[45, 5] = np.nan
    
    result = basic_window_stats(x_win)
    
    # Should return valid result without raising error
    assert not np.any(np.isnan(result)), "Result should not contain NaN values"


def test_basic_window_stats_variance():
    """Test that basic_window_stats captures variance in data."""
    # Create window with high variance
    x_win = np.random.randn(90, 10) * 10
    
    result = basic_window_stats(x_win)
    
    # Std should be high for all features
    std_values = result[10:20]
    assert np.all(std_values > 1.0), "Std should capture high variance"
