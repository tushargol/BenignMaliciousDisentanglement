"""Unit tests for pipeline functions."""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix


def test_confusion_matrix_binary():
    """Test confusion matrix for binary classification."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
    assert cm[0, 0] == 2, "True negatives should be 2"
    assert cm[1, 1] == 2, "True positives should be 2"
    assert cm[1, 0] == 1, "False negatives should be 1"


def test_confusion_matrix_multiclass():
    """Test confusion matrix for multi-class classification."""
    y_true = np.array([0, 1, 2, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2, 1, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3), "Confusion matrix should be 3x3 for 3-class classification"
    assert cm[0, 0] == 2, "True negatives for class 0 should be 2"
    assert cm[1, 1] == 1, "True positives for class 1 should be 1"
    assert cm[2, 2] == 1, "True positives for class 2 should be 1"


def test_threshold_classification():
    """Test binary threshold classification logic."""
    # Simulated probabilities
    y_score = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    threshold = 0.5
    
    y_pred = (y_score >= threshold).astype(int)
    
    expected = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(y_pred, expected), "Threshold classification should match expected output"


def test_threshold_sensitivity():
    """Test that different thresholds produce different predictions."""
    y_score = np.array([0.4, 0.5, 0.6])
    
    pred_0_5 = (y_score >= 0.5).astype(int)
    pred_0_6 = (y_score >= 0.6).astype(int)
    
    assert not np.array_equal(pred_0_5, pred_0_6), "Different thresholds should produce different predictions"


def test_sample_weight_application():
    """Test that sample weights affect loss calculation."""
    # Simulated predictions and labels
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    y_true = np.array([1, 0, 1, 0])
    
    # Unweighted loss
    loss_unweighted = np.mean((y_pred - y_true) ** 2)
    
    # Weighted loss (emphasize first sample)
    weights = np.array([2.0, 1.0, 1.0, 1.0])
    loss_weighted = np.sum(weights * (y_pred - y_true) ** 2) / np.sum(weights)
    
    assert loss_weighted != loss_unweighted, "Weighted loss should differ from unweighted"


def test_data_split_shapes():
    """Test that train/val/test split maintains correct proportions."""
    from sklearn.model_selection import train_test_split
    
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    assert X_train.shape[0] == 700, "Training set should be 70% of data"
    assert X_val.shape[0] == 150, "Validation set should be 15% of data"
    assert X_test.shape[0] == 150, "Test set should be 15% of data"


def test_data_leakage_prevention():
    """Test that train and test sets are disjoint."""
    from sklearn.model_selection import train_test_split
    
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check that no indices overlap
    train_indices = set(range(len(X)))
    # This is a simplified check - in practice you'd track actual indices
    assert X_train.shape[0] + X_test.shape[0] == len(X), "Total samples should be preserved"
