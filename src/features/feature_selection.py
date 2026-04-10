"""SHAP-driven feature selection for the Power Systems IDS."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


def select_features_by_shap(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    keep_percentage: float = 0.8,
    return_indices: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Select features based on SHAP importance scores.
    
    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        keep_percentage: Percentage of features to keep (default: 0.8 for 80%)
        return_indices: If True, return feature indices to keep
        
    Returns:
        Tuple of (feature_indices_to_keep, feature_names_to_keep)
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Determine number of features to keep
    n_features = len(mean_abs_shap)
    n_keep = int(n_features * keep_percentage)
    
    # Get indices of top features by importance
    feature_importance_indices = np.argsort(mean_abs_shap)[::-1][:n_keep]
    feature_importance_indices = np.sort(feature_importance_indices)  # Keep in original order
    
    # Get feature names if provided
    if feature_names is not None:
        feature_names_to_keep = [feature_names[i] for i in feature_importance_indices]
    else:
        feature_names_to_keep = [f"f{i}" for i in feature_importance_indices]
    
    return feature_importance_indices, feature_names_to_keep


def prune_features(
    X: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """
    Prune feature matrix to keep only selected features.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        feature_indices: Indices of features to keep
        
    Returns:
        Pruned feature matrix of shape (n_samples, n_selected_features)
    """
    return X[:, feature_indices]


def get_feature_importance_ranking(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """
    Get feature importance ranking based on SHAP values.
    
    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        
    Returns:
        List of dictionaries with feature names and importance scores
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(mean_abs_shap))]
    
    # Sort by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    
    ranking = [
        {"name": feature_names[i], "importance": float(mean_abs_shap[i])}
        for i in sorted_indices
    ]
    
    return ranking


def apply_feature_selection_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shap_values: np.ndarray,
    keep_percentage: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply SHAP-driven feature selection to train/val/test sets.
    
    Args:
        X_train: Training feature matrix
        X_val: Validation feature matrix
        X_test: Test feature matrix
        shap_values: SHAP values from training data
        keep_percentage: Percentage of features to keep
        
    Returns:
        Tuple of (X_train_pruned, X_val_pruned, X_test_pruned, feature_indices)
    """
    # Select features based on SHAP importance
    feature_indices, _ = select_features_by_shap(
        shap_values, keep_percentage=keep_percentage, return_indices=True
    )
    
    # Prune all datasets
    X_train_pruned = prune_features(X_train, feature_indices)
    X_val_pruned = prune_features(X_val, feature_indices)
    X_test_pruned = prune_features(X_test, feature_indices)
    
    return X_train_pruned, X_val_pruned, X_test_pruned, feature_indices
