"""Cross-validation utilities for statistical validity."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, TensorDataset


def cross_validate_autoencoder(
    X: np.ndarray,
    input_dim: int,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    n_folds: int = 5,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation for autoencoder training.
    
    Args:
        X: Feature matrix (should be normal + benign data only)
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        n_folds: Number of cross-validation folds
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        seed: Random seed
        
    Returns:
        Dictionary with lists of metrics across folds
    """
    from ..models.autoencoder import Autoencoder
    from ..training.train_autoencoder import train_autoencoder
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Create dummy labels for stratification (all same class for normal data)
    y = np.zeros(len(X))
    
    results = {
        "fold": [],
        "train_loss": [],
        "val_loss": [],
        "reconstruction_error_mean": [],
        "reconstruction_error_std": [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"[CV] Fold {fold + 1}/{n_folds}")
        
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        # Train autoencoder on this fold
        model = train_autoencoder(
            X_train_fold,
            X_val_fold,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val_fold).float().to(device)
            reconstructed = model(X_val_tensor)
            reconstruction_error = torch.mean((X_val_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        results["fold"].append(fold + 1)
        results["reconstruction_error_mean"].append(float(np.mean(reconstruction_error)))
        results["reconstruction_error_std"].append(float(np.std(reconstruction_error)))
    
    return results


def cross_validate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dim: int = 128,
    n_folds: int = 5,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    sample_weights: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation for classifier training.
    
    Args:
        X: Feature matrix (anomalous windows only)
        y: Binary labels (0=benign, 1=malicious)
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        n_folds: Number of cross-validation folds
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        sample_weights: Optional sample weights
        seed: Random seed
        
    Returns:
        Dictionary with lists of metrics across folds
    """
    from ..models.classifier import AnomalyClassifier
    from ..training.train_classifier import train_classifier
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    results = {
        "fold": [],
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"[CV] Fold {fold + 1}/{n_folds}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Handle sample weights
        w_train_fold = sample_weights[train_idx] if sample_weights is not None else None
        w_val_fold = sample_weights[val_idx] if sample_weights is not None else None
        
        # Train classifier on this fold
        model = train_classifier(
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sample_weights_train=w_train_fold,
            sample_weights_val=w_val_fold,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val_fold).float().to(device)
            logits = model(X_val_tensor)
            predictions = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_fold, predictions, average='binary', zero_division=0
        )
        
        results["fold"].append(fold + 1)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
    
    return results


def summarize_cv_results(cv_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Summarize cross-validation results with mean and std.
    
    Args:
        cv_results: Dictionary with lists of metrics across folds
        
    Returns:
        Dictionary with mean and std for each metric
    """
    summary = {}
    
    for key, values in cv_results.items():
        if key == "fold":
            continue
        if isinstance(values, list) and len(values) > 0:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    
    return summary


def print_cv_summary(cv_results: Dict[str, List[float]], model_name: str = "Model"):
    """
    Print a formatted summary of cross-validation results.
    
    Args:
        cv_results: Dictionary with lists of metrics across folds
        model_name: Name of the model being evaluated
    """
    summary = summarize_cv_results(cv_results)
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results: {model_name}")
    print(f"{'='*60}")
    
    for metric, stats in summary.items():
        print(f"{metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    print(f"{'='*60}\n")
