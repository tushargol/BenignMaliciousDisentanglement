"""Stage 1: Autoencoder training and inference for anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

from ..models.autoencoder import Autoencoder
from ..training.train_autoencoder import train_autoencoder
from ..config import Paths


def train_stage1(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[str] = None,
    model_save_path: Optional[Path] = None,
) -> Autoencoder:
    """
    Train Stage 1 autoencoder for anomaly detection.
    
    Args:
        X_train: Training data (normal + benign only)
        X_val: Validation data
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use (cuda/cpu)
        model_save_path: Path to save trained model
        
    Returns:
        Trained autoencoder model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Stage 1] Training autoencoder on device: {device}")
    print(f"[Stage 1] Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    model = train_autoencoder(
        X_train,
        X_val,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    
    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"[Stage 1] Saved model to {model_save_path}")
    
    return model


def load_stage1(
    model_path: Path,
    input_dim: int,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    device: Optional[str] = None,
) -> Autoencoder:
    """
    Load trained Stage 1 autoencoder.
    
    Args:
        model_path: Path to saved model weights
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        device: Device to load model on
        
    Returns:
        Loaded autoencoder model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"[Stage 1] Loaded model from {model_path}")
    return model


def infer_stage1(
    model: Autoencoder,
    X: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Run Stage 1 inference to get reconstruction errors.
    
    Args:
        model: Trained autoencoder model
        X: Input data
        device: Device to use
        
    Returns:
        Reconstruction errors for each sample
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        reconstructed = model(X_tensor)
        reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    return reconstruction_error


def detect_anomalies_stage1(
    reconstruction_errors: np.ndarray,
    threshold_percentile: float = 80.0,
) -> Tuple[np.ndarray, float]:
    """
    Detect anomalies using reconstruction error threshold.
    
    Args:
        reconstruction_errors: Reconstruction errors from autoencoder
        threshold_percentile: Percentile for threshold (e.g., 80.0 for 80th percentile)
        
    Returns:
        Tuple of (anomaly_labels, threshold_value)
    """
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomaly_labels = (reconstruction_errors > threshold).astype(int)
    
    return anomaly_labels, threshold
