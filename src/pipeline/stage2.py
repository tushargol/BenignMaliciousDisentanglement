"""Stage 2: Classifier training and inference for benign vs malicious classification."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

from ..models.classifier import AnomalyClassifier
from ..training.train_classifier import train_classifier


def train_stage2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    sample_weights_train: Optional[np.ndarray] = None,
    sample_weights_val: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    model_save_path: Optional[Path] = None,
) -> AnomalyClassifier:
    """
    Train Stage 2 classifier for benign vs malicious classification.
    
    Args:
        X_train: Training data (anomalous windows only)
        y_train: Training labels (0=benign, 1=malicious)
        X_val: Validation data
        y_val: Validation labels
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        sample_weights_train: Optional sample weights for training
        sample_weights_val: Optional sample weights for validation
        device: Device to use (cuda/cpu)
        model_save_path: Path to save trained model
        
    Returns:
        Trained classifier model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Stage 2] Training classifier on device: {device}")
    print(f"[Stage 2] Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    print(f"[Stage 2] Class distribution - Train: Benign={np.sum(y_train==0)}, Malicious={np.sum(y_train==1)}")
    
    model = train_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sample_weights_train=sample_weights_train,
        sample_weights_val=sample_weights_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    
    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"[Stage 2] Saved model to {model_save_path}")
    
    return model


def load_stage2(
    model_path: Path,
    input_dim: int,
    hidden_dim: int = 128,
    device: Optional[str] = None,
) -> AnomalyClassifier:
    """
    Load trained Stage 2 classifier.
    
    Args:
        model_path: Path to saved model weights
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        device: Device to load model on
        
    Returns:
        Loaded classifier model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AnomalyClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"[Stage 2] Loaded model from {model_path}")
    return model


def infer_stage2(
    model: AnomalyClassifier,
    X: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Stage 2 inference to get malicious probabilities.
    
    Args:
        model: Trained classifier model
        X: Input data (anomalous windows)
        device: Device to use
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
    
    return predictions, probabilities


def classify_anomalies_stage2(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify anomalies as benign or malicious using classifier.
    
    Args:
        predictions: Binary predictions from classifier
        probabilities: Probability scores from classifier
        threshold: Classification threshold
        
    Returns:
        Tuple of (binary_labels, probability_scores)
    """
    binary_labels = (probabilities >= threshold).astype(int)
    return binary_labels, probabilities


def apply_rule_based_rescue(
    predictions: np.ndarray,
    attack_family: Optional[np.ndarray] = None,
    rescue_families: Optional[list] = None,
    default_to_malicious: bool = True,
) -> np.ndarray:
    """
    Apply rule-based rescue for specific attack families.
    
    Args:
        predictions: Classifier predictions
        attack_family: Attack family labels (if available)
        rescue_families: List of attack families to rescue (default: ['arp-spoof', 'industroyer'])
        default_to_malicious: If True, default rescued samples to malicious
        
    Returns:
        Adjusted predictions after rule-based rescue
    """
    if rescue_families is None:
        rescue_families = ['arp-spoof', 'industroyer']
    
    adjusted_predictions = predictions.copy()
    
    if attack_family is not None:
        for i, (pred, fam) in enumerate(zip(predictions, attack_family)):
            if pred == 0 and str(fam).lower() in [f.lower() for f in rescue_families]:
                adjusted_predictions[i] = 1 if default_to_malicious else 0
    
    return adjusted_predictions
