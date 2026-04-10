"""
Visualization module for evaluation metrics: confusion matrices, ROC/PR curves, per-attack analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    classification_report,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ["Normal", "Benign", "Malicious"],
    save_path: Optional[Path] = None,
    normalize: bool = False,
) -> plt.Figure:
    """Plot confusion matrix with optional normalization."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot per-class precision, recall, and F1 scores."""
    classes = list(per_class.keys())
    precision = [per_class[c]['precision'] for c in classes]
    recall = [per_class[c]['recall'] for c in classes]
    f1 = [per_class[c]['f1'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_per_attack_recall(
    per_attack: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot malicious recall per attack family."""
    attacks = list(per_attack.keys())
    recalls = [per_attack[attack]['malicious_recall'] for attack in attacks]
    support = [per_attack[attack]['support'] for attack in attacks]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(attacks, recalls, alpha=0.8)
    
    ax.set_xlabel('Attack Family')
    ax.set_ylabel('Malicious Recall')
    ax.set_title('Malicious Detection Rate by Attack Family')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels and support counts
    for i, (bar, recall, supp) in enumerate(zip(bars, recalls, support)):
        height = bar.get_height()
        ax.annotate(f'{recall:.3f}\n(n={supp})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_roc_pr_curves(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC and Precision-Recall curves for binary malicious detection."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = roc_auc_score(y_true_binary, y_score)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve - Malicious Detection')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
    pr_auc = average_precision_score(y_true_binary, y_score)
    
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve - Malicious Detection')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_threshold_sweep(
    thresholds: np.ndarray,
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot performance metrics across different thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        ax.plot(thresholds, values, marker='o', label=metric_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_shap_feature_importance(
    shap_result: Dict[str, Any],
    top_k: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot SHAP feature importance."""
    if 'top_features' not in shap_result:
        raise ValueError("SHAP result must contain 'top_features' key")
    
    features = shap_result['top_features'][:top_k]
    names = [f['name'] for f in features]
    importance = [f['importance'] for f in features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, importance, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('SHAP Importance (Mean Absolute Value)')
    ax.set_title(f'Top {top_k} Feature Importance (SHAP)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        width = bar.get_width()
        ax.annotate(f'{imp:.4f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(3, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot SHAP beeswarm plot showing feature impact distribution across samples."""
    try:
        import shap
    except ImportError:
        print("SHAP library not available for beeswarm plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simple beeswarm-like visualization using matplotlib
    # Calculate mean absolute SHAP values for feature ordering
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20 features
    
    # Plot beeswarm
    for i, feature_idx in enumerate(feature_order):
        feature_shap = shap_values[:, feature_idx]
        y_positions = np.random.normal(i, 0.3, len(feature_shap))
        
        # Color points by SHAP value (red=positive, blue=negative)
        colors = ['red' if val > 0 else 'blue' for val in feature_shap]
        sizes = [abs(val) * 100 for val in feature_shap]
        
        ax.scatter(feature_shap, y_positions, c=colors, s=sizes, alpha=0.6)
    
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels([feature_names[i] for i in feature_order])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    ax.set_title('SHAP Beeswarm Plot - Feature Impact Distribution')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_evaluation_dashboard(
    eval_result: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    shap_result: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Create comprehensive evaluation dashboard with all visualizations."""
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # 1. Confusion Matrix
    cm_path = save_dir / "confusion_matrix.png" if save_dir else None
    fig_cm = plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    saved_plots['confusion_matrix'] = cm_path
    plt.close(fig_cm)
    
    # 2. Normalized Confusion Matrix
    cm_norm_path = save_dir / "confusion_matrix_normalized.png" if save_dir else None
    fig_cm_norm = plot_confusion_matrix(y_true, y_pred, save_path=cm_norm_path, normalize=True)
    saved_plots['confusion_matrix_normalized'] = cm_norm_path
    plt.close(fig_cm_norm)
    
    # 3. Per-class metrics
    per_class_path = save_dir / "per_class_metrics.png" if save_dir else None
    fig_per_class = plot_per_class_metrics(eval_result.per_class, save_path=per_class_path)
    saved_plots['per_class_metrics'] = per_class_path
    plt.close(fig_per_class)
    
    # 4. Per-attack recall
    if eval_result.per_attack:
        per_attack_path = save_dir / "per_attack_recall.png" if save_dir else None
        fig_per_attack = plot_per_attack_recall(eval_result.per_attack, save_path=per_attack_path)
        saved_plots['per_attack_recall'] = per_attack_path
        plt.close(fig_per_attack)
    
    # 5. ROC/PR curves (binary malicious detection)
    y_true_binary = (y_true == 2).astype(int)  # Malicious as positive class
    roc_pr_path = save_dir / "roc_pr_curves.png" if save_dir else None
    fig_roc_pr = plot_roc_pr_curves(y_true_binary, y_score, save_path=roc_pr_path)
    saved_plots['roc_pr_curves'] = roc_pr_path
    plt.close(fig_roc_pr)
    
    # 6. SHAP feature importance
    if shap_result and 'top_features' in shap_result:
        shap_path = save_dir / "shap_feature_importance.png" if save_dir else None
        fig_shap = plot_shap_feature_importance(shap_result, save_path=shap_path)
        saved_plots['shap_feature_importance'] = shap_path
        plt.close(fig_shap)
    
    return saved_plots


def save_visualization_summary(saved_plots: Dict[str, Path], save_path: Path) -> None:
    """Save a summary of generated visualizations."""
    summary = {
        "generated_plots": {name: str(path) for name, path in saved_plots.items()},
        "total_plots": len(saved_plots)
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
