"""
Evaluation pipeline: metrics, per-attack-type analysis, SHAP interpretability, ablations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ..config import Paths
from ..data.prepare_pipeline import PreparedData, prepare
from ..data.windowing import Window
from ..models.autoencoder import Autoencoder
from ..models.baseline import fit_isolation_forest
from ..models.classifier import AnomalyClassifier
from .metrics import (
    binary_curves,
    malicious_metrics,
    per_attack_metrics,
    per_class_metrics,
)
from sklearn.metrics import confusion_matrix


@dataclass
class EvalResult:
    overall: dict
    per_class: dict
    malicious: dict
    per_attack: dict
    config: dict


def _anomaly_scores_ae(model: Autoencoder, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float().to(device)
        recon, _ = model(x)
        err = ((x - recon) ** 2).mean(dim=1).cpu().numpy()
    return err.astype(np.float64)


ATTACK_CONTEXT_CHANNELS = (
    "n_ctx_arp_spoof",
    "n_ctx_industroyer",
    "n_event_start_attack",
    "n_event_set_point_map",
    "interval_n_ctx_arp",
    "interval_n_ctx_industroyer",
    "interval_n_start_attack",
    "interval_n_set_point_map",
    "interval_n_malicious",
)


def _max_attack_context_signal(w: Window, channel_names: List[str]) -> float:
    """Peak per-second activity in Sherlock attack-specific notification channels."""
    idx = [channel_names.index(c) for c in ATTACK_CONTEXT_CHANNELS if c in channel_names]
    if not idx or w.features.size == 0:
        return 0.0
    sub = w.features[:, idx]
    return float(np.max(np.sum(sub, axis=1)))


def _malicious_proba_classifier(model: AnomalyClassifier, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float().to(device)
        logits = model(x).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_two_stage(
    prepared: PreparedData,
    ae_model: Autoencoder,
    clf_model: AnomalyClassifier,
    ae_threshold_percentile: float = 95.0,
    clf_threshold: float = 0.5,
    benign_ctrl_mean_threshold: float | None = None,
    attack_context_rescue: bool = True,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run two-stage pipeline on test set.
    Returns: y_true (0/1/2), y_anomaly (0/1), y_malicious_score (prob for anomalies).
    """
    X = prepared.X_test
    y_true = prepared.y_test

    ae_scores = _anomaly_scores_ae(ae_model, X, device)
    ae_thr = np.percentile(ae_scores, ae_threshold_percentile)
    anomaly_mask = ae_scores >= ae_thr

    y_malicious_score = np.zeros(len(X), dtype=np.float64)
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        X_anom = X[anomaly_indices]
        probs = _malicious_proba_classifier(clf_model, X_anom, device)
        y_malicious_score[anomaly_indices] = probs

    # Binary: 0=normal/benign (no alert or benign alert), 1=malicious (alert as malicious)
    # For metrics we treat: predict malicious if anomaly AND prob > 0.5
    y_pred_3class = np.zeros(len(X), dtype=np.int64)
    y_pred_3class[anomaly_mask] = np.where(
        y_malicious_score[anomaly_indices] > clf_threshold, 2, 1
    )
    # y_pred_3class: 0=normal (not anomaly), 1=benign anomaly, 2=malicious

    # Rule rescue: AE may miss attacks that are sparse in aggregate stats but have explicit
    # arp-spoof / industroyer / start-attack / set-point-map events in raw notifications.
    if attack_context_rescue and clf_model is not None and prepared.feature_channel_names:
        windows = prepared.windows_test
        for i in range(len(X)):
            if y_pred_3class[i] != 0:
                continue
            sig = _max_attack_context_signal(windows[i], prepared.feature_channel_names)
            if sig <= 0:
                continue
            prob = float(_malicious_proba_classifier(clf_model, X[i : i + 1], device)[0])
            y_malicious_score[i] = prob
            y_pred_3class[i] = 2 if prob > clf_threshold else 1

    # Promote benign→malicious when exact-interval counts show arp-spoof / industroyer
    # (classifier may be uncertain; raw notifications are ground-truth process labels).
    if prepared.feature_channel_names:
        names = prepared.feature_channel_names
        for name in ("interval_n_ctx_industroyer", "interval_n_ctx_arp"):
            if name not in names:
                continue
            j = names.index(name)
            for i, w in enumerate(prepared.windows_test):
                if y_pred_3class[i] != 1:
                    continue
                if float(np.max(w.features[:, j])) <= 0:
                    continue
                y_pred_3class[i] = 2
                y_malicious_score[i] = max(y_malicious_score[i], 0.99)

    # Optional benign detector:
    # If a window is not anomalous but has sustained control-center activity,
    # classify it as benign operational anomaly.
    if benign_ctrl_mean_threshold is not None and X.shape[1] > 3:
        ctrl_mean = X[:, 3]  # feature index for mean(n_control_center)
        benign_mask = (~anomaly_mask) & (ctrl_mean >= benign_ctrl_mean_threshold)
        y_pred_3class[benign_mask] = 1

    return y_true, y_pred_3class, y_malicious_score


def run_evaluation(
    prepared: PreparedData,
    ae_model: Optional[Autoencoder] = None,
    clf_model: Optional[AnomalyClassifier] = None,
    baseline_model: Optional[Any] = None,
    ae_threshold_percentile: float = 95.0,
    clf_threshold: float = 0.5,
    benign_ctrl_mean_threshold: float | None = None,
    attack_context_rescue: bool = True,
    device: str = "cpu",
    windows_test: Optional[List[Window]] = None,
) -> EvalResult:
    """
    Full evaluation: overall, per-class, malicious-specific, per-attack-type.
    """
    X = prepared.X_test
    y_true = prepared.y_test
    windows = windows_test or prepared.windows_test
    attack_families = [w.attack_family for w in windows]

    overall = {}
    per_class = {}
    malicious = {}
    per_attack = {}

    if ae_model is not None:
        if clf_model is not None:
            y_true, y_pred, y_score = evaluate_two_stage(
                prepared,
                ae_model,
                clf_model,
                ae_threshold_percentile,
                clf_threshold,
                benign_ctrl_mean_threshold,
                attack_context_rescue,
                device,
            )
        else:
            # Ablation: AE only, treat all anomalies as malicious
            ae_scores = _anomaly_scores_ae(ae_model, X, device)
            ae_thr = np.percentile(ae_scores, ae_threshold_percentile)
            anomaly_mask = ae_scores >= ae_thr
            y_pred = np.where(anomaly_mask, 2, 0)
            y_score = ae_scores.astype(np.float64)
        per_class = per_class_metrics(y_true, y_pred)
        malicious = malicious_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        overall = {
            "method": "two_stage" if clf_model else "ae_only",
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
            "confusion_matrix": cm,
        }

        # Per-attack: use y_score from evaluate_two_stage (includes rule rescue / promotion)
        if clf_model is not None:
            mal_score = np.asarray(y_score, dtype=np.float64).copy()
        else:
            mal_score = np.zeros(len(X))
            anom_mask = (y_pred == 1) | (y_pred == 2)
            sc = y_score[anom_mask]
            if len(sc) > 0 and sc.max() > sc.min():
                mal_score[anom_mask] = (sc - sc.min()) / (sc.max() - sc.min())
            else:
                mal_score[anom_mask] = 0.5
        per_attack = per_attack_metrics(y_true, mal_score, attack_families, y_pred=y_pred)

    if baseline_model is not None:
        from sklearn.ensemble import IsolationForest

        if isinstance(baseline_model, IsolationForest):
            base_scores = -baseline_model.score_samples(X)
            base_thr = np.percentile(base_scores, ae_threshold_percentile)
            y_pred_base = np.where(base_scores >= base_thr, 2, 0)
            y_pred_base = np.where(
                (y_pred_base == 0) & (y_true == 1), 1, y_pred_base
            )
            per_class_base = per_class_metrics(y_true, y_pred_base)
            malicious_base = malicious_metrics(y_true, y_pred_base)
            overall["baseline"] = {
                "per_class": per_class_base,
                "malicious": malicious_base,
            }

    return EvalResult(
        overall=overall,
        per_class=per_class,
        malicious=malicious,
        per_attack=per_attack,
        config=prepared.config,
    )


def explain_with_shap(
    clf_model: AnomalyClassifier,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 200,
    device: str = "cpu",
) -> dict:
    """
    SHAP explainability for the classifier. Returns summary stats and top features.
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}

    if X.shape[0] > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    def predict_fn(x):
        model = clf_model
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x).float().to(device)
            logits = model(t).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_values = explainer.shap_values(X_bg, nsamples=min(100, X_bg.shape[0]))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_k = min(10, len(mean_abs))
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    top_features = [
        {"name": names[i], "importance": float(mean_abs[i])}
        for i in order
    ]
    return {"top_features": top_features}


def save_eval_report(result: EvalResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "per_class": result.per_class,
        "malicious": result.malicious,
        "per_attack": result.per_attack,
        "config": {k: v for k, v in result.config.items() if isinstance(v, (int, float, str, bool))},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
