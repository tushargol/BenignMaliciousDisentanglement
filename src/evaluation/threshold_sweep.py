from __future__ import annotations

"""
Threshold sweep experiments for the two-stage pipeline.

Runs the trained autoencoder + classifier over a grid of
AE reconstruction error percentiles and classifier thresholds,
and writes a CSV with malicious recall / FPR trade-offs.
"""

import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from ..config import Paths
from ..data.prepare_pipeline import PreparedData, prepare
from ..models.autoencoder import Autoencoder
from ..models.classifier import AnomalyClassifier
from .evaluate_pipeline import _anomaly_scores_ae, _malicious_proba_classifier, malicious_metrics


def sweep_thresholds(
    prepared: PreparedData,
    ae_model: Autoencoder,
    clf_model: AnomalyClassifier,
    ae_percentiles: Iterable[float] = (80, 85, 90, 95, 97.5),
    clf_thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    X = prepared.X_test
    y_true = prepared.y_test

    ae_scores = _anomaly_scores_ae(ae_model, X, device)
    results = []

    for p in ae_percentiles:
        ae_thr = np.percentile(ae_scores, p)
        anomaly_mask = ae_scores >= ae_thr
        if not anomaly_mask.any():
            continue
        X_anom = X[anomaly_mask]
        probs = _malicious_proba_classifier(clf_model, X_anom, device)

        for t in clf_thresholds:
            y_pred = np.zeros_like(y_true)
            # mark anomalies as benign or malicious depending on classifier prob
            mal_mask_local = probs > t
            # indices of anomalies in the global index space
            idx_anom = np.where(anomaly_mask)[0]
            y_pred[idx_anom[mal_mask_local]] = 2
            y_pred[idx_anom[~mal_mask_local]] = 1

            m = malicious_metrics(y_true, y_pred)
            results.append(
                {
                    "ae_percentile": float(p),
                    "clf_threshold": float(t),
                    "malicious_recall": m["malicious_recall"],
                    "fpr_on_benign": m["fpr_on_benign"],
                    "fpr_on_normal": m["fpr_on_normal"],
                }
            )

    return results


def main() -> None:
    paths = Paths.auto()
    prepared_path = paths.outputs_dir / "prepared.pkl"
    models_path = paths.outputs_dir / "models"
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    if prepared_path.exists():
        import pickle

        with open(prepared_path, "rb") as f:
            prepared: PreparedData = pickle.load(f)
    else:
        prepared = prepare(paths, paths.repo_root / "experiments" / "example.yaml")

    # Load models
    ae = Autoencoder(
        input_dim=prepared.input_dim,
        hidden_dim=prepared.config.get("stage1", {}).get("hidden_dim", 256),
        latent_dim=prepared.config.get("stage1", {}).get("latent_dim", 64),
    )
    ae.load_state_dict(torch.load(models_path / "autoencoder.pt", map_location="cpu"))

    clf = AnomalyClassifier(
        input_dim=prepared.input_dim,
        hidden_dim=prepared.config.get("stage2", {}).get("hidden_dim", 128),
    )
    clf.load_state_dict(torch.load(models_path / "classifier.pt", map_location="cpu"))

    rows = sweep_thresholds(prepared, ae, clf)

    out_csv = paths.outputs_dir / "threshold_sweep.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ae_percentile",
                "clf_threshold",
                "malicious_recall",
                "fpr_on_benign",
                "fpr_on_normal",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

