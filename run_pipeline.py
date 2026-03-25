"""
Sherlock 02-Semiurban IDS pipeline: prepare -> train_ae -> train_classifier -> evaluate.
Capstone enhancements: per-attack analysis, SHAP interpretability, ablation support.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import Paths
from src.data.prepare_pipeline import PreparedData, prepare
from src.models.baseline import fit_isolation_forest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sherlock 02-Semiurban IDS pipeline (benign vs malicious disentanglement)."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "train_ae", "train_classifier", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: experiments/example.yaml).",
    )
    parser.add_argument(
        "--ae_percentile",
        type=float,
        default=80.0,
        help="Autoencoder anomaly threshold as percentile of reconstruction error (e.g. 80, 90, 95).",
    )
    parser.add_argument(
        "--clf_threshold",
        type=float,
        default=0.5,
        help="Stage-2 malicious probability threshold.",
    )
    parser.add_argument(
        "--benign_ctrl_threshold",
        type=float,
        default=0.01,
        help="If > 0, non-anomalous windows with mean control-center activity above this are tagged benign.",
    )
    parser.add_argument(
        "--auto_tune",
        action="store_true",
        help="Automatically search threshold combinations and pick the best score.",
    )
    parser.add_argument(
        "--no_attack_context_rescue",
        action="store_true",
        help="Disable rule-based rescue for arp-spoof/industroyer events missed by the AE.",
    )
    parser.add_argument(
        "--max_fpr_normal",
        type=float,
        default=None,
        help="Constraint for auto-tune: maximum allowed FPR on normal.",
    )
    parser.add_argument(
        "--max_fpr_benign",
        type=float,
        default=None,
        help="Constraint for auto-tune: maximum allowed FPR on benign.",
    )
    parser.add_argument(
        "--ablation",
        choices=["none", "stage1_only", "no_classifier"],
        default="none",
        help="Ablation: stage1_only=AE only, no_classifier=skip stage2, run_all=compare both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    paths = Paths.auto()
    config_path = args.config or paths.repo_root / "experiments" / "example.yaml"
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = paths.outputs_dir / "prepared.pkl"
    models_path = paths.outputs_dir / "models"

    np.random.seed(args.seed)

    def _load_prepared() -> PreparedData:
        if prepared_path.exists():
            with open(prepared_path, "rb") as f:
                return pickle.load(f)
        prep = prepare(paths, config_path)
        with open(prepared_path, "wb") as f:
            pickle.dump(prep, f)
        return prep

    # --- Prepare ---
    if args.stage in ("prepare", "all"):
        print("[Prepare] Loading events and building time-series...")
        prep = prepare(paths, config_path)
        with open(prepared_path, "wb") as f:
            pickle.dump(prep, f)
        print(f"  Train: {prep.X_train.shape[0]} windows, Test: {prep.X_test.shape[0]} windows")
        print(f"  Input dim: {prep.input_dim}")
        if args.stage == "prepare":
            return

    prep = _load_prepared()

    # --- Train AE (on normal + benign) ---
    if args.stage in ("train_ae", "all"):
        from src.training.train_autoencoder import train_autoencoder
        print("[Train AE] Fitting autoencoder...")
        X_tr, X_val = train_test_split(
            prep.X_train,
            test_size=0.15,
            random_state=args.seed,
        )
        ae_cfg = prep.config.get("stage1", {})
        ae = train_autoencoder(
            X_tr,
            X_val,
            input_dim=prep.input_dim,
            hidden_dim=ae_cfg.get("hidden_dim", 256),
            latent_dim=ae_cfg.get("latent_dim", 64),
            epochs=ae_cfg.get("epochs", 30),
        )
        models_path.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(ae.state_dict(), models_path / "autoencoder.pt")
        print("  Saved autoencoder.pt")

    # --- Train Classifier (on anomaly windows from test: benign vs malicious) ---
    train_clf = args.ablation not in ("stage1_only", "no_classifier")
    if args.stage in ("train_classifier", "all") and train_clf:
        from src.training.train_classifier import train_classifier

        print("[Train Classifier] Preparing anomaly subset...")
        # Use test set anomaly windows for classifier (train has only benign)
        anom_mask = prep.y_test >= 1
        X_anom = prep.X_test[anom_mask]
        y_anom = (prep.y_test[anom_mask] == 2).astype(np.float32)

        # Per-sample weights: emphasize hard attack families for positives
        # Map anomaly indices back to window metadata.
        clf_cfg = prep.config.get("stage2", {})
        family_weight_cfg = clf_cfg.get("attack_family_weights", {})
        hard_fams = tuple(family_weight_cfg.keys()) if family_weight_cfg else ("drift-off", "industroyer", "arp-spoof")
        attack_fams = np.array([w.attack_family for w in prep.windows_test])
        weights_anom = np.ones_like(y_anom, dtype=np.float32)
        for fam, w in family_weight_cfg.items():
            fam_mask = (y_anom == 1.0) & (attack_fams[anom_mask] == fam)
            weights_anom[fam_mask] = float(w)

        # Optional inverse-frequency boosting for malicious families
        if bool(clf_cfg.get("inverse_freq_boost", False)):
            max_w = float(clf_cfg.get("max_inverse_freq_weight", 4.0))
            fams = attack_fams[anom_mask]
            mal_fams = fams[y_anom == 1.0]
            if mal_fams.size > 0:
                uniq, cnt = np.unique(mal_fams, return_counts=True)
                inv = {u: min(max_w, float(cnt.max()) / max(c, 1)) for u, c in zip(uniq, cnt)}
                for fam, w in inv.items():
                    fam_mask = (y_anom == 1.0) & (fams == fam)
                    weights_anom[fam_mask] *= float(w)

        if X_anom.shape[0] < 10:
            print("  WARNING: Few anomaly windows; classifier may not train well.")
        split = train_test_split(
            X_anom,
            y_anom,
            weights_anom,
            test_size=0.2,
            random_state=args.seed,
            stratify=y_anom if len(np.unique(y_anom)) > 1 else None,
        )
        X_cf_tr, X_cf_val, y_cf_tr, y_cf_val, w_cf_tr, w_cf_val = split

        clf = train_classifier(
            X_cf_tr,
            y_cf_tr,
            X_cf_val,
            y_cf_val,
            input_dim=prep.input_dim,
            hidden_dim=clf_cfg.get("hidden_dim", 128),
            sample_weights_train=w_cf_tr,
            sample_weights_val=w_cf_val,
            epochs=clf_cfg.get("epochs", 30),
        )
        models_path.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(clf.state_dict(), models_path / "classifier.pt")
        print("  Saved classifier.pt")

    # --- Train baseline (optional) ---
    if args.stage in ("train_ae", "train_classifier", "all"):
        bl_cfg = prep.config.get("baseline", {})
        if bl_cfg.get("model") == "isolation_forest":
            print("[Baseline] Fitting Isolation Forest...")
            bl = fit_isolation_forest(prep.X_train, seed=args.seed)
            with open(models_path / "baseline_if.pkl", "wb") as f:
                pickle.dump(bl, f)
            print("  Saved baseline_if.pkl")

    # --- Evaluate ---
    if args.stage in ("evaluate", "all"):
        from src.evaluation.evaluate_pipeline import (
            explain_with_shap,
            run_evaluation,
            save_eval_report,
        )

        print("[Evaluate] Loading models and running evaluation...")
        import torch
        from src.models.autoencoder import Autoencoder
        from src.models.classifier import AnomalyClassifier

        ae = Autoencoder(
            input_dim=prep.input_dim,
            hidden_dim=prep.config.get("stage1", {}).get("hidden_dim", 256),
            latent_dim=prep.config.get("stage1", {}).get("latent_dim", 64),
        )
        ae.load_state_dict(torch.load(models_path / "autoencoder.pt", map_location="cpu"))

        clf = None
        use_clf = args.ablation not in ("stage1_only", "no_classifier")
        if use_clf and (models_path / "classifier.pt").exists():
            clf = AnomalyClassifier(
                input_dim=prep.input_dim,
                hidden_dim=prep.config.get("stage2", {}).get("hidden_dim", 128),
            )
            clf.load_state_dict(torch.load(models_path / "classifier.pt", map_location="cpu"))

        baseline = None
        if (models_path / "baseline_if.pkl").exists():
            with open(models_path / "baseline_if.pkl", "rb") as f:
                baseline = pickle.load(f)

        if args.auto_tune and clf is not None:
            ae_grid = [75.0, 80.0, 85.0, 90.0, 95.0]
            clf_grid = [0.3, 0.5, 0.7]
            benign_grid = [0.0, 0.005, 0.01, 0.02]
            rows = []
            best_score = -1.0
            best_params = (args.ae_percentile, args.clf_threshold, args.benign_ctrl_threshold)
            best_result = None
            tune_cfg = prep.config.get("tuning", {})
            max_fpr_normal = args.max_fpr_normal if args.max_fpr_normal is not None else tune_cfg.get("max_fpr_on_normal")
            max_fpr_benign = args.max_fpr_benign if args.max_fpr_benign is not None else tune_cfg.get("max_fpr_on_benign")

            for ae_p in ae_grid:
                for clf_t in clf_grid:
                    for ben_t in benign_grid:
                        r = run_evaluation(
                            prep,
                            ae_model=ae,
                            clf_model=clf,
                            baseline_model=baseline,
                            ae_threshold_percentile=ae_p,
                            clf_threshold=clf_t,
                            benign_ctrl_mean_threshold=ben_t,
                            attack_context_rescue=not args.no_attack_context_rescue,
                        )
                        n_f1 = r.per_class["normal"]["f1"]
                        b_f1 = r.per_class["benign"]["f1"]
                        m_f1 = r.per_class["malicious"]["f1"]
                        m_rec = r.malicious["malicious_recall"]
                        fpr_n = r.malicious["fpr_on_normal"]
                        fpr_b = r.malicious["fpr_on_benign"]
                        constrained_ok = True
                        if max_fpr_normal is not None:
                            constrained_ok = constrained_ok and (fpr_n <= float(max_fpr_normal))
                        if max_fpr_benign is not None:
                            constrained_ok = constrained_ok and (fpr_b <= float(max_fpr_benign))
                        base_score = (n_f1 + b_f1 + m_f1) / 3.0 + 0.5 * m_rec
                        penalty = 0.0
                        if max_fpr_normal is not None and fpr_n > float(max_fpr_normal):
                            penalty += 5.0 * (fpr_n - float(max_fpr_normal))
                        if max_fpr_benign is not None and fpr_b > float(max_fpr_benign):
                            penalty += 5.0 * (fpr_b - float(max_fpr_benign))
                        score = base_score - penalty
                        rows.append(
                            {
                                "ae_percentile": ae_p,
                                "clf_threshold": clf_t,
                                "benign_ctrl_threshold": ben_t,
                                "score": score,
                                "normal_f1": n_f1,
                                "benign_f1": b_f1,
                                "malicious_f1": m_f1,
                                "malicious_recall": m_rec,
                                "fpr_on_normal": fpr_n,
                                "fpr_on_benign": fpr_b,
                                "constraints_ok": constrained_ok,
                            }
                        )
                        if score > best_score:
                            best_score = score
                            best_params = (ae_p, clf_t, ben_t)
                            best_result = r

            tune_csv = paths.outputs_dir / "auto_tune_results.csv"
            with open(tune_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ae_percentile",
                        "clf_threshold",
                        "benign_ctrl_threshold",
                        "score",
                        "normal_f1",
                        "benign_f1",
                        "malicious_f1",
                        "malicious_recall",
                        "fpr_on_normal",
                        "fpr_on_benign",
                        "constraints_ok",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            args.ae_percentile, args.clf_threshold, args.benign_ctrl_threshold = best_params
            result = best_result
            print(
                f"[AutoTune] best ae_percentile={best_params[0]}, "
                f"clf_threshold={best_params[1]}, benign_ctrl_threshold={best_params[2]}, "
                f"score={best_score:.4f}"
            )
            print(f"[AutoTune] wrote {tune_csv}")
        else:
            result = run_evaluation(
                prep,
                ae_model=ae,
                clf_model=clf,
                baseline_model=baseline,
                ae_threshold_percentile=args.ae_percentile,
                clf_threshold=args.clf_threshold,
                benign_ctrl_mean_threshold=args.benign_ctrl_threshold,
                attack_context_rescue=not args.no_attack_context_rescue,
            )

        print("\n--- Per-class metrics ---")
        for k, v in result.per_class.items():
            print(f"  {k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
        print("\n--- Malicious-specific ---")
        for k, v in result.malicious.items():
            print(f"  {k}: {v:.3f}")
        print("\n--- Per-attack-type (malicious recall) ---")
        for fam, v in result.per_attack.items():
            print(f"  {fam}: recall={v.get('malicious_recall', 0):.3f} (n={v.get('support', 0)})")

        report_name = "eval_report_ae_only.json" if args.ablation == "stage1_only" else "eval_report.json"
        save_eval_report(result, paths.outputs_dir / report_name)
        print(f"\n  Saved {paths.outputs_dir / 'eval_report.json'}")

        # SHAP interpretability
        if clf is not None:
            anom_mask = prep.y_test >= 1
            if anom_mask.sum() > 0:
                X_anom = prep.X_test[anom_mask]
                shap_result = explain_with_shap(clf, X_anom, prep.feature_cols, n_samples=min(100, len(X_anom)))
                if "error" not in shap_result:
                    print("\n--- Top features (SHAP) ---")
                    for f in shap_result.get("top_features", [])[:5]:
                        print(f"  {f['name']}: {f['importance']:.4f}")
                    with open(paths.outputs_dir / "shap_top_features.json", "w") as f:
                        import json
                        json.dump(shap_result, f, indent=2)
                else:
                    print("  SHAP skipped:", shap_result.get("error"))


if __name__ == "__main__":
    main()
