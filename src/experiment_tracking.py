"""Experiment tracking module for MLflow and Weights & Biases integration."""

from __future__ import annotations

import os
from typing import Optional, Dict, Any
import numpy as np


class ExperimentTracker:
    """Base class for experiment tracking."""
    
    def __init__(self, tracking_backend: str = "mlflow"):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_backend: Either 'mlflow' or 'wandb'
        """
        self.tracking_backend = tracking_backend
        self._initialized = False
        
    def init_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Initialize experiment tracking."""
        if self.tracking_backend == "mlflow":
            self._init_mlflow(experiment_name, config)
        elif self.tracking_backend == "wandb":
            self._init_wandb(experiment_name, config)
        else:
            raise ValueError(f"Unsupported tracking backend: {self.tracking_backend}")
        
        self._initialized = True
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if not self._initialized:
            return
        if self.tracking_backend == "mlflow":
            self._log_params_mlflow(params)
        elif self.tracking_backend == "wandb":
            self._log_params_wandb(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self._initialized:
            return
        if self.tracking_backend == "mlflow":
            self._log_metrics_mlflow(metrics, step)
        elif self.tracking_backend == "wandb":
            self._log_metrics_wandb(metrics, step)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact (model, visualization, etc.)."""
        if not self._initialized:
            return
        if self.tracking_backend == "mlflow":
            self._log_artifact_mlflow(artifact_path, artifact_name)
        elif self.tracking_backend == "wandb":
            self._log_artifact_wandb(artifact_path, artifact_name)
    
    def end_run(self):
        """End experiment run."""
        if not self._initialized:
            return
        if self.tracking_backend == "mlflow":
            self._end_run_mlflow()
        elif self.tracking_backend == "wandb":
            self._end_run_wandb()
    
    # MLflow-specific methods
    def _init_mlflow(self, experiment_name: str, config: Dict[str, Any]):
        """Initialize MLflow experiment."""
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params(config)
        except ImportError:
            print("MLflow not installed. Install with: pip install mlflow")
        except Exception as e:
            print(f"Failed to initialize MLflow: {e}")
    
    def _log_params_mlflow(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            import mlflow
            mlflow.log_params(params)
        except Exception as e:
            print(f"Failed to log params to MLflow: {e}")
    
    def _log_metrics_mlflow(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Failed to log metrics to MLflow: {e}")
    
    def _log_artifact_mlflow(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact to MLflow."""
        try:
            import mlflow
            mlflow.log_artifact(artifact_path, artifact_name)
        except Exception as e:
            print(f"Failed to log artifact to MLflow: {e}")
    
    def _end_run_mlflow(self):
        """End MLflow run."""
        try:
            import mlflow
            mlflow.end_run()
        except Exception as e:
            print(f"Failed to end MLflow run: {e}")
    
    # Weights & Biases-specific methods
    def _init_wandb(self, experiment_name: str, config: Dict[str, Any]):
        """Initialize Weights & Biases experiment."""
        try:
            import wandb
            wandb.init(project=experiment_name, config=config)
        except ImportError:
            print("Weights & Biases not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
    
    def _log_params_wandb(self, params: Dict[str, Any]):
        """Log parameters to W&B."""
        try:
            import wandb
            wandb.config.update(params)
        except Exception as e:
            print(f"Failed to log params to W&B: {e}")
    
    def _log_metrics_wandb(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"Failed to log metrics to W&B: {e}")
    
    def _log_artifact_wandb(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact to W&B."""
        try:
            import wandb
            wandb.save(artifact_path)
        except Exception as e:
            print(f"Failed to log artifact to W&B: {e}")
    
    def _end_run_wandb(self):
        """End W&B run."""
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            print(f"Failed to end W&B run: {e}")


class SimpleTracker:
    """Simple file-based experiment tracker when MLflow/W&B not available."""
    
    def __init__(self, output_dir: str = "outputs/experiments"):
        """
        Initialize simple tracker.
        
        Args:
            output_dir: Directory to save experiment logs
        """
        self.output_dir = output_dir
        self.current_run = None
        self.metrics_history = []
        
    def init_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Initialize experiment."""
        import json
        from pathlib import Path
        
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_run = {
            "experiment_name": experiment_name,
            "config": config,
            "metrics": {},
            "artifacts": []
        }
        
        # Save initial config
        config_path = self.output_dir / f"{experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[SimpleTracker] Initialized experiment: {experiment_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.current_run:
            self.current_run["config"].update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.current_run:
            if step is not None:
                metrics[f"step_{step}"] = metrics
            self.current_run["metrics"].update(metrics)
            self.metrics_history.append(metrics)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact path."""
        if self.current_run:
            self.current_run["artifacts"].append({
                "path": artifact_path,
                "name": artifact_name
            })
    
    def end_run(self):
        """End run and save results."""
        if self.current_run:
            import json
            from pathlib import Path
            
            results_path = self.output_dir / f"{self.current_run['experiment_name']}_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.current_run, f, indent=2)
            
            print(f"[SimpleTracker] Saved results to {results_path}")


def get_tracker(tracking_backend: Optional[str] = None, output_dir: Optional[str] = None) -> ExperimentTracker:
    """
    Get experiment tracker based on availability.
    
    Args:
        tracking_backend: Preferred backend ('mlflow', 'wandb', or None for auto-detect)
        output_dir: Output directory for simple tracker
        
    Returns:
        Experiment tracker instance
    """
    # Auto-detect if backend not specified
    if tracking_backend is None:
        # Try MLflow first
        try:
            import mlflow
            tracking_backend = "mlflow"
        except ImportError:
            # Try W&B
            try:
                import wandb
                tracking_backend = "wandb"
            except ImportError:
                # Fall back to simple tracker
                return SimpleTracker(output_dir=output_dir or "outputs/experiments")
    
    if tracking_backend in ["mlflow", "wandb"]:
        return ExperimentTracker(tracking_backend=tracking_backend)
    else:
        return SimpleTracker(output_dir=output_dir or "outputs/experiments")
