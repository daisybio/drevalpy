"""Weights & Biases and metric helpers mixed into DRPModel."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import wandb

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import AVAILABLE_METRICS, evaluate


class _DRPLoggingMixin:
    """Shared wandb / evaluation helpers for concrete DRPModel instances."""

    wandb_project: str | None
    wandb_run: Any
    wandb_config: dict[str, Any] | None
    _in_hyperparameter_tuning: bool

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model identity; implemented by ``DRPModel``."""
        raise NotImplementedError

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Return instance hyperparameters; implemented by ``DRPModel``."""
        raise NotImplementedError

    def init_wandb(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        finish_previous: bool = True,
    ) -> None:
        """Initialize wandb logging for this model instance.

        Args:
            project: Weights & Biases project name.
            config: Optional run configuration dict.
            name: Optional run display name; defaults to the model name.
            tags: Optional run tags.
            finish_previous: Finish any active wandb run before starting a new one.
        """
        self.wandb_project = project
        run_config = dict(config or {})
        if self.hyperparameters and "hyperparameters" not in run_config:
            run_config["hyperparameters"] = self.hyperparameters
        self.wandb_config = run_config

        if finish_previous:
            wandb.finish()

        run_name = name or self.get_model_name()
        wandb.init(
            project=project,
            config=self.wandb_config,
            name=run_name,
            tags=tags,
        )
        self.wandb_run = wandb.run

        with suppress(Exception):  # pragma: no cover
            wandb.define_metric("epoch", summary="max")
            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("val_loss", summary="min")
            wandb.define_metric("train_R^2", summary="max")
            wandb.define_metric("val_R^2", summary="max")
            wandb.define_metric("train_Pearson", summary="max")
            wandb.define_metric("val_Pearson", summary="max")

    def is_wandb_enabled(self) -> bool:
        """Return whether wandb logging is active for this instance."""
        return self.wandb_project is not None and (self.wandb_run is not None or wandb.run is not None)

    def get_wandb_logger(self) -> Any | None:
        """Return a Lightning WandbLogger for the active run, if any."""
        if not self.is_wandb_enabled() or self.wandb_project is None:
            return None

        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(project=self.wandb_project, log_model=False)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to wandb."""
        if not self.is_wandb_enabled():
            return
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def compute_performance_metrics(
        self, predictions: np.ndarray, targets: np.ndarray, prefix: str = ""
    ) -> dict[str, float]:
        """Compute R^2 and Pearson metrics with an optional key prefix."""
        try:
            metrics = {
                "R^2": AVAILABLE_METRICS["R^2"](y_pred=predictions, y_true=targets),
                "Pearson": AVAILABLE_METRICS["Pearson"](y_pred=predictions, y_true=targets),
            }
            if prefix:
                metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
            return metrics
        except Exception:
            return {}

    def compute_and_log_final_metrics(
        self,
        dataset: DrugResponseDataset,
        additional_metrics: list[str] | None = None,
        prefix: str = "val_",
    ) -> dict[str, float]:
        """Compute final metrics from a dataset and store them in wandb summary.

        Args:
            dataset: Dataset with ``predictions`` populated.
            additional_metrics: Extra metric names beyond R² and Pearson.
            prefix: Key prefix for logged metric names.

        Returns:
            Mapping from metric name to scalar score.
        """
        if dataset.predictions is None:
            return {}

        metrics_to_compute = ["R^2", "Pearson"]
        if additional_metrics:
            metrics_to_compute.extend(additional_metrics)

        results = evaluate(dataset, metric=metrics_to_compute)
        if self.is_wandb_enabled() and wandb.run is not None:
            wandb_metrics = {f"{prefix}{key}": value for key, value in results.items()}
            self.log_final_metrics(wandb_metrics)
        return results

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        """Store final metrics in the wandb run summary."""
        if not self.is_wandb_enabled() or wandb.run is None:
            return
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def finish_wandb(self) -> None:
        """Finish the wandb run for this model instance."""
        if not self.is_wandb_enabled():
            return
        wandb.finish()
        self.wandb_run = None
