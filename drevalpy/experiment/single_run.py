"""Single model + single fold execution unit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin, clone

from drevalpy.data.structures import EntityScope, SplitMasks
from drevalpy.data.structures.mudataset import MuDataset
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel

from .fold import prepare_mu_fold
from .hpo import select_fold_hyperparameters
from .training import mu_train_and_predict

logger = get_logger(__name__)


@dataclass
class RunResult:
    """Output of a single Run."""

    model_name: str
    fold_index: int
    predictions: np.ndarray
    ground_truth: np.ndarray
    cell_line_ids: np.ndarray
    drug_ids: np.ndarray
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "RunResult",
            f"    Model: {self.model_name}",
            "    Fold:",
            f"        index: {self.split_masks.metadata.get("fold_index", 0)}",
        ]

        for k, v in self.fold_metadata.items():
            if k != "fold_index":
                lines.append(f"        {k}: {v}")

        lines.append(f"    Predictions: {len(self.predictions)} pairs")
        lines.append(f"    Ground truth: {int(np.sum(~np.isnan(self.ground_truth)))} non-NaN values")

        if self.best_hyperparameters:
            lines.append("    Hyperparameters:")
            for k, v in self.best_hyperparameters.items():
                lines.append(f"        {k}: {v}")

        if self.metrics:
            lines.append("    Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"        {k}: {v:.4f}")

        return "\n".join(lines)


class Run:
    """Single model + single fold execution unit.

    Given a model class and a SplitMasks fold, trains the model (optionally
    with HPO) and predicts on the test set. Returns a RunResult.
    """

    def __init__(
        self,
        model_class: type[DRPModel],
        mudataset: MuDataset,
        split_masks: SplitMasks,
        *,
        hyperparameter_tuning: bool = True,
        response_transformation: TransformerMixin | None = None,
        hpo_metric: str = "RMSE",
        hpo_num_samples: int = 16,
        hpo_random_state: int = 42,
    ) -> None:
        """Initialize a Run.

        :param model_class: DRPModel subclass to train.
        :param mudataset: Full dataset with all features.
        :param split_masks: Single fold's train/test/val pair arrays.
        :param hyperparameter_tuning: Whether to run HPO.
        :param response_transformation: Optional sklearn transformer for responses.
        :param hpo_metric: Metric to optimize during HPO.
        :param hpo_num_samples: Number of HPO trials.
        :param hpo_random_state: Random seed for HPO.
        """
        self.model_class = model_class
        self.mudataset = mudataset
        self.split_masks = split_masks
        self.hyperparameter_tuning = hyperparameter_tuning
        self.response_transformation = response_transformation
        self.hpo_metric = hpo_metric
        self.hpo_num_samples = hpo_num_samples
        self.hpo_random_state = hpo_random_state

    def __repr__(self) -> str:
        """Formatted summary of this run configuration."""
        model_name = self.model_class.get_model_name()
        lines = [
            "Run",
            f"    Model: {model_name}",
        ]

        if self.hyperparameter_tuning:
            lines.append("    Hyperparameter Tuning: enabled")
            lines.append(f"        metric: {self.hpo_metric}")
            lines.append(f"        num_samples: {self.hpo_num_samples}")
        else:
            lines.append("    Hyperparameter Tuning: disabled")
            defaults = self.model_class.get_default_hyperparameters()
            if defaults:
                lines.append("    Default Hyperparameters:")
                for k, v in defaults.items():
                    lines.append(f"        {k}: {v}")

        lines.append("    Fold:")
        lines.append(f"        index: {self.split_masks.metadata.get("fold_index", 0)}")
        for k, v in self.split_masks.metadata.items():
            if k != "fold_index":
                lines.append(f"        {k}: {v}")

        lines.append(f"    Train pairs: {len(self.split_masks.train)}")
        lines.append(f"    Test pairs: {len(self.split_masks.test)}")
        lines.append(f"    Val pairs: {len(self.split_masks.val)}")

        return "\n".join(lines)

    def execute(self) -> RunResult:
        """Train the model and predict on the test set.

        :returns: RunResult with predictions, ground truth, and metrics.
        """
        from drevalpy.components.core.tuning.config import build_experiment_hpo_config
        from drevalpy.evaluation import AVAILABLE_METRICS

        model_name = self.model_class.get_model_name()
        logger.info("Run: %s, fold %d", model_name, self.split_masks.metadata.get("fold_index", 0))

        fold_data = prepare_mu_fold(self.mudataset, self.split_masks, self.model_class)

        # HPO or default hyperparameters
        if self.hyperparameter_tuning:
            hpo_cfg = build_experiment_hpo_config(
                self.hpo_metric,
                n_trials=self.hpo_num_samples,
                random_state=self.hpo_random_state,
            )
            best_hpams = select_fold_hyperparameters(
                model_class=self.model_class,
                mudataset=self.mudataset,
                train_scope=fold_data.train_scope,
                val_scope=fold_data.val_scope,
                early_stopping_scope=fold_data.early_stopping_scope,
                response_transformation=self.response_transformation,
                metric=self.hpo_metric,
                model_checkpoint_dir=None,
                hyperparameter_tuning=True,
                hpo_config=hpo_cfg,
            )
        else:
            best_hpams = self.model_class.get_default_hyperparameters()

        logger.info("Best hyperparameters: %s", best_hpams)

        # Train on train+val, predict on test
        from .fold import merge_train_val_scopes

        merged_scope = merge_train_val_scopes(self.split_masks)
        model = self.model_class(best_hpams)
        fold_transform = None if self.response_transformation is None else clone(self.response_transformation)

        predictions = mu_train_and_predict(
            model=model,
            mudataset=self.mudataset,
            train_scope=merged_scope,
            test_scope=fold_data.test_scope,
            early_stopping_scope=fold_data.early_stopping_scope,
            response_transformation=fold_transform,
        )

        # Extract ground truth
        response_matrix = self.mudataset.response_matrix
        test_pairs = self.split_masks.test
        ground_truth = response_matrix[test_pairs[:, 0], test_pairs[:, 1]]

        cl_ids = self.mudataset.cell_line_ids[test_pairs[:, 0]]
        dr_ids = self.mudataset.drug_ids[test_pairs[:, 1]]

        # Compute metrics
        valid = ~np.isnan(predictions) & ~np.isnan(ground_truth)
        metrics: dict[str, float] = {}
        if valid.any():
            from drevalpy.evaluation import _compute_metric_value

            for metric_name in AVAILABLE_METRICS:
                metrics[metric_name] = _compute_metric_value(metric_name, predictions[valid], ground_truth[valid])

        return RunResult(
            model_name=model_name,
            fold_index=self.split_masks.metadata.get("fold_index", 0),
            predictions=predictions,
            ground_truth=ground_truth,
            cell_line_ids=cl_ids,
            drug_ids=dr_ids,
            best_hyperparameters=best_hpams,
            metrics=metrics,
            fold_metadata=self.split_masks.metadata,
        )
