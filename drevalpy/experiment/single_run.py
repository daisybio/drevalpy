"""Single model + single fold execution unit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin, clone
from upath import UPath as Path

from drevalpy.data.structures import SplitMasks
from drevalpy.data.structures.dataset import Dataset
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel

from .fold import prepare_mu_fold
from .training import train_and_predict

logger = get_logger(__name__)


def _available_entity_ids(
    mudataset: Dataset,
    views: list[str],
    *,
    side: str,
) -> frozenset[str] | None:
    """Intersect entity availability across all required views.

    Returns None if no views are required (entity_id_only featurizer).
    """
    if not views:
        return None
    available: frozenset[str] | None = None
    for view in views:
        view_entities = mudataset.entities_with_modality(view, side=side)
        available = view_entities if available is None else available & view_entities
    return available


@dataclass
class TrialResult:
    """Output of a single HPO trial."""

    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    optimization_metric: str
    predictions: np.ndarray

    @property
    def score(self) -> float:
        """The score for the optimization metric."""
        return self.metrics.get(self.optimization_metric, float("nan"))

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = ["TrialResult", "    Hyperparameters:"]
        for k, v in self.hyperparameters.items():
            lines.append(f"        {k}: {v}")
        lines.append("    Metrics:")
        for k, v in self.metrics.items():
            marker = " *" if k == self.optimization_metric else ""
            lines.append(f"        {k}: {v:.4f}{marker}")
        return "\n".join(lines)


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
    trials: list[TrialResult] | None = None

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "RunResult",
            f"    Model: {self.model_name}",
            f"    Fold: {self.fold_index}",
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

        if self.trials:
            lines.append(f"    HPO Trials: {len(self.trials)}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz file.

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "predictions": self.predictions,
            "ground_truth": self.ground_truth,
            "cell_line_ids": self.cell_line_ids,
            "drug_ids": self.drug_ids,
        }
        trials_meta = None
        if self.trials:
            trials_meta = []
            for i, t in enumerate(self.trials):
                arrays[f"trial_{i}_predictions"] = t.predictions
                trials_meta.append(
                    {
                        "hyperparameters": t.hyperparameters,
                        "metrics": t.metrics,
                        "optimization_metric": t.optimization_metric,
                    }
                )
        meta = {
            "model_name": self.model_name,
            "fold_index": self.fold_index,
            "best_hyperparameters": self.best_hyperparameters,
            "metrics": self.metrics,
            "fold_metadata": self.fold_metadata,
            "trials": trials_meta,
        }
        arrays["_metadata"] = np.array(json.dumps(meta))
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        """Load from a .npz file saved by ``save()``.

        :param path: Path to the .npz file.
        :returns: Reconstructed RunResult with trial data.
        """
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["_metadata"]))
        trials = None
        if meta.get("trials"):
            trials = [
                TrialResult(
                    hyperparameters=t["hyperparameters"],
                    metrics=t["metrics"],
                    optimization_metric=t["optimization_metric"],
                    predictions=np.asarray(data[f"trial_{i}_predictions"]),
                )
                for i, t in enumerate(meta["trials"])
            ]
        return cls(
            model_name=meta["model_name"],
            fold_index=meta["fold_index"],
            predictions=np.asarray(data["predictions"]),
            ground_truth=np.asarray(data["ground_truth"]),
            cell_line_ids=np.asarray(data["cell_line_ids"]),
            drug_ids=np.asarray(data["drug_ids"]),
            best_hyperparameters=meta.get("best_hyperparameters", {}),
            metrics=meta.get("metrics", {}),
            fold_metadata=meta.get("fold_metadata", {}),
            trials=trials,
        )


class Run:
    """Single model + single fold execution unit.

    Given a model class and a SplitMasks fold, trains the model (optionally
    with HPO) and predicts on the test set. Returns a RunResult.
    """

    def __init__(
        self,
        model_class: type[DRPModel],
        mudataset: Dataset,
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
        self.dataset = mudataset
        self.hyperparameter_tuning = hyperparameter_tuning
        self.response_transformation = response_transformation
        self.hpo_metric = hpo_metric
        self.hpo_num_samples = hpo_num_samples
        self.hpo_random_state = hpo_random_state

        self.split_masks = self._filter_to_featurizable_pairs(model_class, mudataset, split_masks)

    @staticmethod
    def _filter_to_featurizable_pairs(
        model_class: type[DRPModel],
        mudataset: Dataset,
        split_masks: SplitMasks,
    ) -> SplitMasks:
        """Filter split masks to only include pairs where both entities have features.

        Uses the model's declared views to determine which modalities are required,
        then intersects with available entities in the Dataset.
        """
        config = model_class.model_config()
        cl_views = config.cell_line_views()
        drug_views = config.drug_views()

        available_cl = _available_entity_ids(mudataset, cl_views, side="cell_line")
        available_dr = _available_entity_ids(mudataset, drug_views, side="drug")

        if available_cl is None and available_dr is None:
            return split_masks

        all_cl_ids = mudataset.cell_line_ids
        all_dr_ids = mudataset.drug_ids

        def _filter_pairs(pairs: np.ndarray) -> np.ndarray:
            if len(pairs) == 0:
                return pairs
            mask = np.ones(len(pairs), dtype=bool)
            if available_cl is not None:
                mask &= np.array([all_cl_ids[idx] in available_cl for idx in pairs[:, 0]])
            if available_dr is not None:
                mask &= np.array([all_dr_ids[idx] in available_dr for idx in pairs[:, 1]])
            return pairs[mask]

        train = _filter_pairs(split_masks.train)
        test = _filter_pairs(split_masks.test)
        val = _filter_pairs(split_masks.val)

        n_before = len(split_masks.train) + len(split_masks.test) + len(split_masks.val)
        n_after = len(train) + len(test) + len(val)
        if n_before != n_after:
            logger.info(
                "Filtered %d pairs to %d featurizable pairs (%.1f%% removed)",
                n_before,
                n_after,
                100.0 * (n_before - n_after) / n_before,
            )

        return SplitMasks(train=train, test=test, val=val, metadata=split_masks.metadata)

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
        lines.append(f"        index: {self.split_masks.metadata.get('fold_index', 0)}")
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

        fold_data = prepare_mu_fold(self.dataset, self.split_masks, self.model_class)

        # HPO or default hyperparameters
        trials: list[TrialResult] | None = None
        if self.hyperparameter_tuning:
            from drevalpy.components.core.tuning.hpo import hpam_tune_with_trials

            hpo_cfg = build_experiment_hpo_config(
                self.hpo_metric,
                n_trials=self.hpo_num_samples,
                random_state=self.hpo_random_state,
            )
            best_hpams, raw_trials = hpam_tune_with_trials(
                model_class=self.model_class,
                mudataset=self.dataset,
                train_scope=fold_data.train_scope,
                val_scope=fold_data.val_scope,
                early_stopping_scope=fold_data.early_stopping_scope,
                response_transformation=self.response_transformation,
                metric=self.hpo_metric,
                model_checkpoint_dir=None,
                hpo_config=hpo_cfg,
            )
            trials = [
                TrialResult(
                    hyperparameters=params,
                    metrics=trial_metrics,
                    optimization_metric=self.hpo_metric,
                    predictions=preds,
                )
                for params, trial_metrics, preds in raw_trials
            ]
        else:
            best_hpams = self.model_class.get_default_hyperparameters()

        logger.info("Best hyperparameters: %s", best_hpams)

        # Train on train+val, predict on test
        from .fold import merge_train_val_scopes

        merged_scope = merge_train_val_scopes(self.split_masks)
        model = self.model_class(best_hpams)
        fold_transform = None if self.response_transformation is None else clone(self.response_transformation)

        predictions = train_and_predict(
            model=model,
            mudataset=self.dataset,
            train_scope=merged_scope,
            test_scope=fold_data.test_scope,
            early_stopping_scope=fold_data.early_stopping_scope,
            response_transformation=fold_transform,
        )

        # Extract ground truth
        response_matrix = self.dataset.response_matrix
        test_pairs = self.split_masks.test
        ground_truth = response_matrix[test_pairs[:, 0], test_pairs[:, 1]]

        cl_ids = self.dataset.cell_line_ids[test_pairs[:, 0]]
        dr_ids = self.dataset.drug_ids[test_pairs[:, 1]]

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
            trials=trials,
        )
