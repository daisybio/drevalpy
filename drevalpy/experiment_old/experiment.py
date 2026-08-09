"""Experiment orchestrator: models × folds."""

from __future__ import annotations

from sklearn.base import TransformerMixin

from drevalpy.data.structures import SplitMasks
from drevalpy.data.structures.dataset import Dataset
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel

from .single_run import Run, RunResult

logger = get_logger(__name__)


class Experiment:
    """Orchestrates models × folds.

    Given a list of model classes and a split mode, generates all (model, fold)
    combinations and executes them sequentially.

    Example::

        from drevalpy.data import load, split
        from drevalpy.models import construct_model
        from drevalpy.experiment import Experiment

        mudataset = load("GDSC1")
        folds = split(mudataset, "LCO", n_splits=5)

        ElasticNet = construct_model("ElasticNet")
        RF = construct_model("RandomForest")

        experiment = Experiment(
            models=[ElasticNet, RF],
            mudataset=mudataset,
            folds=folds,
        )
        results = experiment.run()
    """

    def __init__(
        self,
        models: list[type[DRPModel]],
        mudataset: Dataset,
        folds: list[SplitMasks],
        *,
        hyperparameter_tuning: bool = True,
        response_transformation: TransformerMixin | None = None,
        hpo_metric: str = "RMSE",
        hpo_num_samples: int = 16,
        hpo_random_state: int = 42,
    ) -> None:
        """Initialize an Experiment.

        :param models: List of DRPModel subclasses to evaluate.
        :param mudataset: Loaded Dataset with all features.
        :param folds: Pre-split list of SplitMasks (one per CV fold).
        :param hyperparameter_tuning: Whether to run HPO per fold.
        :param response_transformation: Optional sklearn transformer for responses.
        :param hpo_metric: Metric to optimize during HPO.
        :param hpo_num_samples: Number of HPO trials per fold.
        :param hpo_random_state: Random seed for HPO.
        """
        self.models = models
        self.dataset = mudataset
        self._folds = folds
        self.hyperparameter_tuning = hyperparameter_tuning
        self.response_transformation = response_transformation
        self.hpo_metric = hpo_metric
        self.hpo_num_samples = hpo_num_samples
        self.hpo_random_state = hpo_random_state

    @property
    def folds(self) -> list[SplitMasks]:
        """The generated CV folds."""
        return self._folds

    def __repr__(self) -> str:
        """Formatted summary."""
        n_runs = len(self.models) * len(self._folds)
        lines = [
            "Experiment",
            f"    Models: {', '.join(m.get_model_name() for m in self.models)}",
            f"    Folds: {len(self._folds)}",
            f"    Total runs: {n_runs}",
        ]

        if self.hyperparameter_tuning:
            lines.append("    Hyperparameter Tuning: enabled")
            lines.append(f"        metric: {self.hpo_metric}")
            lines.append(f"        num_samples: {self.hpo_num_samples}")
        else:
            lines.append("    Hyperparameter Tuning: disabled")

        return "\n".join(lines)

    @property
    def runs(self) -> list[Run]:
        """Cartesian product of models × folds as Run instances."""
        run_list = []
        for model_class in self.models:
            for _fold_index, split_masks in enumerate(self._folds):
                run_list.append(
                    Run(
                        model_class=model_class,
                        mudataset=self.dataset,
                        split_masks=split_masks,
                        hyperparameter_tuning=self.hyperparameter_tuning,
                        response_transformation=self.response_transformation,
                        hpo_metric=self.hpo_metric,
                        hpo_num_samples=self.hpo_num_samples,
                        hpo_random_state=self.hpo_random_state,
                    )
                )
        return run_list

    def run(self) -> list[RunResult]:
        """Execute all runs sequentially.

        :returns: List of RunResult, one per (model, fold) combination.
        """
        logger.info(
            "Starting experiment: %d models × %d folds = %d runs",
            len(self.models),
            len(self._folds),
            len(self.models) * len(self._folds),
        )
        results = []
        for r in self.runs:
            result = r.execute()
            results.append(result)
            logger.info(
                "Completed: %s fold %d — %s",
                result.model_name,
                result.fold_index,
                {k: f"{v:.4f}" for k, v in result.metrics.items()},
            )
        logger.info("Experiment complete. %d results.", len(results))
        return results
