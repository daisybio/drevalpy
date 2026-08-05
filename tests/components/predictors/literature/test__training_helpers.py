"""Smoke tests for shared literature training helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import FeatureDataset


class _StubAlgorithm(LiteratureTrainingMixin):
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    @classmethod
    def get_model_name(cls) -> str:
        return "Stub"

    def configure(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters


def test_get_concatenated_features_concatenates_views() -> None:
    algorithm = _StubAlgorithm()
    cell_lines = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0])},
        }
    )
    drugs = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([3.0, 4.0])},
        }
    )
    matrix = algorithm.get_concatenated_features(
        "gene_expression",
        "fingerprints",
        np.array(["cl1"]),
        np.array(["d1"]),
        cell_lines,
        drugs,
    )
    assert matrix.shape == (1, 4)


def test_log_metrics_calls_wandb_when_enabled() -> None:
    algorithm = _StubAlgorithm()
    algorithm.wandb_project = "proj"
    algorithm.wandb_run = object()
    metrics = {"loss": 0.5}

    with patch("drevalpy.components.predictors.literature._training_helpers.wandb.log") as mock_log:
        algorithm.log_metrics(metrics, step=1)
        mock_log.assert_called_once_with(metrics, step=1)


def test_log_metrics_swallows_wandb_exceptions() -> None:
    algorithm = _StubAlgorithm()
    algorithm.wandb_project = "proj"
    algorithm.wandb_run = object()
    metrics = {"loss": 0.5}

    with patch("drevalpy.components.predictors.literature._training_helpers.wandb.log") as mock_log:
        mock_log.side_effect = RuntimeError("wandb unavailable")
        algorithm.log_metrics(metrics)
