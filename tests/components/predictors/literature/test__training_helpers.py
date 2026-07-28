"""Smoke tests for shared literature training helpers."""

from __future__ import annotations

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
