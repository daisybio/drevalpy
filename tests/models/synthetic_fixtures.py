"""Tiny in-memory fixtures for model execution gates."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER


def multi_drug_response() -> DrugResponseDataset:
    return DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )


def cell_line_gene_expression() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )


def drug_fingerprints() -> FeatureDataset:
    return FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )


def identity_cell_line_features(*, with_tissue: bool = False) -> FeatureDataset:
    features = {
        "cl1": {CELL_LINE_IDENTIFIER: np.array(["cl1"])},
        "cl2": {CELL_LINE_IDENTIFIER: np.array(["cl2"])},
    }
    if with_tissue:
        features["cl1"][TISSUE_IDENTIFIER] = np.array(["Lung"])
        features["cl2"][TISSUE_IDENTIFIER] = np.array(["Blood"])
    return FeatureDataset(features=features)


def identity_drug_features() -> FeatureDataset:
    return FeatureDataset(
        features={
            "d1": {DRUG_IDENTIFIER: np.array(["d1"])},
            "d2": {DRUG_IDENTIFIER: np.array(["d2"])},
        }
    )
