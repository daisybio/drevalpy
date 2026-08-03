"""Direct tests for drevalpy.components.preprocessing helpers."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.preprocessing import prepare_expression_and_methylation, scale_gene_expression
from drevalpy.datasets.dataset import FeatureDataset


def _cell_line_input() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {
                "gene_expression": np.array([0.0, 1.0, 2.0]),
                "methylation": np.array([0.5, 1.5, 2.5]),
            },
            "cl2": {
                "gene_expression": np.array([1.0, 2.0, 3.0]),
                "methylation": np.array([1.0, 2.0, 3.0]),
            },
        }
    )


def test_scale_gene_expression_fits_and_transforms() -> None:
    cell_line_input = _cell_line_input()
    scaler = StandardScaler()
    transformed = scale_gene_expression(
        cell_line_input=cell_line_input,
        cell_line_ids=np.array(["cl1", "cl2"]),
        training=True,
        gene_expression_scaler=scaler,
    )
    assert transformed.features["cl1"]["gene_expression"].shape == (3,)
    assert np.isfinite(transformed.features["cl1"]["gene_expression"]).all()


def test_prepare_expression_detects_view_via_any_entity() -> None:
    # FeatureDataset.view_names follows the first entity; presence checks must scan all entities.
    cell_line_input = FeatureDataset(
        features={
            "cl0": {
                "tissue": np.array(["lung"], dtype=object),
                "gene_expression": np.array([0.0, 1.0, 2.0]),
            },
            "cl1": {"gene_expression": np.array([0.0, 1.0, 2.0])},
            "cl2": {"gene_expression": np.array([1.0, 2.0, 3.0])},
        }
    )
    scaler = StandardScaler()
    transformed = prepare_expression_and_methylation(
        cell_line_input=cell_line_input,
        cell_line_ids=np.array(["cl1", "cl2"]),
        training=True,
        gene_expression_scaler=scaler,
    )
    assert np.isfinite(transformed.features["cl1"]["gene_expression"]).all()


def test_prepare_expression_and_methylation_training_mode() -> None:
    cell_line_input = _cell_line_input()
    gene_scaler = StandardScaler()
    methylation_scaler = StandardScaler()
    from sklearn.decomposition import PCA

    methylation_pca = PCA(n_components=2)
    transformed = prepare_expression_and_methylation(
        cell_line_input=cell_line_input,
        cell_line_ids=np.array(["cl1", "cl2"]),
        training=True,
        gene_expression_scaler=gene_scaler,
        methylation_scaler=methylation_scaler,
        methylation_pca=methylation_pca,
    )
    assert transformed.features["cl1"]["methylation"].shape == (2,)
