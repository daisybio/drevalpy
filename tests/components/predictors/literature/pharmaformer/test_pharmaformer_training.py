"""Tests for PharmaFormer training helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.pharmaformer.pharmaformer_training import (
    _scale_cell_line_gene_expression,
)
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


def test_scale_cell_line_gene_expression_returns_scalers_and_size() -> None:
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([1.0, 3.0])},
            "cl2": {"gene_expression": np.array([2.0, 4.0])},
        },
        meta_info={"gene_expression": np.array(["g1", "g2"])},
    )
    output = DrugResponseDataset(
        response=np.array([0.5, 0.6]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1"]),
    )

    scaled, scaler, normalizer, gene_input_size = _scale_cell_line_gene_expression(cell_line_input, output)

    assert gene_input_size == 2
    assert scaler is not None
    assert normalizer is not None
    assert scaled.features["cl1"]["gene_expression"].shape == (2,)
