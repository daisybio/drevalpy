"""Tests for raw FeatureDataset view validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.predictors.literature._raw_views import (
    validate_pyg_drug_graphs,
    validate_required_views,
)
from drevalpy.datasets.dataset import FeatureDataset


def test_validate_required_views_none_dataset() -> None:
    with pytest.raises(ValueError, match="requires cell_line FeatureDataset"):
        validate_required_views(
            None,
            ("gene_expression",),
            predictor_name="drugGNN",
            side="cell_line",
        )


def test_validate_required_views_reports_missing_entity_view() -> None:
    features = FeatureDataset(features={"cl1": {"gene_expression": np.array([1.0])}})
    with pytest.raises(ValueError, match="missing cell_line view"):
        validate_required_views(
            features,
            ("gene_expression", "mutations"),
            predictor_name="molir",
            side="cell_line",
        )


def test_validate_pyg_drug_graphs_rejects_wrong_type() -> None:
    features = FeatureDataset(features={"d1": {"drug_graph": object()}})
    with pytest.raises(ValueError, match="torch_geometric.data.Data"):
        validate_pyg_drug_graphs(features, predictor_name="drugGNN")


def test_validate_pyg_drug_graphs_rejects_missing_x() -> None:
    graph = Data(edge_index=torch.tensor([[0], [0]], dtype=torch.long))
    features = FeatureDataset(features={"d1": {"drug_graph": graph}})
    with pytest.raises(ValueError, match="missing attribute 'x'"):
        validate_pyg_drug_graphs(features, predictor_name="drugGNN")


def test_validate_pyg_drug_graphs_rejects_missing_edge_index() -> None:
    graph = Data(x=torch.ones((2, 3), dtype=torch.float32))
    features = FeatureDataset(features={"d1": {"drug_graph": graph}})
    with pytest.raises(ValueError, match="missing attribute 'edge_index'"):
        validate_pyg_drug_graphs(features, predictor_name="drugGNN")


def test_validate_pyg_drug_graphs_rejects_inconsistent_node_dims() -> None:
    g1 = Data(
        x=torch.ones((2, 3), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    g2 = Data(
        x=torch.ones((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    features = FeatureDataset(features={"d1": {"drug_graph": g1}, "d2": {"drug_graph": g2}})
    with pytest.raises(ValueError, match="inconsistent node-feature dimensions"):
        validate_pyg_drug_graphs(features, predictor_name="drugGNN")


def test_validate_pyg_drug_graphs_accepts_consistent_graphs() -> None:
    g1 = Data(
        x=torch.ones((2, 3), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    g2 = Data(
        x=torch.zeros((1, 3), dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    features = FeatureDataset(features={"d1": {"drug_graph": g1}, "d2": {"drug_graph": g2}})
    validate_pyg_drug_graphs(features, predictor_name="drugGNN")
