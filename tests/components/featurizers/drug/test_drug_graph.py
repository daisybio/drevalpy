"""Tests for graph drug featurizer payload handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.utils.torch_io import save_torch_payload


def test_drug_graph_featurizer_preserves_graph_payloads() -> None:
    graph = Data(
        x=torch.ones((2, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    features = FeatureDataset({"d1": {"drug_graph": graph}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    block = featurizer.transform_blocks(features, np.array(["d1"]))["drug_graph"]

    assert block.values.shape == (1,)
    assert block.values[0] is graph


def test_drug_graph_featurizer_load_features_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    graph = Data(
        x=torch.ones((2, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    graph_dir = tmp_path / "TOYv1" / "drug_graphs"
    graph_dir.mkdir(parents=True)
    save_torch_payload(graph, graph_dir / "d1.pt")

    loaded = DrugGraphFeaturizer.load_features("TOYv1")
    assert "d1" in loaded.features
    assert isinstance(loaded.features["d1"]["drug_graph"], Data)
