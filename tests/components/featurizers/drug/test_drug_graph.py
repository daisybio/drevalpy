"""Tests for graph drug featurizer payload handling."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
from tests.conftest import MockFeatureSource


def test_drug_graph_featurizer_preserves_graph_payloads() -> None:
    graph = Data(
        x=torch.ones((2, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    features = MockFeatureSource({"d1": {"drug_graph": graph}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    block = featurizer.transform_blocks(features, np.array(["d1"]))["drug_graph"]

    assert block.values.shape == (1,)
    assert block.values[0] is graph
