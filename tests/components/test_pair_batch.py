"""Tests for PairBatch construction."""

from __future__ import annotations

import numpy as np

from drevalpy.components.pair_batch_build import build_pair_batch
from drevalpy.datasets.dataset import DrugResponseDataset


def test_build_pair_batch_indexes_entities() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    batch = build_pair_batch(
        response,
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        drug_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert batch.cell_line_pair_idx.tolist() == [0, 1]
    assert batch.drug_pair_idx is not None
    assert batch.drug_pair_idx.tolist() == [0, 1]
