"""Tests for shared single-drug routing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.core.batch.feature_block import FeatureBlock
from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
)


def _identity_batch() -> ModelInputBatch:
    return ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1, 0, 1]),
        drug_pair_idx=np.array([0, 0, 1, 1]),
        drug_blocks={
            "identity": FeatureBlock(
                values=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
            "identity_categories": FeatureBlock(
                values=np.array(["d1", "d2"]),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
        },
    )


def test_routing_keys_decode_known_drugs() -> None:
    keys = routing_keys(_identity_batch())
    assert keys.tolist() == ["d1", "d1", "d2", "d2"]


def test_iter_drug_masks_returns_stable_masks() -> None:
    batch = _identity_batch()
    routed = list(iter_drug_masks(batch))
    assert {drug_id for drug_id, _ in routed} == {"d1", "d2"}
    for drug_id, mask in routed:
        assert routing_keys(batch.subset_pairs(mask)).tolist() == [drug_id] * int(mask.sum())


def test_require_known_training_keys_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown drug identities"):
        require_known_training_keys(np.array(["d1", ""]))


def test_routing_keys_requires_identity_blocks() -> None:
    batch = _identity_batch()
    batch.drug_blocks = {}
    with pytest.raises(ValueError, match="require drug identity features"):
        routing_keys(batch)
