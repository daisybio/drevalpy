"""Tests for behavior-neutral block-to-dataset conversion helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.feature_block import (
    graph_feature_block,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.components.predictors.literature._feature_dataset_from_batch import (
    feature_dataset_from_blocks,
    merge_feature_dataset,
)
from drevalpy.datasets.dataset import FeatureDataset


def test_feature_dataset_from_blocks_materializes_float32_views() -> None:
    result = feature_dataset_from_blocks(
        np.array(["c1", "c2"]),
        {
            "expression": numeric_feature_block(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                feature_names=("e1", "e2"),
            ),
            "mutation": numeric_feature_block(np.array([[0], [1]], dtype=np.int64)),
        },
    )

    assert set(result.features) == {"c1", "c2"}
    np.testing.assert_array_equal(result.features["c2"]["expression"], np.array([3.0, 4.0], dtype=np.float32))
    assert result.features["c1"]["mutation"].dtype == np.float32
    assert result.meta_info["expression"] == ["e1", "e2"]


def test_feature_dataset_from_blocks_preserves_graph_and_ragged_payloads() -> None:
    graph_obj = object()
    ragged_row = np.array([1, 2, 3])
    result = feature_dataset_from_blocks(
        np.array(["c1"]),
        {
            "drug_graph": graph_feature_block(np.array([graph_obj], dtype=object)),
            "molgnet_features": ragged_feature_block(np.array([ragged_row], dtype=object)),
        },
    )
    assert result.features["c1"]["drug_graph"] is graph_obj
    np.testing.assert_array_equal(result.features["c1"]["molgnet_features"], ragged_row)


def test_feature_dataset_from_blocks_skips_metadata_blocks() -> None:
    result = feature_dataset_from_blocks(
        np.array(["c1"]),
        {
            "identity": numeric_feature_block(np.array([[1.0, 0.0]], dtype=np.float32)),
            "identity_categories": metadata_feature_block(np.array(["d1", "d2"], dtype=str)),
        },
    )
    assert set(result.features["c1"]) == {"identity"}


def test_feature_dataset_from_blocks_returns_fallback_for_empty_blocks() -> None:
    fallback = FeatureDataset({"c1": {"expression": np.array([1.0])}})

    assert feature_dataset_from_blocks(np.array(["c1"]), {}, fallback=fallback) is fallback


def test_merge_feature_dataset_overlays_blocks_without_mutating_primary() -> None:
    primary = FeatureDataset({"c1": {"expression": np.array([1.0])}})

    merged = merge_feature_dataset(
        primary,
        {"pathways": numeric_feature_block(np.array([[2.0], [3.0]]))},
        np.array(["c1", "c2"]),
    )

    assert "pathways" not in primary.features["c1"]
    np.testing.assert_array_equal(merged.features["c1"]["pathways"], np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(merged.features["c2"]["pathways"], np.array([3.0], dtype=np.float32))
