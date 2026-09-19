"""Tests for the public surface of :mod:`drevalpy.types.data.batch`."""

from __future__ import annotations

from drevalpy.types.data import batch
from tests._barrel_surface import DeclaredSurface


class TestBatchSurface(DeclaredSurface):
    """Pin the batch re-exports used by component and plugin integrations."""

    barrel = batch
    origins = {
        "BlockSpec": "drevalpy.types.data.batch.feature_block",
        "FeatureBlock": "drevalpy.types.data.batch.feature_block",
        "FeaturizerResult": "drevalpy.types.data.batch.feature_block",
        "ModelInputBatch": "drevalpy.types.data.batch.model_input_batch",
        "build_model_input_batch": "drevalpy.types.data.batch.model_input_build",
        "graph_feature_block": "drevalpy.types.data.batch.feature_block",
        "merge_feature_blocks": "drevalpy.types.data.batch.feature_block",
        "metadata_feature_block": "drevalpy.types.data.batch.feature_block",
        "numeric_feature_block": "drevalpy.types.data.batch.feature_block",
        "pair_cell_line_indices": "drevalpy.types.data.batch.model_input_batch",
        "pair_drug_indices": "drevalpy.types.data.batch.model_input_batch",
        "ragged_feature_block": "drevalpy.types.data.batch.feature_block",
    }
