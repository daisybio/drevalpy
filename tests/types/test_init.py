"""Tests for the public :mod:`drevalpy.types` package surface.

:mod:`drevalpy.types` is the shared vocabulary roughly fifty other modules import
from, so its ``__all__`` is a compatibility promise: a rename that forgets the
barrel breaks every one of those dependents at import time. Only the re-export
surface is pinned here - each type's behaviour is tested beside its defining
module.

Origins are recorded against the *leaf* module rather than the intermediate
``types.data`` / ``types.enums`` / ``types.results`` barrels, because comparing a
re-export with the sibling barrel it was imported from cannot fail. The four
assertions themselves live in ``tests/_barrel_surface.py``.
"""

from __future__ import annotations

from drevalpy import types
from tests._barrel_surface import DeclaredSurface

#: ``exported name -> module that defines it``.
EXPECTED_ORIGINS: dict[str, str] = {
    "BlockSpec": "drevalpy.types.data.batch.feature_block",
    "CellLineFeatureSource": "drevalpy.types.data.feature_source",
    "Dataset": "drevalpy.types.data.dataset",
    "DrugFeatureSource": "drevalpy.types.data.feature_source",
    "ExperimentResult": "drevalpy.types.results.experiment",
    "FeatureBlock": "drevalpy.types.data.batch.feature_block",
    "FeatureSource": "drevalpy.types.data.feature_source",
    "LiteratureReference": "drevalpy.types.enums.literature_reference",
    "ModelInputBatch": "drevalpy.types.data.batch.model_input_batch",
    "ModelResult": "drevalpy.types.results.model",
    "ModelScope": "drevalpy.types.enums.model_scope",
    "MuDataLike": "drevalpy.types.data.mudatalike",
    "PredictionMode": "drevalpy.types.enums.prediction_mode",
    "ResponseBatch": "drevalpy.types.data.batch.response_batch",
    "RunResult": "drevalpy.types.results.run",
    "SplitMask": "drevalpy.types.data.split_mask",
    "SplitMasks": "drevalpy.types.data.split_masks",
    "TrialResult": "drevalpy.types.results.trial",
    "build_model_input_batch": "drevalpy.types.data.batch.model_input_build",
    "graph_feature_block": "drevalpy.types.data.batch.feature_block",
    "merge_feature_blocks": "drevalpy.types.data.batch.feature_block",
    "metadata_feature_block": "drevalpy.types.data.batch.feature_block",
    "numeric_feature_block": "drevalpy.types.data.batch.feature_block",
    "pair_cell_line_indices": "drevalpy.types.data.batch.model_input_batch",
    "pair_drug_indices": "drevalpy.types.data.batch.model_input_batch",
    "ragged_feature_block": "drevalpy.types.data.batch.feature_block",
}


class TestTypesSurface(DeclaredSurface):
    barrel = types
    origins = EXPECTED_ORIGINS
