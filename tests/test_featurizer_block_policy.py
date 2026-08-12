"""Policy guards for featurizer-driven literature inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.cell_line_featurizer import list as list_cell_line_featurizers
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers
from drevalpy.registry.predictor import list as list_predictors
from drevalpy.types.data.batch.feature_block import (
    graph_feature_block,
    merge_feature_blocks,
    numeric_feature_block,
)

REPO = Path(__file__).resolve().parents[1]
DREVALPY = REPO / "drevalpy"


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_components()


def test_registry_discovery_counts() -> None:
    assert len(list_cell_line_featurizers()) == 17
    assert len(list_drug_featurizers()) == 10
    assert len(list_predictors()) == 27


def test_no_raw_dataset_predictor_in_executable_source() -> None:
    hits: list[str] = []
    for path in DREVALPY.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "RawDatasetPredictor" in text:
            hits.append(str(path.relative_to(REPO)))
    assert not hits, hits


def test_predictors_do_not_consume_raw_batch_inputs() -> None:
    hits: list[str] = []
    predictors_root = DREVALPY / "components" / "predictors"
    for path in predictors_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("batch.cell_line_input", "batch.drug_input"):
            if needle in text:
                hits.append(f"{path.relative_to(REPO)}:{needle}")
    assert not hits, hits


def test_concat_and_materialization_preserve_graph_and_ragged_payloads() -> None:
    payload = object()
    graph = graph_feature_block(np.array([payload], dtype=object))
    numeric = numeric_feature_block(np.ones((1, 2), dtype=np.float64))
    merged = merge_feature_blocks({"drug_graph": graph}, {"gene_expression": numeric})
    assert merged["drug_graph"].values[0] is payload
    assert merged["drug_graph"].format is FeatureFormat.GRAPH
