"""Tests for model config native."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import ModelConfig, from_spec, validate
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import list_zoo_names
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_zoo_entries_resolve_and_build_models(name: str) -> None:
    config = model_config_for_name(name)
    assert isinstance(config, ModelConfig)
    validate(config)
    assert config.predictor.name
    assert isinstance(from_spec(name), ModelConfig)
    assert DRPModel(ModelSpec.from_name(name)).name == name


def test_no_pair_context_in_production_code() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "drevalpy"
    hits = [p for p in root.rglob("*.py") if "pair_context" in p.read_text(encoding="utf-8")]
    assert not hits


@pytest.mark.parametrize("name", ["ElasticNet", "NaiveDrugMeanPredictor"])
def test_component_stack_save_load_round_trip(name: str) -> None:
    if name == "ElasticNet":
        model = DRPModel(small_data_spec(name), {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1000})
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
    else:
        model = DRPModel(ModelSpec.from_name(name), {})
        mudataset = synthetic_mudataset_identity()
    split = lso_split_masks()
    model.train(mudataset, split.train)
    before = model.predict(mudataset, split.test)
    with tempfile.TemporaryDirectory() as directory:
        model.save(f"{directory}/model")
        after = DRPModel.load(f"{directory}/model").predict(mudataset, split.test)
    assert np.allclose(before, after, rtol=1e-6, atol=1e-6)
