"""Invariants for ModelConfig-native built-in model execution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig, from_spec, validate
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import list_zoo_names
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_model_factory_names_resolve_to_model_config(name: str) -> None:
    config = model_config_for_name(name)
    assert isinstance(config, ModelConfig)
    validate(config)
    assert config.predictor.name


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_zoo_entries_create_runnable_models(name: str) -> None:
    config = from_spec(name)
    assert isinstance(config, ModelConfig)
    validate(config)
    model_cls = construct_model(name)
    assert issubclass(model_cls, DRPModel)
    assert model_cls() is not None


def test_no_pair_context_in_production_code() -> None:
    repo_root = Path(__file__).resolve().parents[2] / "drevalpy"
    hits = [
        path.relative_to(repo_root.parent)
        for path in repo_root.rglob("*.py")
        if "pair_context" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"pair_context found in production code: {hits}"


def test_multiview_baselines_are_construct_model_classes() -> None:
    for name in ("MultiViewRandomForest", "MultiViewXGBoost", "MultiViewLightGBM"):
        cls = construct_model(name)
        assert issubclass(cls, DRPModel)
        config = from_spec(name)
        assert isinstance(config, ModelConfig)
        assert config.cell_line_featurizer is not None


@pytest.mark.parametrize("name", ["ElasticNet", "NaiveDrugMeanPredictor"])
def test_component_stack_save_load_round_trip(name: str) -> None:
    if name == "ElasticNet":
        model = construct_model(name)({"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1000})
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
    else:
        model = construct_model(name)({})
        mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    model.train(mudataset, split)
    preds_before = model.predict(mudataset, split)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = f"{directory}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        preds_after = loaded.predict(mudataset, split)
    assert np.allclose(preds_before, preds_after, rtol=1e-6, atol=1e-6)
