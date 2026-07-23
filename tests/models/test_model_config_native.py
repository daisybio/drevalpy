"""Invariants for ModelConfig-native built-in model execution."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import construct_model
from drevalpy.models._native_drp_model import NativeDRPModel
from drevalpy.models.config import ModelConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import list_zoo_names


def _synthetic_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0, 2.5, 3.5]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3, 0.4])},
            "cl2": {"gene_expression": np.array([0.5, 0.6, 0.7, 0.8])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0, 0.5, 0.2])},
            "d2": {"fingerprints": np.array([0.0, 1.0, 0.3, 0.7])},
        }
    )
    return response, cell_line_input, drug_input


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_model_factory_names_resolve_to_model_config(name: str) -> None:
    config = model_config_for_name(name)
    config.validate()
    assert config.predictor.name


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_zoo_entries_create_runnable_models(name: str) -> None:
    config = ModelConfig.from_spec(name)
    config.validate()
    composed = config.create_model()
    assert composed is not None


def test_literature_impl_engines_do_not_subclass_drp_model() -> None:
    impl_root = Path(__file__).resolve().parents[2] / "drevalpy" / "components" / "predictors" / "literature" / "impl"
    for path in impl_root.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        module_name = "drevalpy.components.predictors.literature.impl." + ".".join(
            path.relative_to(impl_root).with_suffix("").parts
        )
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, LiteratureEngineBase) and obj is not LiteratureEngineBase:
                assert not issubclass(obj, DRPModel)


def test_no_pair_context_in_production_code() -> None:
    repo_root = Path(__file__).resolve().parents[2] / "drevalpy"
    hits = [
        path.relative_to(repo_root.parent)
        for path in repo_root.rglob("*.py")
        if "pair_context" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"pair_context found in production code: {hits}"


def test_multiview_baselines_are_native_facades() -> None:
    for name in ("MultiViewRandomForest", "MultiViewXGBoost", "MultiViewLightGBM"):
        cls = construct_model(name)
        assert issubclass(cls, NativeDRPModel)
        config = ModelConfig.from_spec(name)
        assert config.cell_line_featurizer is not None


@pytest.mark.parametrize("name", ["ElasticNet", "NaiveDrugMeanPredictor"])
def test_component_stack_save_load_round_trip(name: str) -> None:
    response, cell_line_input, drug_input = _synthetic_data()
    model = construct_model(name)()
    if name == "ElasticNet":
        model.build_model({"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1000})
    else:
        model.build_model({})
    model.train(response, cell_line_input, drug_input)
    preds_before = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        loaded = type(model).load(directory)
        preds_after = loaded.predict(
            response.cell_line_ids,
            response.drug_ids,
            cell_line_input,
            drug_input,
        )
    assert np.allclose(preds_before, preds_after, rtol=1e-6, atol=1e-6)
