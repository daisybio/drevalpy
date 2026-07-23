"""Direct ModelConfig / ComposedModel execution for dependency-light models."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from drevalpy.models.config import ModelConfig
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)

_NAIVE_PRESETS = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
)
_SKLEARN_PRESETS = (
    "ElasticNet",
    "RandomForest",
    "Lasso",
    "SVR",
    "SingleDrugElasticNet",
    "SingleDrugRandomForest",
)


@pytest.mark.parametrize("preset", _NAIVE_PRESETS)
def test_naive_direct_component_round_trip(preset: str) -> None:
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    model = ModelConfig.from_spec(preset).create_model()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = type(model).load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


@pytest.mark.parametrize("preset", _SKLEARN_PRESETS)
def test_sklearn_direct_component_round_trip(preset: str) -> None:
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = None if preset.startswith("SingleDrug") else drug_fingerprints()
    model = ModelConfig.from_spec(preset).create_model()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = type(model).load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_multi_drug_sklearn_rejects_missing_drug_featurizer() -> None:
    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        ModelConfig.from_dict(
            {
                "cell_line_featurizer": "scaledGeneExpression",
                "predictor": "elasticNet",
            }
        ).validate()


def test_single_drug_sklearn_accepts_missing_drug_featurizer() -> None:
    config = ModelConfig.from_dict(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "predictor": "singleDrugElasticNet",
            "scope": "single_drug",
        }
    )
    config.validate()


def test_feature_free_naive_accepts_no_featurizers() -> None:
    config = ModelConfig.from_dict({"predictor": "naiveMean"})
    config.validate()


def test_subprocess_blocks_optional_deps_for_simple_models() -> None:
    script = textwrap.dedent(
        """
        import sys
        import types

        blocked = {
            "xgboost": "blocked xgboost",
            "lightgbm": "blocked lightgbm",
            "ray": "blocked ray",
            "wandb": "blocked wandb",
            "drevalpy.components.predictors.literature.structured_predictors": "blocked literature",
            "drevalpy.components.predictors.literature.impl.dipk.dipk": "blocked dipk",
        }

        class Blocker(types.ModuleType):
            def __getattr__(self, name):
                raise ImportError(blocked[self.__name__])

        for name, message in blocked.items():
            module = Blocker(name)
            module.__dict__["__name__"] = name
            sys.modules[name] = module

        from drevalpy.models import MODEL_FACTORY, ElasticNetModel, NaivePredictor
        from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
        import numpy as np

        assert "NaivePredictor" in MODEL_FACTORY
        response = DrugResponseDataset(
            response=np.array([1.0, 2.0]),
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
        cell = FeatureDataset(features={"cl1": {"gene_expression": np.ones(3)}, "cl2": {"gene_expression": np.zeros(3)}})
        drugs = FeatureDataset(features={"d1": {"fingerprints": np.array([1.0, 0.0])}, "d2": {"fingerprints": np.array([0.0, 1.0])}})
        naive = NaivePredictor()
        naive.build_model({})
        naive.train(response, FeatureDataset(features={}), FeatureDataset(features={}))
        elastic = ElasticNetModel()
        elastic.build_model(elastic.get_hyperparameter_set()[0])
        elastic.train(response, cell, drugs)
        try:
            MODEL_FACTORY["DIPK"]().build_model({"epochs": 1})
        except ImportError:
            pass
        else:
            raise AssertionError("DIPK build_model should fail when literature deps are blocked")
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
