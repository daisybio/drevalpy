"""Direct construct_model execution for dependency-light models."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from drevalpy.models import construct_model
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
    model = construct_model(preset)()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


@pytest.mark.parametrize("preset", _SKLEARN_PRESETS)
def test_sklearn_direct_component_round_trip(preset: str) -> None:
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = None if preset.startswith("SingleDrug") else drug_fingerprints()
    model = construct_model(preset)()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
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


def test_single_drug_sklearn_auto_injects_identity() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig.from_dict(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "predictor": "singleDrugElasticNet",
            "scope": "single_drug",
        }
    )
    config.validate()
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"


def test_feature_free_naive_accepts_no_featurizers() -> None:
    config = ModelConfig.from_dict({"predictor": "naiveMean"})
    config.validate()


def test_subprocess_blocks_optional_deps_for_simple_models() -> None:
    script = textwrap.dedent("""
        import importlib.abc
        import importlib.machinery
        import sys

        # Block optional heavy engines/extras; wrapper modules may still load for factory metadata.
        blocked = {
            "xgboost": "blocked xgboost",
            "lightgbm": "blocked lightgbm",
            "drevalpy.components.predictors.literature.dipk.algorithm": "blocked dipk",
            "drevalpy.components.predictors.literature.pharmaformer.algorithm": "blocked pharmaformer",
        }

        class BlockLoader(importlib.abc.Loader):
            def __init__(self, message: str) -> None:
                self.message = message

            def create_module(self, spec):
                raise ImportError(self.message)

            def exec_module(self, module):
                raise ImportError(self.message)

        class BlockFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname in blocked or fullname.split(".", 1)[0] in blocked:
                    key = fullname if fullname in blocked else fullname.split(".", 1)[0]
                    return importlib.machinery.ModuleSpec(fullname, BlockLoader(blocked[key]))
                return None

        sys.meta_path.insert(0, BlockFinder())

        from drevalpy.models import construct_model
        from drevalpy.models._model_lookup import known_model_names
        from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
        import numpy as np

        assert "NaivePredictor" in known_model_names()
        response = DrugResponseDataset(
            response=np.array([1.0, 2.0]),
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
        cell = FeatureDataset(
            features={
                "cl1": {"gene_expression": np.ones(3)},
                "cl2": {"gene_expression": np.zeros(3)},
            }
        )
        drugs = FeatureDataset(
            features={
                "d1": {"fingerprints": np.array([1.0, 0.0])},
                "d2": {"fingerprints": np.array([0.0, 1.0])},
            }
        )
        naive = construct_model("NaivePredictor")({})
        naive.train(response, FeatureDataset(features={}), FeatureDataset(features={}))
        elastic = construct_model("ElasticNet")(construct_model("ElasticNet").get_hyperparameter_set()[0])
        elastic.train(response, cell, drugs)
        try:
            construct_model("DIPK")({"epochs": 1})
        except ImportError:
            pass
        else:
            raise AssertionError("DIPK construction should fail when literature deps are blocked")
        """)
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
