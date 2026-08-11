"""Direct construct_model execution for dependency-light models."""

from __future__ import annotations

import tempfile
import textwrap

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import from_dict, validate
from tests._trusted_subprocess import run_trusted_python
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
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
    mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    model = construct_model(preset)()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds)


@pytest.mark.parametrize("preset", _SKLEARN_PRESETS)
def test_sklearn_direct_component_round_trip(preset: str) -> None:
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()
    model = construct_model(preset)()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds)


def test_multi_drug_sklearn_rejects_missing_drug_featurizer() -> None:
    from drevalpy.models.config import from_dict, validate

    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        validate(
            from_dict(
                {
                    "cell_line_featurizer": "scaledGeneExpression",
                    "predictor": "elasticNet",
                }
            )
        )


def test_single_drug_sklearn_auto_injects_identity() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    config = from_dict(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "predictor": "singleDrugElasticNet",
        }
    )
    validate(config)
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"


def test_feature_free_naive_accepts_no_featurizers() -> None:
    config = from_dict({"predictor": "naiveMean"})
    validate(config)


def test_subprocess_blocks_optional_deps_for_simple_models() -> None:
    script = textwrap.dedent("""
        import importlib.abc
        import importlib.machinery
        import sys

        # Block optional heavy engines/extras not required for built-in sklearn baselines.
        blocked = {
            "xgboost": "blocked xgboost",
            "lightgbm": "blocked lightgbm",
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

        import anndata as ad
        import mudata as md
        import numpy as np
        import pandas as pd

        from drevalpy.types.data.dataset import Dataset
        from drevalpy.types import SplitMask, SplitMasks
        from drevalpy.models import construct_model

        cl_ids = np.array(["cl1", "cl2"])
        drug_ids = np.array(["d1", "d2"])
        response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        response_ad = ad.AnnData(
            X=response_matrix,
            obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
            var=pd.DataFrame(index=drug_ids),
        )
        ge_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        ge_ad = ad.AnnData(
            X=ge_matrix,
            obs=pd.DataFrame(index=cl_ids),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        response_ad.varm["morgan_fingerprint"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mdata = md.MuData({"response": response_ad, "gene_expression": ge_ad})
        mudataset_ge = Dataset(mdata, name="test")

        response_ad2 = ad.AnnData(
            X=response_matrix.copy(),
            obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
            var=pd.DataFrame(index=drug_ids),
        )
        mdata2 = md.MuData({"response": response_ad2})
        mudataset_id = Dataset(mdata2, name="test")

        split = SplitMasks(
            train=SplitMask(np.array([[True, True], [False, False]])),
            test=SplitMask(np.array([[False, False], [True, True]])),
            val=SplitMask(np.zeros((2, 2), dtype=bool)),
        )

        naive = construct_model("NaivePredictor")({})
        naive.train(mudataset_id, split)
        elastic = construct_model("ElasticNet")(construct_model("ElasticNet").get_hyperparameter_set()[0])
        elastic.train(mudataset_ge, split)
        """)
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr
