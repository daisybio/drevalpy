"""Direct ModelSpec/DRPModel execution for dependency-light models."""

from __future__ import annotations

import tempfile
import textwrap

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_dict, validate
from tests._trusted_subprocess import run_trusted_python
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)

_NAIVE_PRESETS = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveSampleMeanPredictor",
)
_SKLEARN_PRESETS = (
    "ElasticNet",
    "RandomForest",
    "Lasso",
    "SVR",
    "SingleDrugElasticNet",
    "SingleDrugRandomForest",
)


@pytest.mark.parametrize(
    ("preset", "naive"),
    [(preset, True) for preset in _NAIVE_PRESETS] + [(preset, False) for preset in _SKLEARN_PRESETS],
)
def test_direct_component_round_trip(preset: str, naive: bool) -> None:
    mudataset = synthetic_mudataset_identity() if naive else synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(ModelSpec.from_name(preset) if naive else small_data_spec(preset))
    model.train(mudataset, split.train)
    preds = model.predict(mudataset, split.test)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(f"{tmp}/model")
        loaded_preds = DRPModel.load(f"{tmp}/model").predict(mudataset, split.test)
    assert np.allclose(preds, loaded_preds)


def test_featurizer_requirements_by_predictor_scope() -> None:
    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        validate(from_dict({"sample_featurizer": "scaledGeneExpression", "predictor": "elasticNet"}))

    single = from_dict({"sample_featurizer": "scaledGeneExpression", "predictor": "singleDrugElasticNet"})
    validate(single)
    assert single.drug_featurizer is not None
    assert single.drug_featurizer.name == "identity"

    validate(from_dict({"predictor": "naiveMean"}))


@pytest.mark.slow
def test_subprocess_blocks_optional_deps_for_simple_models() -> None:
    """Spawns an interpreter because the pristine import graph is the assertion."""
    script = textwrap.dedent("""
        import importlib.abc
        import importlib.machinery
        import sys

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
        from drevalpy.models import DRPModel, ModelSpec
        from drevalpy.models.config import from_spec

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

        naive = DRPModel(ModelSpec.from_name("NaivePredictor"), {})
        naive.train(mudataset_id, split.train)
        elastic = DRPModel(ModelSpec("ElasticNet", from_spec("scaledGeneExpression:fingerprints:elasticNet")))
        elastic.train(mudataset_ge, split.train)
        """)
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr
