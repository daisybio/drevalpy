"""Parity checks for equivalent ModelSpec configurations."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)

_PARITY_CASES = ("NaivePredictor", "NaiveDrugMeanPredictor", "ElasticNet", "RandomForest")


def _minimal_hp(model_name: str) -> dict:
    if model_name.startswith("Naive"):
        return {}
    if model_name == "ElasticNet":
        return {"alpha": 0.1, "l1_ratio": 0.5}
    return {"n_estimators": 8, "max_depth": 3, "max_samples": 1.0, "random_state": 0, "n_jobs": 1}


@pytest.mark.parametrize("model_name", _PARITY_CASES)
def test_equivalent_specs_have_matching_predictions(model_name: str) -> None:
    hp = _minimal_hp(model_name)
    spec = small_data_spec(model_name) if not model_name.startswith("Naive") else ModelSpec.from_name(model_name)
    equivalent = ModelSpec(model_name, spec.config)
    dataset = (
        synthetic_mudataset_identity()
        if model_name.startswith("Naive")
        else synthetic_mudataset_gene_expression_fingerprints()
    )
    split = lso_split_masks()
    first = DRPModel(spec, hp)
    second = DRPModel(equivalent, hp)
    first.train(dataset, split.train)
    second.train(dataset, split.train)
    np.testing.assert_allclose(first.predict(dataset, split.test), second.predict(dataset, split.test), equal_nan=True)


def test_custom_recipe_spec_has_matching_checkpoint_round_trip() -> None:
    config = from_spec("pca[gene_expression]:identity:randomForest")
    spec = ModelSpec("PcaIdentityRF", config)
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(spec, _minimal_hp("RandomForest"))
    model.train(dataset, split.train)
    before = model.predict(dataset, split.test)
    with tempfile.TemporaryDirectory() as directory:
        model.save(f"{directory}/model")
        loaded = DRPModel.load(f"{directory}/model")
    np.testing.assert_allclose(before, loaded.predict(dataset, split.test), equal_nan=True)
    assert loaded.name == "PcaIdentityRF"
