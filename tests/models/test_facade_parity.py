"""Tests for facade parity."""

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

RF_HP = {"n_estimators": 8, "max_depth": 3, "max_samples": 1.0, "random_state": 0, "n_jobs": 1}


@pytest.mark.parametrize(
    ("model_name", "hp"),
    [
        ("NaiveDrugMeanPredictor", {}),
        ("ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5}),
        ("RandomForest", RF_HP),
    ],
)
def test_equivalent_specs_have_matching_predictions(model_name: str, hp: dict) -> None:
    naive = model_name.startswith("Naive")
    spec = ModelSpec.from_name(model_name) if naive else small_data_spec(model_name)
    dataset = synthetic_mudataset_identity() if naive else synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    first, second = DRPModel(spec, hp), DRPModel(ModelSpec(model_name, spec.config), hp)
    first.train(dataset, split.train)
    second.train(dataset, split.train)
    np.testing.assert_allclose(first.predict(dataset, split.test), second.predict(dataset, split.test), equal_nan=True)


def test_custom_recipe_spec_has_matching_checkpoint_round_trip() -> None:
    spec = ModelSpec("PcaIdentityRF", from_spec("pca[gene_expression]:identity:randomForest"))
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(spec, RF_HP)
    model.train(dataset, split.train)
    with tempfile.TemporaryDirectory() as directory:
        model.save(f"{directory}/model")
        loaded = DRPModel.load(f"{directory}/model")
    np.testing.assert_allclose(model.predict(dataset, split.test), loaded.predict(dataset, split.test), equal_nan=True)
    assert loaded.name == "PcaIdentityRF"
