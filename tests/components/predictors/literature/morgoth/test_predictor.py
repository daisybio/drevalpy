"""SingleDrugMorgoth: CurveCurator classes, the quantile fallback and the predictor contract."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from drevalpy.components.predictors.literature.morgoth import predictor as morgoth_predictor
from drevalpy.components.predictors.literature.morgoth._forest import MorgothRegressor
from drevalpy.components.predictors.literature.morgoth.predictor import SingleDrugMorgothPredictor, class_labels
from drevalpy.data import synthetic
from drevalpy.testing.batch import build_synthetic_batch
from drevalpy.testing.conformance import PREDICTOR_CHECKS
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

SMALL = {"n_estimators": 2, "mrmr_k": 3, "min_samples_leaf": 2, "max_depth": 3}


def _batch(*, with_regulation: bool = True) -> ModelInputBatch:
    layers = ("regulation",) if with_regulation else ()
    return build_synthetic_batch(
        synthetic(n_samples=20, n_drugs=2, seed=3), n_sample_features=6, n_drug_features=None, response_layers=layers
    )


@pytest.fixture()
def seen_classes(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray | None]:
    seen: list[np.ndarray | None] = []
    original = MorgothRegressor.fit

    def spy(self, x, y, y_class=None):
        seen.append(None if y_class is None else np.asarray(y_class))
        return original(self, x, y, y_class)

    monkeypatch.setattr(MorgothRegressor, "fit", spy)
    return seen


def test_class_labels_mark_down_as_sensitive_and_drop_pairs_without_a_verdict() -> None:
    keep, y_class = class_labels(np.array([-1.0, 0.0, 1.0, np.nan, -1.0], dtype=np.float32), 5)

    assert keep.tolist() == [True, True, True, False, True]
    assert y_class is not None
    assert y_class.tolist() == [1, 0, 0, 1]


def test_class_labels_without_the_layer_keep_every_pair_and_leave_the_class_to_the_forest() -> None:
    keep, y_class = class_labels(None, 3)

    assert keep.tolist() == [True, True, True]
    assert y_class is None


@pytest.mark.slow
def test_each_drug_trains_on_its_pairs_with_a_verdict(seen_classes) -> None:
    batch = _batch()
    regulation = np.where(np.arange(batch.n_pairs) % 3 == 0, np.nan, np.where(np.arange(batch.n_pairs) % 2, -1.0, 1.0))
    batch = replace(batch, response_layers={"regulation": regulation})

    SingleDrugMorgothPredictor(SMALL).fit(batch)

    expected = []
    for drug_id in sorted(set(batch.drug_ids)):
        drug = regulation[batch.drug_ids == drug_id]
        expected.append((drug[~np.isnan(drug)] == -1).astype(int).tolist())
    assert sorted(c.tolist() for c in seen_classes if c is not None) == sorted(expected)


@pytest.mark.slow
def test_a_drug_without_any_verdict_predicts_nan() -> None:
    batch = _batch()
    first = batch.drug_ids == sorted(set(batch.drug_ids))[0]
    regulation = np.where(first, np.nan, -1.0)
    predictor = SingleDrugMorgothPredictor(SMALL)

    predictor.fit(replace(batch, response_layers={"regulation": regulation}))
    predictions = predictor.predict(replace(batch, response=None, response_layers={}))

    assert np.isnan(predictions[first]).all()
    assert np.isfinite(predictions[~first]).all()


def test_without_the_layer_it_warns_once_and_falls_back_to_the_quantile(caplog, seen_classes) -> None:
    with caplog.at_level(logging.WARNING, logger=morgoth_predictor.__name__):
        SingleDrugMorgothPredictor(SMALL).fit(_batch(with_regulation=False))

    assert [r.message.count("regulation") for r in caplog.records] == [1]
    assert seen_classes == [None, None]


def test_every_default_lies_in_its_space() -> None:
    space = SingleDrugMorgothPredictor.get_hyperparameter_space()
    for name, spec in space.items():
        default = spec["default"]
        if spec["type"] in ("int", "float"):
            assert spec["low"] <= default <= spec["high"], name
        elif spec["type"] == "categorical":
            assert default in spec["choices"], name


@pytest.mark.slow
@pytest.mark.parametrize("check", PREDICTOR_CHECKS, ids=lambda check: check.__name__)
def test_conforms(check) -> None:
    check(SingleDrugMorgothPredictor, hyperparameters=SMALL)
