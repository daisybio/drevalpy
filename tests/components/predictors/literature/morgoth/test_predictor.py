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
from drevalpy.testing.inputs import build_synthetic_input
from drevalpy.types.data.predictor_input import PredictorInput
from tests.synthetic.variants import REQUIRES_MORGOTH, small_hyperparameters

SMALL = small_hyperparameters("SingleDrugMorgoth")


def _batch(*, with_regulation: bool = True) -> PredictorInput:
    layers = ("regulation",) if with_regulation else ()
    return build_synthetic_input(
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


def _with_layers(batch, layers, **changes):
    return replace(batch, pairs=replace(batch.pairs, layers=layers, **changes))


@pytest.mark.slow
@REQUIRES_MORGOTH
def test_each_drug_trains_on_its_pairs_with_a_verdict(seen_classes) -> None:
    batch = _batch()
    regulation = np.where(np.arange(len(batch)) % 3 == 0, np.nan, np.where(np.arange(len(batch)) % 2, -1.0, 1.0))
    batch = _with_layers(batch, {"regulation": regulation})

    SingleDrugMorgothPredictor(SMALL).fit(batch)

    expected = []
    for drug_id in sorted(set(batch.drug_ids)):
        drug = regulation[batch.drug_ids == drug_id]
        expected.append((drug[~np.isnan(drug)] == -1).astype(int).tolist())
    assert sorted(c.tolist() for c in seen_classes if c is not None) == sorted(expected)


@pytest.mark.slow
@REQUIRES_MORGOTH
def test_a_drug_without_any_verdict_predicts_nan() -> None:
    batch = _batch()
    first = batch.drug_ids == sorted(set(batch.drug_ids))[0]
    regulation = np.where(first, np.nan, -1.0)
    predictor = SingleDrugMorgothPredictor(SMALL)

    predictor.fit(_with_layers(batch, {"regulation": regulation}))
    predictions = predictor.predict(_with_layers(batch, {}, response=np.full(len(batch), np.nan)))

    assert np.isnan(predictions[first]).all()
    assert np.isfinite(predictions[~first]).all()


@REQUIRES_MORGOTH
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
