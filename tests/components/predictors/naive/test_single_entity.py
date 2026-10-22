"""Tests for the shared entity-level naive predictor base.

Mirrors the private module
``drevalpy.components.predictors.naive._single_entity`` with the leading
underscore stripped (``AGENTS.md`` rule 4). ``entity_mean.py`` derives both of
its predictors from this base and adds nothing but a ``_feature_side``, so the
lifecycle behaviour is asserted here and ``test_entity_mean.py`` keeps only the
per-side wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from tests.components.predictors.naive._helpers import naive_batch, one_hot


class _SampleSide(SingleEntityNaivePredictor):
    """Concrete probe fixing the feature side to the sample matrix."""


class _DrugSide(SingleEntityNaivePredictor):
    """Concrete probe fixing the feature side to the drug matrix."""

    _feature_side = "drug"


def test_single_entity_predictor_reads_named_blocks_not_flat_matrices() -> None:
    assert issubclass(SingleEntityNaivePredictor, BlockPredictor)
    assert SingleEntityNaivePredictor.input_interface == "block"


def test_single_entity_predictor_defaults_to_the_sample_side() -> None:
    assert SingleEntityNaivePredictor._feature_side == "sample"


def test_single_entity_predictor_is_not_fitted_before_fit() -> None:
    assert _SampleSide().is_fitted() is False


def test_single_entity_predictor_reports_fitted_after_fit() -> None:
    categories = ["cl1", "cl2"]
    predictor = _SampleSide()

    predictor.fit(
        naive_batch(
            response=np.array([2.0, 6.0]),
            sample_features=one_hot(categories, categories),
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )

    assert predictor.is_fitted() is True


def test_single_entity_predictor_recovers_per_entity_means() -> None:
    categories = ["cl1", "cl2"]
    features = one_hot(categories, categories)
    predictor = _SampleSide()

    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0, 7.0]),
            sample_features=features,
            sample_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
        )
    )
    preds = predictor.predict(
        naive_batch(
            sample_features=features,
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )

    np.testing.assert_allclose(preds, [2.0, 6.0])


def test_single_entity_predictor_falls_back_to_the_dataset_mean_for_an_all_zero_row() -> None:
    categories = ["cl1", "cl2"]
    features = one_hot(categories, categories)
    predictor = _SampleSide()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 6.0]),
            sample_features=features,
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )

    preds = predictor.predict(
        naive_batch(
            sample_features=np.vstack([features, np.zeros((1, 2))]),
            sample_pair_idx=np.array([0, 1, 2], dtype=np.int64),
        )
    )

    np.testing.assert_allclose(preds, [2.0, 6.0, 4.0])


def test_single_entity_predictor_honours_the_drug_feature_side() -> None:
    categories = ["d1", "d2"]
    features = one_hot(categories, categories)
    predictor = _DrugSide()

    predictor.fit(
        naive_batch(
            response=np.array([1.0, 9.0]),
            drug_features=features,
            drug_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )
    preds = predictor.predict(
        naive_batch(
            drug_features=features,
            drug_pair_idx=np.array([1, 0], dtype=np.int64),
        )
    )

    np.testing.assert_allclose(preds, [9.0, 1.0])


def test_single_entity_predictor_rejects_fit_without_a_response() -> None:
    predictor = _SampleSide()

    with pytest.raises(ValueError, match="require response values during fit"):
        predictor.fit(naive_batch(n_pairs=2))


def test_single_entity_predictor_inner_fit_guards_against_a_missing_response() -> None:
    predictor = _SampleSide()

    with pytest.raises(RuntimeError, match="batch.response is required"):
        predictor._fit(naive_batch(n_pairs=2))


def test_single_entity_predictor_requires_fit_before_predict() -> None:
    predictor = _SampleSide()

    with pytest.raises(RuntimeError, match="Call fit before predict"):
        predictor.predict(naive_batch(n_pairs=2))


def test_single_entity_predictor_get_state_is_empty_before_fit() -> None:
    assert _SampleSide().get_state() == {}


def test_single_entity_predictor_state_round_trip_reproduces_predictions() -> None:
    categories = ["cl1", "cl2"]
    features = one_hot(categories, categories)
    predictor = _SampleSide()
    batch = naive_batch(
        response=np.array([2.0, 8.0]),
        sample_features=features,
        sample_pair_idx=np.array([0, 1], dtype=np.int64),
    )
    predictor.fit(batch)

    restored = _SampleSide()
    restored.set_state(predictor.get_state())

    assert restored.is_fitted() is True
    np.testing.assert_allclose(restored.predict(batch), predictor.predict(batch))


def test_single_entity_predictor_set_state_ignores_an_empty_payload() -> None:
    predictor = _SampleSide()

    predictor.set_state({})

    assert predictor.is_fitted() is False


def test_single_entity_predictor_state_is_json_friendly() -> None:
    categories = ["cl1", "cl2"]
    predictor = _SampleSide()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 8.0]),
            sample_features=one_hot(categories, categories),
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )

    state = predictor.get_state()

    assert isinstance(state["dataset_mean"], float)
    assert isinstance(state["effects"], list)
