"""Fold preparation and run provenance; robustness variants reshuffle pair order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drevalpy.errors import InvalidInputError
from drevalpy.execution import robustness
from drevalpy.execution._execution import check_mode_supported, prepare_fold
from drevalpy.execution._fit import fit
from drevalpy.models import ModelSpec
from drevalpy.types import Fold, SelectionMask
from tests.models.synthetic_fixtures import synthetic_model_dataset


def _fit(spec, dataset, fold, **overrides):
    settings = {
        "hyperparameter_tuning": False,
        "hpo_metric": "RMSE",
        "hpo_num_trials": 2,
        "hpo_random_state": 9,
        "seed": 5,
    }
    _, result = fit(
        spec,
        dataset,
        fold,
        **{**settings, **overrides},
    )
    return result


def _inputs(monkeypatch, *, stopping=True, seed=17, split_mode="LPO"):
    dataset = synthetic_model_dataset()
    fold = Fold(
        SelectionMask(np.array([[True, False], [False, False]]), seed),
        SelectionMask(np.array([[False, False], [False, True]]), seed),
        SelectionMask(np.array([[False, True], [True, False]]), seed),
        {"split_mode": split_mode, "fold_index": 1, "nested": {"values": [1, 2]}},
    )
    spec = ModelSpec.from_name("NaivePredictor")
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda self: stopping))
    return spec, dataset, fold


def test_stopping_partition_is_reused_for_final_fit(monkeypatch):
    spec, dataset, original = _inputs(monkeypatch)
    fold = prepare_fold(spec, dataset, original, "LPO")
    assert len(fold.stopping) == 1
    assert len(fold.val) == 1
    assert not (fold.train_val.mask & fold.stopping.mask).any()
    assert fold.train_val == original.train | fold.val
    assert fold.fold_id == original.fold_id
    assert fold.train.pair_seed == fold.train_val.pair_seed == 17


def test_non_stopping_model_uses_all_validation_in_final_fit(monkeypatch):
    spec, dataset, original = _inputs(monkeypatch, stopping=False)
    fold = prepare_fold(spec, dataset, original, "LPO")
    assert fold.stopping is None
    assert fold.train_val == original.train | original.val


def test_result_provenance_is_independent_of_fold_metadata(monkeypatch):
    spec, dataset, fold = _inputs(monkeypatch)
    result = _fit(spec, dataset, fold)
    fold.metadata["nested"]["values"].append(3)
    assert result.fold_metadata["nested"]["values"] == [1, 2]


def test_provenance_records_order_as_well_as_mask_membership(monkeypatch):
    first = _fit(*_inputs(monkeypatch, seed=17))
    second = _fit(*_inputs(monkeypatch, seed=18))
    a = first.fold_metadata["execution"]["scopes"]["training"]
    b = second.fold_metadata["execution"]["scopes"]["training"]
    assert a["sha256"] == b["sha256"]
    assert a["pair_seed"] != b["pair_seed"]
    assert first.identity.run_key != second.identity.run_key
    assert first.identity.fold_key == second.identity.fold_key


def test_original_overlap_is_rejected_before_execution(monkeypatch):
    spec, dataset, original = _inputs(monkeypatch)
    with pytest.raises(ValueError):
        _fit(spec, dataset, replace(original, test=original.train))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"hpo_metric": "NoSuchMetric"}, "NoSuchMetric"), ({"hpo_num_trials": -1}, "n_trials must be >= 0 \\(got -1\\)")],
)
def test_tuning_settings_are_validated(monkeypatch, overrides, message):
    spec, dataset, original = _inputs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        _fit(spec, dataset, original, **overrides)


def test_a_stored_lco_mode_is_recorded_as_lso():
    dataset = synthetic_model_dataset()
    fold = Fold(
        SelectionMask(np.array([[True, True], [False, False]]), 17),
        SelectionMask(np.array([[False, False], [False, False]]), 17),
        SelectionMask(np.array([[False, False], [True, True]]), 17),
        {"split_mode": "LCO", "fold_index": 0},
    )
    result = _fit(ModelSpec.from_name("NaivePredictor"), dataset, fold)
    assert result.split_mode == "LSO"
    assert result.fold_metadata["split_mode"] == "LSO"


def test_lsto_folds_are_validated_as_lsto_not_as_lpo(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "drevalpy.execution._execution.validate_folds",
        lambda folds, validation, dataset: seen.append(validation),
    )
    _fit(*_inputs(monkeypatch, stopping=False, split_mode="LStO"))
    assert seen == ["LStO"]


@pytest.mark.parametrize("mode", ["LDO", "LSO", "LPO"])
def test_single_drug_models_are_rejected_only_where_drugs_are_held_out(mode):
    spec = ModelSpec.from_name("SingleDrugElasticNet")

    if mode == "LDO":
        with pytest.raises(InvalidInputError, match="SingleDrugElasticNet is a single-drug model.*LDO"):
            check_mode_supported(spec, mode)
    else:
        check_mode_supported(spec, mode)
    check_mode_supported(ModelSpec.from_name("ElasticNet"), "LDO")


def _one_drug_per_part_fold() -> Fold:
    """An LDO fold of the 2x2 dataset: drug 0 trains, drug 1 validates, nothing is tested."""
    return Fold(
        SelectionMask(np.array([[True, False], [True, False]])),
        SelectionMask(np.zeros((2, 2), dtype=bool)),
        SelectionMask(np.array([[False, True], [False, True]])),
        {"split_mode": "LDO", "fold_index": 0},
    )


def test_fit_rejects_a_single_drug_model_under_ldo():
    with pytest.raises(InvalidInputError, match="single-drug"):
        _fit(ModelSpec.from_name("SingleDrugElasticNet"), synthetic_model_dataset(), _one_drug_per_part_fold())


def test_an_early_stopping_model_needs_two_validation_groups(monkeypatch):
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda self: True))

    with pytest.raises(InvalidInputError, match="at least two drugs"):
        _fit(ModelSpec.from_name("NaivePredictor"), synthetic_model_dataset(), _one_drug_per_part_fold())


@pytest.fixture
def fold() -> Fold:
    rng = np.random.default_rng(0)
    train = rng.random((6, 5)) < 0.6
    test = (~train) & (rng.random((6, 5)) < 0.5)
    return Fold(
        train=SelectionMask(train),
        test=SelectionMask(test),
        val=SelectionMask(~train & ~test),
        metadata={"fold_index": 3, "split_mode": "LPO"},
    )


def test_variant_count_and_metadata(fold: Fold) -> None:
    assert robustness(fold, 0) == []
    variants = robustness(fold, 3)
    assert [v.metadata["robustness_repeat"] for v in variants] == [0, 1, 2]
    assert variants[0].metadata["fold_index"] == 3
    assert variants[0].metadata["split_mode"] == "LPO"
    assert "robustness_repeat" not in fold.metadata


def test_only_pair_order_changes_and_trial_index_seeds_the_shuffle(fold: Fold) -> None:
    first, _, third = robustness(fold, 3)
    for name in ("train", "test", "val"):
        np.testing.assert_array_equal(getattr(third, name).mask, getattr(fold, name).mask)
        np.testing.assert_array_equal(getattr(third, name).pairs, getattr(fold, name).shuffled(seed=2).pairs)
    assert not np.array_equal(third.train.pairs, fold.train.pairs)
    assert not np.array_equal(first.train.pairs, third.train.pairs)


def test_the_sampler_seed_defaults_to_the_root_seed(monkeypatch):
    spec, dataset, fold = _inputs(monkeypatch)
    result = _fit(spec, dataset, fold, hpo_random_state=None)
    provenance = result.fold_metadata["execution"]
    assert provenance["hpo"]["random_state"] == provenance["seed"] == 5
    assert _fit(spec, dataset, fold).fold_metadata["execution"]["hpo"]["random_state"] == 9
