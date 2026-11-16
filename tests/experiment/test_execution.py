"""Execution descriptions freeze scope decisions before fitting; robustness variants reshuffle pair order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drevalpy.errors import InvalidInputError
from drevalpy.experiment import robustness
from drevalpy.experiment._execution import build_execution, check_mode_supported
from drevalpy.models import ModelSpec
from drevalpy.types import Fold, PairMask
from tests.models.synthetic_fixtures import synthetic_model_dataset


def _build(spec, dataset, fold):
    return build_execution(
        spec,
        dataset,
        fold,
        hyperparameter_tuning=False,
        hpo_metric="RMSE",
        hpo_num_trials=2,
        hpo_random_state=9,
        seed=5,
    )


def _execution(monkeypatch, *, stopping=True, seed=17, split_mode="LPO"):
    dataset = synthetic_model_dataset()
    fold = Fold(
        PairMask(np.array([[True, False], [False, False]]), seed),
        PairMask(np.array([[False, False], [False, True]]), seed),
        PairMask(np.array([[False, True], [True, False]]), seed),
        {"split_mode": split_mode, "fold_index": 1, "nested": {"values": [1, 2]}},
    )
    spec = ModelSpec.from_name("NaivePredictor")
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda self: stopping))
    return _build(spec, dataset, fold), fold, dataset


def test_stopping_partition_is_reused_for_final_fit(monkeypatch):
    execution, original, _ = _execution(monkeypatch)
    scopes = execution.scopes
    assert len(scopes.stopping) == 1
    assert len(scopes.validation) == 1
    assert not (scopes.final_training.mask & scopes.stopping.mask).any()
    assert scopes.final_training == original.train | scopes.validation
    assert scopes.training.pair_seed == scopes.final_training.pair_seed == 17
    assert execution.metadata["execution"]["hpo"]["random_state"] == 9
    assert execution.seed == 5
    assert execution.metadata["execution"]["seed"] == 5


def test_non_stopping_model_uses_all_validation_in_final_fit(monkeypatch):
    execution, original, _ = _execution(monkeypatch, stopping=False)
    assert execution.scopes.stopping is None
    assert execution.scopes.final_training == original.train | original.val


def test_masks_and_nested_metadata_are_independent_and_read_only(monkeypatch):
    execution, original, _ = _execution(monkeypatch)
    with pytest.raises(ValueError):
        original.train.mask[:] = False
    original.metadata["nested"]["values"].append(3)
    assert len(execution.scopes.training) == 1
    assert execution.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(ValueError):
        execution.scopes.training.mask[:] = False
    with pytest.raises(TypeError):
        execution.metadata["nested"]["values"] = ()
    metadata = execution.result_metadata()
    metadata["nested"]["values"].append(4)
    assert execution.result_metadata()["nested"]["values"] == [1, 2]


def test_provenance_records_order_as_well_as_mask_membership(monkeypatch):
    first, _, _ = _execution(monkeypatch, seed=17)
    second, _, _ = _execution(monkeypatch, seed=18)
    a, b = first.scopes.provenance()["training"], second.scopes.provenance()["training"]
    assert a["sha256"] == b["sha256"]
    assert a["pair_seed"] != b["pair_seed"]
    assert first.identity.run_key != second.identity.run_key
    assert first.identity.fold_key == second.identity.fold_key


def test_original_overlap_is_rejected_before_execution(monkeypatch):
    execution, original, dataset = _execution(monkeypatch)
    with pytest.raises(ValueError):
        _build(execution.model, dataset, replace(original, test=original.train))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"hpo_metric": "NoSuchMetric"}, "NoSuchMetric"), ({"hpo_num_trials": -1}, "n_trials must be >= 0 \\(got -1\\)")],
)
def test_tuning_settings_are_validated(monkeypatch, overrides, message):
    execution, original, dataset = _execution(monkeypatch)
    settings = {"hyperparameter_tuning": True, "hpo_metric": "RMSE", "hpo_num_trials": 2, "hpo_random_state": 9}
    with pytest.raises(ValueError, match=message):
        build_execution(execution.model, dataset, original, **{**settings, **overrides}, seed=5)


def test_a_stored_lco_mode_is_recorded_as_lso():
    dataset = synthetic_model_dataset()
    fold = Fold(
        PairMask(np.array([[True, True], [False, False]]), 17),
        PairMask(np.array([[False, False], [False, False]]), 17),
        PairMask(np.array([[False, False], [True, True]]), 17),
        {"split_mode": "LCO", "fold_index": 0},
    )
    execution = _build(ModelSpec.from_name("NaivePredictor"), dataset, fold)
    assert execution.identity.split_mode == "LSO"
    assert execution.result_metadata()["split_mode"] == "LSO"


def test_lsto_folds_are_validated_as_lsto_not_as_lpo(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "drevalpy.experiment._execution.validate_folds",
        lambda folds, validation, dataset: seen.append(validation),
    )
    _execution(monkeypatch, stopping=False, split_mode="LStO")
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
        PairMask(np.array([[True, False], [True, False]])),
        PairMask(np.zeros((2, 2), dtype=bool)),
        PairMask(np.array([[False, True], [False, True]])),
        {"split_mode": "LDO", "fold_index": 0},
    )


def test_build_execution_rejects_a_single_drug_model_under_ldo():
    with pytest.raises(InvalidInputError, match="single-drug"):
        _build(ModelSpec.from_name("SingleDrugElasticNet"), synthetic_model_dataset(), _one_drug_per_part_fold())


def test_an_early_stopping_model_needs_two_validation_groups(monkeypatch):
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda self: True))

    with pytest.raises(InvalidInputError, match="at least two drugs"):
        _build(ModelSpec.from_name("NaivePredictor"), synthetic_model_dataset(), _one_drug_per_part_fold())


@pytest.fixture
def fold() -> Fold:
    rng = np.random.default_rng(0)
    train = rng.random((6, 5)) < 0.6
    test = (~train) & (rng.random((6, 5)) < 0.5)
    return Fold(
        train=PairMask(train),
        test=PairMask(test),
        val=PairMask(~train & ~test),
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
