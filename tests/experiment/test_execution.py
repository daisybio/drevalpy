"""Execution descriptions freeze scope decisions before fitting; robustness variants reshuffle pair order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drevalpy.experiment import robustness
from drevalpy.experiment._execution import build_execution
from drevalpy.models import ModelSpec
from drevalpy.types import SplitMask, SplitMasks
from tests.models.synthetic_fixtures import synthetic_mudataset


def _build(spec, dataset, masks):
    return build_execution(
        spec,
        dataset,
        masks,
        hyperparameter_tuning=False,
        hpo_metric="RMSE",
        hpo_num_trials=2,
        hpo_random_state=9,
        seed=5,
    )


def _execution(monkeypatch, *, stopping=True, seed=17, split_mode="LPO"):
    dataset = synthetic_mudataset()
    masks = SplitMasks(
        SplitMask(np.array([[True, False], [False, False]]), seed),
        SplitMask(np.array([[False, False], [False, True]]), seed),
        SplitMask(np.array([[False, True], [True, False]]), seed),
        {"split_mode": split_mode, "fold_index": 1, "nested": {"values": [1, 2]}},
    )
    spec = ModelSpec.from_name("NaivePredictor")
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda self: stopping))
    return _build(spec, dataset, masks), masks, dataset


def test_stopping_partition_is_reused_for_final_fit(monkeypatch):
    execution, original, _ = _execution(monkeypatch)
    scopes = execution.scopes
    assert len(scopes.stopping) == 1
    assert len(scopes.validation) == 1
    assert not (scopes.final_training & scopes.stopping).any()
    assert scopes.final_training == original.train | scopes.validation
    assert scopes.training._pair_seed == scopes.final_training._pair_seed == 17
    assert execution.metadata["execution"]["hpo"]["random_state"] == 9
    assert execution.seed == 5
    assert execution.metadata["execution"]["seed"] == 5


def test_non_stopping_model_uses_all_validation_in_final_fit(monkeypatch):
    execution, original, _ = _execution(monkeypatch, stopping=False)
    assert execution.scopes.stopping is None
    assert execution.scopes.final_training == original.train | original.val


def test_masks_and_nested_metadata_are_independent_and_read_only(monkeypatch):
    execution, original, _ = _execution(monkeypatch)
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
    assert first.identity.execution_key != second.identity.execution_key
    assert first.identity.fold_key == second.identity.fold_key


def test_original_overlap_is_rejected_before_execution(monkeypatch):
    execution, original, dataset = _execution(monkeypatch)
    with pytest.raises(ValueError):
        _build(execution.model, dataset, replace(original, test=original.train))


def test_a_stored_lco_mode_is_recorded_as_lso():
    dataset = synthetic_mudataset()
    masks = SplitMasks(
        SplitMask(np.array([[True, True], [False, False]]), 17),
        SplitMask(np.array([[False, False], [False, False]]), 17),
        SplitMask(np.array([[False, False], [True, True]]), 17),
        {"split_mode": "LCO", "fold_index": 0},
    )
    execution = _build(ModelSpec.from_name("NaivePredictor"), dataset, masks)
    assert execution.identity.split_mode == "LSO"
    assert execution.result_metadata()["split_mode"] == "LSO"


def test_lsto_folds_are_validated_as_lsto_not_as_lpo(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "drevalpy.experiment._execution.validate_folds",
        lambda folds, validation, dataset: seen.append(validation),
    )
    _execution(monkeypatch, split_mode="LStO")
    assert seen == ["LStO"]


@pytest.fixture
def split_masks() -> SplitMasks:
    rng = np.random.default_rng(0)
    train = rng.random((6, 5)) < 0.6
    test = (~train) & (rng.random((6, 5)) < 0.5)
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(~train & ~test),
        metadata={"fold_index": 3, "split_mode": "LPO"},
    )


def test_variant_count_and_metadata(split_masks: SplitMasks) -> None:
    assert robustness(split_masks, 0) == []
    variants = robustness(split_masks, 3)
    assert [v.metadata["robustness_trial"] for v in variants] == [0, 1, 2]
    assert variants[0].metadata["fold_index"] == 3
    assert variants[0].metadata["split_mode"] == "LPO"
    assert "robustness_trial" not in split_masks.metadata


def test_only_pair_order_changes_and_trial_index_seeds_the_shuffle(split_masks: SplitMasks) -> None:
    first, _, third = robustness(split_masks, 3)
    for name in ("train", "test", "val"):
        np.testing.assert_array_equal(getattr(third, name).mask, getattr(split_masks, name).mask)
        np.testing.assert_array_equal(getattr(third, name).pairs, getattr(split_masks, name).shuffled(seed=2).pairs)
    assert not np.array_equal(third.train.pairs, split_masks.train.pairs)
    assert not np.array_equal(first.train.pairs, third.train.pairs)
