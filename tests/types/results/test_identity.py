"""Run identity: the fold key and run key of a run, and how stress tests tell runs of one fold apart."""

from __future__ import annotations

from drevalpy.types.results._identity import RunIdentity
from tests.synthetic import make_run_result


def _identity(metadata=None, randomization=None):
    return RunIdentity.from_metadata("model", "data", "LPO", 2, "fold", randomization, metadata or {})


def test_identity_is_independent_of_metadata_key_order():
    first = _identity({"randomization_provenance": {"type": "permutation", "effective_seed": 7}})
    second = _identity({"randomization_provenance": {"effective_seed": 7, "type": "permutation"}})
    assert first == second


def test_stress_tests_have_distinct_run_keys_and_one_fold_key():
    ordinary = _identity()
    variants = [
        ordinary,
        _identity({"robustness_repeat": 0}),
        _identity({"randomization_provenance": {"effective_seed": 7}}, ("SVRC", "expression")),
        _identity({"randomization_provenance": {"effective_seed": 8}}, ("SVRC", "expression")),
    ]
    assert len({v.run_key for v in variants}) == len(variants)
    assert {v.fold_key for v in variants} == {ordinary.fold_key}


def test_an_index_cannot_collide_with_a_string_fold_id():
    old = RunIdentity("m", "d", "LPO", 2)
    new = RunIdentity("m", "d", "LPO", 2, "2")
    assert old.fold_key != new.fold_key


def test_the_fold_key_prefers_the_stable_fold_id_else_the_index() -> None:
    assert make_run_result(fold_index=7, fold_id="stable").fold_key == ("SyntheticDataset", "LPO", "id", "stable")
    assert make_run_result(fold_index=7, fold_id="").fold_key == ("SyntheticDataset", "LPO", "index", 7)


def test_run_id_distinguishes_stress_tests() -> None:
    ordinary = make_run_result(fold_index=0)
    randomized = make_run_result(fold_index=0, randomization=("SVRC", "gene_expression"))
    robustness = make_run_result(fold_index=0, fold_metadata={"robustness_repeat": 0})

    assert ordinary.run_id != randomized.run_id
    assert ordinary.run_id != robustness.run_id
