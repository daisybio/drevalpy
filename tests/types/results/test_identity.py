"""Report identity includes variants without changing fold grouping."""

from drevalpy.types.results._identity import RunIdentity


def _identity(metadata=None, randomization=None):
    return RunIdentity.from_metadata("model", "data", "LPO", 2, "fold", randomization, metadata or {})


def test_identity_is_independent_of_metadata_key_order():
    first = _identity({"randomization_provenance": {"type": "permutation", "effective_seed": 7}})
    second = _identity({"randomization_provenance": {"effective_seed": 7, "type": "permutation"}})
    assert first == second


def test_variants_have_distinct_execution_keys_and_one_fold_key():
    ordinary = _identity()
    variants = [
        ordinary,
        _identity({"robustness_trial": 0}),
        _identity({"randomization_provenance": {"effective_seed": 7}}, ("SVRC", "expression")),
        _identity({"randomization_provenance": {"effective_seed": 8}}, ("SVRC", "expression")),
    ]
    assert len({v.execution_key for v in variants}) == len(variants)
    assert {v.fold_key for v in variants} == {ordinary.fold_key}


def test_legacy_fold_index_cannot_collide_with_a_string_fold_id():
    old = RunIdentity("m", "d", "LPO", 2)
    new = RunIdentity("m", "d", "LPO", 2, "2")
    assert old.fold_key != new.fold_key
