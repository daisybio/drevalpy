"""Tests for the qualified hyperparameter key grammar.

The grammar is the one place that decides what a qualified key looks like; both
``drevalpy.models.config`` and ``drevalpy.models.tuning`` build and parse keys
through it, so round-tripping ``*_prefix`` against ``split_*`` is asserted here
rather than in either consumer.
"""

from __future__ import annotations

import pytest

from drevalpy.models._hp_key_grammar import (
    CELL_LINE_SLOT,
    DRUG_SLOT,
    FEATURIZER_SLOTS,
    PREDICTOR_SLOT,
    REGISTRY_TO_SLOT,
    SLOT_TO_REGISTRY,
    featurizer_prefix,
    is_featurizer_slot_key,
    predictor_prefix,
    reject_indexed_featurizer_key,
    split_predictor_key,
    split_prefixed_key,
)


class TestSlotConstants:
    """The slot names are the wire format of every persisted config."""

    def test_slot_names(self) -> None:
        assert (CELL_LINE_SLOT, DRUG_SLOT, PREDICTOR_SLOT) == (
            "cell_line_featurizer",
            "drug_featurizer",
            "predictor",
        )

    def test_featurizer_slots_exclude_the_predictor(self) -> None:
        assert FEATURIZER_SLOTS == (CELL_LINE_SLOT, DRUG_SLOT)
        assert PREDICTOR_SLOT not in FEATURIZER_SLOTS

    def test_registry_and_slot_maps_are_inverses(self) -> None:
        assert REGISTRY_TO_SLOT == {"cell_line": CELL_LINE_SLOT, "drug": DRUG_SLOT}
        assert SLOT_TO_REGISTRY == {slot: registry for registry, slot in REGISTRY_TO_SLOT.items()}


class TestPrefixBuilders:
    """Key construction is a pure string join over the slot names."""

    @pytest.mark.parametrize(
        ("registry", "selector", "expected"),
        [
            pytest.param("cell_line", "pca", "cell_line_featurizer.pca.n_components", id="plain-selector"),
            pytest.param(
                "cell_line",
                "pca[methylation]",
                "cell_line_featurizer.pca[methylation].n_components",
                id="view-qualified-selector",
            ),
            pytest.param("drug", "fingerprints", "drug_featurizer.fingerprints.n_components", id="drug-slot"),
        ],
    )
    def test_featurizer_prefix(self, registry: str, selector: str, expected: str) -> None:
        assert featurizer_prefix(registry, selector, "n_components") == expected

    def test_featurizer_prefix_rejects_an_unknown_registry(self) -> None:
        with pytest.raises(KeyError):
            featurizer_prefix("proteome", "pca", "n_components")

    def test_predictor_prefix(self) -> None:
        assert predictor_prefix("elasticNet", "alpha") == "predictor.elasticNet.alpha"


class TestIsFeaturizerSlotKey:
    """Only the two featurizer slots count as already-addressed keys."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("cell_line_featurizer.pca.n_components", True, id="cell-line-slot"),
            pytest.param("drug_featurizer.fingerprints.radius", True, id="drug-slot"),
            pytest.param("predictor.elasticNet.alpha", False, id="predictor-slot"),
            pytest.param("n_components", False, id="short-key"),
            pytest.param("cell_line_featurizer", False, id="bare-slot-without-dot"),
        ],
    )
    def test_classifies_a_key(self, key: str, expected: bool) -> None:
        assert is_featurizer_slot_key(key) is expected


class TestRejectIndexedFeaturizerKey:
    """The removed ``slot.name.<index>.param`` notation is refused outright."""

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("cell_line_featurizer.pca.0.n_components", id="cell-line-slot"),
            pytest.param("drug_featurizer.fingerprints.1.radius", id="drug-slot"),
            pytest.param("cell_line_featurizer.pca.12.n_components", id="multi-digit-index"),
            pytest.param("cell_line_featurizer.pca.0.nested.param", id="dotted-parameter"),
        ],
    )
    def test_rejects_an_indexed_key(self, key: str) -> None:
        with pytest.raises(ValueError, match="no longer supported"):
            reject_indexed_featurizer_key(key)

    def test_error_suggests_the_qualified_selector(self) -> None:
        with pytest.raises(ValueError, match=r"cell_line_featurizer\.pca\[<view>\]\.n_components"):
            reject_indexed_featurizer_key("cell_line_featurizer.pca.0.n_components")

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("cell_line_featurizer.pca.n_components", id="unindexed"),
            pytest.param("cell_line_featurizer.pca[gene_expression].n_components", id="qualified-selector"),
            pytest.param("cell_line_featurizer.pca.view.n_components", id="non-numeric-segment"),
            pytest.param("predictor.elasticNet.0.alpha", id="predictor-slot-is-not-matched"),
            pytest.param("cell_line_featurizer.pca.0", id="no-parameter-segment"),
        ],
    )
    def test_leaves_other_keys_alone(self, key: str) -> None:
        assert reject_indexed_featurizer_key(key) is None


class TestSplitPrefixedKey:
    """Parsing inverts :func:`featurizer_prefix` and refuses everything else."""

    @pytest.mark.parametrize(
        ("registry", "selector", "param"),
        [
            pytest.param("cell_line", "pca", "n_components", id="plain-selector"),
            pytest.param("cell_line", "pca[methylation]", "n_components", id="view-qualified-selector"),
            pytest.param("drug", "fingerprints", "radius", id="drug-slot"),
            pytest.param("drug", "fingerprints", "nested.param", id="dotted-parameter"),
        ],
    )
    def test_round_trips_a_built_key(self, registry: str, selector: str, param: str) -> None:
        assert split_prefixed_key(featurizer_prefix(registry, selector, param)) == (registry, selector, param)

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("predictor.elasticNet.alpha", id="predictor-slot"),
            pytest.param("n_components", id="short-key"),
            pytest.param("cell_line_featurizer.pca", id="missing-parameter"),
        ],
    )
    def test_returns_none_for_a_non_featurizer_key(self, key: str) -> None:
        assert split_prefixed_key(key) is None

    def test_propagates_the_indexed_key_rejection(self) -> None:
        with pytest.raises(ValueError, match="no longer supported"):
            split_prefixed_key("cell_line_featurizer.pca.0.n_components")


class TestSplitPredictorKey:
    """Parsing inverts :func:`predictor_prefix` and refuses everything else."""

    @pytest.mark.parametrize(
        ("name", "param"),
        [
            pytest.param("elasticNet", "alpha", id="plain-parameter"),
            pytest.param("randomForest", "nested.param", id="dotted-parameter"),
        ],
    )
    def test_round_trips_a_built_key(self, name: str, param: str) -> None:
        assert split_predictor_key(predictor_prefix(name, param)) == (name, param)

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("cell_line_featurizer.pca.n_components", id="featurizer-slot"),
            pytest.param("predictor.elasticNet", id="missing-parameter"),
            pytest.param("alpha", id="short-key"),
        ],
    )
    def test_returns_none_for_a_non_predictor_key(self, key: str) -> None:
        assert split_predictor_key(key) is None
