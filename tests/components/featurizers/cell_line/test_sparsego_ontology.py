"""Tests for the SparseGO ontology-aligned cell-line featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line._sparsego_metadata import attach_sparsego_ontology_metadata
from drevalpy.components.featurizers.cell_line.sparsego_ontology import SparseGOOntologyFeaturizer
from tests.conftest import MockFeatureSource


def _source() -> MockFeatureSource:
    return MockFeatureSource(
        {"cl1": {"gene_expression": np.array([1.0, 2.0]), "mutations": np.array([0.0, 1.0])}},
        meta_info={"gene_expression": ["a", "b"], "mutations": ["a", "b"]},
    )


def _with_ontology() -> MockFeatureSource:
    features = _source()
    attach_sparsego_ontology_metadata(
        features,
        {
            "layer_connections": [np.array([["term", "a"]])],
            "gene2id_mapping_ont": {"a": 0, "b": 1},
            "ontology_gene_order": ("a", "b"),
            "gene_dim_input": 2,
        },
    )
    return features


def test_sparsego_ontology_emits_active_block_and_round_trips_state() -> None:
    features = _with_ontology()
    featurizer = SparseGOOntologyFeaturizer().fit(features)
    block = featurizer.transform_blocks(features, np.array(["cl1"]))["gene_expression"]
    assert block.metadata is not None
    assert block.metadata["gene_dim_input"] == 2
    restored = SparseGOOntologyFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl1"])), block.values)


def test_sparsego_mutations_input_type_reads_the_mutations_view() -> None:
    features = _with_ontology()
    featurizer = SparseGOOntologyFeaturizer(input_type="mutations").fit(features)

    blocks = featurizer.transform_blocks(features, np.array(["cl1"]))

    assert set(blocks) == {"mutations"}
    np.testing.assert_allclose(blocks["mutations"].values, [[0.0, 1.0]])


def test_sparsego_rejects_unknown_input_type() -> None:
    with pytest.raises(ValueError, match="input_type must be"):
        SparseGOOntologyFeaturizer(input_type="methylation")


def test_sparsego_fit_without_ontology_metadata_raises() -> None:
    with pytest.raises(ValueError, match="ontology metadata is missing"):
        SparseGOOntologyFeaturizer().fit(_source())


def test_sparsego_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        SparseGOOntologyFeaturizer().transform(_source(), np.array(["cl1"]))


def test_sparsego_state_is_empty_before_fit() -> None:
    assert SparseGOOntologyFeaturizer().get_state() == {}


def test_sparsego_set_state_rejects_unknown_input_type() -> None:
    with pytest.raises(ValueError, match="input_type must be"):
        SparseGOOntologyFeaturizer().set_state({"input_type": "methylation"})


def test_sparsego_output_block_specs_follow_the_configured_input_type() -> None:
    class _Config:
        hyperparameter_space = {"input_type": {"default": "mutations"}}

    specs = SparseGOOntologyFeaturizer.output_block_specs_for_config(_Config())

    assert [spec.name for spec in specs] == ["mutations"]


def test_sparsego_output_block_specs_default_to_expression() -> None:
    specs = SparseGOOntologyFeaturizer.output_block_specs_for_config(None)

    assert [spec.name for spec in specs] == ["gene_expression"]


def test_sparsego_output_block_specs_default_when_the_space_has_no_input_type() -> None:
    class _Config:
        hyperparameter_space = {"other": {"default": 1}}

    specs = SparseGOOntologyFeaturizer.output_block_specs_for_config(_Config())

    assert [spec.name for spec in specs] == ["gene_expression"]


def test_sparsego_prefers_a_precomputed_variant() -> None:
    from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source

    source = precomputed_source(SparseGOOntologyFeaturizer)
    attach_sparsego_ontology_metadata(
        source,
        {
            "layer_connections": [np.array([["term", "gene0"]])],
            "gene2id_mapping_ont": {"gene0": 0, "gene1": 1},
            "ontology_gene_order": ("gene0", "gene1"),
            "gene_dim_input": 2,
        },
    )
    featurizer = SparseGOOntologyFeaturizer().fit(source)

    matrix = featurizer.transform(source, source.identifiers)

    np.testing.assert_allclose(matrix, PRECOMPUTED)


def test_sparsego_output_dim_is_zero_before_fit() -> None:
    assert SparseGOOntologyFeaturizer().output_dim == 0


@pytest.mark.parametrize(
    ("input_type", "expected"),
    [
        pytest.param("expression", ("gene_expression",), id="expression"),
        pytest.param("mutations", ("mutations",), id="mutations"),
    ],
)
def test_sparsego_resolve_input_views(input_type: str, expected: tuple[str, ...]) -> None:
    assert SparseGOOntologyFeaturizer.resolve_input_views(input_type=input_type) == expected
