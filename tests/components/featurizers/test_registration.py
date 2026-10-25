"""Tests for featurizer selectors, leaf kwargs and the per-side binding decorator."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import (
    featurizer_config_block_label,
    qualified_featurizer_selector,
    requires_explicit_view,
)
from drevalpy.components.featurizers._featurizer_tree import ensure_unique_qualified_featurizers
from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.components.featurizers._side_binding import (
    derived_class_name,
    known_sides,
    register_for_sides,
)
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.models.config import (
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    SampleFeaturizerConfig,
    from_spec,
)
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from drevalpy.registry.sample_featurizer import metadata as sample_metadata
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from tests.components.featurizers._helpers import StatelessFeaturizer
from tests.registry._helpers import isolated_component_registries


def test_qualified_selector_uses_view_brackets() -> None:
    assert qualified_featurizer_selector("pca", "gene_expression") == "pca[gene_expression]"
    assert qualified_featurizer_selector("raw", "mutations") == "raw[mutations]"
    assert qualified_featurizer_selector("landmarkGenes") == "landmarkGenes"


def test_block_labels_and_explicit_view_requirement() -> None:
    assert featurizer_config_block_label("pca", "proteomics") == "pca[proteomics]"
    assert featurizer_config_block_label("fingerprints", None) == "fingerprints"
    assert requires_explicit_view("raw") and requires_explicit_view("pca")
    assert not requires_explicit_view("landmarkGenes")


def test_ensure_unique_allows_same_name_different_views() -> None:
    config = FeaturizerConfig.model_validate(
        {
            "name": "concatFeaturizers",
            "registry": "sample",
            "featurizers": [
                {"name": "raw", "view": "gene_expression"},
                {"name": "raw", "view": "mutations"},
            ],
        },
    )
    ensure_unique_qualified_featurizers(config, "sample")


def test_featurizer_config_rejects_duplicate_qualified_selector() -> None:
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[gene_expression\\]'"):
        FeaturizerConfig.model_validate(
            {
                "name": "concatFeaturizers",
                "registry": "sample",
                "featurizers": [
                    {"name": "raw", "view": "gene_expression"},
                    {"name": "raw", "view": "gene_expression"},
                ],
            },
        )


def test_ensure_unique_ignores_a_non_concat_node() -> None:
    config = FeaturizerConfig.model_validate({"name": "raw", "registry": "sample", "view": "gene_expression"})

    ensure_unique_qualified_featurizers(config, "sample")


def test_recipe_string_selector_uniqueness() -> None:
    with pytest.raises(ValueError, match="Duplicate featurizer selector"):
        from_spec("raw[expression]+raw[expression]:fingerprints:randomForest")
    config = from_spec("raw[expression]+raw[mutations]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "concatFeaturizers"


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_leaf_kwargs_fill_hyperparameter_defaults_from_the_registry(model_config: ModelConfig) -> None:
    sample = featurizer_leaf_kwargs(model_config.sample_featurizer, registry="sample", resolved=None)
    drug = featurizer_leaf_kwargs(model_config.drug_featurizer, registry="drug", resolved=None)

    assert sample == {"n_components": 128, "view": "gene_expression"}
    assert drug == {"radius": 2, "n_bits": 2048, "use_chirality": False, "use_counts": False}


@pytest.mark.parametrize(
    ("leaf", "key", "expected"),
    [
        (SampleFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 4}), "n_components", 4),
        (
            SampleFeaturizerConfig(
                name="pca",
                view="gene_expression",
                hyperparameter_space={"n_components": {"type": "int", "low": 2, "high": 4, "default": 3}},
            ),
            "n_components",
            3,
        ),
        (
            SampleFeaturizerConfig(name="raw", view="gene_expression", options={"view": "mutations"}),
            "view",
            "mutations",
        ),
    ],
    ids=["options-over-defaults", "declared-space-over-registry", "explicit-view-option"],
)
def test_leaf_kwargs_precedence(leaf, key, expected) -> None:
    assert featurizer_leaf_kwargs(leaf, registry="sample", resolved=None)[key] == expected


def test_leaf_kwargs_ignore_space_entries_without_a_default() -> None:
    # ``FeaturizerConfig`` rejects such an entry, so only a config-shaped stub reaches the skip branch.
    class _LeafWithoutDefault:
        name = "pca"
        view = "gene_expression"
        options = None
        hyperparameter_space = {"n_components": {"type": "int", "low": 2, "high": 4}}

    kwargs = featurizer_leaf_kwargs(_LeafWithoutDefault(), registry="sample", resolved=None)

    assert kwargs == {"view": "gene_expression"}


def test_leaf_kwargs_let_resolved_values_win(model_config: ModelConfig) -> None:
    resolved = ResolvedModelConfig(
        template=model_config,
        values={"sample_featurizer.pca[gene_expression].n_components": 16},
    )

    kwargs = featurizer_leaf_kwargs(model_config.sample_featurizer, registry="sample", resolved=resolved)

    assert kwargs["n_components"] == 16


@pytest.fixture
def isolated_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _Probe(StatelessFeaturizer):
    entity_id_only = True

    def _transform_features(self, source, entity_ids: np.ndarray) -> FeaturizerResult:
        return FeaturizerResult.from_blocks({}, len(entity_ids))

    @property
    def output_dim(self) -> int:
        return 1


def _bind(name: str, **kwargs) -> type:
    probe = type(_Probe.__name__, (_Probe,), {"__module__": __name__, "__doc__": _Probe.__doc__})
    return register_for_sides(name, output_format=FeatureFormat.NUMERIC_MATRIX, **kwargs)(probe)


@pytest.mark.parametrize(
    ("implementation_name", "side", "expected"),
    [
        ("SharedIdentityFeaturizer", "sample", "SampleIdentityFeaturizer"),
        ("SharedIdentityFeaturizer", "drug", "DrugIdentityFeaturizer"),
        ("PlainFeaturizer", "drug", "DrugPlainFeaturizer"),
    ],
)
def test_derived_class_name_inserts_the_side_prefix(implementation_name: str, side: str, expected: str) -> None:
    assert derived_class_name(implementation_name, side) == expected


def test_known_sides_lists_both_entity_sides_and_derived_names_reject_others() -> None:
    assert known_sides() == ("drug", "sample")
    with pytest.raises(ValueError, match="unknown featurizer side"):
        derived_class_name("SharedProbeFeaturizer", "tissue")


@pytest.mark.usefixtures("isolated_registries")
def test_register_for_sides_binds_a_class_per_side() -> None:
    implementation = _bind("probeBoth", description="Probe.")
    sample_cls = sample_featurizer_registry.get("probeBoth")
    drug_cls = drug_featurizer_registry.get("probeBoth")

    assert sample_cls is not drug_cls
    assert (sample_cls.side, drug_cls.side) == ("sample", "drug")
    assert issubclass(sample_cls, SampleFeaturizer)
    assert issubclass(drug_cls, DrugFeaturizer)
    assert implementation.side == ""
    assert not hasattr(implementation, "registry_name")
    module = sys.modules[implementation.__module__]
    for side in known_sides():
        assert hasattr(module, derived_class_name(implementation.__name__, side))


@pytest.mark.usefixtures("isolated_registries")
def test_register_for_sides_accepts_per_side_descriptions_and_a_single_side() -> None:
    _bind("probeDescribed", description={"sample": "For samples.", "drug": "For drugs."})
    _bind("probeOneSide", description="Probe.", sides=("drug",))

    assert sample_metadata("probeDescribed")["description"] == "For samples."
    assert drug_metadata("probeDescribed")["description"] == "For drugs."
    assert "probeOneSide" in drug_featurizer_registry.list_names()
    assert "probeOneSide" not in sample_featurizer_registry.list_names()


@pytest.mark.usefixtures("isolated_registries")
def test_register_for_sides_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="unknown featurizer side"):
        _bind("probeBadSide", description="Probe.", sides=("tissue",))
