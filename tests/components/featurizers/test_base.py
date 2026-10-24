"""Tests for the ``Featurizer`` base class contract and the registry sweeps over it."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat, featurizer_contract
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)
from drevalpy.registry.sample_featurizer import (
    get as get_sample_featurizer,
)
from drevalpy.registry.sample_featurizer import (
    list as list_sample_featurizers,
)
from drevalpy.types.data.batch.feature_block import (
    FeaturizerResult,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import resolve_omics_accessor
from tests.components.featurizers._helpers import DoublingFeaturizer, StubSource

_PROBE_VIEW = "gene_expression"

#: The NaN warning is emitted by the mixin ``base.py`` delegates the policy to.
_NAN_LOGGER = "drevalpy.components.featurizers._nan_tolerance"


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _featurizer_names(registry: str) -> list[str]:
    register_builtin_components()
    return list_sample_featurizers() if registry == "sample" else list_drug_featurizers()


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_every_featurizer_declares_input_views(registry: str) -> None:
    get = get_sample_featurizer if registry == "sample" else get_drug_featurizer
    names = _featurizer_names(registry)
    assert names
    for name in names:
        cls = get(name)
        if issubclass(cls, ConcatFeaturizersMixin):
            continue
        kwargs = {"view": _PROBE_VIEW} if cls.requires_view else {}
        views = cls.resolve_input_views(**kwargs)
        assert isinstance(views, tuple), name
        assert all(isinstance(view, str) and view.strip() for view in views), name


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_entity_id_only_featurizers_need_no_views(registry: str) -> None:
    get = get_sample_featurizer if registry == "sample" else get_drug_featurizer
    entity_id_only = [name for name in _featurizer_names(registry) if get(name).entity_id_only]
    assert entity_id_only
    for name in entity_id_only:
        assert get(name).resolve_input_views() == (), name


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_concat_featurizer_refuses_standalone_view_resolution(registry: str) -> None:
    get = get_sample_featurizer if registry == "sample" else get_drug_featurizer
    cls = get("concatFeaturizers")
    with pytest.raises(TypeError, match="has no input views of its own"):
        cls.resolve_input_views()


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_sample_featurizer(name)
    assert cls.requires_view
    assert cls.resolve_input_views(view="mutations") == ("mutations",)
    with pytest.raises(TypeError, match="requires an explicit view"):
        cls.resolve_input_views()


_SAMPLE_CONTRACTS = [
    ("landmarkGenes", FeatureFormat.NUMERIC_MATRIX),
    ("landmarkGenesReduced", FeatureFormat.NUMERIC_MATRIX),
    ("pathways", FeatureFormat.NUMERIC_MATRIX),
    ("bionic", FeatureFormat.NUMERIC_MATRIX),
    ("dipkGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("pharmaFormerGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("sparsegoOntology", FeatureFormat.NUMERIC_MATRIX),
    ("molirOmics", FeatureFormat.NUMERIC_MATRIX),
    ("superfeltrOmics", FeatureFormat.NUMERIC_MATRIX),
    ("concatFeaturizers", FeatureFormat.NUMERIC_MATRIX),
    ("raw", FeatureFormat.NUMERIC_MATRIX),
    ("pca", FeatureFormat.NUMERIC_MATRIX),
]

_DRUG_CONTRACTS = [
    ("molgnet", FeatureFormat.RAGGED_SEQUENCE),
    ("bpePharmaformer", FeatureFormat.NUMERIC_MATRIX),
    ("smilesvec", FeatureFormat.NUMERIC_MATRIX),
    ("drugGraph", FeatureFormat.GRAPH),
]


@pytest.mark.parametrize(
    ("name", "getter", "expected_format"),
    [(n, get_sample_featurizer, f) for n, f in _SAMPLE_CONTRACTS]
    + [(n, get_drug_featurizer, f) for n, f in _DRUG_CONTRACTS],
)
def test_literature_featurizer_contracts(
    name: str, getter: Callable[[str], type], expected_format: FeatureFormat
) -> None:
    contract = featurizer_contract(getter(name))
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


#: Featurizers whose declared source views the synthetic fixture deliberately
#: does not carry, mapped to why. ``bionic`` reaches for an S3 artifact CI cannot
#: download, and ``sparsegoOntology`` declares no source views at all.
_NO_FIXTURE_SOURCE = {"bionic", "sparsegoOntology"}


@pytest.mark.parametrize(
    ("name", "getter"),
    [(name, get_sample_featurizer) for name, _ in _SAMPLE_CONTRACTS]
    + [(name, get_drug_featurizer) for name, _ in _DRUG_CONTRACTS],
)
def test_literature_featurizer_source_views_exist_in_the_fixture(
    name: str,
    getter: Callable[[str], type],
    synthetic_dataset: Dataset,
) -> None:
    """Views are resolved through the omics accessors, as the fixture stores omics under the published names."""
    if name in _NO_FIXTURE_SOURCE:
        pytest.skip(f"{name} declares no source views the fixture can supply")

    cls = getter(name)
    declared = tuple(cls.source_views or ()) or tuple(cls.input_views or ())
    if not declared:
        pytest.skip(f"{name} declares neither source_views nor input_views")

    missing = [
        view
        for view in declared
        if view != "canonical_smiles" and not synthetic_dataset._has_required_views((resolve_omics_accessor(view),))
    ]
    assert not missing, f"{name} reads {missing}, which the synthetic fixture does not provide"


@pytest.fixture
def sources() -> dict[str, tuple[StubSource, np.ndarray]]:
    nan_row = [np.nan] * 3
    matrices = {
        "valid": ["ABC", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        "mixed": ["ABCDE", [nan_row, [1, 2, 3], [4, 5, 6], [7, 8, 9], nan_row]],
        "all_nan": ["XYZ", [nan_row] * 3],
    }
    out = {}
    for key, (ids, rows) in matrices.items():
        ids = np.array(list(ids))
        out[key] = (StubSource(np.array(rows, dtype=np.float32), ids), ids)
    return out


@pytest.mark.parametrize("method", ["transform", "transform_blocks"])
@pytest.mark.parametrize("kind", ["valid", "mixed", "all_nan"])
def test_nan_rows_pass_through_as_nan_and_valid_rows_are_transformed(sources, method, kind) -> None:
    source, ids = sources[kind]
    feat = DoublingFeaturizer()
    feat.fit(source, entity_ids=ids)
    out = getattr(feat, method)(source, ids)
    values = out if method == "transform" else out["test_view"].values
    expected = source.get_view_matrix("test_view", ids) * 2
    assert values.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(values), np.isnan(expected))
    np.testing.assert_allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize(("threshold", "warns"), [(0.2, True), (0.5, False)])
def test_transform_warns_only_above_the_nan_threshold(sources, caplog, monkeypatch, threshold, warns) -> None:
    source, ids = sources["mixed"]
    monkeypatch.setattr(DoublingFeaturizer, "nan_threshold", threshold)
    feat = DoublingFeaturizer()
    feat.fit(source, entity_ids=ids)
    with caplog.at_level(logging.WARNING, logger=_NAN_LOGGER):
        feat.transform(source, ids)
    assert any("transform" in record.message for record in caplog.records) is warns


def test_default_hooks_are_no_ops() -> None:
    feat = DoublingFeaturizer()
    assert feat.get_state() == {}
    assert feat.set_state({"anything": 1}) is None
    assert DoublingFeaturizer.get_hyperparameter_space() == {}
    assert DoublingFeaturizer.get_default_hyperparameters() == {}


def test_default_empty_transform_features_transforms_zero_entities() -> None:
    source = StubSource(np.zeros((2, 3), dtype=np.float32), np.array(["A", "B"]))

    result = DoublingFeaturizer()._empty_transform_features(source)

    assert len(result.entity_ids) == 0
    assert result.matrix.shape == (0, 3)


def test_default_transform_concatenates_numeric_blocks_only() -> None:
    class _MixedBlocks(Featurizer):
        input_views = ("test_view",)

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            blocks = {
                "numeric": numeric_feature_block(np.ones((len(entity_ids), 2), dtype=np.float32)),
                "categories": metadata_feature_block(np.asarray(["a"], dtype=str)),
            }
            return FeaturizerResult.from_blocks(blocks, entity_ids)

        @property
        def output_dim(self) -> int:
            return 2

    _MixedBlocks.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    source = StubSource(np.zeros((2, 3), dtype=np.float32), np.array(["A", "B"]))

    matrix = _MixedBlocks().transform(source, np.array(["A", "B"]))

    assert matrix.shape == (2, 2)


def test_default_transform_returns_an_empty_matrix_without_numeric_blocks() -> None:
    class _NoNumericBlocks(Featurizer):
        entity_id_only = True

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            blocks = {"categories": metadata_feature_block(np.asarray(["a"], dtype=str))}
            return FeaturizerResult.from_blocks(blocks, entity_ids)

        @property
        def output_dim(self) -> int:
            return 0

    _NoNumericBlocks.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    matrix = _NoNumericBlocks().transform(StubSource(np.zeros((1, 1)), np.array(["A"])), np.array(["A"]))

    assert matrix.shape == (1, 0)


def test_transform_features_computes_matrix_and_blocks_once() -> None:
    class _CountingFeaturizer(DoublingFeaturizer):
        def __init__(self) -> None:
            self.calls = 0

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            self.calls += 1
            return super()._transform_features(source, entity_ids)

    source = StubSource(np.ones((2, 3), dtype=np.float32), np.array(["A", "B"]))
    featurizer = _CountingFeaturizer()
    result = featurizer.transform_features(source, np.array(["A", "B"]))

    assert result.matrix.shape == (2, 3)
    assert set(result.blocks) == {"test_view"}
    assert result.blocks["test_view"].values is result.matrix
    assert featurizer.calls == 1


def test_transform_features_pads_object_rows_and_preserves_metadata() -> None:
    class _ObjectFeaturizer(Featurizer):
        input_views = ("test_view",)

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            payload = np.empty(len(entity_ids), dtype=object)
            payload[:] = [np.asarray([entity_id]) for entity_id in entity_ids]
            blocks = {
                "payload": ragged_feature_block(payload),
                "labels": metadata_feature_block(np.asarray(["fixed"], dtype=str)),
            }
            return FeaturizerResult(entity_ids, blocks, ("payload",))

        def _empty_transform_features(self, source) -> FeaturizerResult:
            blocks = {"labels": metadata_feature_block(np.asarray(["fixed"]))}
            return FeaturizerResult(np.empty(0, dtype=str), blocks, ())

        @property
        def output_dim(self) -> int:
            return 0

    _ObjectFeaturizer.contract = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
    source = StubSource(np.asarray([[np.nan], [2.0]]), np.array(["bad", "good"]))

    result = _ObjectFeaturizer().transform_features(source, np.array(["bad", "good"]))

    assert result.matrix[0] is None
    assert result.matrix[1].tolist() == ["good"]
    assert result.blocks["payload"].values[0] is None
    assert result.blocks["labels"].values.tolist() == ["fixed"]


@pytest.mark.parametrize("stopping_ids", [None, np.array(["missing"])])
def test_required_featurizer_rejects_absent_or_filtered_holdout(monkeypatch, stopping_ids) -> None:
    source = StubSource(np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]), np.array(["train", "missing"]))
    monkeypatch.setattr(DoublingFeaturizer, "requires_early_stopping", True)
    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        DoublingFeaturizer().fit(source, entity_ids=np.array(["train"]), pair_expanded_es_ids=stopping_ids)
