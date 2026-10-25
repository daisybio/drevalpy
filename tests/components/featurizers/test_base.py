"""Tests for the ``Featurizer`` base contract: registry sweeps, declarations and NaN tolerance."""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers._declarations import FeaturizerDeclarationsMixin
from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers._nan_tolerance import NanToleranceMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.drug.base import DenseViewDrugFeaturizer, DrugFeaturizer
from drevalpy.components.featurizers.precompute import _has_source_data
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer, SampleFeaturizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import list as list_sample_featurizers
from drevalpy.types.data.batch.feature_block import (
    FeaturizerResult,
    global_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.dataset import Dataset
from tests.components.featurizers._helpers import DoublingFeaturizer, StatelessFeaturizer, StubSource

_PROBE_VIEW = "gene_expression"

#: ``base.py`` delegates the NaN policy, and so its warning, to this mixin module.
_NAN_LOGGER = "drevalpy.components.featurizers._nan_tolerance"

_REGISTRIES = {
    "sample": (get_sample_featurizer, list_sample_featurizers),
    "drug": (get_drug_featurizer, list_drug_featurizers),
}
_SIDE_BASES = {
    "sample": (SampleFeaturizer, DenseViewSampleFeaturizer),
    "drug": (DrugFeaturizer, DenseViewDrugFeaturizer),
}


def _registered(registry: str) -> dict[str, type]:
    get, list_names = _REGISTRIES[registry]
    names = list_names()
    assert names
    return {name: get(name) for name in names}


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_every_featurizer_declares_input_views(registry: str) -> None:
    for name, cls in _registered(registry).items():
        if issubclass(cls, ConcatFeaturizersMixin):
            continue
        views = cls.resolve_input_views(**({"view": _PROBE_VIEW} if cls.requires_view else {}))
        assert isinstance(views, tuple), name
        assert all(isinstance(view, str) and view.strip() for view in views), name


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_entity_id_only_featurizers_need_no_views(registry: str) -> None:
    entity_id_only = {name: cls for name, cls in _registered(registry).items() if cls.entity_id_only}
    assert entity_id_only
    for name, cls in entity_id_only.items():
        assert cls.resolve_input_views() == (), name


@pytest.mark.parametrize("registry", ["sample", "drug"])
def test_concat_featurizer_refuses_standalone_view_resolution(registry: str) -> None:
    with pytest.raises(TypeError, match="has no input views of its own"):
        _REGISTRIES[registry][0]("concatFeaturizers").resolve_input_views()


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_sample_featurizer(name)
    assert cls.requires_view
    assert cls.resolve_input_views(view="mutations") == ("mutations",)
    with pytest.raises(TypeError, match="requires an explicit view"):
        cls.resolve_input_views()


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_side_bases_are_abstract_and_own_every_registered_featurizer(side: str) -> None:
    base, dense_view = _SIDE_BASES[side]
    assert issubclass(base, Featurizer)
    with pytest.raises(TypeError, match="abstract"):
        base()
    assert all(issubclass(cls, base) for cls in _registered(side).values())
    mro = dense_view.__mro__
    assert issubclass(dense_view, DenseViewFeaturizer)
    assert mro.index(base) < mro.index(Featurizer)


_SAMPLE_FORMATS = [
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

_DRUG_FORMATS = [
    ("molgnet", FeatureFormat.RAGGED_SEQUENCE),
    ("bpePharmaformer", FeatureFormat.NUMERIC_MATRIX),
    ("smilesvec", FeatureFormat.NUMERIC_MATRIX),
    ("drugGraph", FeatureFormat.GRAPH),
]

_FORMAT_CASES = [(n, get_sample_featurizer, f) for n, f in _SAMPLE_FORMATS] + [
    (n, get_drug_featurizer, f) for n, f in _DRUG_FORMATS
]


@pytest.mark.parametrize(("name", "getter", "expected_format"), _FORMAT_CASES)
def test_literature_featurizer_output_formats(
    name: str, getter: Callable[[str], type], expected_format: FeatureFormat
) -> None:
    assert getter(name).output_format == expected_format


#: Featurizers whose declared source views the synthetic fixture deliberately
#: does not carry. ``bionic`` reaches for an S3 artifact CI cannot download, and
#: ``sparsegoOntology`` declares no source views at all.
_NO_FIXTURE_SOURCE = {"bionic", "sparsegoOntology"}


@pytest.mark.parametrize(("name", "getter"), [case[:2] for case in _FORMAT_CASES])
def test_literature_featurizer_source_views_exist_in_the_fixture(
    name: str, getter: Callable[[str], type], synthetic_dataset: Dataset
) -> None:
    """Views are resolved through their stored names, as the fixture stores omics under the published names."""
    if name in _NO_FIXTURE_SOURCE:
        pytest.skip(f"{name} declares no source views the fixture can supply")

    cls = getter(name)
    declared = tuple(cls.source_views or ()) or tuple(cls.input_views or ())
    if not declared:
        pytest.skip(f"{name} declares neither source_views nor input_views")

    missing = [
        view
        for view in declared
        if view != "canonical_smiles" and not _has_source_data(synthetic_dataset, (view,), cls.side)
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
    class _MixedBlocks(StatelessFeaturizer):
        input_views = ("test_view",)

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            blocks = {
                "numeric": numeric_feature_block(np.ones((len(entity_ids), 2), dtype=np.float32)),
                "categories": global_feature_block(np.asarray(["a"], dtype=str)),
            }
            return FeaturizerResult.from_blocks(blocks, entity_ids)

        @property
        def output_dim(self) -> int:
            return 2

    _MixedBlocks.output_format = FeatureFormat.NUMERIC_MATRIX
    source = StubSource(np.zeros((2, 3), dtype=np.float32), np.array(["A", "B"]))

    matrix = _MixedBlocks().transform(source, np.array(["A", "B"]))

    assert matrix.shape == (2, 2)


def test_default_transform_returns_an_empty_matrix_without_numeric_blocks() -> None:
    class _NoNumericBlocks(StatelessFeaturizer):
        entity_id_only = True

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            blocks = {"categories": global_feature_block(np.asarray(["a"], dtype=str))}
            return FeaturizerResult.from_blocks(blocks, entity_ids)

        @property
        def output_dim(self) -> int:
            return 0

    _NoNumericBlocks.output_format = FeatureFormat.NUMERIC_MATRIX

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
    class _ObjectFeaturizer(StatelessFeaturizer):
        input_views = ("test_view",)

        def _transform_features(self, source, entity_ids) -> FeaturizerResult:
            payload = np.empty(len(entity_ids), dtype=object)
            payload[:] = [np.asarray([entity_id]) for entity_id in entity_ids]
            blocks = {
                "payload": ragged_feature_block(payload),
                "labels": global_feature_block(np.asarray(["fixed"], dtype=str)),
            }
            return FeaturizerResult(entity_ids, blocks, ("payload",))

        def _empty_transform_features(self, source) -> FeaturizerResult:
            blocks = {"labels": global_feature_block(np.asarray(["fixed"]))}
            return FeaturizerResult(np.empty(0, dtype=str), blocks, ())

        @property
        def output_dim(self) -> int:
            return 0

    _ObjectFeaturizer.output_format = FeatureFormat.RAGGED_SEQUENCE
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


class _Declared(StatelessFeaturizer):
    entity_id_only = True

    def _transform_features(self, source, entity_ids) -> FeaturizerResult:
        return FeaturizerResult.from_blocks({}, len(entity_ids))

    @property
    def output_dim(self) -> int:
        return 0


_Declared.output_format = FeatureFormat.NUMERIC_MATRIX


@pytest.mark.parametrize("mixin", [FeaturizerDeclarationsMixin, NanToleranceMixin])
def test_the_mixins_are_part_of_the_featurizer_base(mixin: type) -> None:
    assert issubclass(DoublingFeaturizer, mixin)


def test_output_format_declarations_are_checked_at_class_creation() -> None:
    class _Declares(Featurizer):
        output_format = FeatureFormat.GRAPH

    class _Inheriting(DoublingFeaturizer):
        pass

    assert _Declares.output_format == FeatureFormat.GRAPH
    assert _Inheriting.output_format is DoublingFeaturizer.output_format
    with pytest.raises(TypeError, match="_Bad.output_format must be a FeatureFormat"):

        class _Bad(Featurizer):
            output_format = "graph_but_a_plain_string"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({"view": "mutations"}, ("mutations",)), ({"view": "   "}, ("test_view",)), ({}, ("test_view",))],
)
def test_resolve_input_views(kwargs, expected) -> None:
    assert DoublingFeaturizer.resolve_input_views(**kwargs) == expected
    assert _Declared.resolve_input_views() == ()


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"entity_id_only": False, "requires_view": True}, "requires an explicit view"),
        ({"entity_id_only": False}, "declare input_views on the class body"),
    ],
)
def test_resolve_input_views_rejects_undeclared_views(attrs, match) -> None:
    cls = type("_Bad", (_Declared,), attrs)
    with pytest.raises(TypeError, match=match):
        cls.resolve_input_views()


@pytest.mark.parametrize(("config_view", "expected"), [(None, "test_view"), ("mutations", "mutations")])
def test_output_block_specs_follow_the_config_view_and_contract(config_view, expected) -> None:
    specs = DoublingFeaturizer.output_block_specs_for_config(SimpleNamespace(view=config_view))
    assert [spec.name for spec in specs] == [expected]
    assert [spec.format for spec in specs] == [FeatureFormat.NUMERIC_MATRIX]
    assert _Declared.output_block_specs_for_config(None) == ()


def _valid(matrix, ids, probe_ids=None):
    source = StubSource(np.asarray(matrix, dtype=np.float32), np.array(ids))
    return DoublingFeaturizer()._detect_valid(source, np.array(probe_ids or ids)).tolist()


def test_all_nan_rows_are_invalid() -> None:
    assert _valid([[np.nan] * 2, [1.0, 2.0], [np.nan] * 2], ["A", "B", "C"]) == [False, True, False]


@pytest.mark.parametrize("attr", ["entity_id_only", "input_views"])
def test_featurizers_without_a_probeable_view_are_all_valid(monkeypatch: pytest.MonkeyPatch, attr: str) -> None:
    monkeypatch.setattr(DoublingFeaturizer, attr, True if attr == "entity_id_only" else None)
    assert _valid([[np.nan] * 3], ["A"]) == [True]


def test_an_unreadable_view_is_all_valid() -> None:
    assert _valid(np.zeros((1, 3)), ["A"], ["missing"]) == [True]


def test_a_non_numeric_view_is_all_valid() -> None:
    source = StubSource(np.array([["a", "b"]], dtype=str), np.array(["A"]))
    assert DoublingFeaturizer()._detect_valid(source, np.array(["A"])).tolist() == [True]


def test_expand_blocks_pads_numeric_with_nan_and_keeps_feature_names() -> None:
    block = numeric_feature_block(np.array([[1.0, 2.0]], dtype=np.float32), feature_names=("g1", "g2"))
    expanded = DoublingFeaturizer()._expand_blocks_with_nan({"numeric": block}, np.array([True, False]))
    values = expanded["numeric"].values
    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[0], [1.0, 2.0])
    assert np.all(np.isnan(values[1]))
    assert expanded["numeric"].feature_names == ("g1", "g2")


def test_expand_blocks_leaves_metadata_untouched_and_pads_ragged_with_none() -> None:
    meta = global_feature_block(np.asarray(["lung", "skin"], dtype=str))
    payload = np.empty(1, dtype=object)
    payload[0] = np.ones((2, 3), dtype=np.float32)
    blocks = {"categories": meta, "ragged": ragged_feature_block(payload)}
    expanded = DoublingFeaturizer()._expand_blocks_with_nan(blocks, np.array([True, False]))
    assert expanded["categories"] is meta
    assert expanded["ragged"].values.shape == (2,)
    assert expanded["ragged"].values[1] is None


@pytest.mark.parametrize(
    ("mask", "warns"),
    [([True, False, False, False], True), ([True] * 4, False), ([], False)],
)
def test_warn_if_above_threshold(caplog: pytest.LogCaptureFixture, mask: list[bool], warns: bool) -> None:
    with caplog.at_level(logging.WARNING, logger=_NAN_LOGGER):
        DoublingFeaturizer()._warn_if_above_threshold(np.array(mask, dtype=bool), "probe")
    assert any("invalid" in record.message.lower() for record in caplog.records) is warns
