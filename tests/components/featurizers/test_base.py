"""Tests for the ``Featurizer`` base class contract and view declarations.

Mirrors :mod:`drevalpy.components.featurizers.base`, which is where ``contract``,
``input_views``, ``source_views``, ``requires_view``, ``entity_id_only`` and
``resolve_input_views`` are defined. The registry sweeps below assert those
class-body declarations across every registered featurizer, so they belong to
this module rather than to any single concrete featurizer.

The NaN-tolerance cases at the bottom cover ``fit``/``transform``/
``transform_blocks`` and their ``_detect_valid`` / ``_expand_blocks_with_nan`` /
``_warn_if_above_threshold`` helpers, which are defined in the same module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat, featurizer_contract
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    list as list_cell_line_featurizers,
)
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)
from drevalpy.types.data.batch.feature_block import (
    FeatureBlock,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import resolve_omics_accessor
from tests.components.featurizers._helpers import DoublingFeaturizer, StubSource

_PROBE_VIEW = "gene_expression"


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _featurizer_names(registry: str) -> list[str]:
    register_builtin_components()
    return list_cell_line_featurizers() if registry == "cell_line" else list_drug_featurizers()


def test_featurizer_rejects_class_body_contract() -> None:
    with pytest.raises(TypeError, match="do not set contract on the class body"):

        class BadFeaturizer(Featurizer):  # noqa: B903
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_every_featurizer_declares_input_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
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


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_entity_id_only_featurizers_need_no_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    entity_id_only = [name for name in _featurizer_names(registry) if get(name).entity_id_only]
    assert entity_id_only
    for name in entity_id_only:
        assert get(name).resolve_input_views() == (), name


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_concat_featurizer_refuses_standalone_view_resolution(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    cls = get("concatFeaturizers")
    with pytest.raises(TypeError, match="has no input views of its own"):
        cls.resolve_input_views()


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_cell_line_featurizer(name)
    assert cls.requires_view
    assert cls.resolve_input_views(view="mutations") == ("mutations",)
    with pytest.raises(TypeError, match="requires an explicit view"):
        cls.resolve_input_views()


_CELL_LINE_CONTRACTS = [
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


@pytest.mark.parametrize(("name", "expected_format"), _CELL_LINE_CONTRACTS)
def test_cell_line_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_cell_line_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


@pytest.mark.parametrize(("name", "expected_format"), _DRUG_CONTRACTS)
def test_drug_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_drug_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


#: Featurizers whose declared source views the synthetic fixture deliberately
#: does not carry, mapped to why. ``bionic`` reaches for an S3 artifact CI cannot
#: download, and ``sparsegoOntology`` declares no source views at all.
_NO_FIXTURE_SOURCE = {"bionic", "sparsegoOntology"}


@pytest.mark.parametrize(
    ("name", "getter"),
    [(name, get_cell_line_featurizer) for name, _ in _CELL_LINE_CONTRACTS]
    + [(name, get_drug_featurizer) for name, _ in _DRUG_CONTRACTS],
)
def test_literature_featurizer_source_views_exist_in_the_fixture(
    name: str,
    getter: Callable[[str], type],
    synthetic_dataset: Dataset,
) -> None:
    """Every literature featurizer's raw inputs are present in the synthetic fixture.

    Views are resolved through :data:`OMICS_ACCESSORS` before being looked up,
    because the fixture stores omics under the accessor the published datasets
    use. That resolution is exactly what the library's own read sites still do
    not do, which is why ``molirOmics`` and ``superfeltrOmics`` work here yet
    their models are xfailed in the model tests: the data is present, the lookup
    is what asks for the wrong name.

    :param name: Featurizer registry name.
    :param getter: Registry lookup for the featurizer's side.
    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
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


# ----------------------------------------------------------------------
# NaN tolerance in fit / transform / transform_blocks
# ----------------------------------------------------------------------


@pytest.fixture
def mixed_source() -> tuple[StubSource, np.ndarray]:
    """Source with 5 entities where the first and last rows are all-NaN."""
    ids = np.array(["A", "B", "C", "D", "E"])
    matrix = np.array(
        [
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    return StubSource(matrix, ids), ids


@pytest.fixture
def all_nan_source() -> tuple[StubSource, np.ndarray]:
    """Source where every entity row is all-NaN."""
    ids = np.array(["X", "Y", "Z"])
    matrix = np.full((3, 3), np.nan, dtype=np.float32)
    return StubSource(matrix, ids), ids


@pytest.fixture
def all_valid_source() -> tuple[StubSource, np.ndarray]:
    """Source where every entity row is valid."""
    ids = np.array(["A", "B", "C"])
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    return StubSource(matrix, ids), ids


class TestTransformNaNTolerance:
    """Tests for the transform() NaN tolerance wrapper."""

    def test_all_valid_passes_through(self, all_valid_source):
        source, ids = all_valid_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_valid_invalid(self, mixed_source):
        source, ids = mixed_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        assert result.shape == (5, 3)
        assert np.all(np.isnan(result[0]))
        assert np.all(np.isnan(result[4]))
        np.testing.assert_array_almost_equal(result[1], [2, 4, 6])
        np.testing.assert_array_almost_equal(result[2], [8, 10, 12])
        np.testing.assert_array_almost_equal(result[3], [14, 16, 18])

    def test_all_nan_produces_nan_output(self, all_nan_source):
        source, ids = all_nan_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        assert result.shape[0] == 3
        assert np.all(np.isnan(result))


class TestTransformBlocksNaNTolerance:
    """Tests for the transform_blocks() NaN tolerance wrapper."""

    def test_all_valid_passes_through(self, all_valid_source):
        source, ids = all_valid_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        assert "test_view" in blocks
        expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.float32)
        np.testing.assert_array_almost_equal(blocks["test_view"].values, expected)

    def test_mixed_valid_invalid(self, mixed_source):
        source, ids = mixed_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        values = blocks["test_view"].values
        assert values.shape == (5, 3)
        assert np.all(np.isnan(values[0]))
        assert np.all(np.isnan(values[4]))
        np.testing.assert_array_almost_equal(values[1], [2, 4, 6])
        np.testing.assert_array_almost_equal(values[2], [8, 10, 12])
        np.testing.assert_array_almost_equal(values[3], [14, 16, 18])

    def test_all_nan_produces_nan_blocks(self, all_nan_source):
        source, ids = all_nan_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        values = blocks["test_view"].values
        assert values.shape[0] == 3
        assert np.all(np.isnan(values))


class TestNaNWarning:
    """Tests for the warning threshold logic."""

    def test_warning_above_threshold(self, mixed_source, caplog, monkeypatch):
        source, ids = mixed_source
        monkeypatch.setattr(DoublingFeaturizer, "nan_threshold", 0.2)
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        with caplog.at_level(logging.WARNING, logger="drevalpy.components.featurizers.base"):
            feat.transform(source, ids)
        assert any("invalid" in record.message.lower() for record in caplog.records)

    def test_no_warning_below_threshold(self, mixed_source, caplog, monkeypatch):
        source, ids = mixed_source
        monkeypatch.setattr(DoublingFeaturizer, "nan_threshold", 0.5)
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        with caplog.at_level(logging.WARNING, logger="drevalpy.components.featurizers.base"):
            feat.transform(source, ids)
        nan_warnings = [r for r in caplog.records if "invalid" in r.message.lower()]
        assert not nan_warnings

    def test_empty_mask_never_warns(self, caplog) -> None:
        feat = DoublingFeaturizer()
        with caplog.at_level(logging.WARNING, logger="drevalpy.components.featurizers.base"):
            feat._warn_if_above_threshold(np.array([], dtype=bool), "empty")
        assert not caplog.records


class TestConsistency:
    """Verify transform and transform_blocks produce consistent NaN handling."""

    def test_transform_and_blocks_agree(self, mixed_source):
        source, ids = mixed_source
        feat = DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        matrix = feat.transform(source, ids)
        blocks = feat.transform_blocks(source, ids)
        block_values = blocks["test_view"].values
        np.testing.assert_array_equal(
            np.isnan(matrix),
            np.isnan(block_values),
        )
        valid_mask = ~np.isnan(matrix).all(axis=1)
        np.testing.assert_array_almost_equal(matrix[valid_mask], block_values[valid_mask])


# ----------------------------------------------------------------------
# _detect_valid, _expand_blocks_with_nan and the default hooks
# ----------------------------------------------------------------------


def test_detect_valid_treats_entity_id_only_featurizers_as_all_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    feat = DoublingFeaturizer()
    monkeypatch.setattr(DoublingFeaturizer, "entity_id_only", True)

    mask = feat._detect_valid(StubSource(np.zeros((1, 3)), np.array(["A"])), np.array(["A"]))

    assert mask.tolist() == [True]


def test_detect_valid_treats_a_viewless_featurizer_as_all_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    feat = DoublingFeaturizer()
    monkeypatch.setattr(DoublingFeaturizer, "input_views", None)

    mask = feat._detect_valid(StubSource(np.zeros((1, 3)), np.array(["A"])), np.array(["A"]))

    assert mask.tolist() == [True]


def test_detect_valid_treats_an_unreadable_view_as_all_valid() -> None:
    feat = DoublingFeaturizer()
    source = StubSource(np.zeros((1, 3)), np.array(["A"]))

    mask = feat._detect_valid(source, np.array(["missing"]))

    assert mask.tolist() == [True]


def test_detect_valid_treats_non_numeric_views_as_all_valid() -> None:
    feat = DoublingFeaturizer()
    source = StubSource(np.array([["a", "b"]], dtype=str), np.array(["A"]))

    mask = feat._detect_valid(source, np.array(["A"]))

    assert mask.tolist() == [True]


def test_expand_blocks_passes_non_entity_aligned_blocks_through() -> None:
    feat = DoublingFeaturizer()
    block = metadata_feature_block(np.asarray(["lung", "skin"], dtype=str))

    expanded = feat._expand_blocks_with_nan({"categories": block}, np.array([True, False]), 2)

    assert expanded["categories"] is block


def test_expand_blocks_fills_ragged_payloads_with_none() -> None:
    feat = DoublingFeaturizer()
    payload = np.empty(1, dtype=object)
    payload[0] = np.ones((2, 3), dtype=np.float32)

    expanded = feat._expand_blocks_with_nan(
        {"ragged": ragged_feature_block(payload)},
        np.array([True, False]),
        2,
    )

    values = expanded["ragged"].values
    assert values.shape == (2,)
    assert values[1] is None


def test_default_state_hooks_are_no_ops() -> None:
    feat = DoublingFeaturizer()

    assert feat.get_state() == {}
    assert feat.set_state({"anything": 1}) is None


def test_default_hyperparameter_space_is_empty() -> None:
    assert DoublingFeaturizer.get_hyperparameter_space() == {}
    assert DoublingFeaturizer.get_default_hyperparameters() == {}


def test_default_transform_concatenates_numeric_blocks_only() -> None:
    class _MixedBlocks(Featurizer):
        input_views = ("test_view",)

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_blocks(self, source, entity_ids) -> dict[str, FeatureBlock]:
            return {
                "numeric": numeric_feature_block(np.ones((len(entity_ids), 2), dtype=np.float32)),
                "categories": metadata_feature_block(np.asarray(["a"], dtype=str)),
            }

        @property
        def output_dim(self) -> int:
            return 2

    _MixedBlocks.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    source = StubSource(np.zeros((2, 3), dtype=np.float32), np.array(["A", "B"]))

    matrix = _MixedBlocks()._transform(source, np.array(["A", "B"]))

    assert matrix.shape == (2, 2)


def test_default_transform_returns_an_empty_matrix_without_numeric_blocks() -> None:
    class _NoNumericBlocks(Featurizer):
        entity_id_only = True

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_blocks(self, source, entity_ids) -> dict[str, FeatureBlock]:
            return {"categories": metadata_feature_block(np.asarray(["a"], dtype=str))}

        @property
        def output_dim(self) -> int:
            return 0

    _NoNumericBlocks.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    matrix = _NoNumericBlocks()._transform(StubSource(np.zeros((1, 1)), np.array(["A"])), np.array(["A"]))

    assert matrix.shape == (1, 0)


def test_output_block_specs_for_config_falls_back_to_the_declared_input_view() -> None:
    class _Config:
        view = None

    specs = DoublingFeaturizer.output_block_specs_for_config(_Config())

    assert [spec.name for spec in specs] == ["test_view"]


def test_output_block_specs_for_config_honours_an_explicit_view() -> None:
    class _Config:
        view = "mutations"

    specs = DoublingFeaturizer.output_block_specs_for_config(_Config())

    assert [spec.name for spec in specs] == ["mutations"]


def test_output_block_specs_for_config_is_empty_without_any_view() -> None:
    class _NoViews(Featurizer):
        entity_id_only = True

        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_blocks(self, source, entity_ids) -> dict[str, FeatureBlock]:
            return {}

        @property
        def output_dim(self) -> int:
            return 0

    _NoViews.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    assert _NoViews.output_block_specs_for_config(None) == ()


def test_resolve_input_views_rejects_a_featurizer_declaring_nothing() -> None:
    class _Undeclared(Featurizer):
        def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
            return self

        def _transform_blocks(self, source, entity_ids) -> dict[str, FeatureBlock]:
            return {}

        @property
        def output_dim(self) -> int:
            return 0

    with pytest.raises(TypeError, match="declare input_views on the class body"):
        _Undeclared.resolve_input_views()


# ----------------------------------------------------------------------
# store / fetch / list_stored_variants against a real MuData
# ----------------------------------------------------------------------


class _StoredCellLine(DoublingFeaturizer):
    """Cell-line-side featurizer with a storage key, for variant round-trips."""

    storage_key = "stored_cell_line"
    side = "cell_line"


class _StoredDrug(DoublingFeaturizer):
    """Drug-side featurizer with a storage key, for variant round-trips."""

    storage_key = "stored_drug"
    side = "drug"


def test_fetch_returns_none_when_no_variant_matches() -> None:
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()

    assert _StoredCellLine().fetch(dataset.mdata, dataset.cell_line_ids) is None


def test_store_then_fetch_round_trips_a_cell_line_variant() -> None:
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    featurizer = _StoredCellLine()
    payload = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    featurizer.store(dataset.mdata, dataset.cell_line_ids, payload, {"scale": 2})

    assert list(_StoredCellLine.list_stored_variants(dataset.mdata)) == ["stored_cell_line_0"]
    np.testing.assert_allclose(
        featurizer.fetch(dataset.mdata, dataset.cell_line_ids, {"scale": 2}),
        payload,
    )


def test_store_then_fetch_round_trips_a_drug_variant() -> None:
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    featurizer = _StoredDrug()
    payload = np.array([[5.0], [6.0]], dtype=np.float32)

    featurizer.store(dataset.mdata, dataset.drug_ids, payload)

    np.testing.assert_allclose(featurizer.fetch(dataset.mdata, dataset.drug_ids), payload)


def test_store_allocates_a_fresh_index_per_hyperparameter_setting() -> None:
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    featurizer = _StoredCellLine()
    payload = np.zeros((2, 2), dtype=np.float32)

    featurizer.store(dataset.mdata, dataset.cell_line_ids, payload, {"scale": 2})
    featurizer.store(dataset.mdata, dataset.cell_line_ids, payload, {"scale": 4})

    assert list(_StoredCellLine.list_stored_variants(dataset.mdata)) == [
        "stored_cell_line_0",
        "stored_cell_line_1",
    ]


def test_fetch_prefers_a_modality_over_obsm() -> None:
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()

    class _ModalityBacked(DoublingFeaturizer):
        storage_key = "gene_expression"
        side = "cell_line"

    from drevalpy.components.featurizers.storage import register_variant

    register_variant(dataset.mdata, "gene_expression", "gene_expression", None, side="cell_line")

    matrix = _ModalityBacked().fetch(dataset.mdata, dataset.cell_line_ids)

    assert matrix is not None
    assert matrix.shape == (2, 3)
