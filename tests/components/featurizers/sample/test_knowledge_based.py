"""Tests for the knowledge-based sample featurizers: BIONIC PPI, pathways and SparseGO ontology."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drevalpy.components.featurizers.sample import pathways
from drevalpy.components.featurizers.sample._sparsego_metadata import read_sparsego_ontology_metadata
from drevalpy.components.featurizers.sample.bionic import (
    BionicSampleFeaturizer,
    _aggregate_ppi_for_sample,
)
from drevalpy.components.featurizers.sample.pathways import PathwaysSampleFeaturizer
from drevalpy.components.featurizers.sample.sparsego_ontology import SparseGOOntologyFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

_GENE_NAMES = ("g_high", "g_mid", "g_low")
_EXPR_ROW = np.array([3.0, 2.0, 1.0])
_PPI_LOOKUP = {
    "g_high": np.array([1.0, 5.0], dtype=np.float32),
    "g_mid": np.array([3.0, 1.0], dtype=np.float32),
    "g_low": np.array([100.0, 100.0], dtype=np.float32),
}


def test_hyperparameter_space_and_default_gene_add_num() -> None:
    assert set(BionicSampleFeaturizer.get_hyperparameter_space()) == {"gene_add_num", "aggregation"}
    assert BionicSampleFeaturizer()._gene_add_num == 256


def _aggregate(expr, names, eligible, lookup=_PPI_LOOKUP, *, gene_add_num=2, embed_dim=2, aggregation="mean"):
    return _aggregate_ppi_for_sample(
        expr,
        names,
        eligible,
        lookup,
        gene_add_num=gene_add_num,
        embed_dim=embed_dim,
        aggregation=aggregation,
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("mean", [2.0, 3.0]), ("max", [3.0, 5.0]), ("sum", [4.0, 6.0]), ("unrecognised", [2.0, 3.0])],
)
def test_aggregate_selects_the_top_expressed_eligible_genes(aggregation: str, expected: list[float]) -> None:
    vector = _aggregate(_EXPR_ROW, _GENE_NAMES, {"g_high", "g_mid"}, aggregation=aggregation)

    np.testing.assert_allclose(vector, expected)
    assert vector.dtype == np.float32


def test_aggregate_budget_ties_and_empty_eligibility() -> None:
    np.testing.assert_allclose(_aggregate(_EXPR_ROW, _GENE_NAMES, set(_GENE_NAMES), gene_add_num=1), [1.0, 5.0])
    np.testing.assert_allclose(_aggregate(_EXPR_ROW, _GENE_NAMES, set()), [0.0, 0.0])
    lookup = {name: np.array([float(i)], dtype=np.float32) for i, name in enumerate("abcd")}
    tie = _aggregate(np.zeros(4), tuple("abcd"), set("abcd"), lookup, embed_dim=1)
    np.testing.assert_allclose(tie, [0.5])


def test_bionic_reads_the_bionic_features_view_when_present() -> None:
    source = MockFeatureSource(
        features={"cl1": {"bionic_features": np.array([0.1, 0.2])}, "cl2": {"bionic_features": np.array([0.3, 0.4])}}
    )
    ids = np.array(["cl1", "cl2"], dtype=str)

    featurizer = BionicSampleFeaturizer().fit(source, entity_ids=ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(featurizer.transform(source, ids), [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)


def test_bionic_compute_from_source_aggregates_per_sample_and_needs_feature_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drevalpy.components.featurizers.sample.bionic as bionic_module

    ppi = (np.array([[1.0, 5.0], [3.0, 1.0]], dtype=np.float32), ["g_high", "g_mid"], {"g_high", "g_mid"})
    monkeypatch.setattr(bionic_module, "_load_ppi_data", lambda: ppi)
    named = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([3.0, 2.0])}, "cl2": {"gene_expression": np.array([1.0, 4.0])}},
        meta_info={"gene_expression": ["g_high", "g_mid"]},
    )

    matrix = BionicSampleFeaturizer()._compute_from_source(named, np.array(["cl1", "cl2"], dtype=str))

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], [2.0, 3.0])
    nameless = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})
    with pytest.raises(ValueError, match="must provide feature names"):
        BionicSampleFeaturizer()._compute_from_source(nameless, np.array(["cl1"], dtype=str))


@pytest.mark.network
def test_load_ppi_data_returns_features_gene_names_and_selection() -> None:
    from drevalpy.components.featurizers.sample.bionic import _load_ppi_data

    ppi_features, ppi_gene_names, gene_list_sel = _load_ppi_data()

    assert ppi_features.ndim == 2
    assert len(ppi_gene_names) == ppi_features.shape[0]
    assert gene_list_sel


def test_unfitted_pathways_featurizer() -> None:
    featurizer = PathwaysSampleFeaturizer()
    assert featurizer.output_dim == 0
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(MockFeatureSource(features={}), np.array(["cl1"]))


def test_run_gsva_downloads_the_gene_sets_when_the_dataset_has_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gseapy

    cached = tmp_path / "c2.gmt"
    cached.write_text("SET_A\tcached\tG1\tG2\n", encoding="utf-8")
    monkeypatch.setattr(pathways, "get_precily_gmt", lambda: cached)
    seen: list[str] = []

    def failing_gsva(*, gene_sets: str, **_kwargs: object) -> None:
        seen.append(Path(gene_sets).read_text(encoding="utf-8"))
        raise RuntimeError("stop after reading the gene sets")

    monkeypatch.setattr(gseapy, "gsva", failing_gsva)
    source = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})
    source.get_feature_names = lambda view: ["G1", "G2"]  # type: ignore[method-assign]
    featurizer = PathwaysSampleFeaturizer()

    with pytest.raises(RuntimeError, match="stop after reading"):
        featurizer._run_gsva(source, np.array(["cl1"]))

    assert seen == ["SET_A\tcached\tG1\tG2\n"]
    assert featurizer.get_state()["gmt_text"] == seen[0]


def test_run_gsva_prefers_the_gene_sets_stored_in_the_dataset(
    synthetic_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gseapy

    def no_download() -> Path:
        raise AssertionError("the dataset carries pathways_gmt")

    def failing_gsva(**_kwargs: object) -> None:
        raise RuntimeError("stop")

    monkeypatch.setattr(pathways, "get_precily_gmt", no_download)
    monkeypatch.setattr(gseapy, "gsva", failing_gsva)
    featurizer = PathwaysSampleFeaturizer()
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)

    with pytest.raises(RuntimeError, match="stop"):
        featurizer._run_gsva(source, synthetic_dataset.sample_ids[:3])

    assert featurizer.get_state()["gmt_text"] == str(synthetic_dataset.mdata.uns["pathways_gmt"])


def test_run_gsva_requires_gene_expression_feature_names(synthetic_dataset: Dataset) -> None:
    class _NamelessSource(SampleFeatureSource):
        def get_feature_names(self, view: str) -> None:
            return None

    featurizer = PathwaysSampleFeaturizer()
    source = _NamelessSource(synthetic_dataset, synthetic_dataset.sample_ids)

    with pytest.raises(ValueError, match="must provide feature names"):
        featurizer._run_gsva(source, synthetic_dataset.sample_ids[:2])


def test_run_gsva_removes_its_temporary_gmt_file_when_gsva_fails(
    synthetic_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gseapy

    seen: list[Path] = []

    def failing_gsva(*, gene_sets: str, **_kwargs: object) -> None:
        seen.append(Path(gene_sets))
        assert seen[0].read_text(encoding="utf-8") == str(synthetic_dataset.mdata.uns["pathways_gmt"])
        raise RuntimeError("gsva exploded")

    monkeypatch.setattr(gseapy, "gsva", failing_gsva)
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)

    with pytest.raises(RuntimeError, match="gsva exploded"):
        PathwaysSampleFeaturizer()._run_gsva(source, synthetic_dataset.sample_ids[:3])

    assert len(seen) == 1
    assert not seen[0].exists()


class TestGSVAIntegration:
    pytestmark = pytest.mark.slow

    def test_fit_then_transform_returns_pathway_scores(self, synthetic_dataset: Dataset, monkeypatch) -> None:
        source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
        sample_ids = synthetic_dataset.sample_ids[:6]
        featurizer = PathwaysSampleFeaturizer()

        featurizer.fit(source, entity_ids=sample_ids)
        matrix = featurizer.transform(source, sample_ids)

        assert featurizer.output_dim > 0
        assert matrix.shape == (len(sample_ids), featurizer.output_dim)
        assert matrix.dtype == np.float32
        names = featurizer.transform_blocks(source, sample_ids)["pathways"].feature_names
        assert names is not None and len(names) == featurizer.output_dim
        requested = synthetic_dataset.sample_ids[:7]
        expected = featurizer.transform(source, requested)
        restored = PathwaysSampleFeaturizer()
        restored.set_state(featurizer.get_state())
        monkeypatch.setitem(synthetic_dataset.mdata.uns, "pathways_gmt", "invalid replacement")
        np.testing.assert_allclose(restored.transform(source, requested), expected)
        assert restored.transform_blocks(source, requested)["pathways"].feature_names == names

        restored._pathway_names = ("absent_from_fitted_gene_sets", *names[1:])
        with pytest.raises(ValueError, match="missing pathways"):
            restored.transform(source, requested)

    def test_fit_without_dataset_gene_sets_uses_the_cached_download(
        self, synthetic_dataset: Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        gmt = tmp_path / "c2.gmt"
        gmt.write_text(str(synthetic_dataset.mdata.uns["pathways_gmt"]), encoding="utf-8")
        monkeypatch.delitem(synthetic_dataset.mdata.uns, "pathways_gmt")
        monkeypatch.setattr(pathways, "get_precily_gmt", lambda: gmt)
        source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
        sample_ids = synthetic_dataset.sample_ids[:6]

        featurizer = PathwaysSampleFeaturizer().fit(source, entity_ids=sample_ids)

        assert featurizer.output_dim == 2
        assert featurizer.transform(source, sample_ids).shape == (6, 2)


def test_transform_of_unseen_samples_recomputes_gsva(
    synthetic_dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
    fit_ids = synthetic_dataset.sample_ids[:2]
    featurizer = PathwaysSampleFeaturizer()
    featurizer._fit_scores = np.zeros((2, 3), dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 3

    calls: list[int] = []

    def _fake_run_gsva(_source, entity_ids):
        calls.append(len(entity_ids))
        return np.ones((len(entity_ids), 3), dtype=np.float32)

    monkeypatch.setattr(featurizer, "_run_gsva", _fake_run_gsva)

    matrix = featurizer._transform_features(source, np.array([*fit_ids, synthetic_dataset.sample_ids[5]])).matrix

    assert calls == [3]
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix[:2], featurizer._fit_scores)
    np.testing.assert_array_equal(matrix[2], np.ones(3))


def test_transform_of_known_samples_reuses_fitted_scores_and_emits_one_block(synthetic_dataset: Dataset) -> None:
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
    fit_ids = synthetic_dataset.sample_ids[:2]
    featurizer = PathwaysSampleFeaturizer()
    featurizer._fit_scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 2

    result = featurizer._transform_features(source, np.array([fit_ids[1], fit_ids[0]]))

    np.testing.assert_allclose(result.matrix, [[3.0, 4.0], [1.0, 2.0]])
    assert set(result.blocks) == {"pathways"}
    assert result.blocks["pathways"].feature_names is None


@pytest.mark.parametrize("state", [{}, {"fit_scores": np.zeros((2, 3)), "fit_ids": np.array(["a"]), "output_dim": 3}])
def test_pathways_rejects_incomplete_checkpoint(state) -> None:
    with pytest.raises(ValueError, match="Pathways checkpoint"):
        PathwaysSampleFeaturizer().set_state(state)


def _source() -> MockFeatureSource:
    return MockFeatureSource(
        {"cl1": {"gene_expression": np.array([1.0, 2.0]), "mutations": np.array([0.0, 1.0])}},
        meta_info={"gene_expression": ["a", "b"], "mutations": ["a", "b"]},
    )


def _with_ontology() -> MockFeatureSource:
    features = _source()
    features._meta_info["sparsego_ontology"] = {
        "layer_connections": [np.array([["term", "a"]])],
        "gene2id_mapping_ont": {"a": 0, "b": 1},
        "ontology_gene_order": ("a", "b"),
        "gene_dim_input": 2,
    }
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


def test_sparsego_rejects_unknown_input_type_and_unfitted_use() -> None:
    with pytest.raises(ValueError, match="input_type must be"):
        SparseGOOntologyFeaturizer(input_type="methylation")
    with pytest.raises(ValueError, match="input_type must be"):
        SparseGOOntologyFeaturizer().set_state({"input_type": "methylation"})
    with pytest.raises(ValueError, match="ontology metadata is missing"):
        SparseGOOntologyFeaturizer().fit(_source())
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        SparseGOOntologyFeaturizer().transform(_source(), np.array(["cl1"]))
    assert SparseGOOntologyFeaturizer().get_state() == {}
    assert SparseGOOntologyFeaturizer().output_dim == 0


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        ({"input_type": {"default": "mutations"}}, "mutations"),
        ({"other": {"default": 1}}, "gene_expression"),
        (None, "gene_expression"),
    ],
)
def test_sparsego_output_block_specs_follow_the_configured_input_type(space, expected: str) -> None:
    config = None if space is None else SimpleNamespace(hyperparameter_space=space)

    specs = SparseGOOntologyFeaturizer.output_block_specs_for_config(config)

    assert [spec.name for spec in specs] == [expected]
    assert SparseGOOntologyFeaturizer.resolve_input_views(input_type="expression") == ("gene_expression",)
    assert SparseGOOntologyFeaturizer.resolve_input_views(input_type="mutations") == ("mutations",)


def test_sparsego_prefers_a_precomputed_variant() -> None:
    from tests.components.featurizers.sample._helpers import PRECOMPUTED, precomputed_source

    source = precomputed_source(SparseGOOntologyFeaturizer)
    source._dataset.mdata.uns["sparsego_ontology"] = {
        "layer_connections": [np.array([["term", "gene0"]])],
        "gene2id_mapping_ont": {"gene0": 0, "gene1": 1},
        "ontology_gene_order": ("gene0", "gene1"),
        "gene_dim_input": 2,
    }
    featurizer = SparseGOOntologyFeaturizer().fit(source)

    np.testing.assert_allclose(featurizer.transform(source, source.identifiers), PRECOMPUTED)


def test_ontology_order_controls_values_and_names_and_missing_genes_fail() -> None:
    source = _with_ontology()
    source._meta_info["gene_expression"] = ["b", "a"]
    result = SparseGOOntologyFeaturizer().fit(source).transform_features(source, np.array(["cl1"]))
    np.testing.assert_array_equal(result.matrix, [[2.0, 1.0]])
    assert result.blocks["gene_expression"].feature_names == ("a", "b")

    source._meta_info["gene_expression"] = ["a", "unrelated"]
    with pytest.raises(ValueError, match="missing ontology genes"):
        SparseGOOntologyFeaturizer().fit(source).transform_features(source, np.array(["cl1"]))


def _metadata() -> dict[str, Any]:
    return {
        "layer_connections": [np.array([["term", "a"]])],
        "gene2id_mapping_ont": {"a": 0, "b": 1},
        "ontology_gene_order": ("a", "b"),
        "gene_dim_input": 2,
    }


def test_read_resolves_metadata_from_the_source() -> None:
    source = MockFeatureSource(
        {"cl1": {"gene_expression": np.array([1.0, 2.0])}}, meta_info={"sparsego_ontology": _metadata()}
    )

    restored = read_sparsego_ontology_metadata(source)

    assert restored is not None
    assert restored["gene_dim_input"] == 2
    assert restored["gene2id_mapping_ont"] == {"a": 0, "b": 1}
    assert restored["ontology_gene_order"] == ("a", "b")
    np.testing.assert_array_equal(restored["layer_connections"][0], np.array([["term", "a"]]))


def test_read_returns_none_without_usable_metadata() -> None:
    assert read_sparsego_ontology_metadata(MockFeatureSource({"cl1": {"gene_expression": np.array([1.0])}})) is None
    assert (
        read_sparsego_ontology_metadata(MockFeatureSource({}, meta_info={"sparsego_ontology": {"gene_dim_input": 2}}))
        is None
    )
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    assert read_sparsego_ontology_metadata(SampleFeatureSource(dataset, dataset.sample_ids)) is None


def test_read_resolves_metadata_stored_in_mdata_uns() -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.uns["sparsego_ontology"] = _metadata()

    restored = read_sparsego_ontology_metadata(SampleFeatureSource(dataset, dataset.sample_ids))

    assert restored is not None
    assert restored["gene_dim_input"] == 2
