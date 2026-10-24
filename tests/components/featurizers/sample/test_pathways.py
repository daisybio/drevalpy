"""Tests for the GSVA pathway sample featurizer; exactly one slow test runs ``gseapy.gsva``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample import pathways
from drevalpy.components.featurizers.sample.pathways import PathwaysSampleFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.conftest import MockFeatureSource


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
