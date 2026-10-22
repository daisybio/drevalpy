"""Tests for the GSVA pathway sample featurizer.

Mirrors :mod:`drevalpy.components.featurizers.sample.pathways`. Not network
gated: it needs an ``mdata``-backed source plus ``uns["pathways_gmt"]``, both of
which the synthetic fixture carries. ``gseapy.gsva`` is the slow part, so exactly
one test runs it and the rest cover the guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.pathways import PathwaysSampleFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.conftest import MockFeatureSource


def test_output_dim_is_zero_before_fit() -> None:
    assert PathwaysSampleFeaturizer().output_dim == 0


def test_transform_before_fit_raises() -> None:
    featurizer = PathwaysSampleFeaturizer()

    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(MockFeatureSource(features={}), np.array(["cl1"]))


def test_run_gsva_requires_pathways_gmt_in_uns() -> None:
    featurizer = PathwaysSampleFeaturizer()
    source = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})

    with pytest.raises(ValueError, match="require pathways_gmt"):
        featurizer._run_gsva(source, np.array(["cl1"]))


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


def test_transform_of_known_samples_reuses_fitted_scores(synthetic_dataset: Dataset) -> None:
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
    fit_ids = synthetic_dataset.sample_ids[:2]
    featurizer = PathwaysSampleFeaturizer()
    featurizer._fit_scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 2

    matrix = featurizer._transform_features(source, np.array([fit_ids[1], fit_ids[0]])).matrix

    np.testing.assert_allclose(matrix, [[3.0, 4.0], [1.0, 2.0]])


def test_transform_blocks_emit_one_block_named_after_the_view(synthetic_dataset: Dataset) -> None:
    source = SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids)
    fit_ids = synthetic_dataset.sample_ids[:2]
    featurizer = PathwaysSampleFeaturizer()
    featurizer._fit_scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 2

    blocks = featurizer._transform_features(source, fit_ids).blocks

    assert set(blocks) == {"pathways"}
    assert blocks["pathways"].feature_names is None


@pytest.mark.parametrize("state", [{}, {"fit_scores": np.zeros((2, 3)), "fit_ids": np.array(["a"]), "output_dim": 3}])
def test_pathways_rejects_incomplete_checkpoint(state) -> None:
    with pytest.raises(ValueError, match="Pathways checkpoint"):
        PathwaysSampleFeaturizer().set_state(state)
