"""Tests for the GSVA pathway cell-line featurizer.

Mirrors :mod:`drevalpy.components.featurizers.cell_line.pathways`. Not network
gated: it needs an ``mdata``-backed source plus ``uns["pathways_gmt"]``, both of
which the synthetic fixture carries. ``gseapy.gsva`` is the slow part, so exactly
one test runs it and the rest cover the guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.pathways import PathwaysCellLineFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import CellLineFeatureSource
from tests.conftest import MockFeatureSource


def test_output_dim_is_zero_before_fit() -> None:
    assert PathwaysCellLineFeaturizer().output_dim == 0


def test_transform_before_fit_raises() -> None:
    featurizer = PathwaysCellLineFeaturizer()

    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform(MockFeatureSource(features={}), np.array(["cl1"]))


def test_run_gsva_requires_pathways_gmt_in_uns() -> None:
    featurizer = PathwaysCellLineFeaturizer()
    source = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})

    with pytest.raises(ValueError, match="require pathways_gmt"):
        featurizer._run_gsva(source, np.array(["cl1"]))


def test_run_gsva_requires_gene_expression_feature_names(synthetic_dataset: Dataset) -> None:
    class _NamelessSource(CellLineFeatureSource):
        def get_feature_names(self, view: str) -> None:
            return None

    featurizer = PathwaysCellLineFeaturizer()
    source = _NamelessSource(synthetic_dataset, synthetic_dataset.cell_line_ids)

    with pytest.raises(ValueError, match="must provide feature names"):
        featurizer._run_gsva(source, synthetic_dataset.cell_line_ids[:2])


def test_fit_then_transform_returns_pathway_scores(synthetic_dataset: Dataset) -> None:
    source = CellLineFeatureSource(synthetic_dataset, synthetic_dataset.cell_line_ids)
    cell_line_ids = synthetic_dataset.cell_line_ids[:6]
    featurizer = PathwaysCellLineFeaturizer()

    featurizer.fit(source, entity_ids=cell_line_ids)
    matrix = featurizer.transform(source, cell_line_ids)

    assert featurizer.output_dim > 0
    assert matrix.shape == (len(cell_line_ids), featurizer.output_dim)
    assert matrix.dtype == np.float32


def test_transform_of_unseen_cell_lines_recomputes_gsva(
    synthetic_dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CellLineFeatureSource(synthetic_dataset, synthetic_dataset.cell_line_ids)
    fit_ids = synthetic_dataset.cell_line_ids[:2]
    featurizer = PathwaysCellLineFeaturizer()
    featurizer._fit_scores = np.zeros((2, 3), dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 3

    calls: list[int] = []

    def _fake_run_gsva(_source, entity_ids):
        calls.append(len(entity_ids))
        return np.ones((len(entity_ids), 3), dtype=np.float32)

    monkeypatch.setattr(featurizer, "_run_gsva", _fake_run_gsva)

    matrix = featurizer._transform(source, np.array([*fit_ids, synthetic_dataset.cell_line_ids[5]]))

    assert calls == [3]
    assert matrix.shape == (3, 3)


def test_transform_of_known_cell_lines_reuses_fitted_scores(synthetic_dataset: Dataset) -> None:
    source = CellLineFeatureSource(synthetic_dataset, synthetic_dataset.cell_line_ids)
    fit_ids = synthetic_dataset.cell_line_ids[:2]
    featurizer = PathwaysCellLineFeaturizer()
    featurizer._fit_scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 2

    matrix = featurizer._transform(source, np.array([fit_ids[1], fit_ids[0]]))

    np.testing.assert_allclose(matrix, [[3.0, 4.0], [1.0, 2.0]])


def test_transform_blocks_emit_one_block_named_after_the_view(synthetic_dataset: Dataset) -> None:
    source = CellLineFeatureSource(synthetic_dataset, synthetic_dataset.cell_line_ids)
    fit_ids = synthetic_dataset.cell_line_ids[:2]
    featurizer = PathwaysCellLineFeaturizer()
    featurizer._fit_scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    featurizer._fit_ids = fit_ids
    featurizer._output_dim = 2

    blocks = featurizer._transform_blocks(source, fit_ids)

    assert set(blocks) == {"pathways"}
    assert blocks["pathways"].feature_names is None
