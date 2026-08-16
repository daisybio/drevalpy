"""Tests for raw feature-matrix assembly on ``DRPModel``.

Mirrors :mod:`drevalpy.models.mixins._feature_matrix`. Nothing built by
``construct_model`` calls either method - the component stack featurizes - so
these exist for callers driving featurization by hand, and are asserted against
a stub source rather than through a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models.mixins._feature_matrix import DRPFeatureMatrixMixin


class _StubSource:
    """Feature source stand-in serving one matrix per view name."""

    def __init__(self, matrices: dict[str, np.ndarray]) -> None:
        self._matrices = matrices

    @property
    def view_names(self) -> list[str]:
        return list(self._matrices)

    def get_feature_matrix(self, view: str, identifiers: np.ndarray) -> np.ndarray:
        return np.repeat(self._matrices[view], len(identifiers), axis=0)


class _Model(DRPFeatureMatrixMixin):
    """Model-shaped object declaring the views its config would require."""

    def __init__(self, cell_line_views: list[str], drug_views: list[str]) -> None:
        self._cell_line_views = cell_line_views
        self._drug_views = drug_views

    @property
    def cell_line_views(self) -> list[str]:
        return self._cell_line_views

    @property
    def drug_views(self) -> list[str]:
        return self._drug_views


_IDS = np.array(["a", "b"])


def _cell_line_source() -> _StubSource:
    return _StubSource({"gene_expression": np.array([[1.0, 2.0]]), "mutations": np.array([[9.0]])})


def _drug_source() -> _StubSource:
    return _StubSource({"fingerprints": np.array([[3.0]])})


class TestGetFeatureMatrices:
    """One matrix per required view, from whichever sides were supplied."""

    def test_it_merges_both_sides(self) -> None:
        model = _Model(["gene_expression"], ["fingerprints"])

        matrices = model.get_feature_matrices(_IDS, _IDS, _cell_line_source(), _drug_source())

        assert sorted(matrices) == ["fingerprints", "gene_expression"]
        assert matrices["gene_expression"].shape == (2, 2)
        assert matrices["fingerprints"].shape == (2, 1)

    def test_a_missing_source_contributes_nothing(self) -> None:
        model = _Model(["gene_expression"], ["fingerprints"])

        matrices = model.get_feature_matrices(_IDS, _IDS, _cell_line_source(), None)

        assert sorted(matrices) == ["gene_expression"]

    def test_two_absent_sources_yield_nothing(self) -> None:
        model = _Model(["gene_expression"], ["fingerprints"])

        assert model.get_feature_matrices(_IDS, _IDS, None, None) == {}

    def test_a_missing_cell_line_view_is_reported_by_side(self) -> None:
        model = _Model(["proteomics"], [])

        with pytest.raises(ValueError, match="Cell line input does not contain view proteomics"):
            model.get_feature_matrices(_IDS, _IDS, _cell_line_source(), None)

    def test_a_missing_drug_view_is_reported_by_side(self) -> None:
        model = _Model([], ["chemberta"])

        with pytest.raises(ValueError, match="Drug input does not contain view chemberta"):
            model.get_feature_matrices(_IDS, _IDS, None, _drug_source())


class TestGetConcatenatedFeatures:
    """The requested pair of views, side by side in one matrix."""

    def test_both_sides_are_concatenated_column_wise(self) -> None:
        model = _Model(["gene_expression"], ["fingerprints"])

        matrix = model.get_concatenated_features(
            "gene_expression", "fingerprints", _IDS, _IDS, _cell_line_source(), _drug_source()
        )

        assert matrix.shape == (2, 3)
        np.testing.assert_allclose(matrix[0], [1.0, 2.0, 3.0])

    def test_a_none_drug_view_yields_the_cell_line_side_alone(self) -> None:
        model = _Model(["gene_expression"], [])

        matrix = model.get_concatenated_features("gene_expression", None, _IDS, _IDS, _cell_line_source(), None)

        assert matrix.shape == (2, 2)

    def test_a_none_cell_line_view_yields_the_drug_side_alone(self) -> None:
        model = _Model([], ["fingerprints"])

        matrix = model.get_concatenated_features(None, "fingerprints", _IDS, _IDS, None, _drug_source())

        assert matrix.shape == (2, 1)

    def test_requesting_neither_side_is_rejected(self) -> None:
        model = _Model([], [])

        with pytest.raises(ValueError, match="No features provided"):
            model.get_concatenated_features(None, None, _IDS, _IDS, None, None)

    def test_an_unassembled_cell_line_view_is_rejected(self) -> None:
        """The view was not in ``cell_line_views``, so nothing assembled it."""
        model = _Model([], ["fingerprints"])

        with pytest.raises(ValueError, match="Expected cell_line_view 'gene_expression'"):
            model.get_concatenated_features(
                "gene_expression", "fingerprints", _IDS, _IDS, _cell_line_source(), _drug_source()
            )

    def test_an_unassembled_drug_view_is_rejected_first(self) -> None:
        model = _Model(["gene_expression"], [])

        with pytest.raises(ValueError, match="Expected drug_view 'fingerprints'"):
            model.get_concatenated_features(
                "gene_expression", "fingerprints", _IDS, _IDS, _cell_line_source(), _drug_source()
            )
