"""Tests for the cell-line and drug feature access mixin.

``tests/types/data/test_dataset.py`` covers the happy paths through the real
fixture. This module targets what that cannot reach: the ``name:variant`` varm
prefix resolution, the ``entities_with_modality`` NaN filtering, and the strict
vs warn behaviour, all of which need datasets shaped deliberately wrong.
"""

from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.types.data.dataset import Dataset

CELL_LINES = np.array(["cl1", "cl2", "cl3"])
DRUGS = np.array(["d1", "d2"])


def _dataset(
    *,
    varm: dict[str, np.ndarray] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
    uns: dict[str, object] | None = None,
    gene_expression: np.ndarray | None = None,
    cnv_under_stored_name: bool = False,
) -> Dataset:
    """Build a 3x2 Dataset with exactly the pieces a test needs."""
    response = ad.AnnData(
        X=np.arange(6.0, dtype=np.float32).reshape(3, 2),
        obs=pd.DataFrame({"tissue": ["Lung", "Blood", "Skin"]}, index=CELL_LINES),
        var=pd.DataFrame(index=DRUGS),
    )
    for key, value in (varm or {}).items():
        response.varm[key] = value
    for key, value in (obsm or {}).items():
        response.obsm[key] = value

    mods: dict[str, ad.AnnData] = {"response": response}
    default_expression = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    mods["gene_expression"] = ad.AnnData(
        X=default_expression if gene_expression is None else gene_expression,
        obs=pd.DataFrame(index=CELL_LINES),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    if cnv_under_stored_name:
        mods["copy_number_variation"] = ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obs=pd.DataFrame(index=CELL_LINES),
            var=pd.DataFrame(index=["GENE1", "GENE2"]),
        )

    md.set_options(pull_on_update=False)
    mdata = md.MuData(mods)
    for key, value in (uns or {}).items():
        mdata.uns[key] = value
    return Dataset(mdata, name="feature-access")


class TestModalityResolution:
    def test_a_public_name_resolves_through_the_accessor_map(self):
        dataset = _dataset(cnv_under_stored_name=True)

        features = dataset.get_cell_line_features("copy_number_variation_gistic", CELL_LINES)

        assert features.shape == (3, 2)

    def test_an_unknown_modality_lists_the_public_names_available(self):
        dataset = _dataset(cnv_under_stored_name=True)

        with pytest.raises(KeyError, match="copy_number_variation_gistic"):
            dataset.get_cell_line_features("absent", CELL_LINES)

    def test_feature_names_come_from_var_names(self):
        assert _dataset().get_cell_line_feature_names("gene_expression") == ("GENE1", "GENE2")

    def test_feature_names_are_none_for_an_absent_modality(self):
        assert _dataset().get_cell_line_feature_names("absent") is None

    def test_pathway_features_have_no_column_names(self):
        dataset = _dataset(obsm={"pathway_features": np.zeros((3, 4), dtype=np.float32)})

        assert dataset.get_cell_line_feature_names("pathway_features") is None


class TestObsmFeatures:
    def test_pathway_features_are_read_from_obsm(self):
        dataset = _dataset(obsm={"pathway_features": np.ones((3, 4), dtype=np.float32)})

        features = dataset.get_cell_line_features("pathway_features", CELL_LINES[:2])

        assert features.shape == (2, 4)

    def test_a_missing_obsm_key_raises(self):
        with pytest.raises(KeyError, match="obsm key 'pathway_features' not found"):
            _dataset().get_cell_line_features("pathway_features", CELL_LINES)

    def test_strict_mode_rejects_unknown_cell_lines(self):
        dataset = _dataset(obsm={"pathway_features": np.ones((3, 4), dtype=np.float32)})

        with pytest.raises(KeyError, match="cell line IDs not found"):
            dataset.get_cell_line_features("pathway_features", np.array(["absent"]), strict=True)


class TestVarmKeyResolution:
    def test_an_exact_varm_key_wins(self):
        dataset = _dataset(varm={"pca": np.zeros((2, 3), dtype=np.float32)})

        assert dataset.get_drug_features("pca", DRUGS).shape == (2, 3)

    def test_a_variant_key_is_found_by_prefix(self):
        """Precomputed variants are stored as ``storage_key:index``."""
        dataset = _dataset(varm={"pca:0": np.zeros((2, 3), dtype=np.float32)})

        assert dataset.get_drug_features("pca", DRUGS).shape == (2, 3)

    def test_a_prefix_match_requires_the_colon(self):
        dataset = _dataset(varm={"pca_extra": np.zeros((2, 3), dtype=np.float32)})

        with pytest.raises(KeyError, match="Drug feature 'pca' not found"):
            dataset.get_drug_features("pca", DRUGS)

    def test_available_drug_views_are_sorted(self):
        dataset = _dataset(
            varm={
                "zeta": np.zeros((2, 1), dtype=np.float32),
                "alpha": np.zeros((2, 1), dtype=np.float32),
            }
        )

        assert dataset.available_drug_views == ["alpha", "zeta"]

    def test_drug_feature_names_fall_back_to_positional_labels(self):
        dataset = _dataset(varm={"pca": np.zeros((2, 3), dtype=np.float32)})

        assert dataset.get_drug_feature_names("pca") == ("0", "1", "2")

    def test_drug_feature_names_use_dataframe_columns_when_present(self):
        frame = pd.DataFrame(np.zeros((2, 2), dtype=np.float32), columns=["bit0", "bit1"], index=DRUGS)
        dataset = _dataset(varm={"fingerprint": frame})

        assert dataset.get_drug_feature_names("fingerprint") == ("bit0", "bit1")

    def test_drug_feature_names_are_none_for_an_absent_view(self):
        assert _dataset().get_drug_feature_names("absent") is None

    def test_strict_mode_rejects_unknown_drugs(self):
        dataset = _dataset(varm={"pca": np.zeros((2, 3), dtype=np.float32)})

        with pytest.raises(KeyError, match="drug IDs not found"):
            dataset.get_drug_features("pca", np.array(["absent"]), strict=True)


class TestDrugGraphs:
    def test_graphs_are_aligned_to_the_requested_ids(self):
        dataset = _dataset(uns={"drug_graphs": {"d1": {"x": np.zeros(1)}}})

        graphs = dataset.get_drug_graphs(np.array(["d2", "d1"]))

        assert graphs[0] is None
        assert graphs[1] is not None

    def test_a_dataset_without_graphs_raises(self):
        with pytest.raises(KeyError, match="'drug_graphs' not found"):
            _dataset().get_drug_graphs(DRUGS)


class TestEntitiesWithModality:
    def test_cell_lines_with_an_all_nan_row_are_excluded(self):
        matrix = np.array([[0.0, 1.0], [np.nan, np.nan], [4.0, 5.0]], dtype=np.float32)
        dataset = _dataset(gene_expression=matrix)

        assert dataset.entities_with_modality("gene_expression") == frozenset({"cl1", "cl3"})

    def test_a_partially_nan_row_still_counts(self):
        matrix = np.array([[0.0, np.nan], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        dataset = _dataset(gene_expression=matrix)

        assert "cl1" in dataset.entities_with_modality("gene_expression")

    def test_pathway_features_are_filtered_the_same_way(self):
        obsm = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
        dataset = _dataset(obsm={"pathway_features": obsm})

        assert dataset.entities_with_modality("pathway_features") == frozenset({"cl1", "cl3"})

    def test_absent_pathway_features_yield_an_empty_set(self):
        assert _dataset().entities_with_modality("pathway_features") == frozenset()

    def test_drug_views_are_filtered_by_nan_rows(self):
        varm = np.array([[1.0], [np.nan]], dtype=np.float32)
        dataset = _dataset(varm={"pca": varm})

        assert dataset.entities_with_modality("pca", side="drug") == frozenset({"d1"})

    def test_drug_graph_membership_comes_from_uns(self):
        dataset = _dataset(uns={"drug_graphs": {"d1": {"x": np.zeros(1)}}})

        assert dataset.entities_with_modality("drug_graph", side="drug") == frozenset({"d1"})

    def test_drug_graph_without_uns_yields_an_empty_set(self):
        assert _dataset().entities_with_modality("drug_graph", side="drug") == frozenset()

    def test_an_absent_drug_view_raises(self):
        with pytest.raises(KeyError, match="Drug feature 'absent' not found"):
            _dataset().entities_with_modality("absent", side="drug")
