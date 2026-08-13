"""Tests for :mod:`drevalpy.testing.synthetic`."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.utils import CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.testing.synthetic import (
    DATASET_NAME,
    MEASURE,
    N_CELL_LINES,
    N_DRUGS,
    N_FEATURES,
    _punch_holes,
    build_synthetic_dataset,
)
from drevalpy.types.data.dataset import Dataset


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_synthetic_dataset()


class TestShapeAndNaming:
    def test_it_returns_a_dataset(self, dataset):
        assert isinstance(dataset, Dataset)

    def test_the_default_name_is_recorded(self, dataset):
        assert dataset.name == DATASET_NAME

    def test_the_name_is_configurable(self):
        assert build_synthetic_dataset(name="MYDATA").name == "MYDATA"

    def test_the_response_matrix_has_the_default_shape(self, dataset):
        assert dataset.response_matrix.shape == (N_CELL_LINES, N_DRUGS)

    def test_the_shape_is_configurable(self):
        small = build_synthetic_dataset(n_cell_lines=6, n_drugs=3, n_tissues=3)

        assert small.response_matrix.shape == (6, 3)

    def test_only_the_response_modality_is_present_by_default(self, dataset):
        assert list(dataset.mdata.mod) == ["response"]

    def test_identifiers_are_unique(self, dataset):
        assert len(set(dataset.cell_line_ids)) == N_CELL_LINES
        assert len(set(dataset.drug_ids)) == N_DRUGS


class TestMetadata:
    def test_the_cell_line_name_column_is_populated(self, dataset):
        assert dataset.response.obs[CELL_LINE_IDENTIFIER].notna().all()

    def test_tissue_labels_are_resolvable_through_the_dataset(self, dataset):
        tissues = dataset.get_tissue(np.asarray(dataset.cell_line_ids))

        assert len(tissues) == N_CELL_LINES
        assert all(isinstance(tissue, str) for tissue in tissues)

    def test_the_requested_number_of_tissues_is_produced(self):
        small = build_synthetic_dataset(n_tissues=3)

        assert len(set(small.response.obs[TISSUE_IDENTIFIER])) == 3

    def test_the_published_measure_is_stored_as_a_layer(self, dataset):
        layer = dataset.get_response_layer(MEASURE)

        np.testing.assert_array_equal(np.isnan(layer), np.isnan(dataset.response_matrix))

    def test_drug_names_are_recorded(self, dataset):
        assert dataset.response.var["drug_name"].notna().all()


class TestMissingResponses:
    def test_some_pairs_are_unmeasured_by_default(self, dataset):
        assert np.isnan(dataset.response_matrix).any()

    def test_every_cell_line_keeps_a_measurement(self, dataset):
        assert (~np.isnan(dataset.response_matrix)).any(axis=1).all()

    def test_every_drug_keeps_a_measurement(self, dataset):
        assert (~np.isnan(dataset.response_matrix)).any(axis=0).all()

    def test_zero_fraction_leaves_a_complete_matrix(self):
        complete = build_synthetic_dataset(missing_fraction=0.0)

        assert not np.isnan(complete.response_matrix).any()

    def test_a_larger_fraction_removes_more_pairs(self):
        sparse = build_synthetic_dataset(missing_fraction=0.4)
        dense = build_synthetic_dataset(missing_fraction=0.05)

        assert np.isnan(sparse.response_matrix).sum() > np.isnan(dense.response_matrix).sum()

    def test_an_extreme_fraction_still_honours_the_row_guarantee(self):
        """Clamped rather than obeyed: an empty row breaks the LCO splitters."""
        extreme = build_synthetic_dataset(missing_fraction=1.0)

        assert (~np.isnan(extreme.response_matrix)).any(axis=1).all()
        assert (~np.isnan(extreme.response_matrix)).any(axis=0).all()


class TestPunchHoles:
    def test_a_degenerate_matrix_is_left_alone(self):
        matrix = np.ones((1, 4), dtype=np.float32)

        _punch_holes(matrix, np.random.default_rng(0), 0.5)

        assert not np.isnan(matrix).any()

    def test_it_modifies_in_place(self):
        matrix = np.ones((6, 6), dtype=np.float32)

        _punch_holes(matrix, np.random.default_rng(0), 0.2)

        assert np.isnan(matrix).any()


class TestOmics:
    def test_a_sequence_adds_full_coverage_modalities(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression", "proteomics"])

        assert set(with_omics.mdata.mod) == {"response", "gene_expression", "proteomics"}

    def test_features_are_retrievable_through_the_public_accessor(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"])

        matrix = with_omics.get_cell_line_features("gene_expression", np.asarray(with_omics.cell_line_ids))

        assert matrix.shape == (N_CELL_LINES, N_FEATURES)

    def test_feature_names_are_exposed(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"], feature_names=["A", "B"])

        assert with_omics.get_cell_line_feature_names("gene_expression") == ("A", "B")

    def test_the_feature_width_is_configurable(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"], n_features=3)

        assert with_omics.mdata.mod["gene_expression"].shape[1] == 3

    def test_a_mapping_sets_per_modality_coverage(self):
        """Partial coverage is what makes the NaN-filtering path fire."""
        partial = build_synthetic_dataset(omics={"gene_expression": 10})

        assert partial.mdata.mod["gene_expression"].shape[0] == 10

    def test_uncovered_cell_lines_come_back_as_nan(self):
        partial = build_synthetic_dataset(omics={"gene_expression": 10})

        matrix = partial.get_cell_line_features("gene_expression", np.asarray(partial.cell_line_ids))

        assert np.isnan(matrix[10:]).all()
        assert not np.isnan(matrix[:10]).any()

    def test_the_gistic_alias_resolves_to_the_stored_modality(self):
        """The public name is suffixed; the published files are not."""
        with_cnv = build_synthetic_dataset(omics=["copy_number_variation_gistic"])

        assert "copy_number_variation" in with_cnv.mdata.mod
        assert with_cnv.get_cell_line_features("copy_number_variation_gistic", np.asarray([])).size == 0


class TestDeterminism:
    def test_the_same_seed_gives_the_same_matrix(self):
        first = build_synthetic_dataset()
        second = build_synthetic_dataset()

        np.testing.assert_array_equal(first.response_matrix, second.response_matrix)

    def test_a_different_seed_gives_a_different_matrix(self):
        first = build_synthetic_dataset(seed=1)
        second = build_synthetic_dataset(seed=2)

        assert not np.array_equal(first.response_matrix, second.response_matrix)


class TestSplitterCompatibility:
    """The builder exists so plugin CI can split and train; prove it can."""

    @pytest.mark.parametrize("mode", ["LPO", "LCO", "LDO", "LTO"])
    def test_every_builtin_splitter_accepts_it(self, dataset, mode):
        from drevalpy.registry import splitter

        folds = splitter.get(mode)(dataset, n_splits=2, validation_ratio=0.1)

        assert len(folds) == 2
        for fold in folds:
            assert fold.train.any()
            assert fold.test.any()

    def test_folds_never_select_unmeasured_pairs(self, dataset):
        from drevalpy.registry import splitter

        observed = ~np.isnan(dataset.response_matrix)

        for fold in splitter.get("LPO")(dataset, n_splits=2, validation_ratio=0.1):
            assert not (fold.train.mask & ~observed).any()
            assert not (fold.test.mask & ~observed).any()
