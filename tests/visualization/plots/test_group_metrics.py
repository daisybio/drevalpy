"""Tests for :mod:`drevalpy.visualization.plots._group_metrics`.

The module exists to replace a ``groupby().apply(pearsonr)`` with vectorised
``bincount`` sums, so the tests below pin the numerical agreement with
:func:`scipy.stats.pearsonr` as well as the degenerate cases where the two
disagree deliberately: a group with fewer than two observations, and a group with
no variance on one axis, both of which yield NaN here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._group_metrics import (
    GROUPING_LABELS,
    GROUPINGS,
    GroupCorrelationMatrix,
    group_labels,
    grouped_pearson,
    model_group_correlations,
)
from tests.synthetic import make_experiment_result, make_run_result

N_MODELS = 3
N_FOLDS = 2
N_PAIRS = 20
N_DRUGS = 4
N_CELL_LINES = 5


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result(n_models=N_MODELS, n_folds=N_FOLDS, n_pairs=N_PAIRS)


class TestGroupings:
    def test_drug_and_cell_line_are_the_supported_groupings(self):
        assert GROUPINGS == ("drug", "cell_line")

    def test_every_grouping_has_a_human_label(self):
        assert set(GROUPING_LABELS) == set(GROUPINGS)


class TestGroupLabels:
    def test_drug_grouping_reads_drug_ids(self):
        run = make_run_result(n_pairs=6)

        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)

    def test_cell_line_grouping_reads_cell_line_ids(self):
        run = make_run_result(n_pairs=6)

        np.testing.assert_array_equal(group_labels(run, "cell_line"), run.cell_line_ids)

    def test_an_unknown_grouping_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            group_labels(make_run_result(), "tissue")


class TestGroupedPearson:
    def test_agrees_with_scipy_per_group(self):
        rng = np.random.default_rng(0)
        codes = np.repeat([0, 1], 25)
        x = rng.normal(size=50)
        y = rng.normal(size=50)

        result = grouped_pearson(codes, 2, x, y)

        for group in (0, 1):
            expected = pearsonr(x[codes == group], y[codes == group])[0]
            assert result[group] == pytest.approx(expected, abs=1e-9)

    def test_a_perfect_positive_relationship_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert grouped_pearson(np.zeros(4, dtype=int), 1, x, 2 * x + 1)[0] == pytest.approx(1.0)

    def test_a_perfect_negative_relationship_is_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert grouped_pearson(np.zeros(4, dtype=int), 1, x, -x)[0] == pytest.approx(-1.0)

    def test_a_single_observation_group_is_nan(self):
        result = grouped_pearson(np.array([0, 1, 1]), 2, np.array([1.0, 1.0, 2.0]), np.array([1.0, 3.0, 4.0]))

        assert np.isnan(result[0])
        assert np.isfinite(result[1])

    def test_a_group_with_no_variance_is_nan(self):
        codes = np.zeros(4, dtype=int)

        result = grouped_pearson(codes, 1, np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))

        assert np.isnan(result[0])

    def test_an_empty_group_keeps_its_slot_as_nan(self):
        result = grouped_pearson(np.zeros(3, dtype=int), 3, np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))

        assert result.shape == (3,)
        assert np.isnan(result[1:]).all()

    def test_negative_codes_are_dropped(self):
        codes = np.array([-1, 0, 0, 0])

        result = grouped_pearson(codes, 1, np.array([99.0, 1.0, 2.0, 3.0]), np.array([-99.0, 1.0, 2.0, 3.0]))

        assert result[0] == pytest.approx(1.0)

    def test_nan_observations_are_dropped(self):
        codes = np.zeros(5, dtype=int)
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, np.nan])

        result = grouped_pearson(codes, 1, x, y)

        assert result[0] == pytest.approx(1.0)

    def test_dropping_nans_can_take_a_group_below_the_minimum(self):
        codes = np.zeros(3, dtype=int)

        result = grouped_pearson(codes, 1, np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))

        assert np.isnan(result[0])

    def test_a_higher_min_count_filters_small_groups(self):
        codes = np.zeros(3, dtype=int)
        x = np.array([1.0, 2.0, 3.0])

        assert np.isfinite(grouped_pearson(codes, 1, x, x, min_count=3)[0])
        assert np.isnan(grouped_pearson(codes, 1, x, x, min_count=4)[0])

    def test_results_stay_inside_the_unit_interval(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 5, size=200)
        x = rng.normal(size=200)

        result = grouped_pearson(codes, 5, x, x * 3.0)

        assert np.all(np.abs(result) <= 1.0)

    def test_all_observations_dropped_yields_all_nan(self):
        result = grouped_pearson(np.array([-1, -1]), 2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        assert np.isnan(result).all()


class TestModelGroupCorrelations:
    def test_shape_is_models_by_groups(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert matrix.values.shape == (N_MODELS, N_DRUGS)

    def test_the_cell_line_axis_has_one_column_per_cell_line(self, experiment):
        matrix = model_group_correlations(experiment, "cell_line")

        assert matrix.n_groups == N_CELL_LINES

    def test_values_are_float32_so_the_matrix_stays_small(self, experiment):
        assert model_group_correlations(experiment, "drug").values.dtype == np.float32

    def test_model_names_follow_the_experiment_order(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert list(matrix.model_names) == experiment.model_names

    def test_group_names_are_sorted_strings(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert list(matrix.group_names) == sorted(matrix.group_names)
        assert all(isinstance(name, str) for name in matrix.group_names)

    def test_group_axis_is_the_union_across_models(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A", n_pairs=2, n_drugs=2),
                make_run_result(model_name="B", n_pairs=6, n_drugs=6),
            ]
        )

        assert model_group_correlations(result, "drug").n_groups == 6

    def test_a_group_a_model_never_saw_is_nan_for_that_model(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A", n_pairs=4, n_drugs=2),
                make_run_result(model_name="B", n_pairs=8, n_drugs=4),
            ]
        )

        matrix = model_group_correlations(result, "drug")

        assert np.isnan(matrix.for_model("A")[2:]).all()

    def test_values_match_a_direct_per_group_pearson(self):
        run = make_run_result(model_name="A", n_pairs=20, n_drugs=4)
        matrix = model_group_correlations(ExperimentResult([run]), "drug")

        for index, name in enumerate(matrix.group_names):
            mask = np.asarray(run.drug_ids) == name
            expected = pearsonr(run.predictions[mask], run.ground_truth[mask])[0]
            assert matrix.values[0, index] == pytest.approx(expected, abs=1e-6)

    def test_folds_are_pooled_rather_than_averaged(self):
        pooled = model_group_correlations(
            ExperimentResult([make_run_result(fold_index=i, n_pairs=20) for i in range(2)]), "drug"
        )
        single = model_group_correlations(ExperimentResult([make_run_result(fold_index=0, n_pairs=20)]), "drug")

        assert not np.allclose(pooled.values, single.values, equal_nan=True)

    def test_randomized_runs_are_excluded(self):
        randomized = ExperimentResult(
            [
                make_run_result(model_name="A", fold_index=0),
                make_run_result(model_name="A", fold_index=1, randomization=("gene_expression", "permutation")),
            ]
        )
        plain = ExperimentResult([make_run_result(model_name="A", fold_index=0)])

        np.testing.assert_allclose(
            model_group_correlations(randomized, "drug").values,
            model_group_correlations(plain, "drug").values,
        )

    def test_a_fully_randomized_model_is_omitted(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A"),
                make_run_result(model_name="B", randomization=("gene_expression", "permutation")),
            ]
        )

        assert model_group_correlations(result, "drug").model_names == ("A",)

    def test_an_all_randomized_experiment_is_empty(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        matrix = model_group_correlations(result, "drug")

        assert matrix.is_empty
        assert matrix.values.shape == (0, 0)

    def test_min_count_is_forwarded(self, experiment):
        matrix = model_group_correlations(experiment, "drug", min_count=1000)

        assert np.isnan(matrix.values).all()

    def test_an_unknown_grouping_is_rejected(self, experiment):
        with pytest.raises(ValueError, match="Unknown grouping"):
            model_group_correlations(experiment, "tissue")


class TestGroupCorrelationMatrix:
    def test_reports_its_dimensions(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert (matrix.n_models, matrix.n_groups) == (N_MODELS, N_DRUGS)
        assert matrix.is_empty is False

    def test_for_model_returns_the_matching_row(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        np.testing.assert_array_equal(matrix.for_model(matrix.model_names[1]), matrix.values[1])

    def test_for_model_rejects_an_unknown_name(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        with pytest.raises(KeyError):
            matrix.for_model("NotAModel")

    def test_a_matrix_with_no_groups_is_empty(self):
        matrix = GroupCorrelationMatrix("drug", ("A",), (), np.empty((1, 0), dtype=np.float32))

        assert matrix.is_empty

    def test_drop_all_nan_models_removes_undefined_models(self):
        matrix = GroupCorrelationMatrix(
            "drug",
            ("Good", "Undefined"),
            ("D_0", "D_1"),
            np.array([[0.5, 0.4], [np.nan, np.nan]], dtype=np.float32),
        )

        filtered = matrix.drop_all_nan_models()

        assert filtered.model_names == ("Good",)
        assert filtered.values.shape == (1, 2)

    def test_drop_all_nan_models_keeps_a_partially_defined_model(self):
        matrix = GroupCorrelationMatrix(
            "drug",
            ("Partial",),
            ("D_0", "D_1"),
            np.array([[np.nan, 0.4]], dtype=np.float32),
        )

        assert matrix.drop_all_nan_models().model_names == ("Partial",)

    def test_drop_all_nan_models_returns_self_when_nothing_is_dropped(self, experiment):
        matrix = GroupCorrelationMatrix("drug", ("A",), ("D_0",), np.array([[0.3]], dtype=np.float32))

        assert matrix.drop_all_nan_models() is matrix

    def test_drop_all_nan_models_on_an_empty_matrix_is_a_no_op(self):
        matrix = GroupCorrelationMatrix("drug", (), (), np.empty((0, 0), dtype=np.float32))

        assert matrix.drop_all_nan_models() is matrix

    def test_the_grouping_is_carried_through(self, experiment):
        assert model_group_correlations(experiment, "cell_line").grouping == "cell_line"


class TestBoundedMemory:
    def test_retained_bytes_scale_with_groups_not_predictions(self):
        few_rows = make_experiment_result(n_models=3, n_folds=1, n_pairs=20)
        many_rows = make_experiment_result(n_models=3, n_folds=6, n_pairs=20)

        small = model_group_correlations(few_rows, "drug")
        large = model_group_correlations(many_rows, "drug")

        assert small.values.nbytes == large.values.nbytes

    def test_a_float32_matrix_stays_tiny_at_production_shape(self):
        assert np.empty((96, 524), dtype=np.float32).nbytes < 250_000
