"""Tests for the per-model, per-group correlation matrix behind the comparison scatter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from drevalpy.evaluation._correlation_matrix import GroupCorrelationMatrix, model_group_correlations
from drevalpy.evaluation._grouping import group_labels
from drevalpy.evaluation._scoring import evaluate_per_group
from drevalpy.types.results.experiment import ExperimentResult
from tests.synthetic import make_experiment_result, make_run_result

N_MODELS = 3
N_FOLDS = 2
N_PAIRS = 20
N_DRUGS = 4
N_SAMPLES = 5


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result(n_models=N_MODELS, n_folds=N_FOLDS, n_pairs=N_PAIRS)


class TestModelGroupCorrelations:
    def test_shape_dtype_and_ordering(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert matrix.values.shape == (N_MODELS, N_DRUGS)
        assert matrix.values.dtype == np.float32
        assert list(matrix.model_names) == experiment.model_names
        assert list(matrix.group_names) == sorted(matrix.group_names)
        assert all(isinstance(name, str) for name in matrix.group_names)
        assert matrix.grouping == "drug"
        assert model_group_correlations(experiment, "sample").n_groups == N_SAMPLES

    def test_group_axis_is_the_union_across_models_and_unseen_groups_are_nan(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A", n_pairs=4, n_drugs=2),
                make_run_result(model_name="B", n_pairs=8, n_drugs=4),
            ]
        )

        matrix = model_group_correlations(result, "drug")

        assert matrix.n_groups == 4
        assert np.isnan(matrix.values[matrix.model_names.index("A")][2:]).all()

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

    def test_randomized_runs_and_models_are_excluded(self):
        randomization = ("gene_expression", "permutation")
        mixed = ExperimentResult(
            [
                make_run_result(model_name="A", fold_index=0),
                make_run_result(model_name="A", fold_index=1, randomization=randomization),
                make_run_result(model_name="B", randomization=randomization),
            ]
        )
        plain = ExperimentResult([make_run_result(model_name="A", fold_index=0)])

        matrix = model_group_correlations(mixed, "drug")

        assert matrix.model_names == ("A",)
        np.testing.assert_allclose(matrix.values, model_group_correlations(plain, "drug").values)

    def test_an_all_randomized_experiment_is_empty(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        matrix = model_group_correlations(result, "drug")

        assert matrix.is_empty
        assert matrix.values.shape == (0, 0)

    def test_min_count_is_forwarded_and_unknown_groupings_are_rejected(self, experiment):
        assert np.isnan(model_group_correlations(experiment, "drug", min_count=1000).values).all()
        with pytest.raises(ValueError, match="Unknown grouping"):
            model_group_correlations(experiment, "tissue")

    def test_matches_evaluate_per_group_on_the_concatenated_folds(self):
        runs = [make_run_result(model_name="A", fold_index=i, n_pairs=30, n_drugs=5, n_samples=6) for i in range(3)]
        result = ExperimentResult(runs)

        for grouping in ("drug", "sample"):
            matrix = model_group_correlations(result, grouping)
            scores = evaluate_per_group(
                np.concatenate([run.predictions for run in runs]),
                np.concatenate([run.ground_truth for run in runs]),
                "Pearson",
                labels=np.concatenate([group_labels(run, grouping) for run in runs]),
            )

            assert matrix.group_names == tuple(str(group) for group in scores.groups)
            np.testing.assert_allclose(matrix.values[0], scores.values["Pearson"], atol=1e-6)

    def test_a_per_drug_constant_baseline_scores_zero_in_every_drug(self):
        baseline = make_run_result(model_name="Baseline", n_pairs=20, n_drugs=4)
        for drug in np.unique(baseline.drug_ids):
            mask = baseline.drug_ids == drug
            baseline.predictions[mask] = baseline.ground_truth[mask].mean()
        other = make_run_result(model_name="Other", n_pairs=20, n_drugs=4)

        matrix = model_group_correlations(ExperimentResult([baseline, other]), "drug")

        np.testing.assert_array_equal(matrix.values[matrix.model_names.index("Baseline")], np.zeros(4))
        assert np.isfinite(matrix.values[matrix.model_names.index("Other")]).all()

    def test_groups_with_a_constant_response_are_nan_for_every_model(self):
        runs = []
        for name in ("A", "B"):
            run = make_run_result(model_name=name, n_pairs=20, n_drugs=4)
            run.ground_truth[run.drug_ids == run.drug_ids[0]] = 1.0
            runs.append(run)

        matrix = model_group_correlations(ExperimentResult(runs), "drug")

        column = list(matrix.group_names).index(str(runs[0].drug_ids[0]))
        assert np.isnan(matrix.values[:, column]).all()
        assert np.isfinite(np.delete(matrix.values, column, axis=1)).all()

    def test_min_count_masks_small_groups_only(self):
        result = ExperimentResult([make_run_result(model_name="A", n_pairs=20, n_drugs=4)])
        counts = np.unique(result.models[0].select_runs()[0].drug_ids, return_counts=True)[1]

        matrix = model_group_correlations(result, "drug", min_count=int(counts.max()))

        np.testing.assert_array_equal(np.isfinite(matrix.values[0]), counts >= counts.max())


class TestGroupCorrelationMatrix:
    def test_dimensions(self, experiment):
        matrix = model_group_correlations(experiment, "drug")

        assert (matrix.n_models, matrix.n_groups) == (N_MODELS, N_DRUGS)
        assert matrix.is_empty is False

    def test_a_matrix_with_no_groups_is_empty(self):
        assert GroupCorrelationMatrix("drug", ("A",), (), np.empty((1, 0), dtype=np.float32)).is_empty

    def test_drop_all_nan_models_removes_only_fully_undefined_models(self):
        matrix = GroupCorrelationMatrix(
            "drug",
            ("Good", "Partial", "Undefined"),
            ("D_0", "D_1"),
            np.array([[0.5, 0.4], [np.nan, 0.4], [np.nan, np.nan]], dtype=np.float32),
        )

        filtered = matrix.drop_all_nan_models()

        assert filtered.model_names == ("Good", "Partial")
        assert filtered.values.shape == (2, 2)

    def test_drop_all_nan_models_returns_self_when_nothing_is_dropped(self):
        full = GroupCorrelationMatrix("drug", ("A",), ("D_0",), np.array([[0.3]], dtype=np.float32))
        empty = GroupCorrelationMatrix("drug", (), (), np.empty((0, 0), dtype=np.float32))

        assert full.drop_all_nan_models() is full
        assert empty.drop_all_nan_models() is empty
