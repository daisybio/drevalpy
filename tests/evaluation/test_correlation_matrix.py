"""Tests for the per-model, per-group correlation matrix behind the comparison scatter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from drevalpy.evaluation._correlation_matrix import model_group_correlations
from drevalpy.evaluation._grouping import group_labeller
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

        assert matrix.shape == (N_MODELS, N_DRUGS)
        assert (matrix.dtypes == np.float32).all()
        assert list(matrix.index) == experiment.model_names
        assert list(matrix.columns) == sorted(matrix.columns)
        assert all(isinstance(name, str) for name in matrix.columns)
        assert model_group_correlations(experiment, "sample").shape[1] == N_SAMPLES

    def test_group_axis_is_the_union_across_models_and_unseen_groups_are_nan(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A", n_pairs=4, n_drugs=2),
                make_run_result(model_name="B", n_pairs=8, n_drugs=4),
            ]
        )

        matrix = model_group_correlations(result, "drug")

        assert matrix.shape[1] == 4
        assert matrix.loc["A"].iloc[2:].isna().all()

    def test_values_match_a_direct_per_group_pearson(self):
        run = make_run_result(model_name="A", n_pairs=20, n_drugs=4)
        matrix = model_group_correlations(ExperimentResult([run]), "drug")

        for name, value in matrix.loc["A"].items():
            mask = np.asarray(run.drug_ids) == name
            expected = pearsonr(run.predictions[mask], run.ground_truth[mask])[0]
            assert value == pytest.approx(expected, abs=1e-6)

    def test_folds_are_pooled_rather_than_averaged(self):
        pooled = model_group_correlations(
            ExperimentResult([make_run_result(fold_index=i, n_pairs=20) for i in range(2)]), "drug"
        )
        single = model_group_correlations(ExperimentResult([make_run_result(fold_index=0, n_pairs=20)]), "drug")

        assert not np.allclose(pooled.to_numpy(), single.to_numpy(), equal_nan=True)

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

        assert list(matrix.index) == ["A"]
        np.testing.assert_allclose(matrix.to_numpy(), model_group_correlations(plain, "drug").to_numpy())

    def test_an_all_randomized_experiment_is_empty(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        matrix = model_group_correlations(result, "drug")

        assert matrix.shape == (0, 0)

    def test_unknown_groupings_are_rejected(self, experiment):
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
                labels=np.concatenate([group_labeller(grouping)(run) for run in runs]),
            )

            assert list(matrix.columns) == [str(group) for group in scores.groups]
            np.testing.assert_allclose(matrix.iloc[0], scores.values["Pearson"], atol=1e-6)

    def test_a_per_drug_constant_baseline_scores_zero_in_every_drug(self):
        baseline = make_run_result(model_name="Baseline", n_pairs=20, n_drugs=4)
        for drug in np.unique(baseline.drug_ids):
            mask = baseline.drug_ids == drug
            baseline.predictions[mask] = baseline.ground_truth[mask].mean()
        other = make_run_result(model_name="Other", n_pairs=20, n_drugs=4)

        matrix = model_group_correlations(ExperimentResult([baseline, other]), "drug")

        np.testing.assert_array_equal(matrix.loc["Baseline"], np.zeros(4))
        assert np.isfinite(matrix.loc["Other"]).all()

    def test_groups_with_a_constant_response_are_nan_for_every_model(self):
        runs = []
        for name in ("A", "B"):
            run = make_run_result(model_name=name, n_pairs=20, n_drugs=4)
            run.ground_truth[run.drug_ids == run.drug_ids[0]] = 1.0
            runs.append(run)

        matrix = model_group_correlations(ExperimentResult(runs), "drug")

        constant = str(runs[0].drug_ids[0])
        assert matrix[constant].isna().all()
        assert np.isfinite(matrix.drop(columns=constant).to_numpy()).all()

    def test_a_model_with_no_defined_correlation_is_left_out(self):
        undefined = make_run_result(model_name="Undefined", n_pairs=20, n_drugs=4)
        undefined.ground_truth[:] = 1.0
        good = make_run_result(model_name="Good", n_pairs=20, n_drugs=4)

        matrix = model_group_correlations(ExperimentResult([undefined, good]), "drug")

        assert list(matrix.index) == ["Good"]
        assert matrix.shape == (1, 4)
