"""Tests for :mod:`drevalpy.visualization.plots._utils` (``runs_frame`` and ``compute_ssmd``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._utils import compute_ssmd, runs_frame
from tests.synthetic import make_experiment_result, make_run_result


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


class TestComputeSsmd:
    def test_sign_and_antisymmetry(self):
        a, b = [2.0, 3.0, 5.0], [1.0, 2.0, 2.5]

        assert compute_ssmd(a, a) == 0.0
        assert compute_ssmd(a, b) > 0
        assert compute_ssmd(a, b) == pytest.approx(-compute_ssmd(b, a))

    def test_matches_the_closed_form_and_returns_a_builtin_float(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.5, 1.5])
        expected = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) + b.var(ddof=1))

        assert compute_ssmd(a, b) == pytest.approx(expected)
        assert type(compute_ssmd([1, 2, 3], [0, 1, 2])) is float

    def test_is_nan_when_both_inputs_are_constant(self):
        assert math.isnan(compute_ssmd([2.0, 2.0], [1.0, 1.0]))


class TestRunsFrame:
    def test_one_row_per_run_with_identity_columns_and_every_metric(self, experiment):
        df = runs_frame(experiment)

        assert len(df) == sum(m.n_folds for m in experiment.models)
        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]
        assert {"MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"} <= set(df.columns)
        assert set(df["test_mode"]) == {experiment.split_mode}
        assert set(df["rand_setting"]) == {"predictions"}
        assert list(df.index) == list(range(len(df)))

    def test_randomized_runs_get_a_composite_setting_label_also_in_the_index(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = runs_frame(result, indexed=True, randomization="all")

        assert df["rand_setting"].tolist() == ["gene_expression_permutation"]
        assert "gene_expression_permutation" in df.index[0]

    def test_indexed_encodes_model_setting_mode_and_split_without_changing_the_rows(self, experiment):
        indexed = runs_frame(experiment, indexed=True)

        assert indexed.index[0] == f"{experiment.model_names[0]}_predictions_LPO_split_0"
        assert runs_frame(experiment).to_numpy().tolist() == indexed.to_numpy().tolist()

    def test_default_selection_excludes_randomized_and_robustness_variants(self):
        result = ExperimentResult(
            [
                make_run_result(),
                make_run_result(randomization=("gene_expression", "permutation")),
                make_run_result(fold_metadata={"robustness_trial": 0}),
            ]
        )

        assert len(runs_frame(result)) == 1
        assert len(runs_frame(result, randomization="all", robustness="all")) == 3
