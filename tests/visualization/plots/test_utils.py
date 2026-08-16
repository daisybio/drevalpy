"""Tests for :mod:`drevalpy.visualization.plots._utils`.

Per the underscore-stripping naming convention, this mirrors ``plots/_utils.py``.
``runs_frame`` backs the heatmap, violin and cross-study-table plots; the colour
helpers are a public extension point no shipped plot uses yet, so both are
covered here directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._utils import MODEL_COLORS, compute_ssmd, model_color_palette, runs_frame
from tests.synthetic import make_experiment_result, make_run_result


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


class TestModelColors:
    def test_palette_is_ten_distinct_hex_colors(self):
        assert len(MODEL_COLORS) == 10
        assert len(set(MODEL_COLORS)) == 10

    def test_every_entry_is_a_six_digit_hex_string(self):
        assert all(len(c) == 7 and c.startswith("#") for c in MODEL_COLORS)


class TestModelColorPalette:
    def test_assigns_one_color_per_model_in_order(self):
        palette = model_color_palette(["A", "B", "C"])

        assert palette == {"A": MODEL_COLORS[0], "B": MODEL_COLORS[1], "C": MODEL_COLORS[2]}

    def test_colors_are_distinct_up_to_the_palette_size(self):
        names = [f"m{i}" for i in range(len(MODEL_COLORS))]

        assert len(set(model_color_palette(names).values())) == len(MODEL_COLORS)

    def test_cycles_when_there_are_more_models_than_colors(self):
        names = [f"m{i}" for i in range(len(MODEL_COLORS) + 2)]

        palette = model_color_palette(names)

        assert palette["m10"] == MODEL_COLORS[0]
        assert palette["m11"] == MODEL_COLORS[1]

    def test_empty_input_yields_an_empty_palette(self):
        assert model_color_palette([]) == {}

    def test_duplicate_names_keep_only_the_last_assignment(self):
        palette = model_color_palette(["A", "A"])

        assert palette == {"A": MODEL_COLORS[1]}


class TestComputeSsmd:
    def test_is_zero_for_identical_samples(self):
        values = [1.0, 2.0, 3.0]

        assert compute_ssmd(values, values) == 0.0

    def test_is_positive_when_the_first_sample_is_larger(self):
        assert compute_ssmd([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) > 0

    def test_is_antisymmetric(self):
        a, b = [2.0, 3.0, 5.0], [1.0, 2.0, 2.5]

        assert compute_ssmd(a, b) == pytest.approx(-compute_ssmd(b, a))

    def test_matches_the_closed_form(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.5, 1.5])
        expected = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) + b.var(ddof=1))

        assert compute_ssmd(a, b) == pytest.approx(expected)

    def test_is_nan_when_both_samples_are_constant(self):
        assert math.isnan(compute_ssmd([2.0, 2.0], [1.0, 1.0]))

    def test_accepts_integer_sequences(self):
        assert compute_ssmd([1, 2, 3], [1, 2, 3]) == 0.0

    def test_returns_a_builtin_float(self):
        assert type(compute_ssmd([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])) is float


class TestRunsFrame:
    def test_has_one_row_per_run(self, experiment):
        assert len(runs_frame(experiment)) == sum(m.n_folds for m in experiment.models)

    def test_carries_the_identity_columns_plus_every_metric(self, experiment):
        df = runs_frame(experiment)

        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]
        assert {"MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"} <= set(df.columns)

    def test_test_mode_comes_from_the_experiment_split_mode(self, experiment):
        assert set(runs_frame(experiment)["test_mode"]) == {experiment.split_mode}

    def test_unrandomized_runs_are_labelled_predictions(self, experiment):
        assert set(runs_frame(experiment)["rand_setting"]) == {"predictions"}

    def test_randomized_runs_get_a_composite_setting_label(self):
        result = ExperimentResult([make_run_result(randomization=("methylation", "invariant"))])

        assert runs_frame(result)["rand_setting"].tolist() == ["methylation_invariant"]

    def test_defaults_to_a_positional_index(self, experiment):
        df = runs_frame(experiment)

        assert list(df.index) == list(range(len(df)))

    def test_indexed_encodes_model_setting_mode_and_split(self, experiment):
        df = runs_frame(experiment, indexed=True)

        assert df.index[0] == f"{experiment.model_names[0]}_predictions_LPO_split_0"

    def test_indexed_reuses_the_setting_label_it_puts_in_the_column(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = runs_frame(result, indexed=True)

        assert df["rand_setting"].tolist() == ["gene_expression_permutation"]
        assert "gene_expression_permutation" in df.index[0]

    def test_indexing_does_not_change_the_rows(self, experiment):
        plain = runs_frame(experiment)
        indexed = runs_frame(experiment, indexed=True)

        assert plain.to_numpy().tolist() == indexed.to_numpy().tolist()
