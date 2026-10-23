"""Tests for :mod:`drevalpy.cli.data.synthetic`, the ``drevalpy data synthetic`` command."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data._synthetic import MAX_SYNTHETIC_DRUGS, N_DRUGS, N_SAMPLES, N_TISSUES, SEED
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()


class TestArguments:
    def test_output_is_required(self) -> None:
        result = runner.invoke(app, ["data", "synthetic"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_defaults_match_the_python_function(self) -> None:
        import inspect

        from drevalpy.data import synthetic

        defaults = {name: parameter.default for name, parameter in inspect.signature(synthetic).parameters.items()}

        assert (defaults["seed"], defaults["n_samples"], defaults["n_drugs"], defaults["n_tissues"]) == (
            SEED,
            N_SAMPLES,
            N_DRUGS,
            N_TISSUES,
        )

    def test_help_documents_the_drug_limit(self) -> None:
        result = runner.invoke(app, ["data", "synthetic", "--help"], env=HELP_ENV)

        assert f"at most {MAX_SYNTHETIC_DRUGS}" in plain(result.output)


class TestWriting:
    """Each case builds a dataset and round-trips an ``.h5mu`` file, which costs over 0.2s."""

    pytestmark = pytest.mark.slow

    def test_writes_a_loadable_dataset_with_the_default_shape(self, tmp_path: UPath) -> None:
        import drevalpy

        out = tmp_path / "nested" / "toy.h5mu"
        result = runner.invoke(app, ["data", "synthetic", "--output", str(out)])

        assert result.exit_code == 0, result.output
        assert drevalpy.load(str(out)).response_matrix.shape == (N_SAMPLES, N_DRUGS)

    def test_the_file_carries_gene_expression_for_the_feature_models(self, tmp_path: UPath) -> None:
        import drevalpy

        out = tmp_path / "toy.h5mu"
        runner.invoke(app, ["data", "synthetic", "--output", str(out)])

        assert "gene_expression" in drevalpy.load(str(out)).mdata.mod

    def test_shape_options_set_the_loaded_dataset(self, tmp_path: UPath) -> None:
        import drevalpy
        from drevalpy.data.utils import TISSUE_IDENTIFIER

        out = tmp_path / "big.h5mu"
        result = runner.invoke(
            app,
            ["data", "synthetic", "--output", str(out), "--n-samples", "40", "--n-drugs", "5", "--n-tissues", "4"],
        )
        loaded = drevalpy.load(str(out))

        assert result.exit_code == 0, result.output
        assert loaded.response_matrix.shape == (40, 5)
        assert loaded.response.obs[TISSUE_IDENTIFIER].nunique() == 4

    def test_name_sets_the_dataset_name(self, tmp_path: UPath) -> None:
        import drevalpy

        out = tmp_path / "a.h5mu"
        result = runner.invoke(app, ["data", "synthetic", "--output", str(out), "--name", "STUDY_A"])

        assert result.exit_code == 0, result.output
        assert drevalpy.load(str(out)).name == "STUDY_A"

    def test_the_seed_changes_the_responses(self, tmp_path: UPath) -> None:
        import numpy as np

        import drevalpy

        for seed in (1, 2):
            runner.invoke(app, ["data", "synthetic", "--output", str(tmp_path / f"{seed}.h5mu"), "--seed", str(seed)])
        first, second = (drevalpy.load(str(tmp_path / f"{seed}.h5mu")) for seed in (1, 2))

        assert not np.array_equal(first.response_matrix, second.response_matrix, equal_nan=True)

    def test_the_limit_itself_is_accepted(self, tmp_path: UPath) -> None:
        result = runner.invoke(
            app, ["data", "synthetic", "--output", str(tmp_path / "x.h5mu"), "--n-drugs", str(MAX_SYNTHETIC_DRUGS)]
        )

        assert result.exit_code == 0, result.output

    def test_echoes_the_destination(self, tmp_path: UPath) -> None:
        out = tmp_path / "toy.h5mu"
        result = runner.invoke(app, ["data", "synthetic", "--output", str(out)])

        assert str(out) in plain(result.output)


class TestDrugLimit:
    def test_too_many_drugs_is_an_error_naming_the_limit(self, tmp_path: UPath) -> None:
        result = runner.invoke(
            app, ["data", "synthetic", "--output", str(tmp_path / "x.h5mu"), "--n-drugs", str(MAX_SYNTHETIC_DRUGS + 1)]
        )

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: ")
        assert str(MAX_SYNTHETIC_DRUGS) in plain(result.output)

    def test_no_file_is_written_on_an_error(self, tmp_path: UPath) -> None:
        out = tmp_path / "x.h5mu"
        runner.invoke(app, ["data", "synthetic", "--output", str(out), "--n-drugs", str(MAX_SYNTHETIC_DRUGS + 1)])

        assert not out.exists()
