"""Tests for :mod:`drevalpy.cli.data.synthetic`, the ``drevalpy data synthetic`` command."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data._synthetic import MAX_SYNTHETIC_DRUGS, N_DRUGS, N_SAMPLES
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()


def test_output_is_required_and_help_documents_the_drug_limit() -> None:
    assert runner.invoke(app, ["data", "synthetic"], env=HELP_ENV).exit_code == 2

    helped = runner.invoke(app, ["data", "synthetic", "--help"], env=HELP_ENV)

    assert f"at most {MAX_SYNTHETIC_DRUGS}" in plain(helped.output)


def test_too_many_drugs_is_an_error_naming_the_limit_and_writes_no_file(tmp_path: UPath) -> None:
    out = tmp_path / "x.h5mu"

    result = runner.invoke(app, ["data", "synthetic", "--output", str(out), "--n-drugs", str(MAX_SYNTHETIC_DRUGS + 1)])

    assert result.exit_code == 1
    assert plain(result.output).startswith("Error: ")
    assert str(MAX_SYNTHETIC_DRUGS) in plain(result.output)
    assert not out.exists()


@pytest.mark.slow
class TestWriting:
    def test_writes_a_loadable_dataset_with_the_default_shape(self, tmp_path: UPath) -> None:
        import drevalpy

        out = tmp_path / "nested" / "toy.h5mu"
        result = runner.invoke(app, ["data", "synthetic", "--output", str(out)])

        assert result.exit_code == 0, result.output
        assert str(out) in plain(result.output)
        loaded = drevalpy.load(str(out))
        assert loaded.response_matrix.shape == (N_SAMPLES, N_DRUGS)
        assert "gene_expression" in loaded.mdata.mod

    def test_options_set_the_loaded_dataset(self, tmp_path: UPath) -> None:
        import drevalpy
        from drevalpy.data.utils import TISSUE_IDENTIFIER

        args = ["--n-samples", "40", "--n-drugs", "5", "--n-tissues", "4", "--name", "STUDY_A"]
        result = runner.invoke(app, ["data", "synthetic", "--output", str(tmp_path / "big.h5mu"), *args])
        loaded = drevalpy.load(str(tmp_path / "big.h5mu"))

        assert result.exit_code == 0, result.output
        assert loaded.name == "STUDY_A"
        assert loaded.response_matrix.shape == (40, 5)
        assert loaded.response.obs[TISSUE_IDENTIFIER].nunique() == 4

    def test_the_seed_changes_the_responses_and_the_drug_limit_itself_is_accepted(self, tmp_path: UPath) -> None:
        import drevalpy

        for seed in (1, 2):
            runner.invoke(app, ["data", "synthetic", "--output", str(tmp_path / f"{seed}.h5mu"), "--seed", str(seed)])
        first, second = (drevalpy.load(str(tmp_path / f"{seed}.h5mu")) for seed in (1, 2))
        limit = runner.invoke(
            app, ["data", "synthetic", "--output", str(tmp_path / "x.h5mu"), "--n-drugs", str(MAX_SYNTHETIC_DRUGS)]
        )

        assert not np.array_equal(first.response_matrix, second.response_matrix, equal_nan=True)
        assert limit.exit_code == 0, limit.output
