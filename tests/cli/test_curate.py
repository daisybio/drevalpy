"""Tests for :mod:`drevalpy.cli.curate`, the ``drevalpy curate`` command.

The CSV -> h5ad round-trip here replaces the ``TestCLI`` class that used to live
in ``tests/curation/test_init.py``; it keeps ``fit_speed="fast"`` and
``cores=1`` so the real ``curve_curator`` fit stays cheap and serial. The
remaining tests patch :func:`drevalpy.curation.curate` to isolate argument
plumbing and the format guard from the fitter.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain

runner = make_runner()

CONCENTRATIONS = (0.001, 0.01, 0.1, 1.0, 10.0)
CELL_LINES = ("CL_A", "CL_B", "CL_C")
DRUGS = ("DrugX", "DrugY")


def _sigmoid(x: np.ndarray, top: float, bottom: float, ec50: float, slope: float) -> np.ndarray:
    """4-parameter log-logistic sigmoid."""
    return bottom + (top - bottom) / (1 + (x / ec50) ** slope)


def build_dose_response_df() -> pd.DataFrame:
    """Build artificial dose-response data: 3 cell lines x 2 drugs.

    ``DrugX`` gets a genuine sigmoid so the fit converges; ``DrugY`` is flat.

    Returns:
        Long-format frame with ``drug``/``cell_line``/``concentration``/``intensity``.
    """
    rng = np.random.default_rng(42)
    conc_arr = np.array(CONCENTRATIONS)
    rows: list[dict] = []

    for cell_line in CELL_LINES:
        for drug in DRUGS:
            if drug == "DrugX":
                intensity = _sigmoid(conc_arr, top=1.0, bottom=0.1, ec50=0.5, slope=1.5)
            else:
                intensity = np.ones_like(conc_arr) * 0.95
            intensity = np.clip(intensity + rng.normal(0, 0.02, size=len(conc_arr)), 0.01, 1.5)

            rows.extend(
                {"drug": drug, "cell_line": cell_line, "concentration": conc, "intensity": value}
                for conc, value in zip(CONCENTRATIONS, intensity, strict=True)
            )

    return pd.DataFrame(rows)


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 cell lines x 2 drugs."""
    return build_dose_response_df()


@pytest.fixture()
def input_csv(dose_response_df: pd.DataFrame, tmp_path: UPath) -> UPath:
    """The dose-response frame written as CSV."""
    path = tmp_path / "input.csv"
    dose_response_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Patch :func:`drevalpy.curation.curate` with a recorder returning a 1x1 AnnData.

    Args:
        monkeypatch: Fixture used to replace the source-module worker.

    Returns:
        Recorder standing in for the curve fitter.
    """
    recorder = Recorder(return_value=anndata.AnnData(X=np.zeros((1, 1), dtype=np.float32)))
    patch_worker(monkeypatch, "drevalpy.curation", "curate", recorder)
    return recorder


class TestArguments:
    """Both positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [pytest.param(["curate"], id="none"), pytest.param(["curate", "in.csv"], id="missing-output")],
    )
    def test_missing_positional_arguments_are_usage_errors(self, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestFormatDetection:
    """The suffix decides the reader; anything else is a ``BadParameter``."""

    def test_csv_is_read(self, worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad")])

        assert result.exit_code == 0, result.output
        assert len(worker.args[0]) == len(CONCENTRATIONS) * len(CELL_LINES) * len(DRUGS)

    @pytest.mark.parametrize("suffix", [".parquet", ".pq"], ids=["parquet", "pq"])
    def test_parquet_suffixes_are_read(
        self, worker: Recorder, dose_response_df: pd.DataFrame, tmp_path: UPath, suffix: str
    ) -> None:
        path = tmp_path / f"input{suffix}"
        dose_response_df.to_parquet(path)

        result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")])

        assert result.exit_code == 0, result.output
        assert len(worker.args[0]) == len(dose_response_df)

    def test_suffix_matching_is_case_insensitive(
        self, worker: Recorder, dose_response_df: pd.DataFrame, tmp_path: UPath
    ) -> None:
        path = tmp_path / "input.CSV"
        dose_response_df.to_csv(path, index=False)

        result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")])

        assert result.exit_code == 0, result.output

    def test_unsupported_suffix_is_rejected(self, worker: Recorder, tmp_path: UPath) -> None:
        path = tmp_path / "input.tsv"
        path.write_text("drug\tcell_line\n")

        result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")], env=HELP_ENV)

        assert result.exit_code == 2
        assert "Unsupported file format: .tsv" in plain(result.output)

    def test_unsupported_suffix_does_not_reach_the_fitter(self, worker: Recorder, tmp_path: UPath) -> None:
        path = tmp_path / "input.tsv"
        path.write_text("drug\tcell_line\n")

        runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")], env=HELP_ENV)

        assert worker.call_count == 0


class TestFitOptions:
    """Fit options map onto :func:`drevalpy.curation.curate` keywords."""

    def test_defaults(self, worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
        runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad")])

        assert worker.kwargs == {
            "cores": 4,
            "normalize": False,
            "fit_type": "OLS",
            "fit_speed": "exhaustive",
        }

    def test_overrides(self, worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
        runner.invoke(
            app,
            [
                "curate",
                str(input_csv),
                str(tmp_path / "out.h5ad"),
                "--cores",
                "2",
                "--normalize",
                "--fit-type",
                "MLE",
                "--fit-speed",
                "fast",
            ],
        )

        assert worker.kwargs == {"cores": 2, "normalize": True, "fit_type": "MLE", "fit_speed": "fast"}

    def test_cores_short_option(self, worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
        runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), "-c", "1"])

        assert worker.kwargs["cores"] == 1

    def test_non_integer_cores_is_a_usage_error(self, worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), "--cores", "all"])

        assert result.exit_code == 2


class TestRoundTrip:
    """End-to-end CSV -> .h5ad with the real fitter, kept cheap and serial."""

    @pytest.fixture(scope="class")
    def curated(self, tmp_path_factory: pytest.TempPathFactory) -> UPath:
        """Run the real command once and hand back the written .h5ad path."""
        tmp_path = UPath(tmp_path_factory.mktemp("curate_round_trip"))
        input_path = tmp_path / "input.csv"
        build_dose_response_df().to_csv(input_path, index=False)
        output_path = tmp_path / "output.h5ad"

        result = runner.invoke(
            app,
            ["curate", str(input_path), str(output_path), "--cores", "1", "--fit-speed", "fast"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        return output_path

    def test_writes_the_output_file(self, curated: UPath) -> None:
        assert curated.exists()

    def test_output_is_readable_anndata(self, curated: UPath) -> None:
        adata = anndata.read_h5ad(curated)

        assert adata.shape == (len(CELL_LINES), len(DRUGS))

    def test_output_carries_curve_metric_layers(self, curated: UPath) -> None:
        adata = anndata.read_h5ad(curated)

        assert "EC50" in adata.layers
        assert "AUC" in adata.layers
