"""Tests for :mod:`drevalpy.cli.curate`, the ``drevalpy curate`` command."""

from __future__ import annotations

import logging

import anndata
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.curation import FIT_SPEEDS
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain

runner = make_runner()

CONCENTRATIONS = (0.001, 0.01, 0.1, 1.0, 10.0)
SAMPLES = ("CL_A", "CL_B", "CL_C")
DRUGS = ("DrugX", "DrugY")


def build_dose_response_df() -> pd.DataFrame:
    """Build dose-response data: 3 samples x 2 drugs; ``DrugX`` is a sigmoid, ``DrugY`` is flat."""
    rng = np.random.default_rng(42)
    conc = np.array(CONCENTRATIONS)
    rows: list[dict] = []
    for sample in SAMPLES:
        for drug in DRUGS:
            if drug == "DrugX":
                intensity = 0.1 + 0.9 / (1 + (conc / 0.5) ** 1.5)
            else:
                intensity = np.full_like(conc, 0.95)
            intensity = np.clip(intensity + rng.normal(0, 0.02, size=len(conc)), 0.01, 1.5)
            rows.extend(
                {"drug": drug, "sample": sample, "concentration": c, "intensity": v}
                for c, v in zip(CONCENTRATIONS, intensity, strict=True)
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    return build_dose_response_df()


@pytest.fixture()
def input_csv(dose_response_df: pd.DataFrame, tmp_path: UPath) -> UPath:
    path = tmp_path / "input.csv"
    dose_response_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(return_value=anndata.AnnData(X=np.zeros((1, 1), dtype=np.float32)))
    patch_worker(monkeypatch, "drevalpy.curation", "curate", recorder)
    return recorder


@pytest.mark.parametrize("argv", [["curate"], ["curate", "in.csv"]], ids=["none", "missing-output"])
def test_missing_positional_arguments_are_usage_errors(argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


@pytest.mark.parametrize("name", ["input.csv", "input.parquet", "input.pq", "input.CSV"])
def test_supported_suffixes_are_read(
    worker: Recorder, dose_response_df: pd.DataFrame, tmp_path: UPath, name: str
) -> None:
    path = tmp_path / name
    if name.lower().endswith("csv"):
        dose_response_df.to_csv(path, index=False)
    else:
        dose_response_df.to_parquet(path)

    result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")])

    assert result.exit_code == 0, result.output
    assert len(worker.args[0]) == len(dose_response_df)


def test_unsupported_suffix_is_rejected_before_the_fitter(worker: Recorder, tmp_path: UPath) -> None:
    path = tmp_path / "input.tsv"
    path.write_text("drug\tsample\n")

    result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")], env=HELP_ENV)

    assert result.exit_code == 2
    assert "Unsupported file format: .tsv" in plain(result.output)
    assert worker.call_count == 0


def test_fit_option_defaults_and_overrides(worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
    out = str(tmp_path / "out.h5ad")

    runner.invoke(app, ["curate", str(input_csv), out])
    assert worker.kwargs == {"max_workers": 4, "normalize": False, "fit_speed": "exhaustive"}

    worker.calls.clear()
    runner.invoke(app, ["curate", str(input_csv), out, "-c", "2", "--normalize", "--fit-speed", "fast"])
    assert worker.kwargs == {"max_workers": 2, "normalize": True, "fit_speed": "fast"}


@pytest.mark.parametrize("speed", FIT_SPEEDS)
def test_every_library_fit_speed_is_accepted_case_insensitively(
    worker: Recorder, input_csv: UPath, tmp_path: UPath, speed: str
) -> None:
    runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), "--fit-speed", speed.upper()])

    assert worker.kwargs["fit_speed"] == speed


@pytest.mark.parametrize(
    "extra", [["--fit-speed", "slow"], ["--fit-type", "OLS"], ["--cores", "all"]], ids=["speed", "fit-type", "cores"]
)
def test_invalid_options_are_usage_errors_before_the_fitter(
    worker: Recorder, input_csv: UPath, tmp_path: UPath, extra: list[str]
) -> None:
    result = runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), *extra])

    assert result.exit_code == 2
    assert worker.call_count == 0


@pytest.mark.slow
def test_csv_to_h5ad_round_trip_with_the_real_fitter(tmp_path: UPath) -> None:
    input_path = tmp_path / "input.csv"
    build_dose_response_df().to_csv(input_path, index=False)
    output_path = tmp_path / "output.h5ad"
    logger = logging.getLogger("drevalpy")
    handlers, level = list(logger.handlers), logger.level
    try:
        result = runner.invoke(
            app, ["curate", str(input_path), str(output_path), "--cores", "1", "--fit-speed", "fast"]
        )
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)

    assert result.exit_code == 0, result.output
    adata = anndata.read_h5ad(output_path)
    assert adata.shape == (len(SAMPLES), len(DRUGS))
    assert {"EC50", "AUC"} <= set(adata.layers)
