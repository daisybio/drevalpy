"""Shared dose-response fixtures for the curation tests."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation import curate


def sigmoid(x: np.ndarray, top: float, bottom: float, ec50: float, slope: float) -> np.ndarray:
    """Evaluate a 4-parameter log-logistic sigmoid."""
    return bottom + (top - bottom) / (1 + (x / ec50) ** slope)


def build_dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 samples x 2 drugs, as a plain function for session-scoped fixtures."""
    concentrations = [0.001, 0.01, 0.1, 1.0, 10.0]
    samples = ["CL_A", "CL_B", "CL_C"]
    drugs = ["DrugX", "DrugY"]

    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for cl in samples:
        for drug in drugs:
            conc_arr = np.array(concentrations)
            if drug == "DrugX":
                intensity = sigmoid(conc_arr, top=1.0, bottom=0.1, ec50=0.5, slope=1.5)
            else:
                intensity = np.ones_like(conc_arr) * 0.95

            noise = rng.normal(0, 0.02, size=len(concentrations))
            intensity = np.clip(intensity + noise, 0.01, 1.5)

            for conc, intens in zip(concentrations, intensity, strict=True):
                rows.append({"drug": drug, "sample": cl, "concentration": conc, "intensity": intens})

    return pd.DataFrame(rows)


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 samples x 2 drugs."""
    return build_dose_response_df()


@pytest.fixture(scope="session")
def curated_adata() -> anndata.AnnData:
    """``curate`` run once on :func:`build_dose_response_df`; treat as immutable."""
    return curate(build_dose_response_df(), max_workers=1, fit_speed="fast")


@pytest.fixture()
def dose_response_df_with_replicates(dose_response_df: pd.DataFrame) -> pd.DataFrame:
    """Same data duplicated with replicate column."""
    rng = np.random.default_rng(99)
    rep1 = dose_response_df.copy()
    rep1["replicate"] = 1
    rep2 = dose_response_df.copy()
    rep2["replicate"] = 2
    rep2["intensity"] = rep2["intensity"] + rng.normal(0, 0.01, size=len(rep2))
    return pd.concat([rep1, rep2], ignore_index=True)


def build_replicated_wells() -> pd.DataFrame:
    """Wells with a replicate hierarchy, for :func:`curate` with ``replicate_fits=True``.

    * CL_A x DrugX: two experiments (``day1``, ``day2``) of two plates each.
    * CL_B x DrugX: one full plate and one plate with only two doses.
    * CL_A x DrugY: a single plate.
    """
    concentrations = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
    dates = {"day1": "2020-01-01", "day2": "2020-01-02"}
    rng = np.random.default_rng(7)
    rows: list[dict] = []

    def plate(sample: str, drug: str, experiment: str, replicate: str, doses: np.ndarray, ec50: float) -> None:
        intensity = np.clip(sigmoid(doses, 1.0, 0.1, ec50, 1.5) + rng.normal(0, 0.02, len(doses)), 0.01, 1.5)
        for conc, value in zip(doses, intensity, strict=True):
            rows.append(
                {
                    "sample": sample,
                    "drug": drug,
                    "concentration": conc,
                    "intensity": value,
                    "experiment": experiment,
                    "replicate": replicate,
                    "plate_date": dates[experiment],
                }
            )

    for experiment, plates in (("day1", ("P1", "P2")), ("day2", ("P3", "P4"))):
        for replicate in plates:
            plate("CL_A", "DrugX", experiment, replicate, concentrations, 0.5)
    plate("CL_B", "DrugX", "day1", "P1", concentrations, 2.0)
    plate("CL_B", "DrugX", "day1", "P5", concentrations[:2], 2.0)
    plate("CL_A", "DrugY", "day1", "P1", concentrations, 0.05)
    return pd.DataFrame(rows)
