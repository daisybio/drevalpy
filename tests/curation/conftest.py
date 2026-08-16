"""Shared dose-response fixtures for the curation tests."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation import curate


def sigmoid(x: np.ndarray, top: float, bottom: float, ec50: float, slope: float) -> np.ndarray:
    """Evaluate a 4-parameter log-logistic sigmoid.

    :param x: Concentrations.
    :param top: Response plateau at zero concentration.
    :param bottom: Response plateau at infinite concentration.
    :param ec50: Concentration at half-maximal effect.
    :param slope: Hill slope.
    :returns: Modelled response at each concentration.
    """
    return bottom + (top - bottom) / (1 + (x / ec50) ** slope)


def build_dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 cell lines x 2 drugs.

    Exposed as a plain function so the session-scoped fitted fixtures below can
    build their own private copy without sharing one frame across scopes.

    :returns: Long-form dose-response measurements.
    """
    concentrations = [0.001, 0.01, 0.1, 1.0, 10.0]
    cell_lines = ["CL_A", "CL_B", "CL_C"]
    drugs = ["DrugX", "DrugY"]

    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for cl in cell_lines:
        for drug in drugs:
            conc_arr = np.array(concentrations)
            if drug == "DrugX":
                intensity = sigmoid(conc_arr, top=1.0, bottom=0.1, ec50=0.5, slope=1.5)
            else:
                intensity = np.ones_like(conc_arr) * 0.95

            noise = rng.normal(0, 0.02, size=len(concentrations))
            intensity = np.clip(intensity + noise, 0.01, 1.5)

            for conc, intens in zip(concentrations, intensity, strict=True):
                rows.append({"drug": drug, "cell_line": cl, "concentration": conc, "intensity": intens})

    return pd.DataFrame(rows)


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 cell lines x 2 drugs."""
    return build_dose_response_df()


@pytest.fixture(scope="session")
def curated_adata() -> anndata.AnnData:
    """``curate`` run once on :func:`build_dose_response_df`, shared read-only.

    The six real CurveCurator fits behind this cost ~0.5s, so the end-to-end
    assertions in ``test_init.py`` share one run instead of repeating it. Treat
    the returned object as immutable; a test that needs to mutate it, or that
    asserts on ``curate``'s own arguments, must call ``curate`` itself.
    """
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
