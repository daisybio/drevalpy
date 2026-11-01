"""Fixtures of the golden gate: the golden dataset, its fold and the recorded baseline.

The model-running code lives in :mod:`tests.golden._record` (``train_and_predict``), so the
recorder and the test share it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from tests.golden._record import BASELINE_FILE, PREDICTIONS_FILE, build_dataset, build_fold


@pytest.fixture(scope="session")
def golden_dataset():
    """The golden dataset, built once: building it computes fingerprints, graphs and BPE codes."""
    return build_dataset()


@pytest.fixture(scope="session")
def golden_fold(golden_dataset):
    """The recorded fold of the golden dataset."""
    return build_fold(golden_dataset)


@pytest.fixture(scope="session")
def golden_predictions() -> dict[str, np.ndarray]:
    """Recorded test predictions by baseline key."""
    with PREDICTIONS_FILE.open("rb") as handle, np.load(handle) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="session")
def golden_baseline() -> dict[str, Any]:
    """Recorded metrics and metadata by baseline key."""
    with BASELINE_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)
