"""Fixtures of the golden gate: the native thread pin, the golden dataset, its fold and the recorded baseline.

The model-running code lives in :mod:`tests.golden._record` (``train_and_predict``), so the
recorder and the test share it. The gate runs under :func:`~tests.golden._record.single_threaded`
as the recording did. The pin and the fixtures built under it are module-scoped, so the pin covers
the dataset build too and ends with the gate instead of leaking into the tests after it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from tests.golden._record import BASELINE_FILE, PREDICTIONS_FILE, build_dataset, build_fold, single_threaded


@pytest.fixture(scope="module", autouse=True)
def _single_threaded() -> Iterator[None]:
    """Pin every native thread pool to one thread while the gate runs, as the recording did."""
    with single_threaded():
        yield


@pytest.fixture(scope="module")
def golden_dataset():
    """The golden dataset, built once: building it computes fingerprints, graphs and BPE codes."""
    return build_dataset()


@pytest.fixture(scope="module")
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
