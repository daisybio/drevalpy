"""In-memory synthetic dataset for plugin and component tests.

The builder lives in :mod:`drevalpy.data._synthetic` as :func:`drevalpy.data.synthetic`;
this module keeps the names plugin test suites already import.
"""

from __future__ import annotations

from drevalpy.data._synthetic import (
    DATASET_NAME,
    MAX_SYNTHETIC_DRUGS,
    MEASURE,
    N_DRUGS,
    N_FEATURES,
    N_SAMPLES,
    N_TISSUES,
    SEED,
    synthetic,
)

build_synthetic_dataset = synthetic

__all__ = [
    "DATASET_NAME",
    "MAX_SYNTHETIC_DRUGS",
    "MEASURE",
    "N_DRUGS",
    "N_FEATURES",
    "N_SAMPLES",
    "N_TISSUES",
    "SEED",
    "build_synthetic_dataset",
]
