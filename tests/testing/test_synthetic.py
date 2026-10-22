"""Tests for :mod:`drevalpy.testing.synthetic`, which keeps the names plugin suites already import.

The builder itself is tested in ``tests/data/test_synthetic.py``.
"""

from __future__ import annotations

import drevalpy.data._synthetic as builder
from drevalpy.testing import synthetic as compat


def test_build_synthetic_dataset_is_the_data_function() -> None:
    assert compat.build_synthetic_dataset is builder.synthetic


def test_the_shape_constants_are_the_builders() -> None:
    assert (compat.N_SAMPLES, compat.N_DRUGS, compat.N_TISSUES, compat.N_FEATURES) == (
        builder.N_SAMPLES,
        builder.N_DRUGS,
        builder.N_TISSUES,
        builder.N_FEATURES,
    )


def test_the_dataset_name_seed_and_measure_are_the_builders() -> None:
    assert (compat.DATASET_NAME, compat.SEED, compat.MEASURE) == (builder.DATASET_NAME, builder.SEED, builder.MEASURE)


def test_all_lists_only_names_that_resolve() -> None:
    assert all(hasattr(compat, name) for name in compat.__all__)
