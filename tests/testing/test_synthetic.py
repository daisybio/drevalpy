"""Compat names plugin suites import; the builder itself is tested in ``tests/data``."""

from __future__ import annotations

import drevalpy.data._synthetic as builder
from drevalpy.testing import synthetic as compat


def test_compat_names_are_the_builders() -> None:
    assert compat.build_synthetic_dataset is builder.synthetic
    for name in ("N_SAMPLES", "N_DRUGS", "N_TISSUES", "N_FEATURES", "DATASET_NAME", "SEED", "MEASURE"):
        assert getattr(compat, name) == getattr(builder, name)
    assert all(hasattr(compat, name) for name in compat.__all__)
