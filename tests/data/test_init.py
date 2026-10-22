"""Tests for the public :mod:`drevalpy.data` package surface.

``drevalpy.data`` reaches the dataset and splitter registry singletons through a
module-level ``__getattr__`` rather than a top-level import, because
``drevalpy.registry`` imports back into ``drevalpy.data`` during built-in
registration. That indirection is part of the surface: a rename behind it would
turn into a runtime ``AttributeError`` at the call site instead of an
``ImportError`` here, so both the resolving path and the unknown-name path are
pinned - the first by the shared origin table below, the second by
:func:`test_unknown_name_raises_rather_than_returning_none`.

Only the surface is asserted. ``split``'s fold hashing and ``load``'s dataset
resolution are tested in ``tests/data/splitters/`` and
``tests/data/datasets/`` respectively.
"""

from __future__ import annotations

import pytest

from drevalpy import data
from tests._barrel_surface import DeclaredSurface

#: Names bound at import time -> a module holding the same object. The barrel
#: imports ``load`` from ``datasets._load``, so it is recorded against the
#: ``datasets`` barrel instead: a re-export compared with the module it was
#: imported from cannot fail.
EAGER_ORIGINS: dict[str, str] = {
    "curve_quality_mask": "drevalpy.data.quality",
    "load": "drevalpy.data.datasets",
}

#: Names served by the module-level ``__getattr__`` -> the leaf module that
#: constructs the singleton, not the registry barrel ``__getattr__`` imports from.
LAZY_ORIGINS: dict[str, str] = {
    "dataset_registry": "drevalpy.registry.dataset._registry",
    "splitter_registry": "drevalpy.registry.splitter._registry",
}


class TestDataSurface(DeclaredSurface):
    barrel = data
    origins = {**EAGER_ORIGINS, **LAZY_ORIGINS}
    unpinned_names = ("split",)


def test_split_is_defined_by_the_barrel_itself() -> None:
    assert callable(data.split)
    assert data.split.__module__ == "drevalpy.data"


def test_unknown_name_raises_rather_than_returning_none() -> None:
    """A module-level ``__getattr__`` must not turn a typo into a silent ``None``."""
    with pytest.raises(AttributeError, match="drevalpy.data"):
        getattr(data, "definitely_not_a_real_symbol")  # noqa: B009


def test_split_accepts_the_lco_alias_and_reports_lso(synthetic_dataset: data.Dataset) -> None:
    via_alias = data.split(synthetic_dataset, "LCO", n_splits=2)
    via_mode = data.split(synthetic_dataset, "LSO", n_splits=2)

    assert len(via_alias) == len(via_mode) == 2
    for old, new in zip(via_alias, via_mode, strict=True):
        assert old.metadata["split_mode"] == new.metadata["split_mode"] == "LSO"
        assert old.metadata["mode"] == "LSO"
        assert old.metadata["fold_id"] == new.metadata["fold_id"]
        assert (old.test.mask == new.test.mask).all()


def test_split_still_rejects_an_unknown_mode(synthetic_dataset: data.Dataset) -> None:
    with pytest.raises(ValueError, match="Unknown split mode 'LXO'"):
        data.split(synthetic_dataset, "LXO")
