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

import numpy as np
import pytest
from upath import UPath

from drevalpy import data
from drevalpy.data.splitters import leave_pair_out, splitter_registry
from drevalpy.types import SplitMasks
from tests._barrel_surface import DeclaredSurface

#: Names bound at import time -> a module holding the same object. The barrel
#: imports ``load`` from ``datasets._load``, so it is recorded against the
#: ``datasets`` barrel instead: a re-export compared with the module it was
#: imported from cannot fail.
EAGER_ORIGINS: dict[str, str] = {
    "curve_quality_mask": "drevalpy.data.quality",
    "load": "drevalpy.data.datasets",
    "synthetic": "drevalpy",
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
    unpinned_names = ("CurveQuality", "concat", "split")


def test_curve_quality_is_the_class_defined_in_the_quality_module() -> None:
    from drevalpy.data.quality import CurveQuality

    assert data.CurveQuality is CurveQuality


def test_split_accepts_a_quality_rule_without_changing_the_folds(synthetic_dataset: data.Dataset) -> None:
    plain = data.split(synthetic_dataset, "LPO", n_splits=2)
    with_rule = data.split(synthetic_dataset, "LPO", n_splits=2, quality=data.CurveQuality.off())

    assert [f.metadata["fold_id"] for f in plain] == [f.metadata["fold_id"] for f in with_rule]


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


@pytest.fixture(scope="module")
def low_quality_dataset() -> data.Dataset:
    return data.synthetic(low_quality_fraction=0.3)


def _covered(folds: list) -> np.ndarray:
    covered = np.zeros(folds[0].train.shape, dtype=bool)
    for fold in folds:
        covered |= fold.train.mask | fold.test.mask | fold.val.mask
    return covered


class TestSplitQuality:
    def test_the_default_rule_is_the_one_used_without_an_argument(self, low_quality_dataset: data.Dataset) -> None:
        implicit = data.split(low_quality_dataset, "LPO", n_splits=2)
        explicit = data.split(low_quality_dataset, "LPO", n_splits=2, quality=data.CurveQuality.default())

        assert [f.metadata["fold_id"] for f in implicit] == [f.metadata["fold_id"] for f in explicit]

    @pytest.mark.parametrize("mode", ["LPO", "LSO", "LDO", "LTO"])
    def test_failed_pairs_are_in_no_fold(self, low_quality_dataset: data.Dataset, mode: str) -> None:
        passes = data.CurveQuality.default().mask(low_quality_dataset)
        measured = np.isfinite(low_quality_dataset.response_matrix)
        assert (measured & ~passes).any()

        covered = _covered(data.split(low_quality_dataset, mode, n_splits=2))

        assert not (covered & ~passes).any()

    def test_off_keeps_the_pairs_the_default_rule_drops(self, low_quality_dataset: data.Dataset) -> None:
        default = _covered(data.split(low_quality_dataset, "LPO", n_splits=2))
        off = _covered(data.split(low_quality_dataset, "LPO", n_splits=2, quality=data.CurveQuality.off()))

        np.testing.assert_array_equal(off, np.isfinite(low_quality_dataset.response_matrix))
        assert off.sum() > default.sum()

    def test_a_custom_threshold_drops_more_pairs(self, low_quality_dataset: data.Dataset) -> None:
        median = float(np.nanmedian(low_quality_dataset.response_matrix))
        default = _covered(data.split(low_quality_dataset, "LPO", n_splits=2))

        covered = _covered(
            data.split(low_quality_dataset, "LPO", n_splits=2, quality=data.CurveQuality(min_pec50=median))
        )

        assert 0 < covered.sum() < default.sum()

    def test_the_dataset_is_not_modified(self, low_quality_dataset: data.Dataset) -> None:
        before = low_quality_dataset.response_matrix.copy()

        data.split(low_quality_dataset, "LPO", n_splits=2)

        np.testing.assert_array_equal(low_quality_dataset.response_matrix, before)

    def test_off_splits_a_dataset_without_quality_layers(self) -> None:
        dataset = data.synthetic()
        for name in dataset.response_layer_names():
            del dataset.response.layers[name]

        folds = data.split(dataset, "LSO", n_splits=2, quality=data.CurveQuality.off())

        assert len(folds) == 2
        with pytest.raises(KeyError):
            data.split(dataset, "LSO", n_splits=2)

    @pytest.mark.parametrize(
        "rule",
        [
            pytest.param(None, id="default"),
            pytest.param(data.CurveQuality.off(), id="off"),
            pytest.param(data.CurveQuality(min_relevance_score=5.0, min_r2=0.9), id="custom"),
        ],
    )
    def test_the_rule_is_recorded_and_round_trips_through_save(
        self, low_quality_dataset: data.Dataset, rule: data.CurveQuality | None, tmp_path: UPath
    ) -> None:
        folds = data.split(low_quality_dataset, "LSO", n_splits=2, quality=rule)
        expected = (rule or data.CurveQuality.default()).to_metadata()

        folds[0].save(str(tmp_path / "fold.npz"))
        loaded = SplitMasks.load(str(tmp_path / "fold.npz"))

        assert [f.metadata["quality"] for f in folds] == [expected, expected]
        assert loaded.metadata["quality"] == expected
        assert data.CurveQuality.from_metadata(loaded.metadata["quality"]) == (rule or data.CurveQuality.default())

    def test_a_plugin_splitter_that_ignores_quality_sees_nan_on_failed_pairs(
        self, low_quality_dataset: data.Dataset
    ) -> None:
        seen: list[np.ndarray] = []

        def plugin(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
            seen.append(mudataset.response_matrix)
            return leave_pair_out(mudataset, n_splits, validation_ratio, random_state)

        modes = frozenset(splitter_registry.modes)
        splitter_registry.register("P3_PLUGIN", "Records what it is given", validation="LPO")(plugin)
        try:
            data.split(low_quality_dataset, "P3_PLUGIN", n_splits=2)
        finally:
            splitter_registry.retain_only(modes)

        failed = ~data.CurveQuality.default().mask(low_quality_dataset)
        assert np.isnan(seen[0][failed]).all()
