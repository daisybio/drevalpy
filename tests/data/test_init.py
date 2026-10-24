"""Tests for the public :mod:`drevalpy.data` surface, including the lazy registry singletons and ``split``."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy import data
from drevalpy.data.splitters import leave_pair_out, splitter_registry
from drevalpy.types import SplitMasks
from tests._barrel_surface import DeclaredSurface

EAGER_ORIGINS: dict[str, str] = {
    "curve_quality_mask": "drevalpy.data.quality",
    "load": "drevalpy.data.datasets",
    "synthetic": "drevalpy",
}

LAZY_ORIGINS: dict[str, str] = {
    "dataset_registry": "drevalpy.registry.dataset._registry",
    "splitter_registry": "drevalpy.registry.splitter._registry",
}


class TestDataSurface(DeclaredSurface):
    barrel = data
    origins = {**EAGER_ORIGINS, **LAZY_ORIGINS}
    unpinned_names = ("CurveQuality", "concat", "split")


def test_curve_quality_and_split_are_defined_by_the_barrel() -> None:
    from drevalpy.data.quality import CurveQuality

    assert data.CurveQuality is CurveQuality
    assert data.split.__module__ == "drevalpy.data"


def test_unknown_name_raises_rather_than_returning_none() -> None:
    with pytest.raises(AttributeError, match="drevalpy.data"):
        getattr(data, "definitely_not_a_real_symbol")  # noqa: B009


def test_split_accepts_the_lco_alias_and_reports_lso_and_rejects_an_unknown_mode(
    synthetic_dataset: data.Dataset,
) -> None:
    via_alias = data.split(synthetic_dataset, "LCO", n_splits=2)
    via_mode = data.split(synthetic_dataset, "LSO", n_splits=2)

    assert len(via_alias) == len(via_mode) == 2
    for old, new in zip(via_alias, via_mode, strict=True):
        assert old.metadata["split_mode"] == new.metadata["split_mode"] == "LSO"
        assert old.metadata["mode"] == "LSO"
        assert old.metadata["fold_id"] == new.metadata["fold_id"]
        assert (old.test.mask == new.test.mask).all()
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
        before = low_quality_dataset.response_matrix.copy()

        assert [f.metadata["fold_id"] for f in implicit] == [f.metadata["fold_id"] for f in explicit]
        np.testing.assert_array_equal(low_quality_dataset.response_matrix, before)

    @pytest.mark.parametrize("mode", ["LPO", "LSO", "LDO", "LTO"])
    def test_failed_pairs_are_in_no_fold(self, low_quality_dataset: data.Dataset, mode: str) -> None:
        passes = data.CurveQuality.default().mask(low_quality_dataset)
        measured = np.isfinite(low_quality_dataset.response_matrix)
        assert (measured & ~passes).any()

        covered = _covered(data.split(low_quality_dataset, mode, n_splits=2))

        assert not (covered & ~passes).any()

    def test_off_keeps_the_pairs_the_default_rule_drops_and_a_custom_threshold_drops_more(
        self, low_quality_dataset: data.Dataset
    ) -> None:
        median = float(np.nanmedian(low_quality_dataset.response_matrix))
        default = _covered(data.split(low_quality_dataset, "LPO", n_splits=2))
        off = _covered(data.split(low_quality_dataset, "LPO", n_splits=2, quality=data.CurveQuality.off()))
        strict = _covered(
            data.split(low_quality_dataset, "LPO", n_splits=2, quality=data.CurveQuality(min_pec50=median))
        )

        np.testing.assert_array_equal(off, np.isfinite(low_quality_dataset.response_matrix))
        assert off.sum() > default.sum()
        assert 0 < strict.sum() < default.sum()

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

    def test_the_splitter_gets_the_unfiltered_dataset_and_the_rule(self, low_quality_dataset: data.Dataset) -> None:
        seen: list[tuple[data.Dataset, data.CurveQuality]] = []

        def plugin(mudataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
            seen.append((mudataset, quality))
            return leave_pair_out(mudataset, n_splits, validation_ratio, random_state, quality=quality)

        modes = frozenset(splitter_registry.modes)
        splitter_registry.register("P3_PLUGIN", "Records what it is given", validation="LPO")(plugin)
        try:
            data.split(low_quality_dataset, "P3_PLUGIN", n_splits=2)
            data.split(low_quality_dataset, "P3_PLUGIN", n_splits=2, quality=data.CurveQuality.off())
        finally:
            splitter_registry.retain_only(modes)

        assert seen[0][0] is low_quality_dataset
        assert seen[0][1] == data.CurveQuality.default()
        assert seen[1][1] == data.CurveQuality.off()

    def test_a_splitter_that_ignores_the_rule_still_records_it(self, low_quality_dataset: data.Dataset) -> None:
        def plugin(mudataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
            return leave_pair_out(mudataset, n_splits, validation_ratio, random_state, quality=data.CurveQuality.off())

        modes = frozenset(splitter_registry.modes)
        splitter_registry.register("P3_IGNORES", "Ignores the rule", validation="LPO")(plugin)
        try:
            folds = data.split(low_quality_dataset, "P3_IGNORES", n_splits=2)
        finally:
            splitter_registry.retain_only(modes)

        failed = ~data.CurveQuality.default().mask(low_quality_dataset)
        assert _covered(folds)[failed].any()
        assert folds[0].metadata["quality"] == data.CurveQuality.default().to_metadata()
