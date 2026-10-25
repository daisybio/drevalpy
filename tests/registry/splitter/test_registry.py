"""SplitterRegistry; tests register into a local registry and only read the module singleton."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.registry import splitter as splitter_facade
from drevalpy.registry.splitter._registry import SplitterRegistry, splitter_registry
from drevalpy.registry.splitter._validation import SplitValidationError
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks

_SHAPE = (4, 3)
QUALITY = CurveQuality.default()


class _FakeMuDataset:
    @property
    def sample_ids(self) -> np.ndarray:
        return np.array([f"CL_{i}" for i in range(_SHAPE[0])])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.ones(_SHAPE)

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array([f"tissue_{i}" for i in range(_SHAPE[0])])

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        if column != "tissue":
            raise KeyError(f"Sample column '{column}' not found. Available: ['tissue']")
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        return ["relevance_score", "fold_change"]

    def get_response_layer(self, name: str) -> np.ndarray:
        return np.full(_SHAPE, 9.0 if name == "relevance_score" else -2.0)


def _mask(rows: tuple[int, ...]) -> SplitMask:
    array = np.zeros(_SHAPE, dtype=bool)
    array[list(rows), :] = True
    return SplitMask(array)


def _fold(test_rows: tuple[int, ...] = (2, 3)) -> SplitMasks:
    return SplitMasks(train=_mask((0, 1)), test=_mask(test_rows), val=SplitMask(np.zeros(_SHAPE, dtype=bool)))


def _splitter(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
    return [_fold()]


@pytest.fixture
def registry() -> SplitterRegistry:
    return SplitterRegistry()


@pytest.fixture
def dataset() -> _FakeMuDataset:
    return _FakeMuDataset()


@pytest.fixture
def register(registry: SplitterRegistry) -> Callable[..., Any]:
    def _register(mode: str = "MY_LSO", description: str = "one clean fold", **kwargs: Any):
        def splitter(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
            return [_fold()]

        return registry.register(mode, description, validation="LSO", **kwargs)(splitter)

    return _register


def test_unknown_modes_raise_and_a_new_registry_is_empty(registry: SplitterRegistry) -> None:
    assert registry.modes == []
    assert registry.describe("nope") == ""
    with pytest.raises(UnknownNameError, match=r"Unknown split mode 'nope'\. Registered: \[\]"):
        registry.get("nope")
    with pytest.raises(ValueError, match="Unknown split mode 'nope'"):
        registry.get_metadata("nope")


def test_register_wraps_lists_sorted_and_resolves(registry: SplitterRegistry, register: Callable) -> None:
    wrapped = register(mode="ZZZ")
    register(mode="AAA")

    assert wrapped.__name__ == "splitter"
    assert registry.modes == ["AAA", "ZZZ"]
    assert registry.get("ZZZ") is wrapped
    assert registry.resolve("ZZZ") is wrapped
    assert registry.resolve(_splitter) is _splitter  # type: ignore[arg-type]
    assert registry.describe("ZZZ") == "one clean fold"
    assert [row["name"] for row in registry.list_metadata()] == ["AAA", "ZZZ"]
    assert registry.get_metadata("ZZZ") == {
        "registry": "splitters",
        "name": "ZZZ",
        "description": "one clean fold",
        "validation": "LSO",
        "aliases": [],
    }


def test_re_registering_is_rejected_unless_overridden(registry: SplitterRegistry, register: Callable) -> None:
    register(description="first")
    with pytest.raises(ValueError, match="Splitter mode 'MY_LSO' already registered"):
        register(description="second")
    assert registry.describe("MY_LSO") == "first"

    second = register(description="second", override=True)
    assert registry.describe("MY_LSO") == "second"
    assert registry.get("MY_LSO") is second


def test_quality_parameter_is_required_unless_kwargs(registry: SplitterRegistry) -> None:
    def old_style(dataset, n_splits=5, validation_ratio=0.1, random_state=42):
        return [_fold()]

    with pytest.raises(InvalidInputError, match="must accept a keyword parameter 'quality"):
        registry.register("OLD_STYLE", "no quality", validation="LSO")(old_style)
    assert "OLD_STYLE" not in registry.modes

    registry.register("KWARGS", "takes **kwargs", validation="LSO")(lambda *args, **kwargs: [_fold()])
    assert "KWARGS" in registry.modes


def test_retain_only_drops_unlisted_modes_with_their_aliases(registry: SplitterRegistry, register: Callable) -> None:
    register(mode="KEEP", aliases=("K",))
    register(mode="DROP", aliases=("D",))

    registry.retain_only(frozenset({"KEEP"}))

    assert registry.modes == ["KEEP"]
    assert registry.describe("DROP") == ""
    assert registry.canonical("K") == "KEEP"
    assert registry.canonical("D") == "D"
    with pytest.raises(ValueError, match="Unknown split mode 'D'"):
        registry.get("D")


@pytest.mark.parametrize("validation", ["LSO", "LCO"])
def test_the_wrapper_runs_validation_and_stores_legacy_names_canonically(
    registry: SplitterRegistry, dataset: _FakeMuDataset, validation: str
) -> None:
    leaky = registry.register("LEAKY", "leaks a sample", validation=validation)(  # type: ignore[arg-type]
        lambda dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality: [_fold((1, 2))]
    )

    assert registry.get_metadata("LEAKY")["validation"] == "LSO"
    with pytest.raises(SplitValidationError, match="LSO validation failed"):
        leaky(dataset, quality=QUALITY)


def test_the_wrapper_passes_quality_and_manages_metadata(registry: SplitterRegistry, dataset: _FakeMuDataset) -> None:
    seen: list[CurveQuality] = []

    def records(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
        seen.append(quality)
        fold = _fold()
        if n_splits == 9:
            fold.metadata["split_mode"] = "custom"
        return [fold]

    wrapped = registry.register("RECORDS", "records the rule", validation="LSO")(records)
    folds = wrapped(dataset, n_splits=3, validation_ratio=0.2, random_state=7, quality=CurveQuality.off())

    assert seen == [CurveQuality.off()]
    assert folds[0].metadata == {
        "split_mode": "RECORDS",
        "fold_index": 0,
        "n_splits": 3,
        "validation_ratio": 0.2,
        "random_state": 7,
    }
    assert wrapped(dataset, n_splits=9, quality=QUALITY)[0].metadata["split_mode"] == "custom"


def test_listing_with_aliases(registry: SplitterRegistry, register: Callable) -> None:
    register(aliases=("OLD", "ANCIENT"))

    frame = registry.to_dataframe()
    assert list(frame.columns) == ["Mode", "Description", "Validation", "Aliases"]
    assert frame.iloc[0].tolist() == ["MY_LSO", "one clean fold", "LSO", "ANCIENT, OLD"]
    assert "ANCIENT, OLD" in repr(registry)


def test_facade_delegates_to_the_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    assert {"LSO", "LDO", "LPO", "LTO"}.issubset(splitter_registry.modes)
    assert splitter_facade.list() == splitter_registry.modes
    assert splitter_facade.get("LPO") is splitter_registry.get("LPO")
    assert splitter_facade.metadata("LPO") == splitter_registry.get_metadata("LPO")
    assert list(splitter_facade.table().columns) == ["Mode", "Description", "Validation", "Aliases"]

    assert splitter_registry.canonical("LCO") == "LSO"
    assert splitter_registry.get("LCO") is splitter_registry.get("LSO")
    assert "LCO" not in splitter_registry.modes
    assert splitter_registry.get_metadata("LCO")["aliases"] == ["LCO"]
    assert splitter_facade.metadata("LCO")["name"] == "LSO"

    recorded: dict[str, Any] = {}

    def fake_register(mode, description, validation, *, override=False, aliases=()):
        recorded.update(mode=mode, validation=validation, override=override, aliases=aliases)
        return lambda fn: fn

    monkeypatch.setattr(splitter_registry, "register", fake_register)
    splitter_facade.register("X", "d", "LSO", override=True, aliases=("Y",))
    assert recorded == {"mode": "X", "validation": "LSO", "override": True, "aliases": ("Y",)}


def test_an_alias_resolves_to_its_mode_without_being_listed(
    registry: SplitterRegistry, register: Callable, dataset: _FakeMuDataset
) -> None:
    wrapped = register(aliases=("OLD", "ANCIENT"))

    assert registry.get("OLD") is wrapped
    assert registry.resolve("OLD") is wrapped
    assert registry.canonical("OLD") == "MY_LSO"
    assert registry.canonical("MY_LSO") == "MY_LSO"
    assert registry.canonical("nope") == "nope"
    assert registry.modes == ["MY_LSO"]
    assert registry.get("OLD")(dataset, quality=QUALITY)[0].metadata["split_mode"] == "MY_LSO"
    assert registry.describe("OLD") == "one clean fold"
    assert registry.get_metadata("OLD") == registry.get_metadata("MY_LSO")
    assert registry.get_metadata("OLD")["name"] == "MY_LSO"
    assert registry.get_metadata("OLD")["aliases"] == ["ANCIENT", "OLD"]


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        ({"mode": "TAKEN"}, {"aliases": ("TAKEN",)}, "conflicts with the registered mode 'TAKEN'"),
        ({"mode": "MY_LSO"}, {"mode": "OTHER", "aliases": ("MY_LSO",)}, "conflicts with the registered mode"),
        (
            {"mode": "FIRST", "aliases": ("OLD",)},
            {"mode": "SECOND", "aliases": ("OLD",)},
            "alias 'OLD' already registered for mode 'FIRST'",
        ),
        ({"aliases": ("OLD",)}, {"mode": "OLD"}, "mode 'OLD' conflicts with an alias of 'MY_LSO'"),
    ],
)
def test_alias_and_mode_names_cannot_collide(
    registry: SplitterRegistry, register: Callable, first, second, match
) -> None:
    register(**first)
    before = (registry.modes, registry.canonical("OLD"))
    with pytest.raises(ValueError, match=match):
        register(**second)
    assert (registry.modes, registry.canonical("OLD")) == before


def test_override_replaces_the_aliases_of_the_mode_but_may_keep_one(
    registry: SplitterRegistry, register: Callable
) -> None:
    register(aliases=("OLD", "KEPT"))

    second = register(aliases=("NEWER", "KEPT"), override=True)

    assert registry.get("NEWER") is second
    assert registry.get("KEPT") is second
    assert registry.canonical("OLD") == "OLD"
    assert registry.aliases_of("MY_LSO") == ["KEPT", "NEWER"]
