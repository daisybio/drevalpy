"""The splitter registry: registration, validation wrapping and aliases.

Tests register into the module registry; ``tests/conftest.py`` restores it after each test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.registry.splitter import canonical, get, splitter_registry
from drevalpy.registry.splitter import register as register_splitter
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


@pytest.fixture
def dataset() -> _FakeMuDataset:
    return _FakeMuDataset()


@pytest.fixture
def register() -> Callable[..., Any]:
    def _register(mode: str = "MY_LSO", description: str = "one clean fold", **kwargs: Any):
        def splitter(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
            return [_fold()]

        return register_splitter(mode, description, validation="LSO", **kwargs)(splitter)

    return _register


def test_an_unknown_mode_raises() -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown split mode 'nope'\. Available: LDO"):
        get("nope")
    with pytest.raises(UnknownNameError, match="Unknown split mode 'nope'"):
        splitter_registry.metadata("nope")


def test_register_wraps_keeps_registration_order_and_resolves(register: Callable) -> None:
    wrapped = register(mode="ZZZ")
    register(mode="AAA")

    assert wrapped.__name__ == "splitter"
    assert splitter_registry.names()[-2:] == ("ZZZ", "AAA")
    assert get("ZZZ") is wrapped
    assert dict(splitter_registry.metadata("ZZZ")) == {
        "description": "one clean fold",
        "validation": "LSO",
        "aliases": frozenset(),
    }


def test_re_registering_is_rejected_unless_replaced(register: Callable) -> None:
    register(description="first")
    with pytest.raises(ValueError, match="'MY_LSO' already registered"):
        register(description="second")
    assert splitter_registry.metadata("MY_LSO")["description"] == "first"

    second = register(description="second", replace=True)
    assert splitter_registry.metadata("MY_LSO")["description"] == "second"
    assert get("MY_LSO") is second


def test_quality_parameter_is_required_unless_kwargs() -> None:
    def old_style(dataset, n_splits=5, validation_ratio=0.1, random_state=42):
        return [_fold()]

    with pytest.raises(InvalidInputError, match="must accept a keyword parameter 'quality"):
        register_splitter("OLD_STYLE", "no quality", validation="LSO")(old_style)
    assert "OLD_STYLE" not in splitter_registry

    register_splitter("KWARGS", "takes **kwargs", validation="LSO")(lambda *args, **kwargs: [_fold()])
    assert "KWARGS" in splitter_registry


@pytest.mark.parametrize("validation", ["LSO", "LCO"])
def test_the_wrapper_runs_validation_and_stores_legacy_names_canonically(
    dataset: _FakeMuDataset, validation: str
) -> None:
    leaky = register_splitter("LEAKY", "leaks a sample", validation=validation)(  # type: ignore[arg-type]
        lambda dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality: [_fold((1, 2))]
    )

    assert splitter_registry.metadata("LEAKY")["validation"] == "LSO"
    with pytest.raises(SplitValidationError, match="LSO validation failed"):
        leaky(dataset, quality=QUALITY)


def test_the_wrapper_passes_quality_and_manages_metadata(dataset: _FakeMuDataset) -> None:
    seen: list[CurveQuality] = []

    def records(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
        seen.append(quality)
        fold = _fold()
        if n_splits == 9:
            fold.metadata["split_mode"] = "custom"
        return [fold]

    wrapped = register_splitter("RECORDS", "records the rule", validation="LSO")(records)
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


def test_listing_with_aliases(register: Callable) -> None:
    register(aliases=("OLD", "ANCIENT"))

    frame = splitter_registry.to_dataframe()
    assert list(frame.columns) == ["Name", "Description", "Validation", "Aliases"]
    assert frame.iloc[-1].tolist() == ["MY_LSO", "one clean fold", "LSO", "ANCIENT, OLD"]
    assert "ANCIENT, OLD" in repr(splitter_registry)


def test_the_builtin_lco_alias_resolves_to_lso() -> None:
    assert {"LSO", "LDO", "LPO", "LTO", "LStO"} <= set(splitter_registry.names())
    assert canonical("LCO") == "LSO"
    assert get("LCO") is get("LSO")
    assert "LCO" not in splitter_registry
    assert splitter_registry.metadata("LSO")["aliases"] == frozenset({"LCO"})


def test_an_alias_resolves_to_its_mode_without_being_listed(register: Callable, dataset: _FakeMuDataset) -> None:
    wrapped = register(aliases=("OLD", "ANCIENT"))

    assert get("OLD") is wrapped
    assert canonical("OLD") == "MY_LSO"
    assert canonical("MY_LSO") == "MY_LSO"
    assert canonical("nope") == "nope"
    assert "OLD" not in splitter_registry
    assert get("OLD")(dataset, quality=QUALITY)[0].metadata["split_mode"] == "MY_LSO"


def test_a_snapshot_restores_modes_with_their_aliases(register: Callable) -> None:
    snapshot = splitter_registry.snapshot()
    register(aliases=("OLD",))

    splitter_registry.restore(snapshot)

    assert "MY_LSO" not in splitter_registry
    assert canonical("OLD") == "OLD"


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
def test_alias_and_mode_names_cannot_collide(register: Callable, first, second, match) -> None:
    register(**first)
    before = (splitter_registry.names(), canonical("OLD"))
    with pytest.raises(ValueError, match=match):
        register(**second)
    assert (splitter_registry.names(), canonical("OLD")) == before


def test_replace_replaces_the_aliases_of_the_mode_but_may_keep_one(register: Callable) -> None:
    register(aliases=("OLD", "KEPT"))

    second = register(aliases=("NEWER", "KEPT"), replace=True)

    assert get("NEWER") is second
    assert get("KEPT") is second
    assert canonical("OLD") == "OLD"
    assert splitter_registry.metadata("MY_LSO")["aliases"] == frozenset({"KEPT", "NEWER"})
