"""Tests for :class:`~drevalpy.registry.splitter._registry.SplitterRegistry`.

Registration mutates registry state, so every test registers into a locally
constructed ``SplitterRegistry``. The module singleton is only read, never
written, which keeps the built-in modes intact for the rest of the suite.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from drevalpy.registry import splitter as splitter_facade
from drevalpy.registry.splitter._registry import SplitterRegistry, splitter_registry
from drevalpy.registry.splitter._validation import SplitValidationError
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks

_SHAPE = (4, 3)
_BUILTIN_MODES = ("LCO", "LDO", "LPO", "LTO")


class _FakeMuDataset:
    """Minimal ``MuDataLike`` stand-in with one tissue per cell line."""

    @property
    def cell_line_ids(self) -> np.ndarray:
        """Row identifiers."""
        return np.array([f"CL_{i}" for i in range(_SHAPE[0])])

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers."""
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        """Fully observed response matrix."""
        return np.ones(_SHAPE)

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """One distinct tissue per cell line."""
        return np.array([f"tissue_{i}" for i in range(_SHAPE[0])])

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        return ["relevance_score", "fold_change"]

    def get_response_layer(self, name: str) -> np.ndarray:
        """Quality layers on which every curve passes the default thresholds."""
        return np.full(_SHAPE, 9.0 if name == "relevance_score" else -2.0)


def _mask(rows: tuple[int, ...]) -> SplitMask:
    array = np.zeros(_SHAPE, dtype=bool)
    array[list(rows), :] = True
    return SplitMask(array)


def _lco_fold() -> SplitMasks:
    return SplitMasks(train=_mask((0, 1)), test=_mask((2, 3)), val=SplitMask(np.zeros(_SHAPE, dtype=bool)))


def _leaking_fold() -> SplitMasks:
    return SplitMasks(train=_mask((0, 1)), test=_mask((1, 2)), val=SplitMask(np.zeros(_SHAPE, dtype=bool)))


@pytest.fixture
def registry() -> SplitterRegistry:
    return SplitterRegistry()


@pytest.fixture
def mudataset() -> _FakeMuDataset:
    return _FakeMuDataset()


@pytest.fixture
def register_valid(registry: SplitterRegistry) -> Callable[..., Any]:
    """Register a one-fold LCO splitter and return the wrapped callable."""

    def _register(mode: str = "MY_LCO", description: str = "one clean fold"):
        @registry.register(mode, description, validation="LCO")
        def splitter(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
            """Return a single leakage-free fold."""
            return [_lco_fold()]

        return splitter

    return _register


def test_a_new_registry_has_no_modes(registry: SplitterRegistry) -> None:
    assert registry.modes == []


def test_register_returns_the_wrapped_splitter(register_valid: Callable[..., Any]) -> None:
    wrapped = register_valid()

    assert callable(wrapped)


def test_register_preserves_the_wrapped_function_name(register_valid: Callable[..., Any]) -> None:
    wrapped = register_valid()

    assert wrapped.__name__ == "splitter"


def test_registered_mode_is_listed(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid()

    assert registry.modes == ["MY_LCO"]


def test_modes_are_sorted(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid(mode="ZZZ")
    register_valid(mode="AAA")

    assert registry.modes == ["AAA", "ZZZ"]


def test_get_returns_the_wrapped_splitter(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    wrapped = register_valid()

    assert registry.get("MY_LCO") is wrapped


def test_get_rejects_an_unknown_mode(registry: SplitterRegistry) -> None:
    with pytest.raises(ValueError, match=r"Unknown split mode 'nope'\. Registered: \[\]"):
        registry.get("nope")


def test_re_registering_a_mode_is_rejected(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid(description="first")

    with pytest.raises(ValueError, match="Splitter mode 'MY_LCO' already registered"):
        register_valid(description="second")


def test_the_rejected_re_registration_leaves_the_original(
    registry: SplitterRegistry, register_valid: Callable[..., Any]
) -> None:
    register_valid(description="first")

    with pytest.raises(ValueError):
        register_valid(description="second")

    assert registry.describe("MY_LCO") == "first"


def test_override_replaces_an_existing_mode(registry: SplitterRegistry) -> None:
    @registry.register("MY_LCO", "first", validation="LCO")
    def first(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
        """Return a single leakage-free fold."""
        return [_lco_fold()]

    @registry.register("MY_LCO", "second", validation="LCO", override=True)
    def second(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
        """Return a single leakage-free fold."""
        return [_lco_fold()]

    assert registry.describe("MY_LCO") == "second"
    assert registry.get("MY_LCO") is second


def test_the_module_facade_forwards_override(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_register(mode, description, validation, *, override=False):
        recorded.update(mode=mode, description=description, validation=validation, override=override)
        return lambda fn: fn

    monkeypatch.setattr(splitter_registry, "register", fake_register)

    splitter_facade.register("X", "d", "LCO", override=True)

    assert recorded == {"mode": "X", "description": "d", "validation": "LCO", "override": True}


def test_describe_returns_the_registered_description(
    registry: SplitterRegistry, register_valid: Callable[..., Any]
) -> None:
    register_valid()

    assert registry.describe("MY_LCO") == "one clean fold"


def test_describe_of_an_unknown_mode_is_empty(registry: SplitterRegistry) -> None:
    assert registry.describe("nope") == ""


def test_resolve_maps_a_mode_name_to_a_splitter(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    wrapped = register_valid()

    assert registry.resolve("MY_LCO") is wrapped


def test_resolve_passes_a_callable_through(registry: SplitterRegistry) -> None:
    def splitter(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
        return []

    assert registry.resolve(splitter) is splitter  # type: ignore[arg-type]


def test_the_wrapper_runs_validation(registry: SplitterRegistry, mudataset: _FakeMuDataset) -> None:
    @registry.register("LEAKY", "leaks a cell line", validation="LCO")
    def leaky(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
        """Return a fold whose train and test share a cell line."""
        return [_leaking_fold()]

    with pytest.raises(SplitValidationError, match="LCO validation failed"):
        leaky(mudataset)


def test_the_wrapper_injects_default_metadata(register_valid: Callable[..., Any], mudataset: _FakeMuDataset) -> None:
    wrapped = register_valid()

    folds = wrapped(mudataset, n_splits=3, validation_ratio=0.2, random_state=7)

    assert folds[0].metadata == {
        "mode": "MY_LCO",
        "fold_index": 0,
        "n_splits": 3,
        "validation_ratio": 0.2,
        "random_state": 7,
    }


def test_the_wrapper_does_not_overwrite_splitter_metadata(
    registry: SplitterRegistry, mudataset: _FakeMuDataset
) -> None:
    @registry.register("ANNOTATED", "sets its own mode", validation="LCO")
    def annotated(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
        """Return a fold that already declares its mode."""
        fold = _lco_fold()
        fold.metadata["mode"] = "custom"
        return [fold]

    folds = annotated(mudataset)

    assert folds[0].metadata["mode"] == "custom"


def test_retain_only_drops_unlisted_modes(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid(mode="KEEP")
    register_valid(mode="DROP")

    registry.retain_only(frozenset({"KEEP"}))

    assert registry.modes == ["KEEP"]


def test_retain_only_forgets_the_dropped_description(
    registry: SplitterRegistry, register_valid: Callable[..., Any]
) -> None:
    register_valid(mode="DROP")

    registry.retain_only(frozenset())

    assert registry.describe("DROP") == ""


def test_to_dataframe_lists_mode_description_and_validation(
    registry: SplitterRegistry, register_valid: Callable[..., Any]
) -> None:
    register_valid()

    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Mode", "Description", "Validation"]
    assert frame.iloc[0].tolist() == ["MY_LCO", "one clean fold", "LCO"]


def test_repr_renders_without_an_index(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid()

    rendered = repr(registry)

    assert "MY_LCO" in rendered
    assert not rendered.startswith("0")


def test_repr_html_emits_a_table(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid()

    assert "<table" in registry._repr_html_()


def test_the_singleton_holds_every_builtin_mode() -> None:
    assert set(_BUILTIN_MODES).issubset(splitter_registry.modes)


def test_module_list_delegates_to_the_singleton() -> None:
    assert splitter_facade.list() == splitter_registry.modes


def test_module_get_delegates_to_the_singleton() -> None:
    assert splitter_facade.get("LPO") is splitter_registry.get("LPO")


def test_module_table_delegates_to_the_singleton() -> None:
    assert list(splitter_facade.table().columns) == ["Mode", "Description", "Validation"]


def test_get_metadata_reports_the_registry_fields(
    registry: SplitterRegistry, register_valid: Callable[..., Any]
) -> None:
    register_valid()

    assert registry.get_metadata("MY_LCO") == {
        "registry": "splitters",
        "name": "MY_LCO",
        "description": "one clean fold",
        "validation": "LCO",
    }


def test_get_metadata_rejects_an_unknown_mode(registry: SplitterRegistry) -> None:
    with pytest.raises(ValueError, match="Unknown split mode 'nope'"):
        registry.get_metadata("nope")


def test_list_metadata_covers_every_mode(registry: SplitterRegistry, register_valid: Callable[..., Any]) -> None:
    register_valid(mode="ZZZ")
    register_valid(mode="AAA")

    assert [row["name"] for row in registry.list_metadata()] == ["AAA", "ZZZ"]


def test_module_metadata_delegates_to_the_singleton() -> None:
    assert splitter_facade.metadata("LPO") == splitter_registry.get_metadata("LPO")
