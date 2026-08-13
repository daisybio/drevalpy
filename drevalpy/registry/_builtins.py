"""Register built-in components by scanning package directories.

No explicit name-to-module mapping required -- modules are discovered by
convention (any .py file without a leading underscore in the component dirs).
"""

from __future__ import annotations

import importlib
import traceback

from drevalpy.log import get_logger

logger = get_logger(__name__)

_SKIPPED_MODULES: dict[str, str] = {}


def get_skipped_builtin_modules() -> dict[str, str]:
    """Return modules that could not be imported during built-in registration.

    :returns: Mapping of dotted module name to the formatted traceback of the
        import failure. Empty when every built-in module imported cleanly.
    """
    return dict(_SKIPPED_MODULES)


def _discover_modules(package_path: str, package_name: str) -> list[str]:
    """Return dotted module names for all public .py files in a package directory.

    :param package_path: Filesystem path of the package directory.
    :param package_name: Dotted package name prefix.
    :returns: List of importable module names.
    """
    from upath import UPath

    pkg_dir = UPath(package_path)
    modules = []
    for py_file in sorted(pkg_dir.glob("*.py")):
        name = py_file.stem
        if name.startswith("_") or name in ("base", "__init__"):
            continue
        modules.append(f"{package_name}.{name}")
    return modules


def _import_modules(module_names: list[str]) -> None:
    """Import each module, logging and skipping failures.

    If a module is already imported, scan it for registered classes and
    re-register them (handles the case where clear() was called).
    Modules that fail on first attempt are retried at the end; modules that
    still fail are recorded in :func:`get_skipped_builtin_modules` and reported
    at WARNING level, because a silently skipped module means a component
    silently disappears from the registries.
    """
    import sys

    deferred: list[str] = []
    for module_name in module_names:
        if module_name in sys.modules:
            _reregister_from_module(sys.modules[module_name])
            continue
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            deferred.append(module_name)
            logger.debug("Deferring %s (import failed on first pass)", module_name)
        else:
            _SKIPPED_MODULES.pop(module_name, None)

    for module_name in deferred:
        if module_name in sys.modules:
            _reregister_from_module(sys.modules[module_name])
            continue
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError) as exc:
            _SKIPPED_MODULES[module_name] = traceback.format_exc()
            logger.warning(
                "Skipping built-in component module %s: its components will be unavailable (%s: %s)",
                module_name,
                type(exc).__name__,
                exc,
            )
        else:
            _SKIPPED_MODULES.pop(module_name, None)


def _reregister_from_module(module) -> None:
    """Re-register classes from an already-imported module."""
    import inspect

    from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry
    from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry
    from drevalpy.registry.predictor._registry import predictor_registry

    for value in vars(module).values():
        if not inspect.isclass(value):
            continue
        registry_name = getattr(value, "registry_name", None)
        if not registry_name:
            continue
        side = getattr(value, "side", None)
        if side == "cell_line":
            cell_line_featurizer_registry.register_existing(registry_name, value)
        elif side == "drug":
            drug_featurizer_registry.register_existing(registry_name, value)
        elif hasattr(value, "cell_line_contract"):
            predictor_registry.register_existing(registry_name, value)
        elif hasattr(value, "contract"):
            if side == "drug":
                drug_featurizer_registry.register_existing(registry_name, value)
            else:
                cell_line_featurizer_registry.register_existing(registry_name, value)


def _discover_literature_predictor_modules() -> list[str]:
    """Find all literature/<name>/predictor.py modules."""
    from upath import UPath

    import drevalpy.components.predictors as pkg

    lit_dir = UPath(pkg.__path__[0]) / "literature"
    modules = []
    if lit_dir.is_dir():
        for sub_dir in sorted(lit_dir.iterdir()):
            if sub_dir.is_dir() and not sub_dir.name.startswith("_"):
                modules.append(f"{pkg.__name__}.literature.{sub_dir.name}.predictor")
    # Also the neural_network predictor
    modules.append(f"{pkg.__name__}.neural_network.predictor")
    return modules


def _cell_line_featurizer_modules() -> list[str]:
    import drevalpy.components.featurizers.cell_line as pkg

    return _discover_modules(pkg.__path__[0], pkg.__name__)


def _drug_featurizer_modules() -> list[str]:
    import drevalpy.components.featurizers.drug as pkg

    return _discover_modules(pkg.__path__[0], pkg.__name__)


def _native_predictor_modules() -> list[str]:
    """Discover non-literature predictor modules."""
    import drevalpy.components.predictors as pkg
    import drevalpy.components.predictors.naive as naive_pkg

    top_level = _discover_modules(pkg.__path__[0], pkg.__name__)
    naive = _discover_modules(naive_pkg.__path__[0], naive_pkg.__name__)
    return top_level + naive


def register_native_components() -> None:
    """Register dependency-light native components (featurizers + non-literature predictors)."""
    _import_modules(_cell_line_featurizer_modules())
    _import_modules(_drug_featurizer_modules())
    _import_modules(_native_predictor_modules())


def register_literature_components() -> None:
    """Register literature predictors and their neural dependencies."""
    _import_modules(_discover_literature_predictor_modules())


def _register_builtin_splitters() -> None:
    """Import splitter modules to trigger their @register decorators."""
    import sys

    from drevalpy.registry.splitter._registry import splitter_registry

    if splitter_registry.modes:
        return

    splitter_modules = [
        "drevalpy.data.splitters.lco",
        "drevalpy.data.splitters.ldo",
        "drevalpy.data.splitters.lpo",
        "drevalpy.data.splitters.lto",
    ]
    # Evict every module before importing any of them: importing the first one runs
    # the package __init__, which imports all four. Interleaving eviction and import
    # would re-execute the later modules and register their modes twice.
    for mod_name in splitter_modules:
        sys.modules.pop(mod_name, None)
    for mod_name in splitter_modules:
        importlib.import_module(mod_name)


def _register_builtin_visualizations() -> None:
    """Import visualization plot modules to trigger their @register decorators."""
    import sys

    from drevalpy.registry.visualization._registry import visualization_registry

    if visualization_registry.names:
        return

    viz_pkg = "drevalpy.visualization.plots"
    modules_to_reload = [key for key in sys.modules if key.startswith(viz_pkg)]
    for mod_name in modules_to_reload:
        del sys.modules[mod_name]
    importlib.import_module(viz_pkg)


def register_builtin_components() -> None:
    """Register every built-in component by scanning package directories.

    Safe to call multiple times. Detects cleared registries and re-populates.
    """
    from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry
    from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry
    from drevalpy.registry.predictor._registry import predictor_registry

    all_populated = (
        predictor_registry.list_names()
        and cell_line_featurizer_registry.list_names()
        and drug_featurizer_registry.list_names()
    )
    if all_populated:
        return

    logger.debug("Registering built-in components...")
    register_native_components()
    register_literature_components()
    _register_builtin_splitters()
    _register_builtin_visualizations()
    logger.debug("Built-in registration complete.")


def reregister_builtin_components() -> None:
    """Force re-registration of all builtins (used after `clear()` in tests)."""
    register_builtin_components()


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------


def ensure_predictor_registered(name: str) -> None:
    """No-op with eager loading -- all predictors are registered at startup."""


def ensure_cell_line_featurizer_registered(name: str) -> None:
    """No-op with eager loading -- all featurizers are registered at startup."""


def ensure_drug_featurizer_registered(name: str) -> None:
    """No-op with eager loading -- all featurizers are registered at startup."""


def is_known_builtin_predictor(name: str) -> bool:
    """Return whether *name* is a registered predictor."""
    from drevalpy.registry.predictor._registry import predictor_registry

    return name in predictor_registry.list_names()


def is_known_builtin_cell_line_featurizer(name: str) -> bool:
    """Return whether *name* is a registered cell-line featurizer."""
    from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry

    return name in cell_line_featurizer_registry.list_names()


def is_known_builtin_drug_featurizer(name: str) -> bool:
    """Return whether *name* is a registered drug featurizer."""
    from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry

    return name in drug_featurizer_registry.list_names()


# ---------------------------------------------------------------------------
# Lazy built-in name sets (computed on first access)
# ---------------------------------------------------------------------------


def _get_builtin_cell_line_featurizer_names() -> frozenset[str]:
    from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry

    return frozenset(cell_line_featurizer_registry.list_names())


def _get_builtin_drug_featurizer_names() -> frozenset[str]:
    from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry

    return frozenset(drug_featurizer_registry.list_names())


def _get_builtin_predictor_names() -> frozenset[str]:
    from drevalpy.registry.predictor._registry import predictor_registry

    return frozenset(predictor_registry.list_names())


class _LazyFrozenset:
    """Module-level lazy frozenset that computes on first access."""

    def __init__(self, getter):
        self._getter = getter
        self._value = None

    def _resolve(self):
        if self._value is None:
            self._value = self._getter()
        return self._value

    def __iter__(self):
        return iter(self._resolve())

    def __contains__(self, item):
        return item in self._resolve()

    def __len__(self):
        return len(self._resolve())

    def __eq__(self, other):
        return self._resolve() == other

    def __repr__(self):
        return repr(self._resolve())


BUILTIN_CELL_LINE_FEATURIZER_NAMES = _LazyFrozenset(_get_builtin_cell_line_featurizer_names)
BUILTIN_DRUG_FEATURIZER_NAMES = _LazyFrozenset(_get_builtin_drug_featurizer_names)
BUILTIN_PREDICTOR_NAMES = _LazyFrozenset(_get_builtin_predictor_names)
