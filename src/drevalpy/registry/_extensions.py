"""Load external featurizers, predictors, splitters, visualizations, and zoo entries.

Example::

    from drevalpy.registry._extensions import load_extensions

    load_extensions(
        directories=["./my_components"],
        zoo_files=["./my_zoo.yaml"],
    )

When ``directories`` is omitted, ``load_extensions`` falls back to the
``DREVALPY_EXTENSIONS_DIR`` environment variable: one directory, or several joined
with :data:`os.pathsep` (``:`` on POSIX, ``;`` on Windows). Passing ``directories``
explicitly, even as an empty list, ignores the variable.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from typing import Any

from upath import UPath as Path

#: Environment variable naming extension directories, joined with :data:`os.pathsep`.
EXTENSIONS_DIR_ENV_VAR = "DREVALPY_EXTENSIONS_DIR"

#: Module name -> registry names the file added when it was loaded, so a repeat load can be skipped.
_LOADED_FILES: dict[str, dict[str, frozenset[str]]] = {}


def _extension_module_name(file_path: Path) -> str:
    digest = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    return f"drevalpy_user_extension_{file_path.stem}_{digest}"


def _get_registries() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Lazily import all registry singletons to avoid circular imports."""
    from drevalpy.registry.dataset._registry import dataset_registry
    from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry
    from drevalpy.registry.predictor._registry import predictor_registry
    from drevalpy.registry.sample_featurizer._registry import sample_featurizer_registry
    from drevalpy.registry.splitter._registry import splitter_registry
    from drevalpy.registry.visualization._registry import visualization_registry

    return (
        predictor_registry,
        sample_featurizer_registry,
        drug_featurizer_registry,
        splitter_registry,
        visualization_registry,
        dataset_registry,
    )


def _snapshot_all_registries() -> dict[str, frozenset[str]]:
    """Capture current state of all in-memory registries for rollback."""
    (
        predictor_registry,
        sample_featurizer_registry,
        drug_featurizer_registry,
        splitter_registry,
        visualization_registry,
        _dataset_registry,
    ) = _get_registries()

    return {
        "predictor": frozenset(predictor_registry.list_names()),
        "sample_featurizer": frozenset(sample_featurizer_registry.list_names()),
        "drug_featurizer": frozenset(drug_featurizer_registry.list_names()),
        "splitter": frozenset(splitter_registry.modes),
        "visualization": frozenset(visualization_registry.names),
    }


def _restore_all_registries(snapshot: dict[str, frozenset[str]]) -> None:
    """Roll back all in-memory registries to a prior snapshot."""
    (
        predictor_registry,
        sample_featurizer_registry,
        drug_featurizer_registry,
        splitter_registry,
        visualization_registry,
        _dataset_registry,
    ) = _get_registries()

    predictor_registry.retain_only(snapshot["predictor"])
    sample_featurizer_registry.retain_only(snapshot["sample_featurizer"])
    drug_featurizer_registry.retain_only(snapshot["drug_featurizer"])
    splitter_registry.retain_only(snapshot["splitter"])
    visualization_registry.retain_only(snapshot["visualization"])


def _added_since(snapshot: dict[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    now = _snapshot_all_registries()
    return {kind: now[kind] - snapshot[kind] for kind in now}


def _is_still_registered(added: dict[str, frozenset[str]]) -> bool:
    now = _snapshot_all_registries()
    return all(names <= now[kind] for kind, names in added.items())


def load_extension_module(module_name: str) -> None:
    """Import a Python module so its registration decorators run.

    :param module_name: Dotted import path of an installed or ``PYTHONPATH`` module.
    :raises ValueError: If *module_name* is empty.
    :raises ImportError: If the module cannot be imported or registration fails mid-load.
    """
    if not module_name:
        msg = "module_name must be a non-empty string"
        raise ValueError(msg)
    snapshot = _snapshot_all_registries()
    try:
        importlib.import_module(module_name)
    except ImportError:
        _restore_all_registries(snapshot)
        raise
    except Exception as exc:
        _restore_all_registries(snapshot)
        msg = f"Failed to import extension module '{module_name}'"
        raise ImportError(msg) from exc


def load_extension_file(path: Path | str) -> None:
    """Import one Python file so its registration decorators run.

    Loading the same file twice is harmless: while everything the first load registered is
    still registered, the second call does nothing.

    :param path: Path to a ``.py`` file containing ``@register_*`` decorators.
    :raises FileNotFoundError: If *path* does not exist.
    :raises ImportError: If the file cannot be executed or registration fails mid-load.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        msg = f"Extension file not found: {file_path}"
        raise FileNotFoundError(msg)
    module_name = _extension_module_name(file_path)
    previous = _LOADED_FILES.get(module_name)
    if previous is not None and module_name in sys.modules and _is_still_registered(previous):
        return
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load extension file: {file_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    snapshot = _snapshot_all_registries()
    try:
        spec.loader.exec_module(module)
        _LOADED_FILES[module_name] = _added_since(snapshot)
    except ImportError:
        sys.modules.pop(module_name, None)
        _restore_all_registries(snapshot)
        raise
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _restore_all_registries(snapshot)
        msg = f"Failed to import extension file '{file_path}'"
        raise ImportError(msg) from exc


def _load_yaml_extension(path: Path) -> None:
    """Inspect YAML top-level keys: dataset config vs zoo entry."""
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return

    if "sources" in data or "datasets" in data:
        _register_datasets_from_yaml(data)
    else:
        from drevalpy.models.zoo import load_external_zoo_file

        load_external_zoo_file(path)


def _register_datasets_from_yaml(data: dict) -> None:
    """Register sources and datasets from a parsed YAML dict."""
    from drevalpy.registry.dataset._registry import dataset_registry

    for name, source_info in (data.get("sources") or {}).items():
        dataset_registry.register_source(
            name,
            base_url=source_info["url"],
            storage_options=source_info.get("storage_options"),
        )
    for name, ds_info in (data.get("datasets") or {}).items():
        dataset_registry.register_dataset(
            name,
            source=ds_info["source"],
            file=ds_info["file"],
        )


def load_extension_dir(path: Path | str) -> None:
    """Import all ``*.py`` files and load all ``*.yaml`` files from a directory.

    Python files are imported in sorted order (skipping ``__init__.py``).
    YAML files are inspected and dispatched to either the dataset registry
    or the zoo loader. Loading the same directory twice is harmless.

    :param path: Directory containing extension modules (non-recursive).
    :raises FileNotFoundError: If *path* is not a directory.
    """
    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        msg = f"Extension directory not found: {dir_path}"
        raise FileNotFoundError(msg)

    snapshot = _snapshot_all_registries()
    try:
        py_files = sorted(
            file_path
            for file_path in dir_path.glob("*.py")
            if file_path.name != "__init__.py" and "__pycache__" not in file_path.parts
        )
        for file_path in py_files:
            load_extension_file(file_path)

        yaml_files = sorted(dir_path.glob("*.yaml"))
        for yaml_file in yaml_files:
            _load_yaml_extension(yaml_file)
    except Exception:
        _restore_all_registries(snapshot)
        raise


def extension_dirs_from_env() -> list[str]:
    """Return the directories named by ``DREVALPY_EXTENSIONS_DIR``.

    The value is split on :data:`os.pathsep`; empty entries are dropped.

    :returns: Directories in the order written, or an empty list when the variable is unset or empty.
    """
    raw = os.environ.get(EXTENSIONS_DIR_ENV_VAR, "")
    return [entry for entry in raw.split(os.pathsep) if entry]


def load_extensions(
    *,
    modules: list[str] | None = None,
    files: list[Path | str] | None = None,
    directories: list[Path | str] | None = None,
    zoo_files: list[Path | str] | None = None,
) -> None:
    """Load extension modules/files/directories and optional external zoo YAML.

    :param modules: Installed module names to import.
    :param files: Individual ``.py`` extension files.
    :param directories: Directories scanned for ``*.py`` and ``*.yaml`` extension files. ``None`` means
        the directories listed in ``DREVALPY_EXTENSIONS_DIR`` (split on :data:`os.pathsep`), if set; any
        explicit list, including ``[]``, means the variable is not read.
    :param zoo_files: External zoo YAML files resolved into ``ModelSpec`` recipes.
    """
    for module_name in modules or []:
        load_extension_module(module_name)
    for file_path in files or []:
        load_extension_file(file_path)
    if directories is None:
        directories = extension_dirs_from_env()
    for directory in directories:
        load_extension_dir(directory)
    if zoo_files:
        from drevalpy.models.zoo import load_external_zoo_file

        for zoo_path in zoo_files:
            load_external_zoo_file(zoo_path)
