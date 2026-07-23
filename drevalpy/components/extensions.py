"""Load external featurizers, predictors, and zoo entries.

Example::

    from drevalpy.components import load_extensions
    from drevalpy.models.config import ModelConfig

    load_extensions(
        directories=["./my_components"],
        zoo_files=["./my_zoo.yaml"],
    )
    config = ModelConfig.from_spec("myZooEntry")
    model = config.create_model()

Orchestration helpers such as ``build_model_config_from_spec`` and zoo loading live
under ``drevalpy.models``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)


def _extension_module_name(file_path: Path) -> str:
    digest = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    return f"drevalpy_user_extension_{file_path.stem}_{digest}"


def _snapshot_registry_names() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return (
        frozenset(cell_line_featurizer_registry.list_names()),
        frozenset(drug_featurizer_registry.list_names()),
        frozenset(predictor_registry.list_names()),
    )


def _restore_registry_names(
    snapshot: tuple[frozenset[str], frozenset[str], frozenset[str]],
) -> None:
    cell_line_names, drug_names, predictor_names = snapshot
    cell_line_featurizer_registry.retain_only(cell_line_names)
    drug_featurizer_registry.retain_only(drug_names)
    predictor_registry.retain_only(predictor_names)


def load_extension_module(module_name: str) -> None:
    """Import a Python module so its registration decorators run.

    Args:
        module_name: Dotted import path of an installed or ``PYTHONPATH`` module.

    Raises:
        ValueError: If *module_name* is empty.
        ImportError: If the module cannot be imported or registration fails mid-load.
    """
    if not module_name:
        msg = "module_name must be a non-empty string"
        raise ValueError(msg)
    registry_snapshot = _snapshot_registry_names()
    try:
        importlib.import_module(module_name)
    except ImportError:
        _restore_registry_names(registry_snapshot)
        raise
    except Exception as exc:
        _restore_registry_names(registry_snapshot)
        msg = f"Failed to import extension module '{module_name}'"
        raise ImportError(msg) from exc


def load_extension_file(path: Path | str) -> None:
    """Import one Python file so its registration decorators run.

    Args:
        path: Path to a ``.py`` file containing ``@register_*`` decorators.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ImportError: If the file cannot be executed or registration fails mid-load.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        msg = f"Extension file not found: {file_path}"
        raise FileNotFoundError(msg)
    module_name = _extension_module_name(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load extension file: {file_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    registry_snapshot = _snapshot_registry_names()
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.modules.pop(module_name, None)
        _restore_registry_names(registry_snapshot)
        raise
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _restore_registry_names(registry_snapshot)
        msg = f"Failed to import extension file '{file_path}'"
        raise ImportError(msg) from exc


def load_extension_dir(path: Path | str) -> None:
    """Import all ``*.py`` files in a directory in sorted order.

    Args:
        path: Directory containing extension modules (non-recursive).

    Raises:
        FileNotFoundError: If *path* is not a directory.
        ImportError: Propagated from ``load_extension_file`` on failure.
    """
    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        msg = f"Extension directory not found: {dir_path}"
        raise FileNotFoundError(msg)
    py_files = sorted(
        file_path
        for file_path in dir_path.glob("*.py")
        if file_path.name != "__init__.py" and "__pycache__" not in file_path.parts
    )
    for file_path in py_files:
        load_extension_file(file_path)


def load_extensions(
    *,
    modules: list[str] | None = None,
    files: list[Path | str] | None = None,
    directories: list[Path | str] | None = None,
    zoo_files: list[Path | str] | None = None,
) -> None:
    """Load extension modules/files/directories and optional external zoo YAML.

    Args:
        modules: Installed module names to import.
        files: Individual ``.py`` extension files.
        directories: Directories scanned for ``*.py`` extension files.
        zoo_files: External zoo YAML files resolved via ``ModelConfig`` / ``construct_model``.
    """
    for module_name in modules or []:
        load_extension_module(module_name)
    for file_path in files or []:
        load_extension_file(file_path)
    for directory in directories or []:
        load_extension_dir(directory)
    if zoo_files:
        from drevalpy.models.zoo import load_external_zoo_file

        for zoo_path in zoo_files:
            load_external_zoo_file(zoo_path)
