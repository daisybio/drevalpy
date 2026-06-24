"""Load external featurizers, predictors, and zoo entries.

Example::

    from drevalpy.components import load_extensions, ModelConfig

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

import importlib
import importlib.util
import sys
from pathlib import Path


def load_extension_module(module_name: str) -> None:
    """Import a Python module so its registration decorators run."""
    if not module_name:
        msg = "module_name must be a non-empty string"
        raise ValueError(msg)
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"Failed to import extension module '{module_name}'"
        raise ImportError(msg) from exc


def load_extension_file(path: Path | str) -> None:
    """Import one Python file so its registration decorators run."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        msg = f"Extension file not found: {file_path}"
        raise FileNotFoundError(msg)
    module_name = f"drevalpy_user_extension_{file_path.stem}_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load extension file: {file_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        msg = f"Failed to import extension file '{file_path}'"
        raise ImportError(msg) from exc


def load_extension_dir(path: Path | str) -> None:
    """Import all ``*.py`` files in a directory in sorted order."""
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
    """Load extension modules/files/directories and optional external zoo YAML."""
    from drevalpy.components.register_builtins import ensure_components_registered

    ensure_components_registered()
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
