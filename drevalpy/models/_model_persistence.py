"""Versioned persistence for concrete DRPModel instances.

Checkpoints are a single ZIP archive written atomically to an archive file path.
Callers must only load artifacts they created with ``save_model`` in the same
drevalpy version family.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from drevalpy.models.config import ModelConfig

if TYPE_CHECKING:
    from drevalpy.models.drp_model import DRPModel

FORMAT_NAME = "drevalpy-model"
FORMAT_VERSION = 1
PAYLOAD_MEMBER = "payload.joblib"


class ModelCheckpointError(Exception):
    """Base error for DRPModel checkpoint problems."""


class UnsupportedCheckpointFormatError(ModelCheckpointError, ValueError):
    """Raised when checkpoint format or version is not supported."""


class CorruptedCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint payload structure or content is invalid."""


class IncompatibleModelCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint model identity does not match the loader class."""


def _as_path(path: str | Path) -> Path:
    """Normalize a user path to ``Path``, rejecting trailing separators."""
    if isinstance(path, str) and path.endswith(("/", "\\")):
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)
    return Path(path)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Return the archive file path, appending ``.zip`` when missing.

    Args:
        path: Checkpoint archive path; ``.zip`` is appended when missing.

    Returns:
        Normalized archive ``Path``.

    Raises:
        ValueError: If *path* ends with a directory separator.
    """
    target = _as_path(path)
    if target.name.lower().endswith(".zip"):
        return target
    return target.with_name(f"{target.name}.zip")


def _reject_directory_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)


def _write_archive_atomically(archive_path: Path, payload: dict[str, Any]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{archive_path.name}.", suffix=".tmp", dir=archive_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        buffer = io.BytesIO()
        joblib.dump(payload, buffer)
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
        os.replace(tmp_path, archive_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_payload_from_archive(archive_path: Path) -> Any:
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            try:
                info = archive.getinfo(PAYLOAD_MEMBER)
            except KeyError as exc:
                raise CorruptedCheckpointError(
                    f"checkpoint archive {archive_path} is missing {PAYLOAD_MEMBER!r}"
                ) from exc
            with archive.open(info) as handle:
                return joblib.load(handle)
    except zipfile.BadZipFile as exc:
        raise CorruptedCheckpointError(f"checkpoint archive {archive_path} is not a valid zip file") from exc
    except CorruptedCheckpointError:
        raise
    except Exception as exc:
        raise CorruptedCheckpointError(f"Failed to deserialize checkpoint {archive_path}: {exc}") from exc


def save_model(model: DRPModel, path: str | Path) -> None:
    """Save model identity, config, and component state as one ZIP archive.

    Args:
        model: Trained ``DRPModel`` instance to persist.
        path: Archive file path; ``.zip`` is appended when missing.

    Raises:
        RuntimeError: If the model is not trained or lacks a ``ModelConfig``.
    """
    stack = model._stack
    if stack is None or not stack.is_fitted():
        raise RuntimeError("Cannot save: component stack is not trained")
    config = model._resolved_model_config
    if config is None:
        raise RuntimeError("Cannot save a model without its ModelConfig")
    target = _as_path(path)
    _reject_directory_path(target)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model.get_model_name(),
        "config": config.model_dump(mode="json"),
        "state": stack.component_state(),
    }
    _write_archive_atomically(resolve_checkpoint_path(target), payload)


def load_model_payload(path: str | Path) -> tuple[str, ModelConfig, dict[str, object]]:
    """Load and validate a ``DRPModel`` checkpoint payload from an archive path.

    Args:
        path: Archive file path; ``.zip`` is appended when missing.

    Returns:
        Tuple of ``(model_name, config, component_state)``.

    Raises:
        FileNotFoundError: If the archive does not exist.
        UnsupportedCheckpointFormatError: If the format or version is unsupported.
        CorruptedCheckpointError: If the payload structure is invalid.
    """
    target = _as_path(path)
    _reject_directory_path(target)
    archive_path = resolve_checkpoint_path(target)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {archive_path}")

    payload = _read_payload_from_archive(archive_path)
    if not isinstance(payload, dict):
        raise CorruptedCheckpointError("checkpoint payload is not a mapping")
    if payload.get("format") != FORMAT_NAME or payload.get("version") != FORMAT_VERSION:
        raise UnsupportedCheckpointFormatError(
            f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}"
        )
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise CorruptedCheckpointError("checkpoint model_name is missing or invalid")
    try:
        config = ModelConfig.model_validate(payload["config"])
    except Exception as exc:
        raise CorruptedCheckpointError("checkpoint config is invalid") from exc
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptedCheckpointError("checkpoint state is not a mapping")
    return model_name, config, state


def load_model(path: str | Path) -> DRPModel:
    """Reconstruct a fitted ``DRPModel`` from a checkpoint archive path.

    Reads the stored model name and ``ModelConfig``, builds the matching class
    via ``construct_model``, then restores fitted state. Use this when you do not
    already have a class handle for ``ModelClass.load(path)``.

    Custom featurizers and predictors must already be registered (same as for
    training). Load only artifacts created with ``save_model`` in the same
    drevalpy version family.

    Args:
        path: Archive file path; ``.zip`` is appended when missing.

    Returns:
        Fitted ``DRPModel`` instance.

    Raises:
        FileNotFoundError: If the archive does not exist.
        UnsupportedCheckpointFormatError: If the format or version is unsupported.
        CorruptedCheckpointError: If the payload structure is invalid.
        IncompatibleModelCheckpointError: If restoration fails on the rebuilt class.
    """
    from drevalpy.models._construct_model_api import construct_model

    model_name, config, _state = load_model_payload(path)
    return construct_model(model_name, config).load(path)
