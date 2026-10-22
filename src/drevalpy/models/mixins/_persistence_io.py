"""Low-level versioned checkpoint I/O for DRPModel instances.

Checkpoints are a single ZIP archive written atomically to an archive file path,
which may be local or a remote ``UPath`` (``s3://``, ``memory://``, ...).
Callers must only load artifacts they created with ``save_model`` in the same
drevalpy version family.

.. warning::
   The payload is deserialized with ``joblib`` (pickle). Loading a checkpoint
   executes arbitrary code from the file, so load checkpoints only from sources
   you trust.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any

import joblib
from upath import UPath as Path

from drevalpy.models.config.resolved import ResolvedModelConfig

if TYPE_CHECKING:
    from drevalpy.models.drp_model import DRPModel

FORMAT_NAME = "drevalpy-model"
FORMAT_VERSION = 4
PAYLOAD_MEMBER = "payload.joblib"
_PRE_SAMPLE_RENAME_VERSION = 3


class ModelCheckpointError(Exception):
    """Base error for DRPModel checkpoint problems."""


class UnsupportedCheckpointFormatError(ModelCheckpointError, ValueError):
    """Raised when checkpoint format or version is not supported."""


class CorruptedCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint payload structure or content is invalid."""


def _as_path(path: str | Path) -> Path:
    """Normalize a user path to ``Path``, rejecting trailing separators.

    :param path: Checkpoint archive path string or ``Path``.
    :returns: Normalized path without a trailing directory separator.
    :raises ValueError: If ``path`` ends with a directory separator.
    """
    if isinstance(path, str) and path.endswith(("/", "\\")):
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)
    return Path(path)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Return the archive file path, appending ``.zip`` when missing.

    :param path: Checkpoint archive path; ``.zip`` is appended when missing.
    :returns: Normalized archive ``Path``.
    """
    target = _as_path(path)
    if target.name.lower().endswith(".zip"):
        return target
    return target.with_name(f"{target.name}.zip")


def _reject_directory_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)


def _is_local(path: Path) -> bool:
    return path.protocol in ("", "file", "local")


def _write_archive_atomically(archive_path: Path, payload: dict[str, Any]) -> None:
    """Write the archive so readers never see a partial file.

    Local targets are written beside the destination and moved with ``os.replace``.
    Remote targets (``s3://``, ``memory://``, ...) are built in the local temp
    directory and then uploaded in one streamed copy.

    :param archive_path: Final archive location.
    :param payload: Checkpoint payload to serialize.
    """
    local = _is_local(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.", suffix=".tmp", dir=os.fspath(archive_path.parent) if local else None
    )
    os.close(fd)
    try:
        buffer = io.BytesIO()
        joblib.dump(payload, buffer)
        with zipfile.ZipFile(tmp_name, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
        if local:
            os.replace(tmp_name, os.fspath(archive_path))
        else:
            with open(tmp_name, "rb") as source, archive_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _read_payload_from_archive(archive_path: Path) -> Any:
    try:
        with archive_path.open("rb") as raw, zipfile.ZipFile(raw, mode="r") as archive:
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

    :param model: Trained ``DRPModel`` instance to persist.
    :param path: Archive file path; ``.zip`` is appended when missing.
    :raises RuntimeError: If the model is not trained or lacks a ``ModelConfig``.
    """
    pipeline = model._pipeline
    if not pipeline.is_fitted():
        raise RuntimeError(f"Cannot save model {model.name!r}: pipeline must be FITTED")
    config = model.config
    target = _as_path(path)
    _reject_directory_path(target)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model.name,
        "config": config.model_dump(mode="json"),
        "state": pipeline.component_state(),
    }
    _write_archive_atomically(resolve_checkpoint_path(target), payload)


def _resolved_config_from_checkpoint_payload(payload: dict[str, Any]) -> ResolvedModelConfig:
    version = payload.get("version")
    if version == _PRE_SAMPLE_RENAME_VERSION:
        raise UnsupportedCheckpointFormatError(
            f"unsupported checkpoint format/version: {payload.get('format')!r}/{version!r}; "
            f"this checkpoint predates the cell line -> sample rename (format {FORMAT_VERSION}) and its "
            "configuration and state keys are no longer readable; retrain the model and save it again"
        )
    if version == FORMAT_VERSION:
        return ResolvedModelConfig.model_validate(payload["config"])
    raise UnsupportedCheckpointFormatError(
        f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}; "
        "retrain with the current model API because older checkpoints omit target transformation state"
    )


def load_model_payload(path: str | Path) -> tuple[str, ResolvedModelConfig, dict[str, object]]:
    """Load and validate a ``DRPModel`` checkpoint payload from an archive path.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :returns: Tuple of ``(model_name, resolved_config, component_state)``.
    :raises FileNotFoundError: If the archive does not exist.
    :raises UnsupportedCheckpointFormatError: If the format or version is unsupported.
    :raises CorruptedCheckpointError: If the payload structure is invalid.
    """
    target = _as_path(path)
    _reject_directory_path(target)
    archive_path = resolve_checkpoint_path(target)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {archive_path}")

    payload = _read_payload_from_archive(archive_path)
    if not isinstance(payload, dict):
        raise CorruptedCheckpointError("checkpoint payload is not a mapping")
    if payload.get("format") != FORMAT_NAME:
        raise UnsupportedCheckpointFormatError(
            f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}; "
            "retrain with the current model API because older checkpoints omit target transformation state"
        )
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise CorruptedCheckpointError("checkpoint model_name is missing or invalid")
    try:
        config = _resolved_config_from_checkpoint_payload(payload)
    except UnsupportedCheckpointFormatError:
        raise
    except Exception as exc:
        raise CorruptedCheckpointError("checkpoint config is invalid") from exc
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptedCheckpointError("checkpoint state is not a mapping")
    return model_name, config, state


def load_model(path: str | Path) -> DRPModel:
    """Reconstruct a fitted ``DRPModel`` from a checkpoint archive path.

    Reads the stored model name and resolved configuration, constructs a concrete
    ``DRPModel``, and restores components and target transformation state.

    Custom featurizers and predictors must already be registered (same as for
    training). Load only artifacts created with ``save_model`` in the same
    drevalpy version family.

    .. warning::
       Checkpoints are unpickled (``joblib``, and ``torch.load`` with
       ``weights_only=False`` for neural weights). Loading one can execute
       arbitrary code, so load checkpoints only from trusted sources.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :returns: Fitted ``DRPModel`` instance.
    """
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config

    model_name, config, state = load_model_payload(path)
    model = DRPModel(ModelSpec(model_name, config.template), public_hyperparameters_from_config(config))
    try:
        model._pipeline.restore_component_state(state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CorruptedCheckpointError(f"checkpoint pipeline state is invalid: {exc}") from exc
    if not model._pipeline.is_fitted():
        raise CorruptedCheckpointError("checkpoint did not restore a fitted predictor")
    return model
