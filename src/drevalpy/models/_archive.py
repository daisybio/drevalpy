"""The versioned single-ZIP checkpoint archive of a ``DRPModel``.

Checkpoints are a single ZIP archive written atomically to an archive file path,
which may be local or a remote ``UPath`` (``s3://``, ``memory://``, ...).
The payload holds the model spec as plain data and the fitted components as pickled objects.
Only archives of format version 2 are read.

.. warning::
   The payload is deserialized with ``joblib`` (pickle). Loading a checkpoint
   executes arbitrary code from the file, so load checkpoints only from sources
   you trust.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import joblib
from upath import UPath as Path

from drevalpy.errors import DrEvalError

FORMAT_NAME = "drevalpy-model"
FORMAT_VERSION = 3
PAYLOAD_MEMBER = "payload.joblib"


class ModelCheckpointError(DrEvalError):
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


@contextmanager
def _modules_on_cpu(components: Mapping[str, Any]) -> Iterator[None]:
    """Move every torch module held by *components* to the CPU, restoring the devices afterwards.

    torch is imported only when it is already loaded: without it no component holds a module.

    :param components: Fitted component objects.
    :yields: While the modules are on the CPU.
    """
    if "torch" not in sys.modules:
        yield
        return
    import torch

    devices: list[tuple[torch.nn.Module, torch.device]] = []
    for module in _torch_modules(components, torch.nn.Module):
        tensor = next(module.parameters(), None)
        if tensor is None:
            tensor = next(module.buffers(), None)
        if tensor is not None and tensor.device.type != "cpu":
            devices.append((module, tensor.device))
            module.to("cpu")
    try:
        yield
    finally:
        for module, device in devices:
            module.to(device)


def _torch_modules(root: Any, module_type: type) -> Iterator[Any]:
    """Yield the outermost torch modules reachable from *root* through containers and drevalpy objects.

    :param root: Object to search.
    :param module_type: ``torch.nn.Module``.
    :yields: Each module once.
    """
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, module_type):
            yield value
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif type(value).__module__.startswith("drevalpy") and hasattr(value, "__dict__"):
            stack.extend(vars(value).values())


def save_archive(path: str | Path, spec: dict[str, Any], components: dict[str, Any], *, measure: str) -> None:
    """Write a model spec, its fitted components and its response measure as one ZIP archive.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :param spec: Plain-data model spec (``ModelSpec.dump()``).
    :param components: Fitted component objects by role; torch modules are saved on the CPU.
    :param measure: The response measure the model was trained on.
    """
    target = _as_path(path)
    _reject_directory_path(target)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "spec": spec,
        "components": components,
        "measure": measure,
    }
    with _modules_on_cpu(components):
        _write_archive_atomically(resolve_checkpoint_path(target), payload)


def _unsupported_format(payload: dict[str, Any]) -> UnsupportedCheckpointFormatError:
    """Build the error for a checkpoint whose format name or version this release cannot read.

    :param payload: Decoded checkpoint payload.
    :returns: Error naming the stored format and version, and the version this release writes.
    """
    return UnsupportedCheckpointFormatError(
        f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}; "
        f"this release reads {FORMAT_NAME!r} version {FORMAT_VERSION}; retrain the model and save it again"
    )


def load_archive(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Read a model archive written by :func:`save_archive`.

    .. warning::
       The payload is unpickled, which can execute arbitrary code. Load archives only from
       trusted sources.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :returns: The plain-data spec, the component objects by role and the response measure.
    :raises FileNotFoundError: If the archive does not exist.
    :raises UnsupportedCheckpointFormatError: If the format or version is unsupported.
    :raises CorruptedCheckpointError: If the payload structure is invalid.
    """  # noqa: DOC503
    target = _as_path(path)
    _reject_directory_path(target)
    archive_path = resolve_checkpoint_path(target)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {archive_path}")
    payload = _read_payload_from_archive(archive_path)
    if not isinstance(payload, dict):
        raise CorruptedCheckpointError("checkpoint payload is not a mapping")
    if payload.get("format") != FORMAT_NAME or payload.get("version") != FORMAT_VERSION:
        raise _unsupported_format(payload)
    spec, components = payload.get("spec"), payload.get("components")
    if not isinstance(spec, dict):
        raise CorruptedCheckpointError("checkpoint spec is not a mapping")
    if not isinstance(components, dict):
        raise CorruptedCheckpointError("checkpoint components are not a mapping")
    measure = payload.get("measure")
    if not isinstance(measure, str) or not measure:
        raise CorruptedCheckpointError("checkpoint has no response measure")
    return spec, components, measure
