"""Trusted PyTorch serialization boundary for drevalpy."""

from __future__ import annotations

import io
from typing import Any, BinaryIO

import torch
from upath import UPath as Path

TorchSource = bytes | bytearray | Path | str | BinaryIO


def _coerce_source(source: TorchSource) -> Path | BinaryIO:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if isinstance(source, str):
        return Path(source)
    return source


def load_torch_payload(
    source: TorchSource,
    *,
    map_location: Any | None = None,
    weights_only: bool = True,
) -> Any:
    """Load a PyTorch-serialized payload from a trusted local source.

    :param source: Bytes, path, or open binary stream containing a torch checkpoint.
    :param map_location: Optional device remapping passed through to ``torch.load``.
    :param weights_only: When ``True``, restrict deserialization to tensor payloads.
    :returns: Deserialized checkpoint object.
    """
    coerced = _coerce_source(source)
    kwargs: dict[str, Any] = {"weights_only": weights_only}
    if map_location is not None:
        kwargs["map_location"] = map_location
    return torch.load(coerced, **kwargs)  # noqa: S614


def save_torch_payload(payload: Any, destination: Path | str | BinaryIO) -> None:
    """Serialize ``payload`` with ``torch.save`` to a path or buffer.

    :param payload: Object to serialize.
    :param destination: Output path or binary stream.
    """
    torch.save(payload, destination)


def load_state_dict(
    source: TorchSource,
    *,
    map_location: Any | None = None,
) -> dict[str, Any]:
    """Load a PyTorch state dict mapping from a trusted local source.

    :param source: Bytes, path, or open binary stream containing a state dict.
    :param map_location: Optional device remapping passed through to ``torch.load``.
    :returns: Mapping of parameter names to tensors.
    :raises TypeError: If the deserialized payload is not a mapping.
    """
    data = load_torch_payload(source, map_location=map_location, weights_only=True)
    if not isinstance(data, dict):
        msg = "torch payload must be a state dict mapping"
        raise TypeError(msg)
    return data


def load_trusted_payload(
    source: TorchSource,
    *,
    map_location: Any | None = None,
) -> Any:
    """Load a trusted checkpoint that may contain arbitrary pickled Python objects.

    :param source: Bytes, path, or open binary stream containing a trusted checkpoint.
    :param map_location: Optional device remapping passed through to ``torch.load``.
    :returns: Deserialized checkpoint object.
    """
    return load_torch_payload(source, map_location=map_location, weights_only=False)


def load_trusted_mapping(
    source: TorchSource,
    *,
    map_location: Any | None = None,
) -> dict[str, Any]:
    """Load a trusted mapping previously written with ``save_trusted_mapping``.

    :param source: Bytes, path, or open binary stream containing a mapping checkpoint.
    :param map_location: Optional device remapping passed through to ``torch.load``.
    :returns: Deserialized mapping object.
    :raises TypeError: If the deserialized payload is not a mapping.
    """
    data = load_trusted_payload(source, map_location=map_location)
    if not isinstance(data, dict):
        msg = "torch payload must be a mapping"
        raise TypeError(msg)
    return data


def save_state_dict(state_dict: dict[str, Any]) -> bytes:
    """Serialize a PyTorch state dict to bytes.

    :param state_dict: Mapping of parameter names to tensors.
    :returns: Serialized checkpoint bytes.
    """
    buffer = io.BytesIO()
    save_torch_payload(state_dict, buffer)
    return buffer.getvalue()


def save_trusted_mapping(payload: dict[str, Any]) -> bytes:
    """Serialize an arbitrary mapping with ``torch.save``.

    :param payload: Mapping to serialize.
    :returns: Serialized checkpoint bytes.
    """
    buffer = io.BytesIO()
    save_torch_payload(payload, buffer)
    return buffer.getvalue()


# Backward-compatible aliases used by literature predictor state modules.
load_object_mapping = load_trusted_mapping
save_object_mapping = save_trusted_mapping
