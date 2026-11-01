"""Trusted PyTorch serialization boundary for drevalpy.

``torch`` is imported inside the functions that call it. Predictor modules
reach this module while ``drevalpy.registry`` registers builtins, so a
module-scope ``import torch`` would put ~0.33s on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO

from upath import UPath as Path

TorchSource = bytes | bytearray | Path | str | BinaryIO


def _coerce_source(source: TorchSource) -> Path | BinaryIO:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if isinstance(source, str):
        return Path(source)
    return source


def save_torch_payload(payload: Any, destination: Path | str | BinaryIO) -> None:
    """Serialize ``payload`` with ``torch.save`` to a path or buffer.

    :param payload: Object to serialize.
    :param destination: Output path or binary stream.
    """
    import torch

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
    import torch

    kwargs: dict[str, Any] = {} if map_location is None else {"map_location": map_location}
    data = torch.load(_coerce_source(source), weights_only=True, **kwargs)  # noqa: S614
    if not isinstance(data, dict):
        msg = "torch payload must be a state dict mapping"
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
