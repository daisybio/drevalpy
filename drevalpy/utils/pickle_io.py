"""Trusted pickle serialization boundary for drevalpy pipeline artifacts."""

from __future__ import annotations

import pathlib
import pickle  # noqa: S403
from typing import Any, BinaryIO

from upath import UPath as Path

PickleDestination = Path | str | BinaryIO
PickleSource = Path | str | BinaryIO
UnpicklingError = pickle.UnpicklingError


def load_trusted_pickle(source: PickleSource) -> Any:
    """Load a pickle from a trusted local path or stream.

    :param source: Path or open binary stream written by ``dump_trusted_pickle``.
    :returns: Deserialized Python object.
    """
    if isinstance(source, (str, pathlib.PurePath)):
        with Path(source).open("rb") as handle:
            return pickle.load(handle)  # noqa: S301
    return pickle.load(source)  # noqa: S301


def dump_trusted_pickle(payload: Any, destination: PickleDestination) -> None:
    """Serialize ``payload`` with pickle to a trusted local path or stream.

    :param payload: Object to serialize.
    :param destination: Output path or binary stream.
    """
    if isinstance(destination, (str, pathlib.PurePath)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return
    pickle.dump(payload, destination)
