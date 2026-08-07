"""Resolution of the optional model checkpoint directory."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEMPORARY_CHECKPOINT_DIR = "TEMPORARY"


def resolve_checkpoint_dir(model_checkpoint_dir: str | Path | None) -> Path | None:
    """Normalize a checkpoint directory argument.

    ``None`` means "no directory was requested", and callers are expected to fall back
    to a temporary directory. The literal string ``"TEMPORARY"`` is accepted as a
    deprecated alias for ``None`` so that existing command lines keep working.

    :param model_checkpoint_dir: Directory for checkpoints, or ``None``.

    :returns: The directory as a ``Path``, or ``None`` when a temporary one should be used.
    """
    if model_checkpoint_dir is None:
        return None
    if isinstance(model_checkpoint_dir, str) and model_checkpoint_dir == TEMPORARY_CHECKPOINT_DIR:
        return None
    return Path(model_checkpoint_dir)


@contextmanager
def checkpoint_dir_or_temporary(model_checkpoint_dir: str | Path | None) -> Iterator[Path]:
    """Yield an existing checkpoint directory, creating a temporary one when none was requested.

    The temporary directory is removed when the context exits, so callers must finish
    reading any checkpoints before leaving the block.

    :param model_checkpoint_dir: Directory for checkpoints, or ``None`` for a temporary one.

    :yields: An existing directory to write checkpoints into.
    """
    resolved = resolve_checkpoint_dir(model_checkpoint_dir)
    if resolved is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir} for model checkpoints")
            yield Path(temp_dir)
        return
    resolved.mkdir(parents=True, exist_ok=True)
    print(f"Using directory: {resolved} for model checkpoints")
    yield resolved
