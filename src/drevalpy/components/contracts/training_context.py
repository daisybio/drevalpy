"""Small, serializable context for one component training call."""

import secrets
import tempfile
from dataclasses import dataclass, field

from upath import UPath as Path


def _default_checkpoint_dir() -> Path:
    """Return a unique, not yet created directory path under the system temp directory.

    :returns: A fresh path that never points into the working directory.
    """
    return Path(tempfile.gettempdir()) / f"drevalpy-checkpoints-{secrets.token_hex(8)}"


@dataclass(frozen=True)
class TrainingContext:
    """Runtime metadata that does not belong in predictor hyperparameters.

    ``DRPModel.train`` always supplies a checkpoint directory (its own temporary one
    unless the caller passes one). The default only serves hand-built contexts.
    """

    checkpoint_dir: Path = field(default_factory=_default_checkpoint_dir)
    logging_metadata: dict[str, str] = field(default_factory=dict)
