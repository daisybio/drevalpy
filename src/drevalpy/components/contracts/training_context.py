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

    ``n_jobs`` is the number of worker processes a single-drug predictor may fit its per-drug models
    in. It changes how long a fit takes, never what it produces.
    """

    checkpoint_dir: Path = field(default_factory=_default_checkpoint_dir)
    n_jobs: int = 1
