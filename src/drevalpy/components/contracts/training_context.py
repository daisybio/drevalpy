"""Small, serializable context for one component training call."""

from dataclasses import dataclass, field

from upath import UPath as Path

_DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


@dataclass(frozen=True)
class TrainingContext:
    """Runtime metadata that does not belong in predictor hyperparameters."""

    checkpoint_dir: Path = _DEFAULT_CHECKPOINT_DIR
    logging_metadata: dict[str, str] = field(default_factory=dict)
