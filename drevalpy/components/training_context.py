"""Small, serializable context for one component training call."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingContext:
    """Runtime metadata that does not belong in predictor hyperparameters."""

    checkpoint_dir: str = "checkpoints"
    logging_metadata: dict[str, str] = field(default_factory=dict)
