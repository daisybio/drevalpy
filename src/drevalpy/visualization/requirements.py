"""Plot requirement declarations for capability-based plot selection."""

from enum import Enum, auto


class PlotRequirement(Enum):
    """Requirements that a plot class may declare."""

    MULTIPLE_MODELS = auto()
    MULTIPLE_FOLDS = auto()
    RANDOMIZATION = auto()
    ROBUSTNESS = auto()
