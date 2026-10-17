"""HPO trial result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrialResult:
    """Output of a single HPO trial."""

    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    optimization_metric: str
    predictions: np.ndarray

    @property
    def score(self) -> float:
        """The score for the optimization metric."""
        return self.metrics.get(self.optimization_metric, float("nan"))

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = ["TrialResult", "    Hyperparameters:"]
        for k, v in self.hyperparameters.items():
            lines.append(f"        {k}: {v}")
        lines.append("    Metrics:")
        for k, v in self.metrics.items():
            marker = " *" if k == self.optimization_metric else ""
            lines.append(f"        {k}: {v:.4f}{marker}")
        return "\n".join(lines)
