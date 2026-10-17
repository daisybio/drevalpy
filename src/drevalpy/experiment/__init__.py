"""Experiment sub-module: randomization and robustness utilities."""

# Re-exported for convenience only. These types are owned and documented by
# drevalpy.types.results, so they are deliberately kept out of __all__: listing
# them would make autodoc document the same classes a second time under this
# package, producing ambiguous cross-references.
from drevalpy.types.results.run import RunResult as RunResult
from drevalpy.types.results.trial import TrialResult as TrialResult

from ._randomization import randomization
from ._robustness import robustness

__all__ = ["randomization", "robustness"]
