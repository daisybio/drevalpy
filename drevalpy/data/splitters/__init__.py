"""Cross-validation splitting strategies for drug response prediction.

Splitters are plain callables with the signature::

    (mudataset: MuDataLike, n_splits: int, validation_ratio: float, random_state: int) -> list[SplitMasks]

Built-in modes (LPO, LCO, LDO, LTO) are registered via decorator on import.
Register custom splitters with::

    @splitter_registry.register("MY_MODE", "Description", validation="LCO")
    def my_splitter(mudataset, n_splits=5, validation_ratio=0.1, random_state=42): ...
"""

from .lco import leave_cell_line_out as leave_cell_line_out
from .ldo import leave_drug_out as leave_drug_out
from .lpo import leave_pair_out as leave_pair_out
from .lto import leave_tissue_out as leave_tissue_out
from .registry import Splitter, SplitterRegistry, splitter_registry
from .validation import SplitValidationError, Validation

get_splitter = splitter_registry.get

__all__ = [
    "Splitter",
    "SplitValidationError",
    "SplitterRegistry",
    "Validation",
    "get_splitter",
    "leave_cell_line_out",
    "leave_drug_out",
    "leave_pair_out",
    "leave_tissue_out",
    "splitter_registry",
]
