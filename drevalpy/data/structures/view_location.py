"""View location enum for Dataset storage resolution."""

from __future__ import annotations

from enum import Enum


class ViewLocation(Enum):
    """Where a named view is stored within the MuData structure."""

    MODALITY = "modality"
    VARM = "varm"
    OBSM = "obsm"
    UNS = "uns"
