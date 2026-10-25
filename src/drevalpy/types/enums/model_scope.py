"""Training scope for global vs single-drug models."""

from __future__ import annotations

from enum import StrEnum


class ModelScope(StrEnum):
    """Whether one model is fitted globally or independently per drug."""

    MULTI_DRUG = "multi_drug"
    SINGLE_DRUG = "single_drug"
