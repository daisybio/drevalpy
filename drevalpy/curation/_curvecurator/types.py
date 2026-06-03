"""Typed in-memory objects for CurveCurator curation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CurationWorkItem:
    """One CurveCurator fit job prepared from raw viability data."""

    work_id: str
    dataset_name: str
    group_key: str
    chunk_index: int | None
    input_table: pd.DataFrame
    config: dict
    n_curves: int
    input_filename: str


@dataclass(frozen=True)
class CurationSplitResult:
    """Prepared CurveCurator work items for one raw viability dataset."""

    dataset_name: str
    input_filename: str
    work_items: tuple[CurationWorkItem, ...]


@dataclass(frozen=True)
class CurationFitResult:
    """Fitted CurveCurator output for one work item."""

    work_id: str
    curves: pd.DataFrame
    work_item: CurationWorkItem
