"""CSV export helpers for FeatureDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import FeatureDataset


def _feature_column_names(meta_info: dict[str, Any], view_name: str, vector_length: int) -> list[str]:
    meta_names = meta_info.get(view_name)
    if isinstance(meta_names, list) and len(meta_names) == vector_length:
        return meta_names
    return [f"feature_{i}" for i in range(vector_length)]


def _feature_row(identifier: str, vector: Any, feature_names: list[str], id_column: str) -> dict[str, Any]:
    row: dict[str, Any] = {id_column: identifier}
    row.update({name: value for name, value in zip(feature_names, vector, strict=True)})
    return row


def feature_dataset_to_csv(
    dataset: FeatureDataset,
    path: str | Path,
    id_column: str,
    view_name: str,
) -> None:
    """Write one view of a FeatureDataset to CSV.

    :param dataset: Feature dataset to export.
    :param path: Output CSV path.
    :param id_column: Column name for row identifiers.
    :param view_name: Feature view to serialize.
    :raises ValueError: If *view_name* is missing for an identifier.
    """
    data: list[dict[str, Any]] = []
    feature_names: list[str] | None = None

    for identifier, feature_dict in dataset.features.items():
        vector = feature_dict.get(view_name)
        if vector is None:
            raise ValueError(f"View {view_name!r} not found for identifier {identifier!r}.")
        if feature_names is None:
            feature_names = _feature_column_names(dataset.meta_info, view_name, len(vector))
        data.append(_feature_row(identifier, vector, feature_names, id_column))

    pd.DataFrame(data).to_csv(path, index=False)
