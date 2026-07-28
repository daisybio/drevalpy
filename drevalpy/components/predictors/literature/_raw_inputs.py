"""Validate raw FeatureDataset views for literature predictors."""

from __future__ import annotations

from typing import cast

from drevalpy.components.predictors.literature._raw_views import (
    validate_pyg_drug_graphs,
    validate_required_views,
)
from drevalpy.datasets.dataset import FeatureDataset


def validate_raw_inputs(
    predictor: object,
    cell_line_input: FeatureDataset | None,
    drug_input: FeatureDataset | None,
    *,
    cell_line_views: tuple[str, ...],
    drug_views: tuple[str, ...],
    validate_drug_graphs: bool = False,
) -> tuple[FeatureDataset, FeatureDataset | None]:
    """Validate required raw featurizer views before train or predict."""
    name = getattr(predictor, "registry_name", predictor.__class__.__name__)
    validate_required_views(
        cell_line_input,
        cell_line_views,
        predictor_name=str(name),
        side="cell_line",
    )
    cell_lines = cast(FeatureDataset, cell_line_input)
    drugs: FeatureDataset | None = None
    if drug_views:
        validate_required_views(
            drug_input,
            drug_views,
            predictor_name=str(name),
            side="drug",
        )
        drugs = cast(FeatureDataset, drug_input)
        if validate_drug_graphs and "drug_graph" in drug_views:
            validate_pyg_drug_graphs(drugs, predictor_name=str(name))
    return cell_lines, drugs
