"""Hyperparameter search space utilities for internal modular composition."""

from __future__ import annotations

from typing import Any


def merge_search_spaces(
    cell_line_featurizer_space: dict[str, Any] | None = None,
    drug_featurizer_space: dict[str, Any] | None = None,
    predictor_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge component spaces into a single dict with dot-notation prefixed keys.

    Keys are prefixed as ``featurizer.cell_line.*``, ``featurizer.drug.*``,
    and ``predictor.*`` so :func:`split_hyperparameters` can invert the merge.
    """
    merged: dict[str, Any] = {}
    if cell_line_featurizer_space:
        for key, value in cell_line_featurizer_space.items():
            merged[f"featurizer.cell_line.{key}"] = value
    if drug_featurizer_space:
        for key, value in drug_featurizer_space.items():
            merged[f"featurizer.drug.{key}"] = value
    if predictor_space:
        for key, value in predictor_space.items():
            merged[f"predictor.{key}"] = value
    return merged


def split_hyperparameters(
    merged_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Invert :func:`merge_search_spaces`.

    Returns ``(cell_line_hp, drug_hp, predictor_hp)``.
    Keys without a recognised prefix are put into ``predictor_hp``.
    """
    cell_line_hp: dict[str, Any] = {}
    drug_hp: dict[str, Any] = {}
    predictor_hp: dict[str, Any] = {}
    for key, value in merged_config.items():
        if key.startswith("featurizer.cell_line."):
            cell_line_hp[key.removeprefix("featurizer.cell_line.")] = value
        elif key.startswith("featurizer.drug."):
            drug_hp[key.removeprefix("featurizer.drug.")] = value
        elif key.startswith("predictor."):
            predictor_hp[key.removeprefix("predictor.")] = value
        else:
            predictor_hp[key] = value
    return cell_line_hp, drug_hp, predictor_hp


def extract_defaults(
    cell_line_featurizer_space: dict[str, Any] | None = None,
    drug_featurizer_space: dict[str, Any] | None = None,
    predictor_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pull ``"default"`` values from spec dicts, returning a merged flat dict."""
    defaults: dict[str, Any] = {}

    def _pull(space: dict[str, Any], prefix: str) -> None:
        for name, spec in space.items():
            if isinstance(spec, dict) and "default" in spec:
                defaults[f"{prefix}.{name}"] = spec["default"]

    if cell_line_featurizer_space:
        _pull(cell_line_featurizer_space, "featurizer.cell_line")
    if drug_featurizer_space:
        _pull(drug_featurizer_space, "featurizer.drug")
    if predictor_space:
        _pull(predictor_space, "predictor")
    return defaults
