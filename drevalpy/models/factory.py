"""Resolve zoo/spec names to `~drevalpy.models.config.ModelConfig` objects."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.featurizer_mapping import cell_line_featurizer_from_views, drug_featurizer_from_view


def _get_view_as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def featurizer_configs_from_view_hyperparameters(
    hyperparameters: dict[str, Any],
) -> tuple[Any, Any]:
    """Build featurizer configs when view hyperparameters are explicitly set.

    Prefer ``drevalpy.models.flat_hyperparameters.apply_public_flat_hyperparameters``
    for full flat-HP application. This helper remains for view-only construction
    and is deprecated alongside ``cell_line_views`` / ``drug_views``.
    """
    if "cell_line_views" in hyperparameters or "drug_views" in hyperparameters:
        from drevalpy._deprecations import warn_deprecated

        warn_deprecated(
            what="featurizer_configs_from_view_hyperparameters",
            replacement=(
                "ModelConfig featurizer blocks, recipe strings, or "
                "apply_public_flat_hyperparameters with explicit featurizer configs"
            ),
            stacklevel=3,
        )

    cell_line_featurizer = None
    drug_featurizer = None

    if "cell_line_views" in hyperparameters:
        cell_line_views = _get_view_as_list(hyperparameters.get("cell_line_views", ["gene_expression"]))
        cell_line_featurizer = cell_line_featurizer_from_views(cell_line_views, hyperparameters)

    if "drug_views" in hyperparameters:
        drug_views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))
        if drug_views:
            drug_featurizer = drug_featurizer_from_view(drug_views[0])

    return cell_line_featurizer, drug_featurizer


def model_config_for_name(model_name: str, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
    """Resolve a factory/zoo name to a modular config with public flat HP applied."""
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    if model_name not in list_zoo_names(include_external=True):
        msg = f"Unknown model name: {model_name}"
        raise KeyError(msg)
    return zoo_model_config(model_name, hyperparameters)
