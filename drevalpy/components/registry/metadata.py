"""Build discovery/catalog metadata dicts for registered components."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import featurizer_contract
from drevalpy.types.literature_reference import LiteratureReference


def _reference_fields(cls: type[Any]) -> dict[str, str]:
    reference = getattr(cls, "reference", None)
    if not isinstance(reference, LiteratureReference):
        return {
            "repo_url": "",
            "citation": "",
            "citation_doi": "",
            "citation_text": "",
            "deviations": "",
        }
    cite = reference.citation_doi or reference.citation_text
    if cite.startswith("10."):
        cite = f"https://doi.org/{cite}"
    return {
        "repo_url": reference.repo_url,
        "citation": cite,
        "citation_doi": reference.citation_doi,
        "citation_text": reference.citation_text,
        "deviations": reference.deviations,
    }


def _predictor_input_interface(cls: type[Any]) -> str:
    # Local imports avoid circular dependencies during package import.
    from drevalpy.components.predictors.block import BlockPredictor
    from drevalpy.components.predictors.feature_free import FeatureFreePredictor
    from drevalpy.components.predictors.matrix import MatrixPredictor

    if issubclass(cls, FeatureFreePredictor):
        return "feature_free"
    if issubclass(cls, MatrixPredictor):
        return "matrix"
    if issubclass(cls, BlockPredictor):
        return "block"
    return ""


def base_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return shared discovery fields for a registered component.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered component class.
    :returns: Catalog metadata dict.
    """
    fields: dict[str, Any] = {
        "registry": registry_name,
        "name": name,
        "class_name": cls.__name__,
        "description": str(getattr(cls, "description", "") or ""),
        "tags": getattr(cls, "tags", frozenset()),
    }
    fields.update(_reference_fields(cls))
    return fields


def featurizer_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `base_component_metadata` plus featurizer output format.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered featurizer class.
    :returns: Catalog metadata dict.
    """
    meta = base_component_metadata(registry_name, name, cls)
    meta["output_format"] = featurizer_contract(cls).format.value
    return meta


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `base_component_metadata` plus predictor input interface.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered predictor class.
    :returns: Catalog metadata dict.
    """
    meta = base_component_metadata(registry_name, name, cls)
    meta["input_interface"] = _predictor_input_interface(cls)
    return meta
