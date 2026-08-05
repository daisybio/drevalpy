"""Flatten component metadata for featurizers and predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat, featurizer_contract, predictor_contracts
from drevalpy.types.literature_reference import LiteratureReference


def _contract_to_str(contract: FeatureContract | FeatureFormat | str | None) -> str:
    if contract is None:
        return ""
    if isinstance(contract, FeatureContract):
        return contract.format.value
    if isinstance(contract, FeatureFormat):
        return contract.value
    return str(contract)


def _normalize_tags(tags: object) -> frozenset[str]:
    if not tags:
        return frozenset()
    if isinstance(tags, (frozenset, set, list, tuple)):
        return frozenset(str(tag) for tag in tags)
    msg = f"tags must be a collection of strings, got {type(tags).__name__}"
    raise TypeError(msg)


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
    from drevalpy.components.predictors.feature_free import FeatureFreePredictor
    from drevalpy.components.predictors.matrix import MatrixPredictor
    from drevalpy.components.predictors.structured import BlockPredictor

    if issubclass(cls, FeatureFreePredictor):
        return "feature_free"
    if issubclass(cls, MatrixPredictor):
        return "matrix"
    if issubclass(cls, BlockPredictor):
        return "block"
    return ""


def metadata_record(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return metadata for a registered component.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered component class.
    :returns: Result.
    """
    fields: dict[str, Any] = {
        "registry": registry_name,
        "name": name,
        "class_name": cls.__name__,
        "description": str(getattr(cls, "description", "") or ""),
        "tags": _normalize_tags(getattr(cls, "tags", frozenset())),
    }
    fields.update(_reference_fields(cls))
    return fields


def featurizer_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `metadata_record` plus featurizer contract summary.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered featurizer class.
    :returns: Result.
    """
    meta = metadata_record(registry_name, name, cls)
    meta["output_format"] = _contract_to_str(featurizer_contract(cls))
    return meta


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `metadata_record` plus predictor capability and contract summaries.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered predictor class.
    :returns: Result.
    """
    meta = metadata_record(registry_name, name, cls)
    cell_line, drug = predictor_contracts(cls)
    meta["input_interface"] = _predictor_input_interface(cls)
    meta["cell_line_format"] = _contract_to_str(cell_line)
    meta["drug_format"] = _contract_to_str(drug)
    modes: object = getattr(cls, "supported_modes", frozenset())
    scopes: object = getattr(cls, "supported_scopes", frozenset())
    mode_values = modes if isinstance(modes, (frozenset, set, list, tuple)) else ()
    scope_values = scopes if isinstance(scopes, (frozenset, set, list, tuple)) else ()
    meta["supported_modes"] = ",".join(sorted(str(mode) for mode in mode_values))
    meta["supported_scopes"] = ",".join(sorted(str(scope) for scope in scope_values))
    meta["supports_early_stopping"] = str(bool(getattr(cls, "supports_early_stopping", False))).lower()
    meta["requires_drug_featurizer"] = str(bool(getattr(cls, "requires_drug_featurizer", True))).lower()
    cell_views = getattr(cls, "required_cell_line_views", ())
    drug_views = getattr(cls, "required_drug_views", ())
    meta["required_cell_line_views"] = ",".join(cell_views) if cell_views else ""
    meta["required_drug_views"] = ",".join(drug_views) if drug_views else ""
    return meta
