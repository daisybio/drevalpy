"""Flatten registry metadata rows for featurizers and predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureKind, featurizer_contract, predictor_contracts


def _contract_to_str(contract: FeatureContract | FeatureKind | str | None) -> str:
    if contract is None:
        return ""
    if isinstance(contract, FeatureContract):
        return contract.kind.value
    if isinstance(contract, FeatureKind):
        return contract.value
    return str(contract)


def metadata_record(registry_name: str, name: str, cls: type[Any]) -> dict[str, str]:
    """Flattened metadata dict for internal consumers."""
    cite = (
        str(getattr(cls, "citation", "") or "").strip()
        or str(getattr(cls, "citation_doi", "") or "").strip()
        or str(getattr(cls, "citation_text", "") or "").strip()
    )
    if cite.startswith("10."):
        cite = f"https://doi.org/{cite}"
    return {
        "registry": registry_name,
        "name": name,
        "class_name": cls.__name__,
        "description": str(getattr(cls, "description", "") or ""),
        "category": str(getattr(cls, "category", "") or ""),
        "component_type": str(getattr(cls, "component_type", "") or ""),
        "template_repo_url": str(getattr(cls, "template_repo_url", "") or ""),
        "citation": cite,
        "citation_doi": str(getattr(cls, "citation_doi", "") or ""),
        "citation_text": str(getattr(cls, "citation_text", "") or ""),
        "deviations": str(getattr(cls, "deviations", "") or ""),
    }


def featurizer_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, str]:
    """Like `metadata_record` plus featurizer contract summary."""
    meta = metadata_record(registry_name, name, cls)
    meta["output_type"] = _contract_to_str(featurizer_contract(cls))
    return meta


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, str]:
    """Like `metadata_record` plus predictor input contract summaries."""
    meta = metadata_record(registry_name, name, cls)
    cell_line, drug = predictor_contracts(cls)
    meta["required_cell_line_input"] = _contract_to_str(cell_line)
    meta["required_drug_input"] = _contract_to_str(drug)
    return meta
