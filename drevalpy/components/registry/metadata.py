"""Flatten registry metadata rows for featurizers and predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureKind


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
    """Like :func:`metadata_record` plus ``output_contract`` summary."""
    meta = metadata_record(registry_name, name, cls)
    output = getattr(cls, "output_contract", None)
    if output is None:
        output = FeatureContract(kind=FeatureKind.DENSE)
    meta["output_type"] = _contract_to_str(output)
    return meta


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, str]:
    """Like :func:`metadata_record` plus required input contract summaries."""
    meta = metadata_record(registry_name, name, cls)
    cell_line = getattr(cls, "required_cell_line_contract", None)
    drug = getattr(cls, "required_drug_contract", None)
    if cell_line is None:
        cell_line = FeatureContract(kind=FeatureKind.DENSE)
    if drug is None:
        drug = FeatureContract(kind=FeatureKind.DENSE)
    meta["required_cell_line_input"] = _contract_to_str(cell_line)
    meta["required_drug_input"] = _contract_to_str(drug)
    return meta
