"""Build discovery/catalog metadata dicts for registered components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drevalpy.components.predictors.inputs import predictor_inputs


@dataclass(frozen=True)
class LiteratureReference:
    """Repository, citation, and integration-deviation notes for a literature port."""

    repo_url: str
    citation_text: str = ""
    citation_doi: str = ""
    deviations: str = ""

    def __post_init__(self) -> None:
        """Normalize surrounding whitespace in every field."""
        object.__setattr__(self, "repo_url", self.repo_url.strip())
        object.__setattr__(self, "citation_text", self.citation_text.strip())
        object.__setattr__(self, "citation_doi", self.citation_doi.strip())
        object.__setattr__(self, "deviations", self.deviations.strip())


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


def base_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return shared discovery fields for a registered component.

    :param registry_name: Identifier of the registry that holds the class.
    :param name: Name the class is registered under.
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


def _output_format(cls: type[Any]) -> str:
    try:
        blocks = cls.output_blocks({})
    except Exception:  # noqa: BLE001 - a config-dependent featurizer has no default-config blocks
        return ""
    return ", ".join(dict.fromkeys(block.format.value for block in blocks))


def featurizer_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `base_component_metadata` plus the featurizer's sides and output format.

    A featurizer that declares its blocks through ``output_blocks`` reports the formats of its
    default-config blocks, joined; one that emits none reports ``""``.

    :param registry_name: Identifier of the registry that holds the class.
    :param name: Name the class is registered under.
    :param cls: Registered featurizer class.
    :returns: Catalog metadata dict.
    """
    meta = base_component_metadata(registry_name, name, cls)
    meta["sides"] = frozenset(getattr(cls, "sides", ()))
    meta["output_format"] = _output_format(cls)
    meta["precompute"] = getattr(cls, "precompute", False)
    return meta


def _input_interface(cls: type[Any]) -> str:
    """Summarize a predictor's declared inputs as one catalog label.

    :param cls: Registered predictor class.
    :returns: ``"block"`` when either side reads named blocks, else ``"matrix"`` when either
        side reads a matrix, else ``"feature_free"``.
    """
    kinds = {predictor_inputs(cls, "sample").kind, predictor_inputs(cls, "drug").kind}
    if "blocks" in kinds:
        return "block"
    return "matrix" if "matrix" in kinds else "feature_free"


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `base_component_metadata` plus the predictor's input interface.

    :param registry_name: Identifier of the registry that holds the class.
    :param name: Name the class is registered under.
    :param cls: Registered predictor class.
    :returns: Catalog metadata dict.
    """
    meta = base_component_metadata(registry_name, name, cls)
    meta["input_interface"] = _input_interface(cls)
    return meta
