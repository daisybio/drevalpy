"""Generate built-in component tables from registry metadata."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TypedDict, TypeVar, cast

from drevalpy.components.register_builtins import (
    _CELL_LINE_MODULES,
    _DRUG_MODULES,
    _PREDICTOR_MODULES,
    register_builtin_components,
)
from drevalpy.components.registry import (
    list_cell_line_featurizer_metadata,
    list_drug_featurizer_metadata,
    list_predictor_metadata,
)

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_CATALOGS = {
    "cell_line": DOCS_DIR / "concepts" / "_generated_cell_line_featurizers.rst",
    "drug": DOCS_DIR / "concepts" / "_generated_drug_featurizers.rst",
    "predictor": DOCS_DIR / "concepts" / "_generated_predictors.rst",
}


class ComponentCatalogMetadata(TypedDict):
    """Registry fields shared by every generated component row."""

    name: str
    description: str


class FeaturizerCatalogMetadata(ComponentCatalogMetadata):
    """Registry fields rendered for a featurizer."""

    output_format: str


class PredictorCatalogMetadata(ComponentCatalogMetadata):
    """Registry fields rendered for a predictor."""

    input_interface: str


MetadataT = TypeVar("MetadataT", bound=ComponentCatalogMetadata)


def _builtin_rows(
    rows: list[MetadataT],
    builtin_names: Collection[str],
    component_label: str,
) -> list[MetadataT]:
    """Select and sort built-in rows, failing if discovery is incomplete.

    :param rows: discovered registry metadata rows
    :param builtin_names: names expected from built-in module registration
    :param component_label: component kind used in validation errors
    :returns: built-in metadata rows sorted by name
    :raises RuntimeError: if a built-in component has no discovered metadata
    """
    rows_by_name = {row["name"]: row for row in rows if row["name"] in builtin_names}
    missing = sorted(set(builtin_names) - rows_by_name.keys())
    if missing:
        raise RuntimeError(f"Built-in {component_label} missing registry metadata: {missing}")
    return [rows_by_name[name] for name in sorted(rows_by_name)]


def _description(row: ComponentCatalogMetadata) -> str:
    description = " ".join(row["description"].split())
    if not description:
        raise RuntimeError(f"Component {row['name']!r} is missing a description")
    return description


def _render_featurizers(rows: list[FeaturizerCatalogMetadata]) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 20 56",
        "",
        "   * - Name",
        "     - Output format",
        "     - Role",
    ]
    for row in rows:
        if not row["output_format"]:
            raise RuntimeError(f"Featurizer {row['name']!r} is missing an output format")
        lines.extend(
            [
                f"   * - ``{row['name']}``",
                f"     - ``{row['output_format']}``",
                f"     - {_description(row)}",
            ]
        )
    return "\n".join([*lines, ""])


def _render_predictors(rows: list[PredictorCatalogMetadata]) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 20 56",
        "",
        "   * - Name",
        "     - Interface",
        "     - Role",
    ]
    for row in rows:
        interface = row["input_interface"]
        if not interface:
            raise RuntimeError(f"Predictor {row['name']!r} is missing an input interface")
        lines.extend(
            [
                f"   * - ``{row['name']}``",
                f"     - {interface.replace('_', '-').capitalize()}",
                f"     - {_description(row)}",
            ]
        )
    return "\n".join([*lines, ""])


def generate_component_catalog_rsts() -> dict[str, str]:
    """Return deterministic RST tables for every built-in component registry.

    :returns: generated RST keyed by component registry
    """
    register_builtin_components()
    cell_line_rows = _builtin_rows(
        cast(list[FeaturizerCatalogMetadata], list_cell_line_featurizer_metadata()),
        _CELL_LINE_MODULES,
        "cell-line featurizers",
    )
    drug_rows = _builtin_rows(
        cast(list[FeaturizerCatalogMetadata], list_drug_featurizer_metadata()),
        _DRUG_MODULES,
        "drug featurizers",
    )
    predictor_rows = _builtin_rows(
        cast(list[PredictorCatalogMetadata], list_predictor_metadata()),
        _PREDICTOR_MODULES,
        "predictors",
    )
    return {
        "cell_line": _render_featurizers(cell_line_rows),
        "drug": _render_featurizers(drug_rows),
        "predictor": _render_predictors(predictor_rows),
    }


def write_generated_component_catalogs() -> tuple[Path, ...]:
    """Write generated RST includes consumed by the component catalog.

    :returns: paths to the generated RST files
    """
    generated = generate_component_catalog_rsts()
    for key, path in GENERATED_CATALOGS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated[key], encoding="utf-8")
    return tuple(GENERATED_CATALOGS.values())
