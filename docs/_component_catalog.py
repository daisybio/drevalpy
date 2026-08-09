"""Generate built-in component tables from registry metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import TypedDict, TypeVar, cast

from _generated_io import write_text_if_changed

from drevalpy.components.core.plugins.register_builtins import (
    BUILTIN_CELL_LINE_FEATURIZER_NAMES,
    BUILTIN_DRUG_FEATURIZER_NAMES,
    BUILTIN_PREDICTOR_NAMES,
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
EXPECTED_BUILTIN_COMPONENT_COUNTS = {"cell_line": 17, "drug": 9, "predictor": 27}
EXPECTED_PREDICTOR_INTERFACE_COUNTS = {"feature_free": 1, "matrix": 13, "block": 13}


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


def _validate_builtin_catalog(
    *,
    cell_line_rows: list[FeaturizerCatalogMetadata],
    drug_rows: list[FeaturizerCatalogMetadata],
    predictor_rows: list[PredictorCatalogMetadata],
) -> None:
    """Fail generation when built-in catalogs diverge from supported interfaces.

    :param cell_line_rows: registered cell-line featurizer metadata rows
    :param drug_rows: registered drug featurizer metadata rows
    :param predictor_rows: registered predictor metadata rows
    :raises RuntimeError: if component or interface counts diverge from expectations
    """
    observed_counts = {
        "cell_line": len(cell_line_rows),
        "drug": len(drug_rows),
        "predictor": len(predictor_rows),
    }
    if observed_counts != EXPECTED_BUILTIN_COMPONENT_COUNTS:
        raise RuntimeError(
            "Built-in component catalog counts do not match the supported set: "
            f"expected {EXPECTED_BUILTIN_COMPONENT_COUNTS}, got {observed_counts}"
        )

    interface_counts = dict(Counter(row["input_interface"] for row in predictor_rows))
    if interface_counts != EXPECTED_PREDICTOR_INTERFACE_COUNTS:
        raise RuntimeError(
            "Predictor interface counts do not match the supported set: "
            f"expected {EXPECTED_PREDICTOR_INTERFACE_COUNTS}, got {interface_counts}"
        )


def generate_component_catalog_rsts() -> dict[str, str]:
    """Return deterministic RST tables for every built-in component registry.

    :returns: generated RST keyed by component registry
    """
    register_builtin_components()
    cell_line_rows = _builtin_rows(
        cast(list[FeaturizerCatalogMetadata], list_cell_line_featurizer_metadata()),
        BUILTIN_CELL_LINE_FEATURIZER_NAMES,
        "cell-line featurizers",
    )
    drug_rows = _builtin_rows(
        cast(list[FeaturizerCatalogMetadata], list_drug_featurizer_metadata()),
        BUILTIN_DRUG_FEATURIZER_NAMES,
        "drug featurizers",
    )
    predictor_rows = _builtin_rows(
        cast(list[PredictorCatalogMetadata], list_predictor_metadata()),
        BUILTIN_PREDICTOR_NAMES,
        "predictors",
    )
    _validate_builtin_catalog(
        cell_line_rows=cell_line_rows,
        drug_rows=drug_rows,
        predictor_rows=predictor_rows,
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
        write_text_if_changed(path, generated[key])
    return tuple(GENERATED_CATALOGS.values())
