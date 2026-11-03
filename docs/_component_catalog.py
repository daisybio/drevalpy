"""Generate built-in component tables from the registries and the component classes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from _generated_io import write_text_if_changed

from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.components.predictors.inputs import Inputs, predictor_inputs
from drevalpy.plugin import PrecomputableFeaturizer
from drevalpy.registry._builtins import builtin_names, get_skipped_builtin_modules, register_builtins
from drevalpy.registry._component import component_license
from drevalpy.registry.featurizer import featurizer_registries
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.data.feature_source import Side
from drevalpy.types.enums.accelerator import Accelerator

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_CATALOGS = {
    "sample_featurizer": DOCS_DIR / "reference" / "_generated_sample_featurizers.rst",
    "drug_featurizer": DOCS_DIR / "reference" / "_generated_drug_featurizers.rst",
    "predictor": DOCS_DIR / "reference" / "_generated_predictors.rst",
}
EXPECTED_BUILTIN_COMPONENT_COUNTS = {"sample_featurizer": 18, "drug_featurizer": 12, "predictor": 31}
#: Built-in predictors per input kind; a predictor counts under ``matrix`` or ``blocks`` when either side has it.
EXPECTED_PREDICTOR_INPUT_COUNTS = {"none": 1, "matrix": 15, "blocks": 15}

#: View a featurizer that requires an explicit view is built with to read its declarations.
_PLACEHOLDER_VIEW = "gene_expression"


def _description(name: str, metadata: Any) -> str:
    description = " ".join((metadata.get("description") or "").split())
    if not description:
        raise RuntimeError(f"Component {name!r} is missing a description")
    return description


def _featurizer_config(name: str, cls: type[Any]) -> dict[str, Any]:
    """Return the construction kwargs of a built-in featurizer with its defaults.

    :param name: Registry name.
    :param cls: Featurizer class.
    :returns: Hyperparameter defaults, plus a placeholder view when the featurizer needs one.
    """
    config: dict[str, Any] = dict(cls.default_hyperparameters())
    if requires_explicit_view(name, cls.side):
        config["view"] = _PLACEHOLDER_VIEW
    return config


def _accelerator_cell(cls: type[Any]) -> str:
    level = cls.accelerator
    return "CPU only" if level is Accelerator.NONE else level.value.capitalize()


def _license_cell(cls: type[Any]) -> str:
    """Render the code license, and the license of the downloaded files when there are any."""
    cell = f"``{component_license(cls)}``"
    return f"{cell}; files: ``{cls.artifact_license}``" if cls.artifact_license else cell


def _output_formats(cls: type[Any], config: dict[str, Any]) -> str:
    formats = dict.fromkeys(spec.format.value for spec in cls.output_blocks(config))
    return ", ".join(f"``{fmt}``" for fmt in formats) or "Those of its children"


def _render_featurizers(side: Side, rows: list[tuple[str, type[Any]]]) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 14 12 10 14 30",
        "",
        "   * - Name",
        "     - Output format",
        "     - Precomputable",
        "     - Accelerator",
        "     - License",
        "     - Description",
    ]
    for name, cls in rows:
        config = _featurizer_config(name, cls)
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - {_output_formats(cls, config)}",
                f"     - {chr(0x2705) if issubclass(cls, PrecomputableFeaturizer) else chr(0x274C)}",
                f"     - {_accelerator_cell(cls)}",
                f"     - {_license_cell(cls)}",
                f"     - {_description(name, featurizer_registries[side].metadata(name))}",
            ]
        )
    return "\n".join([*lines, ""])


def _inputs_cell(inputs: Inputs) -> str:
    """Render one side's input declaration."""
    if inputs.kind != "blocks":
        return inputs.kind.capitalize()
    if not inputs.specs:
        return "Blocks: any"
    parts = []
    for spec in inputs.specs:
        names = [option.name for option in spec] if isinstance(spec, tuple) else [spec.name]
        parts.append(" or ".join(f"``{name}``" for name in names))
    return "Blocks: " + ", ".join(parts)


def _input_kind(cls: type[Any]) -> str:
    kinds = {predictor_inputs(cls, "sample").kind, predictor_inputs(cls, "drug").kind}
    return next((kind for kind in ("blocks", "matrix") if kind in kinds), "none")


def _render_predictors(rows: list[tuple[str, type[Any]]]) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 16 17 17 10 14 26",
        "",
        "   * - Name",
        "     - Sample inputs",
        "     - Drug inputs",
        "     - Accelerator",
        "     - License",
        "     - Description",
    ]
    for name, cls in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - {_inputs_cell(predictor_inputs(cls, 'sample'))}",
                f"     - {_inputs_cell(predictor_inputs(cls, 'drug'))}",
                f"     - {_accelerator_cell(cls)}",
                f"     - {_license_cell(cls)}",
                f"     - {_description(name, predictor_registry.metadata(name))}",
            ]
        )
    return "\n".join([*lines, ""])


def _skipped_module_report() -> str:
    """Return a human-readable report of built-in modules that failed to import.

    :returns: report text, or the empty string when every module imported cleanly
    """
    skipped = get_skipped_builtin_modules()
    if not skipped:
        return ""
    sections = [f"{name}:\n{tb.rstrip()}" for name, tb in sorted(skipped.items())]
    header = "The following built-in component modules failed to import, so their components are missing:"
    return "\n\n" + header + "\n\n" + "\n\n".join(sections)


def _builtin_rows(registry: Any) -> list[tuple[str, type[Any]]]:
    builtin = frozenset(builtin_names(registry))
    return sorted((name, cls) for name, cls in registry.items() if name in builtin)


def _validate_builtin_catalog(rows: dict[str, list[tuple[str, type[Any]]]]) -> None:
    """Fail generation when the built-in catalogs diverge from the supported set.

    :param rows: built-in components keyed like :data:`EXPECTED_BUILTIN_COMPONENT_COUNTS`
    :raises RuntimeError: if component or input-kind counts diverge from expectations
    """
    observed_counts = {key: len(entries) for key, entries in rows.items()}
    if observed_counts != EXPECTED_BUILTIN_COMPONENT_COUNTS:
        raise RuntimeError(
            "Built-in component catalog counts do not match the supported set: "
            f"expected {EXPECTED_BUILTIN_COMPONENT_COUNTS}, got {observed_counts}"
            f"{_skipped_module_report()}"
        )
    input_counts = dict(Counter(_input_kind(cls) for _, cls in rows["predictor"]))
    if input_counts != EXPECTED_PREDICTOR_INPUT_COUNTS:
        raise RuntimeError(
            "Predictor input counts do not match the supported set: "
            f"expected {EXPECTED_PREDICTOR_INPUT_COUNTS}, got {input_counts}"
        )


def generate_component_catalog_rsts() -> dict[str, str]:
    """Return deterministic RST tables for the built-in featurizers of each side and the predictors.

    :returns: generated RST keyed by ``sample_featurizer``, ``drug_featurizer`` and ``predictor``
    """
    register_builtins()
    rows = {f"{side}_featurizer": _builtin_rows(featurizer_registries[side]) for side in Side}
    rows["predictor"] = _builtin_rows(predictor_registry)
    _validate_builtin_catalog(rows)
    generated = {f"{side}_featurizer": _render_featurizers(side, rows[f"{side}_featurizer"]) for side in Side}
    generated["predictor"] = _render_predictors(rows["predictor"])
    return generated


def write_generated_component_catalogs() -> tuple[Path, ...]:
    """Write generated RST includes consumed by the component catalog.

    :returns: paths to the generated RST files
    """
    generated = generate_component_catalog_rsts()
    for key, path in GENERATED_CATALOGS.items():
        write_text_if_changed(path, generated[key])
    return tuple(GENERATED_CATALOGS.values())
