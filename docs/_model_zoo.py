"""Docs-only helpers for generating the model zoo catalog from zoo YAML + registries."""

from __future__ import annotations

from pathlib import Path

from _generated_io import write_text_if_changed

from drevalpy.models.zoo import get_zoo_config, list_zoo_names
from drevalpy.registry._builtins import register_builtins
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.enums.model_scope import ModelScope

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_MODEL_ZOO = DOCS_DIR / "reference" / "_generated_model_zoo.rst"


def _predictor_description(predictor_name: str) -> str:
    meta = predictor_registry.metadata(predictor_name)
    description = (meta.get("description") or "").strip()
    return description or "No description."


def _render_scope_table(scope: ModelScope) -> list[str]:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 28 42 30",
        "",
        "   * - Name",
        "     - Description",
        "     - Composition",
    ]
    for name in list_zoo_names(include_external=False, scope=scope):
        config = get_zoo_config(name)
        description = _predictor_description(config.predictor.name)
        recipe = config.model_id
        if recipe is None:
            raise RuntimeError(f"Zoo preset {name!r} has no recipe")
        lines.extend(
            [
                f"   * - {name}",
                f"     - {description}",
                f"     - ``{recipe}``",
            ]
        )
    lines.append("")
    return lines


def generate_model_zoo_rst() -> str:
    """Return RST tables for built-in multi-drug and single-drug zoo presets.

    :returns: RST source with multi-drug and single-drug list tables
    """
    register_builtins()
    lines = [
        "Multi-drug models",
        "-----------------",
        "",
        *(_render_scope_table(ModelScope.MULTI_DRUG)),
        "Single-drug models",
        "------------------",
        "",
        *(_render_scope_table(ModelScope.SINGLE_DRUG)),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_generated_model_zoo() -> Path:
    """Write the generated model zoo RST consumed by ``reference/model_zoo.rst``.

    :returns: path to the written ``_generated_model_zoo.rst`` file
    """
    write_text_if_changed(GENERATED_MODEL_ZOO, generate_model_zoo_rst())
    return GENERATED_MODEL_ZOO
