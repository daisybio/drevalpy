"""Docs-only helpers for generating the model zoo catalog from zoo YAML + registries."""

from __future__ import annotations

from pathlib import Path

from _generated_io import list_table, registry_description, write_text_if_changed

from drevalpy.models.zoo import get_zoo_config, list_zoo_names
from drevalpy.registry._builtins import register_builtins
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.enums import ModelScope

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_MODEL_ZOO = DOCS_DIR / "reference" / "_generated_model_zoo.rst"


def _render_scope_table(scope: ModelScope) -> list[str]:
    rows = []
    for name in list_zoo_names(include_external=False, scope=scope):
        config = get_zoo_config(name)
        if config.model_id is None:
            raise RuntimeError(f"Zoo preset {name!r} has no recipe")
        rows.append((name, registry_description(predictor_registry, config.predictor.name), f"``{config.model_id}``"))
    return [*list_table((28, 42, 30), ("Name", "Description", "Composition"), rows), ""]


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
