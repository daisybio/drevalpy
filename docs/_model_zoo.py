"""Docs-only helpers for generating the model zoo catalog from zoo YAML + registries."""

from __future__ import annotations

from pathlib import Path

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor, get_predictor_metadata
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.zoo import get_zoo_config, list_zoo_names
from drevalpy.types.model_scope import ModelScope

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_MODEL_ZOO = DOCS_DIR / "concepts" / "_generated_model_zoo.rst"


def _featurizer_recipe(feat: FeaturizerConfig | None) -> str:
    if feat is None:
        return ""
    if feat.name == "concatFeaturizers":
        children = feat.hyperparameters.get("featurizers", [])
        parts: list[str] = []
        for child in children:
            if isinstance(child, FeaturizerConfig):
                parts.append(_featurizer_recipe(child))
            else:
                parts.append(_featurizer_recipe(FeaturizerConfig.model_validate(child)))
        return "+".join(parts) if parts else "concatFeaturizers"
    if feat.view:
        return f"{feat.name}[{feat.view}]"
    return feat.name


def _model_recipe(config: ModelConfig) -> str:
    if config.cell_line_featurizer is None and config.drug_featurizer is None:
        return config.predictor.name
    cell = _featurizer_recipe(config.cell_line_featurizer)
    drug = _featurizer_recipe(config.drug_featurizer)
    return f"{cell}:{drug}:{config.predictor.name}"


def _predictor_description(predictor_name: str) -> str:
    get_predictor(predictor_name)  # ensure builtins are loaded for this name
    meta = get_predictor_metadata(predictor_name)
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
        recipe = _model_recipe(config)
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
    register_builtin_components()
    lines = [
        "Multi-drug models",
        "-----------------",
        "",
        *(_render_scope_table(ModelScope.MULTI_DRUG)),
        "Single-drug models",
        "------------------",
        "",
        "Feature-based single-drug presets use ``identity`` to create and route one",
        "model per drug; identity is not appended to the predictor matrix. Raw",
        "single-drug literature models remain predictor-only.",
        "",
        *(_render_scope_table(ModelScope.SINGLE_DRUG)),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_generated_model_zoo() -> Path:
    """Write the generated model zoo RST consumed by ``concepts/model_zoo.rst``.

    :returns: path to the written ``_generated_model_zoo.rst`` file
    """
    GENERATED_MODEL_ZOO.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_MODEL_ZOO.write_text(generate_model_zoo_rst(), encoding="utf-8")
    return GENERATED_MODEL_ZOO
