"""Architecture policy: no legacy bridges, adapters, or deleted runtime modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from drevalpy.models import construct_model
from drevalpy.models._model_lookup import known_model_names
from drevalpy.models.drp_model import DRPModel

_FORBIDDEN_MODULE_FRAGMENTS = (
    "_legacy_",
    "legacy_checkpoint_migration",
    "_component_bridge",
    "predictors.baselines",
    "models.baselines",
    "literature.public_models",
    "_native_drp_model",
    "composed_model",
    "_factory_classes",
    "_component_persistence",
)


def test_no_forbidden_modules_exist() -> None:
    repo = Path(__file__).resolve().parents[2] / "drevalpy"
    forbidden_paths = []
    for path in repo.rglob("*.py"):
        text = str(path.relative_to(repo.parent))
        if any(fragment in text for fragment in _FORBIDDEN_MODULE_FRAGMENTS):
            forbidden_paths.append(text)
    assert not forbidden_paths, f"Forbidden modules remain: {forbidden_paths}"


def test_no_forbidden_runtime_imports_in_source() -> None:
    repo = Path(__file__).resolve().parents[2] / "drevalpy"
    hits = []
    needles = (
        "drevalpy.models._component_bridge",
        "drevalpy.models._legacy_",
        "legacy_checkpoint_migration",
        "predictors.baselines",
        "literature.public_models",
        "restore_naive_to_components",
        "restore_literature_to_components",
        "drevalpy.models._native_drp_model",
        "drevalpy.models.composed_model",
        "drevalpy.models._factory_classes",
        "drevalpy.models._component_persistence",
    )
    for path in repo.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in content:
                hits.append(f"{path.relative_to(repo.parent)}:{needle}")
    assert not hits, f"Forbidden imports/references remain: {hits}"


def test_production_code_does_not_import_model_factory_except_models_init() -> None:
    import re

    repo = Path(__file__).resolve().parents[2] / "drevalpy"
    allowed = repo / "models" / "__init__.py"
    import_pattern = re.compile(
        r"^\s*(?:from\s+\S+\s+import\s+.*MODEL_FACTORY|import\s+.*MODEL_FACTORY)",
        re.MULTILINE,
    )
    hits = []
    for path in repo.rglob("*.py"):
        if path == allowed:
            continue
        content = path.read_text(encoding="utf-8")
        if import_pattern.search(content):
            hits.append(str(path.relative_to(repo.parent)))
    assert not hits, f"MODEL_FACTORY imported outside drevalpy.models.__init__: {hits}"


@pytest.mark.parametrize("model_name", known_model_names(include_external=False))
def test_construct_model_classes_are_drp_models(model_name: str) -> None:
    cls = construct_model(model_name)
    assert issubclass(cls, DRPModel)
    assert cls.__module__ == "drevalpy.models"
    assert cls._model_name == model_name
