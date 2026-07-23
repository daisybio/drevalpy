"""Architecture policy: no legacy bridges, adapters, or non-facade factory classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models._native_drp_model import NativeDRPModel

_FORBIDDEN_MODULE_FRAGMENTS = (
    "_legacy_",
    "legacy_checkpoint_migration",
    "_component_bridge",
    "predictors.baselines",
    "models.baselines",
    "literature.public_models",
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
    )
    for path in repo.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in content:
                hits.append(f"{path.relative_to(repo.parent)}:{needle}")
    assert not hits, f"Forbidden imports/references remain: {hits}"


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_factory_classes_are_canonical_facades(model_name: str) -> None:
    cls = MODEL_FACTORY[model_name]
    assert issubclass(cls, NativeDRPModel)
    assert cls.__module__ == "drevalpy.models"
    assert getattr(cls, "_model_spec", None) == model_name
