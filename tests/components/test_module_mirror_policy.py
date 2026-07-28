"""Policy test: executable drevalpy modules must have mirrored tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "drevalpy"
TESTS_ROOT = REPO_ROOT / "tests"

REQUIRED_MIRRORS = (
    "components/tuning/config_resolution.py",
    "components/tuning/public_flat.py",
    "components/tuning/compatibility_keys.py",
    "components/register_builtins.py",
    "components/data_loading.py",
    "components/pair_features.py",
    "components/predictors/naive/mean.py",
    "components/predictors/naive/entity_mean.py",
    "components/predictors/naive/tissue.py",
    "components/predictors/naive/effects.py",
    "components/predictors/feature_free.py",
    "components/predictors/raw_dataset.py",
    "components/predictors/literature/structured_engine_adapter.py",
    "components/predictors/literature/block_engine_adapter.py",
    "components/predictors/literature/raw_engine_adapter.py",
    "components/predictors/literature/_engine_resolve.py",
    "components/predictors/literature/_engine_mixin.py",
    "components/predictors/literature/_raw_views.py",
    "components/predictors/literature/precily_predictor.py",
    "components/predictors/literature/srmf_predictor.py",
    "components/predictors/literature/molir_predictor.py",
    "components/predictors/literature/superfeltr_predictor.py",
    "components/predictors/literature/pharmaformer_predictor.py",
    "components/predictors/literature/dipk_predictor.py",
    "components/predictors/literature/sparsego_predictor.py",
    "types/literature_reference.py",
    "components/featurizers/cell_line/scaled_gene_expression.py",
    "components/featurizers/cell_line/landmark.py",
)


def _mirrored_test_path(relative_module: str) -> Path:
    rel = Path(relative_module)
    return TESTS_ROOT / rel.parent / f"test_{rel.name}"


@pytest.mark.parametrize("relative_module", REQUIRED_MIRRORS)
def test_executable_module_has_mirrored_test(relative_module: str) -> None:
    module_path = PACKAGE_ROOT / relative_module
    expected = _mirrored_test_path(relative_module)
    assert module_path.is_file(), f"missing source module: {relative_module}"
    assert (
        expected.is_file()
    ), f"missing mirrored test for {relative_module}: expected {expected.relative_to(REPO_ROOT)}"
