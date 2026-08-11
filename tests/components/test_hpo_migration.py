"""Guard tests for the structured HPO migration."""

from __future__ import annotations

import inspect
from pathlib import Path


def test_experiment_tuning_does_not_use_parameter_grid() -> None:
    from drevalpy.models.tuning.hpo import hpam_tune

    source = inspect.getsource(hpam_tune)
    assert "ParameterGrid" not in source
    assert "grid_search" not in source


def test_drp_model_does_not_load_yaml_hyperparameters() -> None:
    from drevalpy.models import drp_model

    source = inspect.getsource(drp_model.DRPModel.get_hyperparameter_set)
    assert "yaml" not in source.lower()
    assert "ParameterGrid" not in source


def test_predictor_hyperparameters_yaml_removed() -> None:
    """v2 ParameterGrid YAML is unused; HPO uses Python get_*_hyperparameters."""
    predictors_root = Path(__file__).resolve().parents[2] / "drevalpy" / "components" / "predictors"
    leftover = sorted(predictors_root.rglob("hyperparameters.yaml"))
    assert leftover == [], f"remove unused v2 YAML: {leftover}"


def test_package_has_no_v2_hyperparameters_yaml() -> None:
    package_root = Path(__file__).resolve().parents[2] / "drevalpy"
    leftover = sorted(package_root.rglob("hyperparameters.yaml"))
    assert leftover == [], f"remove unused v2 YAML under drevalpy/: {leftover}"


_LEGACY_ENGINE_ADAPTER_NAMES = (
    "structured_engine_adapter.py",
    "raw_engine_adapter.py",
    "block_engine_adapter.py",
    "_engine_base.py",
    "_engine_mixin.py",
    "_engine_resolve.py",
)


def test_literature_has_no_legacy_engine_adapters() -> None:
    literature_root = Path(__file__).resolve().parents[2] / "drevalpy" / "components" / "predictors" / "literature"
    present = [name for name in _LEGACY_ENGINE_ADAPTER_NAMES if (literature_root / name).is_file()]
    assert present == [], f"remove legacy engine adapters: {present}"


def test_literature_impl_tree_removed() -> None:
    impl_root = Path(__file__).resolve().parents[2] / "drevalpy" / "components" / "predictors" / "literature" / "impl"
    assert not impl_root.exists(), f"remove unused literature/impl tree: {impl_root}"


def test_literature_has_no_flat_predictor_wrappers() -> None:
    literature_root = Path(__file__).resolve().parents[2] / "drevalpy" / "components" / "predictors" / "literature"
    flat_wrappers = sorted(path.name for path in literature_root.glob("*_predictor.py") if path.is_file())
    legacy_modules = sorted(
        name
        for name in ("druggnn.py", "neural_network.py", "structured_predictors.py")
        if (literature_root / name).is_file()
    )
    assert flat_wrappers == [], f"remove flat literature wrappers: {flat_wrappers}"
    assert legacy_modules == [], f"remove legacy literature modules: {legacy_modules}"
