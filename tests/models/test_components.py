"""Tests for fresh component construction behind the model pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models._components import build_components
from drevalpy.models.config import from_spec
from drevalpy.models.tuning.search_space import resolve_model_config


def test_build_components_creates_predictor_and_configured_featurizers() -> None:
    resolved = resolve_model_config(from_spec("ElasticNet"))
    sample, drug, predictor = build_components(resolved)

    assert sample is not None
    assert drug is not None
    assert predictor.__class__.__name__


def test_build_components_returns_independent_instances() -> None:
    resolved = resolve_model_config(from_spec("ElasticNet"))
    first = build_components(resolved)
    second = build_components(resolved)

    assert first[0] is not second[0]
    assert first[1] is not second[1]
    assert first[2] is not second[2]


def test_build_components_passes_prediction_mode_as_a_plain_string() -> None:
    _, _, predictor = build_components(resolve_model_config(from_spec("ElasticNet")))

    mode = predictor._hyperparameters["prediction_mode"]
    assert mode == "regression"
    assert type(mode) is str


def _three_omics_dataset():
    """Build an eight-sample, one-drug dataset with the three views MOLIR and SuperFELTR read."""
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset

    rng = np.random.default_rng(0)
    sample_ids = np.array([f"cl{i}" for i in range(8)])
    response = ad.AnnData(
        X=rng.normal(size=(8, 1)).astype(np.float32),
        obs=pd.DataFrame({"cell_line_name": sample_ids, "tissue": ["Lung", "Blood"] * 4}, index=sample_ids),
        var=pd.DataFrame(index=["d1"]),
    )
    response.varm["morgan_fingerprint"] = np.ones((1, 2), dtype=np.float32)
    modalities = {"response": response}
    for view, prefix in (("gene_expression", "g"), ("mutations", "m"), ("copy_number_variation_gistic", "c")):
        modalities[view] = ad.AnnData(
            X=rng.normal(size=(8, 4)).astype(np.float32),
            obs=pd.DataFrame(index=sample_ids),
            var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(4)]),
        )
    return Dataset(md.MuData(modalities), name="test")


def _spy_on_best_epoch_reload(monkeypatch: pytest.MonkeyPatch, model_name: str) -> list[object]:
    """Record every best-epoch checkpoint the model reloads, and what the reload returned."""
    reloaded: list[object] = []
    if model_name == "MOLIR":
        from drevalpy.components.predictors.literature.molir import utils

        original = utils.load_state_dict

        def spy(path, *args, **kwargs):
            reloaded.append(original(path, *args, **kwargs))
            return reloaded[-1]

        monkeypatch.setattr(utils, "load_state_dict", spy)
    else:
        from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder, SuperFELTRegressor

        for network in (SuperFELTEncoder, SuperFELTRegressor):
            original = network.load_from_checkpoint.__func__

            def spy(cls, *args, _original=original, **kwargs):
                reloaded.append(_original(cls, *args, **kwargs))
                return reloaded[-1]

            monkeypatch.setattr(network, "load_from_checkpoint", classmethod(spy))
    return reloaded


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["MOLIR", "SuperFELTR"])
def test_fit_with_early_stopping_scope_reloads_the_best_epoch(
    model_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from drevalpy.models import DRPModel
    from drevalpy.registry._builtins import register_builtin_components
    from drevalpy.types.data.split_mask import SplitMask
    from tests.models.synthetic_fixtures import small_data_spec

    register_builtin_components()
    hyperparameters = {"epochs": 2, "mini_batch": 2, "dropout_rate": 0.0}
    if model_name == "MOLIR":
        hyperparameters.update({"h_dim1": 2, "h_dim2": 2, "h_dim3": 2})
    else:
        hyperparameters.update({"out_dim_expr_encoder": 2, "out_dim_mutation_encoder": 2, "out_dim_cnv_encoder": 2})
    mask = np.zeros((8, 1), dtype=bool)
    train, stop = mask.copy(), mask.copy()
    train[:6] = True
    stop[6:] = True

    reloaded = _spy_on_best_epoch_reload(monkeypatch, model_name)
    dataset = _three_omics_dataset()
    model = DRPModel(small_data_spec(model_name), hyperparameters)
    model.train(dataset, SplitMask(train), early_stopping_scope=SplitMask(stop), model_checkpoint_dir=tmp_path)

    assert reloaded, "the fit never reached the best-epoch reload"
    assert np.isfinite(model.predict(dataset, SplitMask(stop))).all()
