"""Train/predict/round-trip gate for the models that are not single-drug models."""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import DrugFeaturizerConfig, ModelConfig, SampleFeaturizerConfig
from drevalpy.models.zoo import get_zoo_config
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset
from tests.synthetic.variants import (
    SAVE_LOAD_DEFECTS,
    SUPPORTED_GLOBAL_MODELS,
    build_partial_coverage_dataset,
    model_param,
)

#: Extended tier: every test here trains a real model. PharmaFormer alone is 2.1s
#: and the file is ~2.6s. The dependency-light models keep their fast-tier
#: coverage in ``test_direct_component_harness.py``.
pytestmark = pytest.mark.slow


def _zoo_config_variant(name: str, **updates: Any) -> ModelConfig:
    """Build a variant of a zoo preset by re-validating an updated dump.

    :param name: Zoo preset name.
    :param updates: ``ModelConfig`` field overrides.
    :returns: Newly validated ``ModelConfig``.
    """
    payload = get_zoo_config(name).model_dump(mode="python")
    payload.update(updates)
    return ModelConfig.model_validate(payload)


def _resolve_global_model_name(model_name: str) -> tuple[str, str]:
    whole_name = model_name
    if model_name.startswith("SimpleNeuralNetwork"):
        model_name = "SimpleNeuralNetwork"
    return whole_name, model_name


def _construct_global_model_spec(whole_name: str, model_name: str) -> ModelSpec:
    if whole_name == "SimpleNeuralNetwork[chemberta]":
        config = _zoo_config_variant(
            "SimpleNeuralNetwork",
            drug_featurizer=DrugFeaturizerConfig(
                name="view",
                options={"view": "chemberta"},
            ),
        )
        return ModelSpec(model_name, config)
    return ModelSpec.from_name(model_name)


def _apply_global_model_hpam_tweaks(model_name: str, hpam_combi: dict) -> None:
    """Shrink the model to the smallest configuration that still exercises it."""
    if model_name == "DIPK":
        hpam_combi["batch_size"] = 8
        hpam_combi["epochs"] = 1
        hpam_combi["epochs_autoencoder"] = 1
        hpam_combi["heads"] = 1
    elif model_name in ["SimpleNeuralNetwork", "MultiViewNeuralNetwork"]:
        hpam_combi["units_per_layer"] = [2, 2]
        hpam_combi["max_epochs"] = 1
    elif model_name == "PharmaFormer":
        hpam_combi["epochs"] = 1
        hpam_combi["patience"] = 2
    elif model_name in {"Precily", "SparseGO", "DrugGNN"}:
        hpam_combi["epochs"] = 1
        hpam_combi["batch_size"] = 32
    elif model_name == "SRMF":
        hpam_combi["max_iter"] = 2
    elif model_name == "AdaBoostDecisionTree":
        hpam_combi["max_depth"] = 2
        hpam_combi["min_samples_split"] = 2
        hpam_combi["min_samples_leaf"] = 2
        hpam_combi["n_estimators"] = 2


def _first_lto_fold(mudataset: Dataset) -> SplitMasks:
    """Return the first Leave-Tissue-Out fold of *mudataset*.

    :param mudataset: Dataset to split.
    :returns: The first fold's split masks.
    """
    from drevalpy.registry.splitter import get as get_splitter

    return get_splitter("LTO")(mudataset, n_splits=2, validation_ratio=0.4)[0]


def _assert_round_trips(
    model: DRPModel,
    model_name: str,
    mudataset: Dataset,
    split: SplitMask,
    preds: np.ndarray,
) -> None:
    """Assert a save/load cycle reproduces the prediction shape.

    Models with a known set-dependent-featurizer defect are asserted to still
    raise it, so the defect cannot be fixed without this test noticing.

    :param model: Trained model instance.
    :param model_name: Model name, used to look up known defects.
    :param mudataset: Dataset the model was trained on.
    :param split: Split the predictions were made over.
    :param preds: Predictions from the in-memory model.
    """
    defect = SAVE_LOAD_DEFECTS.get(model_name)
    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded_model = DRPModel.load(checkpoint)
        assert isinstance(loaded_model, DRPModel)
        if defect is not None:
            with pytest.raises(RuntimeError, match=defect):
                loaded_model.predict(mudataset, split)
            return
        np.testing.assert_allclose(preds, loaded_model.predict(mudataset, split), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model_name", [model_param(name) for name in SUPPORTED_GLOBAL_MODELS])
def test_global_models(synthetic_dataset: Dataset, model_name: str) -> None:
    """Each global model trains, predicts and reloads on a Leave-Tissue-Out fold.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    :param model_name: Model name, possibly with a ``[view]`` suffix.
    """
    split = _first_lto_fold(synthetic_dataset)

    whole_name, model_name = _resolve_global_model_name(model_name)
    spec = _construct_global_model_spec(whole_name, model_name)
    hpam_combi = dict(spec.default_hyperparameters)
    _apply_global_model_hpam_tweaks(model_name, hpam_combi)
    model = DRPModel(spec, hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(
            synthetic_dataset,
            split.train,
            early_stopping_scope=split.early_stopping_mask()[0] if model.spec.supports_early_stopping else None,
            model_checkpoint_dir=tmpdirname,
        )

    preds = model.predict(synthetic_dataset, split.test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    _assert_round_trips(model, model_name, synthetic_dataset, split.test, preds)


def test_multi_view_neural_network_custom_views(synthetic_dataset: Dataset) -> None:
    """MultiViewNeuralNetwork runs with a non-default sample view.

    Uses an existing modality (methylation) through the raw featurizer to verify
    the flexible input pipeline works end-to-end including save/load. Overriding
    the preset's featurizer list also drops its copy-number view, so this variant
    covers the single-view path rather than the preset's multi-omics one.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
    split = _first_lto_fold(synthetic_dataset)

    spec = ModelSpec(
        "MultiViewNeuralNetwork",
        _zoo_config_variant(
            "MultiViewNeuralNetwork",
            sample_featurizer=SampleFeaturizerConfig(
                name="raw",
                view="methylation",
            ),
        ),
    )

    model = DRPModel(spec, {"units_per_layer": [2, 2], "dropout_prob": 0.3, "max_epochs": 1})

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(
            synthetic_dataset,
            split.train,
            early_stopping_scope=split.early_stopping_mask()[0] if model.spec.supports_early_stopping else None,
            model_checkpoint_dir=tmpdirname,
        )

    preds = model.predict(synthetic_dataset, split.test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    _assert_round_trips(model, "MultiViewNeuralNetwork", synthetic_dataset, split.test, preds)


def test_partial_omics_coverage_reaches_the_nan_filtering_path() -> None:
    """The partial-coverage variant really does leave samples without omics data.

    Guards the premise of the test below: if coverage silently became complete,
    that test would start passing for the wrong reason.
    """
    mudataset = build_partial_coverage_dataset()
    covered = mudataset.entities_with_modality("gene_expression")
    assert len(covered) < len(mudataset.sample_ids)


def test_partial_omics_coverage_trains() -> None:
    """Training over partial omics coverage exercises ``PredictorBase``' NaN filtering.

    ``fit`` routes the batch through ``ModelInputBatch.subset_pairs``, which
    narrows the early-stopping pairs to the drugs surviving the NaN mask. That
    mask spans every drug, so this is the multi-drug path; ``train`` completing
    is the assertion. Every early-stopping predictor takes the same route, so
    one representative is enough.
    """
    mudataset = build_partial_coverage_dataset()
    split = _first_lto_fold(mudataset)
    model = DRPModel(ModelSpec.from_name("SimpleNeuralNetwork"), {"units_per_layer": [2, 2], "max_epochs": 1})

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(mudataset, split.train, model_checkpoint_dir=tmpdirname)


@pytest.mark.network
class TestDIPKArtifactIntegration:
    """Train the full facade with actual pretrained PPI and MolGNet artifacts."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_raw_features_train_and_restore(self, seed: int, tmp_path) -> None:
        import hashlib
        import json

        from upath import UPath

        from drevalpy.data.artifacts import get_artifact
        from tests.components.predictors.literature._helpers import _isolated_reference_rng
        from tests.synthetic.builders import build_synthetic_dataset

        manifest = json.loads((UPath(__file__).parents[2] / "tools/literature_reference/manifest.json").read_text())
        for name in ("MolGNet.pt", "human_ppi_features.tsv", "gene_list_sel.txt"):
            # Actual cached/downloaded bytes, with no artifact or backend mocking.
            assert hashlib.sha256(get_artifact(name).read_bytes()).hexdigest() == manifest["artifacts"][name]["sha256"]

        with _isolated_reference_rng(seed):
            dataset = build_synthetic_dataset(seed=seed)
            split = _first_lto_fold(dataset)
            model = DRPModel(
                ModelSpec.from_name("DIPK"), {"epochs": 2, "epochs_autoencoder": 2, "batch_size": 8, "heads": 1}
            )
            model.train(
                dataset,
                split.train,
                early_stopping_scope=split.early_stopping_mask()[0],
                model_checkpoint_dir=str(tmp_path),
            )
            predictions = model.predict(dataset, split.test)
            assert np.isfinite(predictions).all()
            checkpoint = str(tmp_path / "model")
            model.save(checkpoint)
            restored = DRPModel.load(checkpoint)
            np.testing.assert_allclose(predictions, restored.predict(dataset, split.test), rtol=1e-5, atol=1e-6)
