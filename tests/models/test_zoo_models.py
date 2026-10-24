"""Train/predict/round-trip gate for the zoo models, global and single-drug, on a Leave-Tissue-Out fold."""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import DrugFeaturizerConfig, ModelConfig, SampleFeaturizerConfig, from_spec
from drevalpy.models.zoo import get_zoo_config
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset
from drevalpy.utils.seed import seed_everything
from tests.synthetic.variants import (
    SAVE_LOAD_DEFECTS,
    SUPPORTED_GLOBAL_MODELS,
    SUPPORTED_SINGLE_DRUG_MODELS,
    build_partial_coverage_dataset,
    model_param,
)

pytestmark = pytest.mark.slow


def _zoo_config_variant(name: str, **updates: Any) -> ModelConfig:
    """Build a variant of a zoo preset by re-validating an updated dump."""
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
    if whole_name == "PaccMann":
        config = _zoo_config_variant(
            "PaccMann",
            sample_featurizer=SampleFeaturizerConfig(
                name="landmarkGenes",
                options={
                    "gene_list_stem": "gene_list_paccmann_network_prop_reduced",
                    "arcsinh": False,
                    "fit_on_pairs": True,
                },
            ),
            drug_featurizer=DrugFeaturizerConfig(name="smilesTokens", options={"padding_length": 64, "n_variants": 4}),
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
    elif model_name == "EnsembleMF":
        hpam_combi["n_ensemble"] = 2
        hpam_combi["max_epochs"] = 2
        hpam_combi["hidden_dim"] = 16
        hpam_combi["emb_dim"] = 8
        hpam_combi["mlp_hidden"] = 8
    elif model_name == "PaccMann":
        hpam_combi["epochs"] = 1
        hpam_combi["batch_size"] = 32
        hpam_combi["filters"] = [4, 4, 4]
        hpam_combi["molecule_heads"] = [1, 1, 1, 1]
        hpam_combi["gene_heads"] = [1, 1, 1, 1]
        hpam_combi["stacked_dense_hidden_sizes"] = [16]
    elif model_name == "SRMF":
        hpam_combi["max_iter"] = 2
    elif model_name == "RandomForestAEMolGNet":
        hpam_combi["epochs_autoencoder"] = 1
        hpam_combi["n_estimators"] = 5
    elif model_name == "AdaBoostDecisionTree":
        hpam_combi["max_depth"] = 2
        hpam_combi["min_samples_split"] = 2
        hpam_combi["min_samples_leaf"] = 2
        hpam_combi["n_estimators"] = 2


def _stub_molgnet_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve seeded node embeddings in place of the MolGNet checkpoint, which needs a download."""

    def embed(smiles: str) -> np.ndarray:
        return np.random.default_rng(len(smiles)).normal(size=(5 + len(smiles) % 7, 768)).astype(np.float32)

    monkeypatch.setattr("drevalpy.components.featurizers.drug.molgnet._compute_molgnet_embedding", embed)


def _first_lto_fold(mudataset: Dataset) -> SplitMasks:
    """Return the first Leave-Tissue-Out fold of *mudataset*."""
    from drevalpy.data import CurveQuality
    from drevalpy.registry.splitter import get as get_splitter

    return get_splitter("LTO")(mudataset, n_splits=2, validation_ratio=0.4, quality=CurveQuality.off())[0]


def _train(model: DRPModel, mudataset: Dataset, split: SplitMasks) -> None:
    """Train *model* on the fold's training scope, early-stopping on its validation split when supported."""
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.train(
            mudataset,
            split.train,
            early_stopping_scope=split.early_stopping_mask()[0] if model.spec.supports_early_stopping else None,
            model_checkpoint_dir=checkpoint_dir,
        )


def _assert_round_trips(
    model: DRPModel,
    model_name: str,
    mudataset: Dataset,
    split: SplitMask,
    preds: np.ndarray,
) -> None:
    """Assert a save/load cycle reproduces the prediction shape."""
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


def _resolve_single_drug_model_name(whole_name: str) -> str:
    if whole_name.startswith("SingleDrugRandomForest"):
        return "SingleDrugRandomForest"
    if whole_name.startswith("SingleDrugElasticNet"):
        return "SingleDrugElasticNet"
    return whole_name


def _construct_single_drug_spec(whole_name: str, model_name: str) -> ModelSpec:
    if whole_name.endswith("[proteomics]"):
        predictor_token = model_name.replace("SingleDrug", "singleDrug")
        return ModelSpec(model_name, from_spec(f"normalizedProteomics:{predictor_token}"))
    return ModelSpec.from_name(model_name)


def _configure_single_drug_hpam(model_name: str, hpam_combi: dict) -> None:
    """Shrink the model to the smallest configuration that still exercises it."""
    if model_name == "SingleDrugRandomForest":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1


@pytest.mark.parametrize("model_name", [model_param(name) for name in SUPPORTED_GLOBAL_MODELS])
def test_global_models(synthetic_dataset: Dataset, model_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each global model trains, predicts and reloads on a Leave-Tissue-Out fold."""
    if model_name == "RandomForestAEMolGNet":
        _stub_molgnet_embeddings(monkeypatch)
    split = _first_lto_fold(synthetic_dataset)

    whole_name, model_name = _resolve_global_model_name(model_name)
    spec = _construct_global_model_spec(whole_name, model_name)
    hpam_combi = dict(spec.default_hyperparameters)
    _apply_global_model_hpam_tweaks(model_name, hpam_combi)
    model = DRPModel(spec, hpam_combi)

    _train(model, synthetic_dataset, split)

    preds = model.predict(synthetic_dataset, split.test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    _assert_round_trips(model, model_name, synthetic_dataset, split.test, preds)


def test_multi_view_neural_network_custom_views(synthetic_dataset: Dataset) -> None:
    split = _first_lto_fold(synthetic_dataset)
    spec = ModelSpec(
        "MultiViewNeuralNetwork",
        _zoo_config_variant(
            "MultiViewNeuralNetwork", sample_featurizer=SampleFeaturizerConfig(name="raw", view="methylation")
        ),
    )
    model = DRPModel(spec, {"units_per_layer": [2, 2], "dropout_prob": 0.3, "max_epochs": 1})

    _train(model, synthetic_dataset, split)

    preds = model.predict(synthetic_dataset, split.test)
    assert preds.shape[0] > 0
    _assert_round_trips(model, "MultiViewNeuralNetwork", synthetic_dataset, split.test, preds)


def test_partial_omics_coverage_trains() -> None:
    mudataset = build_partial_coverage_dataset()
    assert len(mudataset.entities_with_modality("gene_expression")) < len(mudataset.sample_ids)
    split = _first_lto_fold(mudataset)
    model = DRPModel(ModelSpec.from_name("SimpleNeuralNetwork"), {"units_per_layer": [2, 2], "max_epochs": 1})

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(mudataset, split.train, model_checkpoint_dir=tmpdirname)


@pytest.mark.parametrize("model_name", [model_param(name) for name in SUPPORTED_SINGLE_DRUG_MODELS])
def test_single_drug_models(synthetic_dataset: Dataset, model_name: str) -> None:
    """Each single-drug model trains, predicts and reloads on a Leave-Tissue-Out fold."""
    seed_everything(42)

    whole_name = model_name
    model_name = _resolve_single_drug_model_name(whole_name)

    split = _first_lto_fold(synthetic_dataset)

    spec = _construct_single_drug_spec(whole_name, model_name)
    hpam_combi = dict(spec.default_hyperparameters)
    _configure_single_drug_hpam(model_name, hpam_combi)
    model = DRPModel(spec, hpam_combi)

    _train(model, synthetic_dataset, split)

    preds = model.predict(synthetic_dataset, split.test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded_model = DRPModel.load(checkpoint)
        assert isinstance(loaded_model, DRPModel)
        np.testing.assert_allclose(preds, loaded_model.predict(synthetic_dataset, split.test), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "model_name",
    [model_param(name) for name in SUPPORTED_SINGLE_DRUG_MODELS if not name.endswith("[proteomics]")],
)
def test_n_jobs_changes_the_speed_not_the_model(synthetic_dataset: Dataset, model_name: str, tmp_path) -> None:
    """Fitting the drugs in two worker processes gives the model of fitting them one after the other."""
    name = _resolve_single_drug_model_name(model_name)
    split = _first_lto_fold(synthetic_dataset)
    spec = _construct_single_drug_spec(model_name, name)
    hpam_combi = dict(spec.default_hyperparameters)
    _configure_single_drug_hpam(name, hpam_combi)
    if name in ("MOLIR", "SuperFELTR"):
        hpam_combi["mini_batch"] = 6  # at most the 7 pairs of a drug, so every drug really trains

    predictions, restored = {}, {}
    for n_jobs in (1, 2):
        seed_everything(42)
        model = DRPModel(spec, hpam_combi)
        model.train(
            synthetic_dataset,
            split.train,
            early_stopping_scope=split.early_stopping_mask()[0] if spec.supports_early_stopping else None,
            model_checkpoint_dir=tmp_path / f"checkpoints_{n_jobs}",
            n_jobs=n_jobs,
        )
        predictions[n_jobs] = model.predict(synthetic_dataset, split.test)
        model.save(tmp_path / f"model_{n_jobs}")
        restored[n_jobs] = DRPModel.load(tmp_path / f"model_{n_jobs}").predict(synthetic_dataset, split.test)

    assert np.isfinite(predictions[1]).all()
    np.testing.assert_array_equal(predictions[1], predictions[2])
    np.testing.assert_array_equal(restored[1], restored[2])
    np.testing.assert_allclose(predictions[2], restored[2], rtol=1e-5, atol=1e-5)


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
