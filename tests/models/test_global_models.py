"""Train/predict/round-trip gate for the models that are not single-drug models."""

from __future__ import annotations

import tempfile
from typing import Any, cast

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import get_zoo_config
from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset
from tests.synthetic.variants import (
    SAVE_LOAD_DEFECTS,
    SUPPORTED_GLOBAL_MODELS,
    build_partial_coverage_dataset,
    model_param,
)


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


def _construct_global_model_class(whole_name: str, model_name: str) -> type[DRPModel]:
    if whole_name == "SimpleNeuralNetwork[chemberta]":
        config = _zoo_config_variant(
            "SimpleNeuralNetwork",
            drug_featurizer=DrugFeaturizerConfig(
                name="view",
                options={"view": "chemberta"},
            ),
        )
        return cast(type[DRPModel], construct_model(model_name, config))
    return cast(type[DRPModel], construct_model(model_name))


def _apply_global_model_hpam_tweaks(model_name: str, hpam_combi: dict) -> None:
    """Shrink the model to the smallest configuration that still exercises it."""
    if model_name == "DIPK":
        hpam_combi["batch_size"] = 1
        hpam_combi["epochs"] = 1
        hpam_combi["epochs_autoencoder"] = 1
        hpam_combi["heads"] = 1
    elif model_name in ["SimpleNeuralNetwork", "MultiViewNeuralNetwork"]:
        hpam_combi["units_per_layer"] = [2, 2]
        hpam_combi["max_epochs"] = 1
    elif model_name == "PharmaFormer":
        hpam_combi["epochs"] = 1
        hpam_combi["patience"] = 2
    elif model_name in {"Precily", "SparseGO"}:
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
    model_class: type[DRPModel],
    model_name: str,
    mudataset: Dataset,
    split: SplitMasks,
    preds: np.ndarray,
) -> None:
    """Assert a save/load cycle reproduces the prediction shape.

    Models with a known set-dependent-featurizer defect are asserted to still
    raise it, so the defect cannot be fixed without this test noticing.

    :param model: Trained model instance.
    :param model_class: Class used to reload the checkpoint.
    :param model_name: Model name, used to look up known defects.
    :param mudataset: Dataset the model was trained on.
    :param split: Split the predictions were made over.
    :param preds: Predictions from the in-memory model.
    """
    defect = SAVE_LOAD_DEFECTS.get(model_name)
    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded_model = model_class.load(checkpoint)
        assert isinstance(loaded_model, DRPModel)
        if defect is not None:
            with pytest.raises(RuntimeError, match=defect):
                loaded_model.predict(mudataset, split)
            return
        assert preds.shape == loaded_model.predict(mudataset, split).shape


@pytest.mark.parametrize("model_name", [model_param(name) for name in SUPPORTED_GLOBAL_MODELS])
def test_global_models(synthetic_dataset: Dataset, model_name: str) -> None:
    """Each global model trains, predicts and reloads on a Leave-Tissue-Out fold.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    :param model_name: Model name, possibly with a ``[view]`` suffix.
    """
    split = _first_lto_fold(synthetic_dataset)

    whole_name, model_name = _resolve_global_model_name(model_name)
    model_class = _construct_global_model_class(whole_name, model_name)
    hpam_combi = dict(model_class.get_hyperparameter_set()[0])
    _apply_global_model_hpam_tweaks(model_name, hpam_combi)
    model = model_class(hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(synthetic_dataset, split, model_checkpoint_dir=tmpdirname)

    preds = model.predict(synthetic_dataset, split)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    _assert_round_trips(model, model_class, model_name, synthetic_dataset, split, preds)


def test_multi_view_neural_network_custom_views(synthetic_dataset: Dataset) -> None:
    """MultiViewNeuralNetwork runs with a non-default cell-line view.

    Uses an existing modality (methylation) through the raw featurizer to verify
    the flexible input pipeline works end-to-end including save/load. Overriding
    the preset's featurizer list also drops its copy-number view, so this variant
    covers the single-view path rather than the preset's multi-omics one.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
    split = _first_lto_fold(synthetic_dataset)

    model_class = cast(
        type[DRPModel],
        construct_model(
            "MultiViewNeuralNetwork",
            _zoo_config_variant(
                "MultiViewNeuralNetwork",
                cell_line_featurizer=CellLineFeaturizerConfig(
                    name="raw",
                    view="methylation",
                ),
            ),
        ),
    )

    model = model_class({"units_per_layer": [2, 2], "dropout_prob": 0.3, "max_epochs": 1})

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(synthetic_dataset, split, model_checkpoint_dir=tmpdirname)

    preds = model.predict(synthetic_dataset, split)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    _assert_round_trips(model, model_class, "MultiViewNeuralNetwork", synthetic_dataset, split, preds)


def test_partial_omics_coverage_reaches_the_nan_filtering_path() -> None:
    """The partial-coverage variant really does leave cell lines without omics data.

    Guards the premise of the test below: if coverage silently became complete,
    that test would start passing for the wrong reason.
    """
    mudataset = build_partial_coverage_dataset()
    covered = mudataset.entities_with_modality("gene_expression")
    assert len(covered) < len(mudataset.cell_line_ids)


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
    model_class = cast(type[DRPModel], construct_model("SimpleNeuralNetwork"))
    model = model_class({"units_per_layer": [2, 2], "max_epochs": 1})

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(mudataset, split, model_checkpoint_dir=tmpdirname)
