"""Capture deterministic tiny-fit evidence for all literature predictors.

Run this file with the repository's absolute virtualenv Python and select the
implementation through ``PYTHONPATH``.  The benchmark deliberately contains no
imports from ``tests`` so the same file can execute against a source archive.

The default mode preserves the compact one-epoch fixture used by the frozen v2
references. ``--extended`` runs four multi-batch epochs, records the actual
training/validation delegates and Lightning checkpoint traces, and labels models
without early stopping as fixed-budget fits (SRMF uses eight iterations).
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
from upath import UPath

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import (
    FeaturizerResult,
    graph_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch

EXTENDED = False


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _parameters(module: Any) -> np.ndarray:
    parts = [parameter.detach().cpu().numpy().ravel() for parameter in module.parameters() if parameter.requires_grad]
    return np.concatenate(parts)


def _fit_evidence(
    predictor: Any,
    batch: ModelInputBatch,
    initial: np.ndarray | None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predictor.fit(batch)
    predictions = np.asarray(predictor.predict(batch), dtype=np.float64).ravel()
    evidence: dict[str, Any] = {
        "n_train": int(batch.n_pairs),
        "n_validation": int(len(batch.early_stopping_response)) if batch.early_stopping_response is not None else 0,
        "predictions": predictions.tolist(),
        "mse": float(np.mean((predictions - np.asarray(batch.response)) ** 2)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
    }
    fitted_module = getattr(predictor, "_model", None)
    if initial is not None and fitted_module is not None:
        delta = np.abs(_parameters(fitted_module) - initial)
        evidence["parameter_updates"] = int(np.count_nonzero(delta))
        evidence["max_parameter_delta"] = float(delta.max(initial=0.0))
    if checkpoint:
        evidence["checkpoint"] = checkpoint

    restored = type(predictor)()
    try:
        restored.set_state(predictor.get_state())
        restored_predictions = np.asarray(restored.predict(batch), dtype=np.float64).ravel()
        if np.isfinite(restored_predictions).all():
            evidence["state_roundtrip_max_error"] = float(np.max(np.abs(restored_predictions - predictions)))
        else:
            evidence["state_roundtrip_max_error"] = "non-finite"
    except Exception as exc:  # evidence collector: preserve the concrete failure
        evidence["state_roundtrip_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _two_by_two(
    cell_line_blocks: dict[str, Any],
    drug_blocks: dict[str, Any],
    checkpoint_dir: UPath,
    *,
    early_stopping: bool = False,
) -> ModelInputBatch:
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    return ModelInputBatch.from_response(
        response,
        cell_line=FeaturizerResult.from_blocks(cell_line_blocks, np.array(["cl1", "cl2"])),
        drug=FeaturizerResult(
            np.array(["d1", "d2"]),
            drug_blocks,
            tuple(name for name, block in drug_blocks.items() if block.values.dtype.kind in "biufc"),
        ),
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=(
            ResponseBatch(
                response=np.array([1.5, 2.5], dtype=np.float32),
                cell_line_ids=np.array(["cl1", "cl2"]),
                drug_ids=np.array(["d1", "d2"]),
            )
            if early_stopping
            else None
        ),
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


def _early_stopping_batch(
    cell_line_blocks: dict[str, Any],
    drug_blocks: dict[str, Any],
    checkpoint_dir: UPath,
) -> ModelInputBatch:
    training = ResponseBatch(
        response=np.array([-1.0, 0.5, 1.5, 3.0], dtype=np.float32),
        cell_line_ids=np.array(["train-1", "train-1", "train-2", "train-2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    validation = ResponseBatch(
        response=np.array([0.25, 2.25], dtype=np.float32),
        cell_line_ids=np.array(["val-1", "val-2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    return ModelInputBatch.from_response(
        training,
        cell_line=FeaturizerResult.from_blocks(cell_line_blocks, np.array(["train-1", "train-2", "val-1", "val-2"])),
        drug=FeaturizerResult(
            np.array(["d1", "d2"]),
            drug_blocks,
            tuple(name for name, block in drug_blocks.items() if block.values.dtype.kind in "biufc"),
        ),
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=validation,
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


def _omics_batch(checkpoint_dir: UPath) -> ModelInputBatch:
    rng = np.random.default_rng(2025)
    cell_ids = np.array([*(f"train-{idx}" for idx in range(6)), "val-0", "val-1"])
    training = ResponseBatch(
        response=np.array([0.0, 0.02, 0.3, 0.7, 0.98, 1.0], dtype=np.float32),
        cell_line_ids=cell_ids[:6],
        drug_ids=np.repeat("d1", 6),
    )
    validation = ResponseBatch(
        response=np.array([0.15, 0.85], dtype=np.float32),
        cell_line_ids=cell_ids[6:],
        drug_ids=np.repeat("d1", 2),
    )
    widths = {"gene_expression": 6, "mutations": 5, "copy_number_variation_gistic": 4}
    blocks = {
        name: numeric_feature_block(rng.normal(size=(8, width)).astype(np.float32)) for name, width in widths.items()
    }
    return ModelInputBatch.from_response(
        training,
        cell_line=FeaturizerResult.from_blocks(blocks, cell_ids),
        drug=FeaturizerResult(
            np.array(["d1"]),
            {
                "identity": numeric_feature_block(np.ones((1, 1), dtype=np.float32)),
                "identity_categories": numeric_feature_block(np.array([["d1"]], dtype=object)),
            },
            ("identity",),
        ),
        cell_line_pair_idx=np.arange(6),
        drug_pair_idx=np.zeros(6, dtype=np.int64),
        early_stopping_response=validation,
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


def _dipk(seed: int, root: UPath) -> dict[str, Any]:
    from drevalpy.components.predictors.literature.dipk.predictor import DIPKPredictor

    rng = np.random.default_rng(321)
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [rng.normal(size=(2, 768)).astype(np.float32), rng.normal(size=(3, 768)).astype(np.float32)]
    batch = _early_stopping_batch(
        {
            "gene_expression": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
            "bionic_features": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
        },
        {"molgnet_features": ragged_feature_block(molgnet)},
        root,
    )
    hp = {
        "batch_size": 2 if EXTENDED else 4,
        "lr": 0.001,
        "heads": 1,
        "fc_layer_num": 3,
        "fc_layer_dim": [8, 8, 8, 8, 8, 8],
        "dropout_rate": 0.0,
        "epochs": 4 if EXTENDED else 1,
        "patience": 1,
    }
    _seed(seed)
    initial = _parameters(DIPKPredictor(hp)._build_model())
    _seed(seed)
    predictor = DIPKPredictor(hp)
    if not EXTENDED:
        return _fit_evidence(predictor, batch, initial)

    traces: dict[str, list[float]] = {"training": [], "validation": []}
    run_epoch = predictor._run_epoch

    def traced_epoch(model: Any, loader: Any, loss_func: Any, optimizer: Any) -> float:
        loss = run_epoch(model, loader, loss_func, optimizer)
        traces["training" if optimizer is not None else "validation"].append(float(loss))
        return loss

    predictor._run_epoch = traced_epoch
    result = _fit_evidence(predictor, batch, initial)
    result.update(
        {
            "training_policy": "early_stopping",
            "epochs_configured": 4,
            "patience": 1,
            "training_loss_history": traces["training"],
            "validation_loss_history": traces["validation"],
        }
    )
    return result


def _druggnn(seed: int, root: UPath) -> dict[str, Any]:
    import pytorch_lightning as pl
    import torch
    from torch_geometric.data import Data

    from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor

    def graph(offset: float) -> Data:
        return Data(
            x=torch.arange(36, dtype=torch.float32).reshape(4, 9) / 10 + offset,
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.zeros(4, dtype=torch.long),
        )

    graphs = np.empty(2, dtype=object)
    graphs[:] = [graph(0.0), graph(0.5)]
    batch = _two_by_two(
        {"gene_expression": numeric_feature_block(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))},
        {"drug_graph": graph_feature_block(graphs)},
        root,
        early_stopping=True,
    )
    hp = {"epochs": 4, "batch_size": 2, "hidden_dim": 8, "dropout": 0.0, "learning_rate": 0.5}
    _seed(seed)
    pristine = DrugGNNPredictor(hp)
    initial = _parameters(pristine._build_model(batch.cell_line.blocks["gene_expression"].values, graphs))
    _seed(seed)
    selected = DrugGNNPredictor(hp)
    selected.fit(batch)
    if selected._model is None:
        raise RuntimeError("DrugGNN fit did not produce a model")

    class ValidationHistory(pl.Callback):
        def __init__(self) -> None:
            self.losses: list[float] = []

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if not trainer.sanity_checking:
                self.losses.append(float(trainer.callback_metrics["val_loss"]))

    _seed(seed)
    final = DrugGNNPredictor(hp)
    cell_matrix = batch.cell_line.blocks["gene_expression"].values
    final._model = final._build_model(cell_matrix, graphs)
    train_loader = final._graph_loader(
        batch.cell_line_pair_idx,
        final._resolve_drug_pair_idx(batch),
        cell_matrix,
        graphs,
        batch.response,
        batch_size=2,
        shuffle=True,
    )
    val_loader = final._build_val_loader(batch, cell_matrix, graphs)
    history = ValidationHistory()
    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="cpu",
        devices=1,
        callbacks=[history],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(final._model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    evaluator = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    selected_loss = evaluator.validate(selected._model, dataloaders=val_loader, verbose=False)[0]["val_loss"]
    final_loss = evaluator.validate(final._model, dataloaders=val_loader, verbose=False)[0]["val_loss"]
    predictions = np.asarray(selected.predict(batch), dtype=np.float64)
    delta = np.abs(_parameters(selected._model) - initial)
    restored = DrugGNNPredictor()
    restored.set_state(selected.get_state())
    result = {
        "n_train": 4,
        "n_validation": 2,
        "predictions": predictions.tolist(),
        "mse": float(np.mean((predictions - batch.response) ** 2)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "parameter_updates": int(np.count_nonzero(delta)),
        "max_parameter_delta": float(delta.max(initial=0.0)),
        "validation_history": history.losses,
        "selected_validation_loss": float(selected_loss),
        "final_validation_loss": float(final_loss),
        "state_roundtrip_max_error": float(np.max(np.abs(restored.predict(batch) - predictions))),
    }
    if EXTENDED:
        result.update(
            {
                "training_policy": "validation_checkpoint",
                "epochs_configured": 4,
                "batch_size": 2,
            }
        )
    return result


def _molir(seed: int, root: UPath) -> dict[str, Any]:
    from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor

    hp = {
        "epochs": 4 if EXTENDED else 1,
        "mini_batch": 3 if EXTENDED else 6,
        "h_dim1": 4,
        "h_dim2": 4,
        "h_dim3": 4,
        "dropout_rate": 0.0,
    }
    batch = _omics_batch(root)
    _seed(seed)
    initial = _parameters(MOLIRPredictor(hp)._build_model((6, 5, 4)))
    _seed(seed)
    predictor = MOLIRPredictor(hp)
    validation_trace: list[float] = []
    if EXTENDED:
        build_model = predictor._build_model

        def traced_build(widths: tuple[int, int, int]) -> Any:
            model = build_model(widths)
            validation_step = model.validation_step
            fit_model = model.fit

            def traced_validation(batch: Any, batch_idx: int) -> Any:
                loss = validation_step(batch, batch_idx)
                validation_trace.append(float(loss.detach()))
                return loss

            def traced_fit(*args: Any, **kwargs: Any) -> Any:
                kwargs["patience"] = 1
                return fit_model(*args, **kwargs)

            model.validation_step = traced_validation
            model.fit = traced_fit
            return model

        predictor._build_model = traced_build
    predictor.fit(batch)
    model = predictor._models["d1"]
    checkpoint = model.checkpoint_callback
    predictions = np.asarray(predictor.predict(batch), dtype=np.float64)
    delta = np.abs(_parameters(model) - initial)
    result = {
        "n_train": 6,
        "n_validation": 2,
        "predictions": predictions.tolist(),
        "mse": float(np.mean((predictions - batch.response) ** 2)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "parameter_updates": int(np.count_nonzero(delta)),
        "max_parameter_delta": float(delta.max(initial=0.0)),
        "checkpoint_score": float(checkpoint.best_model_score),
    }
    if EXTENDED:
        result.update(
            {
                "training_policy": "early_stopping",
                "epochs_configured": 4,
                "patience": 1,
                "validation_loss_history": validation_trace[1:],
            }
        )
    restored = MOLIRPredictor()
    restored.set_state(predictor.get_state())
    result["state_roundtrip_max_error"] = float(np.max(np.abs(restored.predict(batch) - predictions)))
    return result


def _pharmaformer(seed: int, root: UPath) -> dict[str, Any]:
    import importlib

    from drevalpy.components.predictors.literature.pharmaformer.predictor import (
        PharmaFormerPredictor,
        _build_combined_model,
    )

    rng = np.random.default_rng(987)
    batch = _early_stopping_batch(
        {"gene_expression": numeric_feature_block(rng.normal(size=(4, 6)).astype(np.float32))},
        {"bpe_smiles": numeric_feature_block(rng.normal(size=(2, 128)).astype(np.float32))},
        root,
    )
    hp = {
        "epochs": 4 if EXTENDED else 1,
        "patience": 1,
        "batch_size": 2 if EXTENDED else 4,
        "lr": 0.001,
        "dropout": 0.0,
        "gene_hidden_size": 16,
        "drug_hidden_size": 16,
        "feature_dim": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
    }
    _seed(seed)
    pristine = PharmaFormerPredictor(hp)
    initial = _parameters(_build_combined_model(6, hp, pristine._device))
    _seed(seed)
    predictor = PharmaFormerPredictor(hp)
    if not EXTENDED:
        return _fit_evidence(predictor, batch, initial)

    module = importlib.import_module("drevalpy.components.predictors.literature.pharmaformer.predictor")
    run_epoch = module._run_epoch
    traces: dict[str, list[float]] = {"training": [], "validation": []}

    def traced_epoch(model: Any, loader: Any, loss_func: Any, optimizer: Any, device: Any) -> float:
        loss = run_epoch(model, loader, loss_func, optimizer, device)
        traces["training" if optimizer is not None else "validation"].append(float(loss))
        return loss

    module._run_epoch = traced_epoch
    try:
        result = _fit_evidence(predictor, batch, initial)
    finally:
        module._run_epoch = run_epoch
    result.update(
        {
            "training_policy": "early_stopping",
            "epochs_configured": 4,
            "patience": 1,
            "training_loss_history": traces["training"],
            "validation_loss_history": traces["validation"],
        }
    )
    return result


def _precily(seed: int, root: UPath) -> dict[str, Any]:
    from drevalpy.components.predictors.literature.precily.predictor import PrecilyPredictor

    batch = _two_by_two(
        {"pathways": numeric_feature_block(np.array([[0.1, 0.4, 0.8, 1.2], [1.3, 0.7, -0.2, 0.5]], dtype=np.float32))},
        {
            "smilesvec": numeric_feature_block(
                np.array([[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.4, 0.6]], dtype=np.float32)
            )
        },
        root,
    )
    hp = {
        "epochs": 4 if EXTENDED else 1,
        "batch_size": 2 if EXTENDED else 4,
        "dropout": 0.0,
        "learning_rate": 0.01,
    }
    _seed(seed)
    pristine = PrecilyPredictor(hp)
    initial = _parameters(pristine._build_model(8, 0.0))
    _seed(seed)
    predictor = PrecilyPredictor(hp)
    if not EXTENDED:
        return _fit_evidence(predictor, batch, initial)
    training_trace: list[float] = []
    run_epoch = predictor._run_epoch

    def traced_epoch(model: Any, loader: Any, loss_func: Any, optimizer: Any) -> float:
        loss = run_epoch(model, loader, loss_func, optimizer)
        training_trace.append(float(loss))
        return loss

    predictor._run_epoch = traced_epoch
    result = _fit_evidence(predictor, batch, initial)
    result.update(
        {
            "training_policy": "fixed_budget",
            "epochs_configured": 4,
            "training_loss_history": training_trace,
        }
    )
    return result


def _sparsego(seed: int, root: UPath) -> dict[str, Any]:
    import torch

    from drevalpy.components.predictors.literature.sparsego.predictor import SparseGOPredictor

    metadata = {
        "layer_connections": [
            np.array([["term-1", "gene-1"], ["term-1", "gene-2"], ["term-2", "gene-3"]]),
            np.array([["root", "term-1"], ["root", "term-2"]]),
        ],
        "gene2id_mapping_ont": {"gene-1": 0, "gene-2": 1, "gene-3": 2},
        "ontology_gene_order": ["gene-1", "gene-2", "gene-3"],
    }
    batch = _two_by_two(
        {
            "gene_expression": numeric_feature_block(
                np.array([[0.1, 0.7, 1.4], [1.3, -0.2, 0.4]], dtype=np.float32), metadata=metadata
            )
        },
        {
            "fingerprints": numeric_feature_block(
                np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
            )
        },
        root,
    )
    hp = {
        "num_neurons_per_GO": 2,
        "num_neurons_per_final_GO": 2,
        "num_neurons_drug": [3, 2],
        "num_neurons_final": 3,
        "drug_dim": 4,
        "learning_rate": 0.05,
        "momentum": 0.0,
        "decay_rate": 0.0,
        "p_drop_genes": 0.0,
        "p_drop_terms": 0.0,
        "p_drop_drugs": 0.0,
        "p_drop_final": 0.0,
        "epochs": 4 if EXTENDED else 1,
        "batch_size": 2 if EXTENDED else 4,
    }
    _seed(seed)
    pristine = SparseGOPredictor(hp)
    pristine._layer_connections = metadata["layer_connections"]
    pristine._gene2id_mapping_ont = metadata["gene2id_mapping_ont"]
    pristine._build_network()
    initial = _parameters(pristine._model)
    _seed(seed)
    predictor = SparseGOPredictor(hp)
    if not EXTENDED:
        return _fit_evidence(predictor, batch, initial)

    batch_losses: list[float] = []
    mse_forward = torch.nn.MSELoss.forward

    def traced_mse(loss_module: Any, inputs: Any, targets: Any) -> Any:
        loss = mse_forward(loss_module, inputs, targets)
        batch_losses.append(float(loss.detach()))
        return loss

    torch.nn.MSELoss.forward = traced_mse
    try:
        result = _fit_evidence(predictor, batch, initial)
    finally:
        torch.nn.MSELoss.forward = mse_forward
    batches_per_epoch = 2
    result.update(
        {
            "training_policy": "fixed_budget",
            "epochs_configured": 4,
            "training_loss_history": [
                float(np.mean(batch_losses[start : start + batches_per_epoch]))
                for start in range(0, len(batch_losses), batches_per_epoch)
            ],
        }
    )
    return result


def _srmf(seed: int, root: UPath) -> dict[str, Any]:
    from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor

    batch = _two_by_two(
        {"gene_expression": numeric_feature_block(np.array([[0.1, 0.4, 0.9], [1.1, -0.2, 0.5]], dtype=np.float32))},
        {
            "fingerprints": numeric_feature_block(
                np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
            )
        },
        root,
    )
    iterations = 8 if EXTENDED else 3
    predictor = SRMFPredictor({"K": 2, "max_iter": iterations, "n_features": 4, "seed": seed})
    predictor.fit(batch)
    predictions = np.asarray(predictor.predict(batch), dtype=np.float64)
    factors = np.concatenate([predictor._best_u.to_numpy().ravel(), predictor._best_v.to_numpy().ravel()])
    restored = SRMFPredictor()
    restored.set_state(predictor.get_state())
    result = {
        "n_train": 4,
        "n_validation": 0,
        "predictions": predictions.tolist(),
        "mse": float(np.mean((predictions - batch.response) ** 2)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "finite_factors": bool(np.isfinite(factors).all()),
        "factor_count": int(factors.size),
        "state_roundtrip_max_error": float(np.max(np.abs(restored.predict(batch) - predictions))),
    }
    if EXTENDED:
        result.update({"training_policy": "fixed_budget", "iterations_configured": iterations})
    return result


def _superfeltr(seed: int, root: UPath) -> dict[str, Any]:
    import importlib

    from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor

    hp = {
        "epochs": 4 if EXTENDED else 1,
        "mini_batch": 3 if EXTENDED else 2,
        "out_dim_expr_encoder": 3,
        "out_dim_mutation_encoder": 3,
        "out_dim_cnv_encoder": 2,
        "dropout_rate": 0.0,
    }
    batch = _omics_batch(root)
    _seed(seed)
    predictor = SuperFELTRPredictor(hp)
    traces: dict[str, list[float]] = {}
    checkpoint_trace: list[dict[str, Any]] = []
    if not EXTENDED:
        predictor.fit(batch)
    else:
        utils_module = importlib.import_module("drevalpy.components.predictors.literature.superfeltr.utils")
        train_model = utils_module.train_superfeltr_model

        def traced_train(model: Any, hpams: Any, train: Any, val: Any = None, **kwargs: Any) -> Any:
            label = getattr(model, "omic_type", type(model).__name__)
            trace = traces.setdefault(str(label), [])
            validation_step = model.validation_step

            def traced_validation(batch: Any, batch_idx: int) -> Any:
                loss = validation_step(batch, batch_idx)
                trace.append(float(loss.detach()))
                return loss

            model.validation_step = traced_validation
            kwargs["patience"] = 1
            callback = train_model(model, hpams, train, val, **kwargs)
            checkpoint_trace.append(
                {
                    "model": str(label),
                    "score": float(callback.best_model_score),
                    "selected": bool(callback.best_model_path),
                }
            )
            return callback

        utils_module.train_superfeltr_model = traced_train
        try:
            predictor.fit(batch)
        finally:
            utils_module.train_superfeltr_model = train_model
    predictions = np.asarray(predictor.predict(batch), dtype=np.float64)
    dm = predictor._drug_models["d1"]
    callback = dm.best_checkpoint
    result: dict[str, Any] = {
        "n_train": 6,
        "n_validation": 2,
        "predictions": predictions.tolist(),
        "mse": float(np.mean((predictions - batch.response) ** 2)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "checkpoint_score": float(callback.best_model_score),
        "checkpoint_selected": bool(callback.best_model_path),
    }
    if EXTENDED:
        result.update(
            {
                "training_policy": "early_stopping",
                "epochs_configured": 4,
                "patience": 1,
                "validation_loss_history": {name: values[1:] for name, values in traces.items()},
                "checkpoint_trace": checkpoint_trace,
            }
        )
    restored = SuperFELTRPredictor()
    restored.set_state(predictor.get_state())
    restored_predictions = np.asarray(restored.predict(batch), dtype=np.float64)
    result["state_roundtrip_max_error"] = (
        float(np.max(np.abs(restored_predictions - predictions)))
        if np.isfinite(restored_predictions).all()
        else "non-finite"
    )
    return result


RUNNERS: dict[str, Callable[[int, UPath], dict[str, Any]]] = {
    "dipk": _dipk,
    "druggnn": _druggnn,
    "molir": _molir,
    "pharmaformer": _pharmaformer,
    "precily": _precily,
    "sparsego": _sparsego,
    "srmf": _srmf,
    "superfeltr": _superfeltr,
}


def main() -> None:
    """Run all seeded fits and write their compact JSON evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run four-epoch, multi-batch fits and capture training/validation traces.",
    )
    args = parser.parse_args()

    global EXTENDED
    EXTENDED = args.extended
    evidence: dict[str, Any] = {
        "implementation": args.implementation,
        "protocol": "extended" if EXTENDED else "default",
        "seeds": {},
    }
    if EXTENDED:
        evidence["training_regimes"] = {
            "early_stopping": ["dipk", "molir", "pharmaformer", "superfeltr"],
            "validation_checkpoint": ["druggnn"],
            "fixed_budget": ["precily", "sparsego", "srmf"],
        }
    for seed in (0, 1, 2):
        seed_results: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix=f"literature-{seed}-") as directory:
            root = UPath(directory)
            for name, runner in RUNNERS.items():
                seed_results[name] = runner(seed, root / name)
        evidence["seeds"][str(seed)] = seed_results
    UPath(args.output).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
