"""SuperFELTR training orchestration (encoders + regressor).

This module is retained for backward compatibility but the main training logic
has been inlined into the predictor's ``_fit_single_drug`` method.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from upath import UPath as Path

from .utils import SuperFELTEncoder, SuperFELTRegressor, train_superfeltr_model


class _SuperFELTRLike(Protocol):
    hyperparameters: dict[str, Any]
    wandb_project: str | None
    ranges: Any
    expr_encoder: SuperFELTEncoder | None
    mut_encoder: SuperFELTEncoder | None
    cnv_encoder: SuperFELTEncoder | None
    regressor: SuperFELTRegressor | None
    best_checkpoint: Any
    early_stopping: bool


def run_superfeltr_training(
    model: _SuperFELTRLike,
    gene_expression: np.ndarray,
    mutations: np.ndarray,
    copy_number: np.ndarray,
    response: np.ndarray,
    val_gene_expression: np.ndarray | None = None,
    val_mutations: np.ndarray | None = None,
    val_copy_number: np.ndarray | None = None,
    val_response: np.ndarray | None = None,
    model_checkpoint_dir: str | Path = "checkpoints",
) -> None:
    """Train encoders and regressor on pre-expanded pair-level omics matrices.

    :param model: SuperFELTR algorithm instance being trained.
    :param gene_expression: pair-level gene expression matrix (n_samples, n_features)
    :param mutations: pair-level mutation matrix (n_samples, n_features)
    :param copy_number: pair-level copy number matrix (n_samples, n_features)
    :param response: training response values (n_samples,)
    :param val_gene_expression: validation gene expression matrix
    :param val_mutations: validation mutation matrix
    :param val_copy_number: validation copy number matrix
    :param val_response: validation response values
    :param model_checkpoint_dir: Directory for encoder checkpoint persistence.
    """
    n_samples = gene_expression.shape[0]
    if n_samples <= 0:
        model.best_checkpoint = None
        model.expr_encoder, model.mut_encoder, model.cnv_encoder, model.regressor = None, None, None, None
        return

    pair_idx = np.arange(n_samples)
    dim_gex, dim_mut, dim_cnv = gene_expression.shape[1], mutations.shape[1], copy_number.shape[1]
    std = float(np.std(response))
    model.ranges = (std * 0.1, std)

    if model.early_stopping and val_response is not None and len(val_response) < 2:
        val_gene_expression = None
        val_mutations = None
        val_copy_number = None
        val_response = None

    encoder_dims = {
        "expression": dim_gex,
        "mutation": dim_mut,
        "copy_number_variation_gistic": dim_cnv,
    }
    encoders = {}
    for omic_type, dim in encoder_dims.items():
        encoder = SuperFELTEncoder(
            input_size=dim, hpams=model.hyperparameters, omic_type=omic_type, ranges=model.ranges
        )
        if n_samples >= model.hyperparameters["mini_batch"]:
            best_checkpoint = train_superfeltr_model(
                model=encoder,
                hpams=model.hyperparameters,
                gene_expression=gene_expression,
                mutations=mutations,
                copy_number=copy_number,
                response=response,
                pair_idx=pair_idx,
                val_gene_expression=val_gene_expression,
                val_mutations=val_mutations,
                val_copy_number=val_copy_number,
                val_response=val_response,
                val_pair_idx=np.arange(len(val_response)) if val_response is not None else None,
                patience=5,
                model_checkpoint_dir=model_checkpoint_dir,
                wandb_project=model.wandb_project,
            )
            encoder = SuperFELTEncoder.load_from_checkpoint(best_checkpoint.best_model_path)
        encoders[omic_type] = encoder

    model.expr_encoder = encoders["expression"]
    model.mut_encoder = encoders["mutation"]
    model.cnv_encoder = encoders["copy_number_variation_gistic"]

    regressor_input_size = (
        model.hyperparameters["out_dim_expr_encoder"]
        + model.hyperparameters["out_dim_mutation_encoder"]
        + model.hyperparameters["out_dim_cnv_encoder"]
    )
    model.regressor = SuperFELTRegressor(
        input_size=regressor_input_size,
        hpams=model.hyperparameters,
        encoders=(model.expr_encoder, model.mut_encoder, model.cnv_encoder),
    )
    if n_samples >= model.hyperparameters["mini_batch"]:
        model.best_checkpoint = train_superfeltr_model(
            model=model.regressor,
            hpams=model.hyperparameters,
            gene_expression=gene_expression,
            mutations=mutations,
            copy_number=copy_number,
            response=response,
            pair_idx=pair_idx,
            val_gene_expression=val_gene_expression,
            val_mutations=val_mutations,
            val_copy_number=val_copy_number,
            val_response=val_response,
            val_pair_idx=np.arange(len(val_response)) if val_response is not None else None,
            patience=5,
            model_checkpoint_dir=model_checkpoint_dir,
            wandb_project=model.wandb_project,
        )
    else:
        model.best_checkpoint = None

    if model.best_checkpoint is not None:
        model.regressor = SuperFELTRegressor.load_from_checkpoint(
            model.best_checkpoint.best_model_path,
            input_size=regressor_input_size,
            hpams=model.hyperparameters,
            encoders=(model.expr_encoder, model.mut_encoder, model.cnv_encoder),
        )
