"""SuperFELTR training orchestration (encoders + regressor)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.components.predictors.literature.molir.utils import get_dimensions_of_omics_data, make_ranges
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .utils import SuperFELTEncoder, SuperFELTRegressor, train_superfeltr_model

if TYPE_CHECKING:
    from .algorithm import SuperFELTR


def _train_encoder_for_omic(
    model: SuperFELTR,
    omic_type: str,
    dim: int,
    output: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: DrugResponseDataset | None,
    model_checkpoint_dir: str,
) -> SuperFELTEncoder:
    encoder = SuperFELTEncoder(input_size=dim, hpams=model.hyperparameters, omic_type=omic_type, ranges=model.ranges)
    if len(output) >= model.hyperparameters["mini_batch"]:
        print(f"Training SuperFELTR Encoder for {omic_type} ... ")
        best_checkpoint = train_superfeltr_model(
            model=encoder,
            hpams=model.hyperparameters,
            output_train=output,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
            patience=5,
            model_checkpoint_dir=model_checkpoint_dir,
            wandb_project=model.wandb_project,
        )
        return SuperFELTEncoder.load_from_checkpoint(best_checkpoint.best_model_path)
    print(f"Not enough training data provided for SuperFELTR Encoder for {omic_type}. Using random initialization.")
    return encoder


def run_superfeltr_training(
    model: SuperFELTR,
    output: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: DrugResponseDataset | None,
    model_checkpoint_dir: str,
) -> None:
    """Train encoders and regressor on featurizer-preprocessed omics.

    :param model: SuperFELTR algorithm instance being trained.
    :param output: Training responses and pair identifiers.
    :param cell_line_input: Cell-line omics feature dataset.
    :param output_earlystopping: Optional validation responses for early stopping.
    :param model_checkpoint_dir: Directory for encoder checkpoint persistence.
    """
    if len(output) <= 0:
        print("No training data provided, skipping model")
        model.best_checkpoint = None
        model.expr_encoder, model.mut_encoder, model.cnv_encoder, model.regressor = None, None, None, None
        return

    model.record_feature_names(cell_line_input)
    if output_earlystopping is not None and model.early_stopping and len(output_earlystopping) < 2:
        output_earlystopping = None
    dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)
    model.ranges = make_ranges(output)

    encoder_dims = {
        "expression": dim_gex,
        "mutation": dim_mut,
        "copy_number_variation_gistic": dim_cnv,
    }
    encoders = {
        omic_type: _train_encoder_for_omic(
            model,
            omic_type,
            dim,
            output,
            cell_line_input,
            output_earlystopping,
            model_checkpoint_dir,
        )
        for omic_type, dim in encoder_dims.items()
    }

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
    if len(output) >= model.hyperparameters["mini_batch"]:
        print("Training SuperFELTR Regressor ... ")
        model.best_checkpoint = train_superfeltr_model(
            model=model.regressor,
            hpams=model.hyperparameters,
            output_train=output,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
            patience=5,
            model_checkpoint_dir=model_checkpoint_dir,
            wandb_project=model.wandb_project,
        )
    else:
        print("Not enough training data provided for SuperFELTR Regressor. Using random initialization.")
        model.best_checkpoint = None

    if model.best_checkpoint is not None:
        model.regressor = SuperFELTRegressor.load_from_checkpoint(
            model.best_checkpoint.best_model_path,
            input_size=regressor_input_size,
            hpams=model.hyperparameters,
            encoders=(model.expr_encoder, model.mut_encoder, model.cnv_encoder),
        )
