"""Training loop for the PharmaFormer literature engine."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .model_utils import CombinedModel

if TYPE_CHECKING:
    from .pharmaformer import PharmaFormerModel, _PharmaFormerDataset


def _build_combined_model(gene_input_size: int, hyperparameters: dict[str, Any], device: torch.device) -> CombinedModel:
    return CombinedModel(
        gene_input_size=gene_input_size,
        gene_hidden_size=hyperparameters["gene_hidden_size"],
        drug_hidden_size=hyperparameters["drug_hidden_size"],
        feature_dim=hyperparameters["feature_dim"],
        nhead=hyperparameters["nhead"],
        num_layers=hyperparameters.get("num_layers", 3),
        dim_feedforward=hyperparameters.get("dim_feedforward", 2048),
        dropout=hyperparameters.get("dropout", 0.1),
    ).to(device)


def _run_epoch(
    model: CombinedModel,
    loader: DataLoader,
    loss_func: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Run one train or validation epoch; optimizer None skips backward pass."""
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    batch_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for gene_inputs, smiles_inputs, batch_targets in loader:
            gene_inputs = gene_inputs.to(device)
            smiles_inputs = smiles_inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(gene_inputs, smiles_inputs)
            loss = loss_func(outputs.squeeze(), batch_targets)

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            else:
                epoch_loss += loss.item()

            batch_count += 1
            predictions.append(outputs.squeeze().detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())

    return epoch_loss / batch_count, predictions, targets


def _log_epoch_metrics(
    engine: PharmaFormerModel,
    train_loss: float,
    train_predictions: list[np.ndarray],
    train_targets: list[np.ndarray],
    val_loss: float,
    val_predictions: list[np.ndarray],
    val_targets: list[np.ndarray],
    epoch: int,
) -> None:
    train_metrics: dict[str, float] = {"train_loss": train_loss}
    if train_predictions:
        all_train_preds = np.concatenate(train_predictions)
        all_train_targets = np.concatenate(train_targets)
        train_metrics.update(engine.compute_performance_metrics(all_train_preds, all_train_targets, prefix="train_"))

    val_metrics: dict[str, float] = {"val_loss": val_loss}
    if val_predictions:
        all_val_preds = np.concatenate(val_predictions)
        all_val_targets = np.concatenate(val_targets)
        val_metrics.update(engine.compute_performance_metrics(all_val_preds, all_val_targets, prefix="val_"))

    if engine.is_wandb_enabled():
        engine.log_metrics(train_metrics, step=epoch)
        engine.log_metrics(val_metrics, step=epoch)


def run_pharmaformer_training(
    engine: PharmaFormerModel,
    output: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    drug_input: FeatureDataset,
    output_earlystopping: DrugResponseDataset,
    model_checkpoint_dir: str,
    pharmaformer_dataset_cls: type[_PharmaFormerDataset],
) -> None:
    """Train PharmaFormer with early stopping and reload the best checkpoint."""
    gene_input_size = cell_line_input.get_feature_matrix(
        view="gene_expression", identifiers=output.cell_line_ids
    ).shape[1]
    engine._saved_gene_input_size = gene_input_size
    engine.model = _build_combined_model(gene_input_size, engine.hyperparameters, engine.DEVICE)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(engine.model.parameters(), lr=engine.hyperparameters["lr"])

    train_dataset = pharmaformer_dataset_cls(
        response=output.response,
        cell_line_ids=output.cell_line_ids,
        drug_ids=output.drug_ids,
        cell_line_features=cell_line_input,
        drug_features=drug_input,
    )
    early_stopping_dataset = pharmaformer_dataset_cls(
        response=output_earlystopping.response,
        cell_line_ids=output_earlystopping.cell_line_ids,
        drug_ids=output_earlystopping.drug_ids,
        cell_line_features=cell_line_input,
        drug_features=drug_input,
    )

    train_loader = DataLoader(train_dataset, batch_size=engine.hyperparameters["batch_size"], shuffle=True)
    early_stopping_loader = DataLoader(
        early_stopping_dataset, batch_size=engine.hyperparameters["batch_size"], shuffle=False
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    version = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(20)])
    checkpoint_path = os.path.join(model_checkpoint_dir, f"{version}_best_PharmaFormer_model.pth")

    print("Training PharmaFormer model")
    for epoch in range(engine.hyperparameters["epochs"]):
        epoch_loss, train_predictions, train_targets = _run_epoch(
            engine.model, train_loader, loss_func, optimizer, engine.DEVICE
        )
        print(
            f"PharmaFormer: Epoch [{epoch + 1}/{engine.hyperparameters['epochs']}] " f"Training Loss: {epoch_loss:.4f}"
        )

        val_loss, val_predictions, val_targets = _run_epoch(
            engine.model, early_stopping_loader, loss_func, None, engine.DEVICE
        )
        print(
            f"PharmaFormer: Epoch [{epoch + 1}/{engine.hyperparameters['epochs']}] " f"Validation Loss: {val_loss:.4f}"
        )

        _log_epoch_metrics(
            engine,
            epoch_loss,
            train_predictions,
            train_targets,
            val_loss,
            val_predictions,
            val_targets,
            epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(engine.model.state_dict(), checkpoint_path)  # noqa: S614
            print(f"PharmaFormer: Saved best model at epoch {epoch + 1}")
        else:
            epochs_without_improvement += 1
            patience = engine.hyperparameters.get("patience", 10)
            if epochs_without_improvement >= patience:
                print(f"PharmaFormer: Early stopping triggered at epoch {epoch + 1}")
                break

    print("PharmaFormer: Reloading the best model")
    engine.model.load_state_dict(
        torch.load(checkpoint_path, map_location=engine.DEVICE, weights_only=True)
    )  # noqa: S614
    engine.model.to(engine.DEVICE)
