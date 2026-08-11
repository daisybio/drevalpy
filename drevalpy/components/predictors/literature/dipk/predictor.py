"""DIPK predictor consuming ModelInputBatch directly."""

from __future__ import annotations

import secrets
from typing import Any, ClassVar

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from upath import UPath as Path

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import DIPK_REFERENCE
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import load_state_dict, save_state_dict, save_torch_payload

from .model_utils import Predictor as DIPKNetwork


@register(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.RAGGED_SEQUENCE,
    reference=DIPK_REFERENCE,
)
class DIPKPredictor(BlockPredictor):
    """DIPK predictor that trains an attention-based model on gene expression, BIONIC, and MolGNet features."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression", "bionic_features")
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("bionic_features", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE),
    )
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the DIPK predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: DIPKNetwork | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters for DIPK.

        :returns: Default hyperparameter mapping.
        """
        return {
            "batch_size": 64,
            "lr": 0.0001,
            "heads": 2,
            "fc_layer_num": 3,
            "fc_layer_dim": [256, 128, 64, 32, 16, 1],
            "dropout_rate": 0.3,
            "epochs": 100,
            "patience": 10,
        }

    def _build_model(self) -> DIPKNetwork:
        """Construct the DIPK network from hyperparameters.

        :returns: Initialized DIPKNetwork placed on the target device.
        """
        hp = self._hyperparameters
        return DIPKNetwork(
            heads=hp["heads"],
            fc_layer_num=hp["fc_layer_num"],
            fc_layer_dim=hp["fc_layer_dim"],
            dropout_rate=hp["dropout_rate"],
        ).to(self._device)

    def _build_samples(
        self,
        cell_line_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray,
        batch: ModelInputBatch,
        response: np.ndarray | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Construct per-sample dictionaries from batch blocks using pair indices.

        :param cell_line_pair_idx: Indices mapping pairs to cell-line entities.
        :param drug_pair_idx: Indices mapping pairs to drug entities.
        :param batch: Model input batch with feature blocks.
        :param response: Optional response values per pair.
        :returns: List of sample dictionaries with tensor values.
        """
        gene_block = batch.cell_line_blocks["gene_expression"].values
        bionic_block = batch.cell_line_blocks["bionic_features"].values
        drug_block = batch.drug_blocks["molgnet_features"].values

        samples: list[dict[str, torch.Tensor]] = []
        for i in range(len(cell_line_pair_idx)):
            cl_idx = cell_line_pair_idx[i]
            dr_idx = drug_pair_idx[i]

            sample: dict[str, torch.Tensor] = {
                "molgnet_features": torch.tensor(np.asarray(drug_block[dr_idx], dtype=np.float32)),
                "gene_expression": torch.tensor(np.asarray(gene_block[cl_idx], dtype=np.float32)),
                "bionic_features": torch.tensor(np.asarray(bionic_block[cl_idx], dtype=np.float32)),
            }
            if response is not None:
                sample["ic50"] = torch.tensor([response[i]], dtype=torch.float32)
            samples.append(sample)
        return samples

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the DIPK model on a batch of pairs.

        :param batch: Featurized batch with gene_expression, bionic_features, molgnet_features blocks.
        :raises ValueError: If drug features or early stopping data is missing.
        """
        if batch.drug_pair_idx is None:
            msg = "DIPK requires drug features"
            raise ValueError(msg)
        if batch.early_stopping_response is None:
            msg = "DIPK requires early stopping data"
            raise ValueError(msg)

        self._model = self._build_model()
        hp = self._hyperparameters

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=hp["lr"])

        # Build training samples
        train_samples = self._build_samples(
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            batch,
            response=batch.response,
        )

        # Build early stopping samples
        es_response = batch.early_stopping_response
        cl_pair_idx_es, drug_pair_idx_es = batch._pair_indices_for(es_response)
        if drug_pair_idx_es is None:
            msg = "DIPK requires drug pair indices for early stopping"
            raise ValueError(msg)
        es_samples = self._build_samples(
            cl_pair_idx_es,
            drug_pair_idx_es,
            batch,
            response=es_response.response,
        )

        collate_train = _CollateFn(train=True)
        collate_val = _CollateFn(train=True)
        train_loader: DataLoader = DataLoader(
            _DIPKDataset(train_samples),
            batch_size=hp["batch_size"],
            shuffle=True,
            collate_fn=collate_train,
        )
        es_loader: DataLoader = DataLoader(
            _DIPKDataset(es_samples),
            batch_size=hp["batch_size"],
            shuffle=True,
            collate_fn=collate_val,
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        checkpoint_dir = Path(batch.training_context.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        version = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(20)])
        checkpoint_path = checkpoint_dir / f"{version}_best_DIPK_model.pth"

        for _epoch in range(hp["epochs"]):
            self._model.train()
            epoch_loss = 0.0
            batch_count = 0

            for dl_batch in train_loader:
                drug_features = dl_batch["molgnet_features"].to(self._device)
                gene_features = dl_batch["gene_features"].to(self._device)
                bionic_features = dl_batch["bionic_features"].to(self._device)
                molgnet_mask = dl_batch["molgnet_mask"].to(self._device)
                ic50_values = dl_batch["ic50_values"].to(self._device)

                prediction = self._model(
                    molgnet_drug_features=drug_features,
                    gene_expression=gene_features,
                    bionic=bionic_features,
                    molgnet_mask=molgnet_mask,
                )
                loss = loss_func(torch.squeeze(prediction), torch.squeeze(ic50_values))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                batch_count += 1

            epoch_loss /= batch_count

            # Validation phase
            self._model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for dl_batch in es_loader:
                    drug_features = dl_batch["molgnet_features"].to(self._device)
                    gene_features = dl_batch["gene_features"].to(self._device)
                    bionic_features = dl_batch["bionic_features"].to(self._device)
                    molgnet_mask = dl_batch["molgnet_mask"].to(self._device)
                    ic50_values = dl_batch["ic50_values"].to(self._device)

                    prediction = self._model(
                        molgnet_drug_features=drug_features,
                        gene_expression=gene_features,
                        bionic=bionic_features,
                        molgnet_mask=molgnet_mask,
                    )
                    loss = loss_func(torch.squeeze(prediction), torch.squeeze(ic50_values))
                    val_loss += loss.item()
                    val_batch_count += 1

            val_loss /= val_batch_count

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_torch_payload(self._model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= hp["patience"]:
                    break

        # Reload best model
        self._model.load_state_dict(load_state_dict(checkpoint_path, map_location=self._device))
        self._model.to(self._device)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Run DIPK inference on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If drug pair indices are missing.
        """
        if self._model is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        if batch.drug_pair_idx is None:
            msg = "DIPK requires drug features for prediction"
            raise ValueError(msg)

        samples = self._build_samples(
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            batch,
        )

        collate = _CollateFn(train=False)
        test_loader: DataLoader = DataLoader(
            _DIPKDataset(samples),
            batch_size=self._hyperparameters["batch_size"],
            shuffle=False,
            collate_fn=collate,
        )

        self._model.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for dl_batch in test_loader:
                drug_features = dl_batch["molgnet_features"].to(self._device)
                gene_features = dl_batch["gene_features"].to(self._device)
                bionic_features = dl_batch["bionic_features"].to(self._device)
                molgnet_mask = dl_batch["molgnet_mask"].to(self._device)

                prediction = self._model(
                    molgnet_drug_features=drug_features,
                    gene_expression=gene_features,
                    bionic=bionic_features,
                    molgnet_mask=molgnet_mask,
                )
                if prediction.numel() > 1:
                    predictions += torch.squeeze(prediction).cpu().tolist()
                else:
                    predictions += [prediction.item()]

        return np.asarray(predictions)

    def is_fitted(self) -> bool:
        """Return whether the model has been trained.

        :returns: True when the model is initialized.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted model state for persistence.

        :returns: Mapping with binary payload and hyperparameters.
        """
        if self._model is None:
            return {}
        return {
            "payload": save_state_dict(self._model.state_dict()),
            "hyperparameters": dict(self._hyperparameters),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted model from serialized state.

        :param state: State mapping from ``get_state``.
        :raises PredictorStateError: If state is malformed.
        """
        hp = state.get("hyperparameters")
        if not isinstance(hp, dict):
            msg = "DIPKPredictor state requires hyperparameters dict"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hp)
        model_bytes = state.get("payload")
        if not isinstance(model_bytes, (bytes, bytearray)):
            msg = "DIPKPredictor state requires payload bytes"
            raise PredictorStateError(msg)
        self._model = self._build_model()
        self._model.load_state_dict(load_state_dict(bytes(model_bytes)))
        self._model.to(self._device)


class _CollateFn:
    """Collate function for DIPK DataLoader batches."""

    def __init__(self, train: bool = True) -> None:
        """Initialize collation function.

        :param train: Whether to include target (ic50) values.
        """
        self.train = train

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate a list of sample dicts into padded batch tensors.

        :param batch: List of per-sample dictionaries.
        :returns: Collated batch dictionary.
        """
        max_atoms = max(sample["molgnet_features"].size(0) for sample in batch)

        padded_molgnet: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for sample in batch:
            num_atoms = sample["molgnet_features"].size(0)
            padding_size = max_atoms - num_atoms
            padded = torch.cat(
                [sample["molgnet_features"], torch.zeros(padding_size, sample["molgnet_features"].size(1))],
                dim=0,
            )
            padded_molgnet.append(padded)
            mask = torch.cat(
                [torch.ones(num_atoms, dtype=torch.bool), torch.zeros(padding_size, dtype=torch.bool)],
                dim=0,
            )
            masks.append(mask)

        result: dict[str, torch.Tensor] = {
            "molgnet_features": torch.stack(padded_molgnet),
            "molgnet_mask": torch.stack(masks),
            "gene_features": torch.stack([s["gene_expression"] for s in batch]),
            "bionic_features": torch.stack([s["bionic_features"] for s in batch]),
        }
        if self.train:
            result["ic50_values"] = torch.stack([s["ic50"] for s in batch])
        return result


class _DIPKDataset(Dataset):
    """Simple list-backed dataset for DIPK samples."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        """Initialize dataset from a list of sample dicts.

        :param samples: Pre-built sample dictionaries.
        """
        self._samples = samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return sample at the given index.

        :param idx: Sample index.
        :returns: Sample dictionary.
        """
        return self._samples[idx]

    def __len__(self) -> int:
        """Return number of samples.

        :returns: Dataset length.
        """
        return len(self._samples)
