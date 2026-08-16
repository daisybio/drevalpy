"""DIPK predictor consuming ModelInputBatch directly.

``torch`` is imported inside the methods that use it, and the DIPK network comes
from ``.model_utils`` only when a model is actually built. ``drevalpy.registry``
imports this module to register the ``dipk`` predictor on ``import drevalpy``, so
a module-scope ``import torch`` put ~0.35s on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    train_with_early_stopping,
)
from drevalpy.components.predictors.literature._metadata import DIPK_REFERENCE
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import load_state_dict, save_state_dict

if TYPE_CHECKING:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

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
        import torch

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
        from .model_utils import Predictor as DIPKNetwork

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
        import torch

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

    def _build_loaders(self, batch: ModelInputBatch) -> tuple[DataLoader, DataLoader]:
        """Build the training and early-stopping loaders.

        :param batch: Featurized training batch.
        :returns: Tuple of ``(train_loader, es_loader)``.
        :raises ValueError: If drug features or early stopping data is missing.
        """
        if batch.drug_pair_idx is None:
            msg = "DIPK requires drug features"
            raise ValueError(msg)
        if batch.early_stopping_response is None:
            msg = "DIPK requires early stopping data"
            raise ValueError(msg)

        train_samples = self._build_samples(
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            batch,
            response=batch.response,
        )

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

        return (
            self._loader(train_samples, train=True, shuffle=True),
            self._loader(es_samples, train=True, shuffle=True),
        )

    def _loader(self, samples: list[dict[str, torch.Tensor]], *, train: bool, shuffle: bool) -> DataLoader:
        """Wrap *samples* in a padding-collated DataLoader.

        :param samples: Per-sample dictionaries from :meth:`_build_samples`.
        :param train: Whether the collated batches carry target values.
        :param shuffle: Whether to shuffle each epoch.
        :returns: A DataLoader over the samples.
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            _DIPKDataset(samples),
            batch_size=self._hyperparameters["batch_size"],
            shuffle=shuffle,
            collate_fn=_CollateFn(train=train),
        )

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the DIPK model on a batch of pairs.

        :param batch: Featurized batch with gene_expression, bionic_features, molgnet_features blocks.
        """
        from torch import nn, optim

        train_loader, es_loader = self._build_loaders(batch)

        model = self._build_model()
        self._model = model
        hp = self._hyperparameters

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hp["lr"])

        train_with_early_stopping(
            model,
            EarlyStoppingRun(
                epochs=hp["epochs"],
                patience=hp["patience"],
                checkpoint_dir=batch.training_context.checkpoint_dir,
                model_name="DIPK",
            ),
            train_epoch=lambda: self._run_epoch(model, train_loader, loss_func, optimizer),
            val_epoch=lambda: self._run_epoch(model, es_loader, loss_func, None),
            device=self._device,
        )

    def _run_epoch(
        self,
        model: DIPKNetwork,
        loader: DataLoader,
        loss_func: nn.Module,
        optimizer: optim.Optimizer | None,
    ) -> float:
        """Run one train or validation epoch.

        :param model: The DIPK network.
        :param loader: Training or validation loader.
        :param loss_func: Loss function.
        :param optimizer: Optimizer for training; ``None`` for eval-only.
        :returns: Mean epoch loss.
        """
        import torch

        is_training = optimizer is not None
        if is_training:
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        batch_count = 0
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for dl_batch in loader:
                prediction = self._forward(model, dl_batch)
                loss = loss_func(torch.squeeze(prediction), torch.squeeze(dl_batch["ic50_values"].to(self._device)))

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                else:
                    epoch_loss += loss.item()
                batch_count += 1

        return epoch_loss / max(batch_count, 1)

    def _forward(self, model: DIPKNetwork, dl_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Move one collated batch onto the device and run the network.

        :param model: The DIPK network.
        :param dl_batch: One collated batch from the loader.
        :returns: The network's raw output.
        """
        return model(
            molgnet_drug_features=dl_batch["molgnet_features"].to(self._device),
            gene_expression=dl_batch["gene_features"].to(self._device),
            bionic=dl_batch["bionic_features"].to(self._device),
            molgnet_mask=dl_batch["molgnet_mask"].to(self._device),
        )

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Run DIPK inference on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If drug pair indices are missing.
        """
        import torch

        if self._model is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        if batch.drug_pair_idx is None:
            msg = "DIPK requires drug features for prediction"
            raise ValueError(msg)

        samples = self._build_samples(batch.cell_line_pair_idx, batch.drug_pair_idx, batch)
        test_loader = self._loader(samples, train=False, shuffle=False)

        self._model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for dl_batch in test_loader:
                prediction = self._forward(self._model, dl_batch)
                chunks.append(prediction.detach().cpu().numpy().reshape(-1))

        if not chunks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

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
        import torch

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


class _DIPKDataset:
    """Simple list-backed dataset for DIPK samples.

    Deliberately not a ``torch.utils.data.Dataset`` subclass: that base class
    contributes only ``__add__``, which nothing here uses, and inheriting from it
    would force ``import torch`` at module scope - which is exactly what keeps
    this predictor off the ``import drevalpy`` critical path. ``DataLoader``
    treats any object with ``__getitem__`` and ``__len__`` as a map-style dataset.
    """

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
