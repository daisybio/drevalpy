"""DIPK predictor consuming PredictorInput directly.

``torch`` is imported inside the methods that use it, and the DIPK network comes
from ``.model_utils`` only when a model is actually built. ``drevalpy.registry``
imports this module to register the ``dipk`` predictor on ``import drevalpy``, so
a module-scope ``import torch`` put ~0.35s on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.predictors._pair_loader import collect_predictions, require_drug_pair_idx
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._torch_training import (
    DevicePicklingMixin,
    EarlyStoppingRun,
    run_epoch,
    train_with_early_stopping,
)
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.feature_format import BlockSpec, FeatureFormat
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from .model_utils import Predictor as DIPKNetwork

_REFERENCE = LiteratureReference(
    repo_url="https://github.com/user15632/DIPK",
    citation_doi="10.1093/bib/bbae153",
    citation_text="DIPK deep integration model with BIONIC and MolGNet features.",
    deviations=(
        "Preserve the upstream autoencoder architecture, 20% masking and final-epoch encoder return. Add "
        "disjoint validation stopping to autoencoder/predictor; predictor restores best validation "
        "weights. Real MolGNet/PPI artifacts are required for raw-input integration."
    ),
)


@register(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    reference=_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class DIPKPredictor(DevicePicklingMixin, Predictor):
    """DIPK predictor that trains an attention-based model on gene expression, BIONIC, and MolGNet features."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(
        BlockSpec("gene_expression"),
        BlockSpec("bionic_features"),
    )
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE))
    supports_early_stopping: ClassVar[bool] = True
    requires_early_stopping: ClassVar[bool] = True

    fixed_hyperparameters = {
        "batch_size": 64,
        "lr": 0.0001,
        "heads": 2,
        "fc_layer_num": 3,
        "fc_layer_dim": [256, 128, 64, 32, 16, 1],
        "dropout_rate": 0.3,
        "epochs": 100,
        "patience": 10,
    }

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the DIPK predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: DIPKNetwork | None = None

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

    def _build_pairs(self, data: PredictorInput, *, response: bool = False) -> list[dict[str, torch.Tensor]]:
        """Construct one dictionary of feature tensors per pair of *data*.

        :param data: Pairs to build, with the feature blocks they index into.
        :param response: Whether to add each pair's response as its ``ic50`` target.
        :returns: List of pair dictionaries with tensor values.
        """
        import torch

        sample_pair_idx = data.sample_pair_idx
        drug_pair_idx = require_drug_pair_idx(data.drug_pair_idx)
        targets = data.response if response else None

        gene_block = data.sample.blocks["gene_expression"].values
        bionic_block = data.sample.blocks["bionic_features"].values
        drug_block = data.drug.blocks["molgnet_features"].values

        pairs: list[dict[str, torch.Tensor]] = []
        for i in range(len(sample_pair_idx)):
            cl_idx = sample_pair_idx[i]
            dr_idx = drug_pair_idx[i]

            pair: dict[str, torch.Tensor] = {
                "molgnet_features": torch.tensor(np.asarray(drug_block[dr_idx], dtype=np.float32)),
                "gene_expression": torch.tensor(np.asarray(gene_block[cl_idx], dtype=np.float32)),
                "bionic_features": torch.tensor(np.asarray(bionic_block[cl_idx], dtype=np.float32)),
            }
            if targets is not None:
                pair["ic50"] = torch.tensor([targets[i]], dtype=torch.float32)
            pairs.append(pair)
        return pairs

    def _build_loaders(self, data: PredictorInput) -> tuple[DataLoader, DataLoader]:
        """Build the training and early-stopping loaders.

        :param data: Featurized training batch.
        :returns: Tuple of ``(train_loader, es_loader)``.
        :raises ValueError: If drug features or early stopping data is missing.
        """
        if data.drug_pair_idx is None:
            msg = "DIPK requires drug features"
            raise ValueError(msg)
        stopping = data.stopping
        if stopping is None:
            msg = "DIPK requires early stopping data"
            raise ValueError(msg)

        train_pairs = self._build_pairs(data, response=True)
        es_pairs = self._build_pairs(stopping, response=True)

        return (
            self._loader(train_pairs, shuffle=True),
            self._loader(es_pairs, shuffle=True),
        )

    def _loader(self, pairs: list[dict[str, torch.Tensor]], *, shuffle: bool) -> DataLoader:
        """Wrap *pairs* in a padding-collated DataLoader.

        :param pairs: Per-pair dictionaries from :meth:`_build_pairs`.
        :param shuffle: Whether to shuffle each epoch.
        :returns: A DataLoader over the pairs.
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            pairs,
            batch_size=self._hyperparameters["batch_size"],
            shuffle=shuffle,
            collate_fn=_collate,
        )

    def _fit(self, data: PredictorInput) -> None:
        """Train the DIPK model on a batch of pairs.

        :param data: Featurized batch with gene_expression, bionic_features, molgnet_features blocks.
        """
        import torch
        from torch import nn, optim

        train_loader, es_loader = self._build_loaders(data)

        model = self._build_model()
        self._model = model
        hp = self._hyperparameters

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hp["lr"])

        def batch_loss(dl_batch: dict[str, torch.Tensor]) -> torch.Tensor:
            prediction = self._forward(model, dl_batch)
            return loss_func(torch.squeeze(prediction), torch.squeeze(dl_batch["ic50_values"].to(self._device)))

        train_with_early_stopping(
            model,
            EarlyStoppingRun(
                epochs=hp["epochs"],
                patience=hp["patience"],
                checkpoint_dir=data.training_context.checkpoint_dir,
                model_name="DIPK",
            ),
            train_epoch=lambda: run_epoch(model, train_loader, batch_loss, optimizer),
            val_epoch=lambda: run_epoch(model, es_loader, batch_loss, None),
            device=self._device,
        )

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

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Run DIPK inference on the given batch.

        :param data: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If drug pair indices are missing.
        """
        if self._model is None:
            return np.full(len(data), np.nan, dtype=np.float64)
        if data.drug_pair_idx is None:
            msg = "DIPK requires drug features for prediction"
            raise ValueError(msg)

        pairs = self._build_pairs(data)
        test_loader = self._loader(pairs, shuffle=False)

        model = self._model
        return collect_predictions(model, test_loader, lambda dl_batch: self._forward(model, dl_batch))


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate pair dicts from :meth:`DIPKPredictor._build_pairs` into one zero-padded batch.

    :param batch: Per-pair dictionaries; they carry ``ic50`` only when built with ``response=True``.
    :returns: Batch with the padded atom features, their boolean mask, and ``ic50_values`` when the pairs hold targets.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    atoms = [pair["molgnet_features"] for pair in batch]
    result: dict[str, torch.Tensor] = {
        "molgnet_features": pad_sequence(atoms, batch_first=True),
        "molgnet_mask": pad_sequence([torch.ones(len(a), dtype=torch.bool) for a in atoms], batch_first=True),
        "gene_features": torch.stack([pair["gene_expression"] for pair in batch]),
        "bionic_features": torch.stack([pair["bionic_features"] for pair in batch]),
    }
    if "ic50" in batch[0]:
        result["ic50_values"] = torch.stack([pair["ic50"] for pair in batch])
    return result
