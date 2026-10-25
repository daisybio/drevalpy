"""PaccMann block literature predictor.

Multimodal attention-based convolutional encoders (Manica et al., 2019): tokenized SMILES and gene
expression attend over each other. ``torch`` and ``.network`` are imported inside the methods that need
them: ``drevalpy.registry`` imports this module to register the ``paccMann`` predictor on
``import drevalpy``, so a module-scope ``import torch`` would put the training stack on the startup path
of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict, save_trusted_mapping
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors._pair_loader import make_pair_loader
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    run_epoch,
    train_with_early_stopping,
)
from drevalpy.components.predictors.literature._metadata import PACCMANN_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from .network import PaccMannNetwork

TOKENS_BLOCK = "smiles_tokens"
VARIANTS_BLOCK = "smiles_token_variants"
_ARCHITECTURE_KEYS = ("smiles_vocabulary_size", "smiles_padding_length", "number_of_genes", "n_variants")


def _drug_blocks(batch: ModelInputBatch) -> tuple[FeatureBlock, FeatureBlock]:
    """Return the SMILES token and variant blocks of a batch.

    :param batch: Batch to read.
    :returns: ``(smiles_tokens, smiles_token_variants)`` feature blocks.
    :raises RuntimeError: If the batch carries no drug features.
    """
    if batch.drug is None:
        msg = "PaccMann requires drug features (smilesTokens)."
        raise RuntimeError(msg)
    return batch.drug.blocks[TOKENS_BLOCK], batch.drug.blocks[VARIANTS_BLOCK]


def _build_network(
    hyperparameters: dict[str, Any], architecture: dict[str, int], device: torch.device
) -> PaccMannNetwork:
    """Construct the PaccMann network on *device*.

    :param hyperparameters: Hyperparameters holding the architecture keys.
    :param architecture: Sizes derived from the data: vocabulary size, padding length and gene count.
    :param device: Device the network is moved to.
    :returns: Initialized network.
    """
    from .network import PaccMannConfig, PaccMannNetwork

    return PaccMannNetwork(PaccMannConfig.from_hyperparameters({**hyperparameters, **architecture})).to(device)


@register(
    "paccMann",
    description="PaccMann: SMILES convolutions and gene expression linked by context attention.",
    sample_input_format=FeatureFormat.NUMERIC_MATRIX,
    drug_input_format=FeatureFormat.NUMERIC_MATRIX,
    reference=PACCMANN_REFERENCE,
)
class PaccMannPredictor(BlockPredictor):
    """Registered PaccMann predictor consuming ModelInputBatch directly."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = (TOKENS_BLOCK, VARIANTS_BLOCK)
    required_sample_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec(TOKENS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
        BlockSpec(VARIANTS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
    )
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the PaccMann predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        import torch

        super().__init__(hyperparameters)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: PaccMannNetwork | None = None
        self._architecture: dict[str, int] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return the DrEvalPy 1.6.1 defaults.

        :returns: Default hyperparameter mapping.
        """
        return {
            "epochs": 10,
            "patience": 10,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "smiles_embedding_size": 8,
            "filters": [16, 16, 16],
            "molecule_heads": [2, 2, 2, 2],
            "gene_heads": [2, 2, 2, 2],
            "dropout": 0.5,
            "batch_norm": True,
            "smiles_attention_size": 64,
            "gene_attention_size": 1,
            "molecule_temperature": 1.0,
            "gene_temperature": 1.0,
            "stacked_dense_hidden_sizes": [512, 256],
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Empty mapping; 1.6.1 searched a single point.
        """
        return {}

    @staticmethod
    def _architecture_from(batch: ModelInputBatch) -> dict[str, int]:
        """Read the data-derived network sizes from the featurized batch.

        :param batch: Batch holding the gene-expression and SMILES token blocks.
        :returns: Vocabulary size, padding length, gene count and variants per drug.
        :raises ValueError: If the SMILES token block carries no vocabulary metadata.
        """
        tokens = _drug_blocks(batch)[0]
        metadata = tokens.annotations or {}
        if not {"vocabulary_size", "padding_length", "n_variants"} <= metadata.keys():
            msg = "PaccMann needs the vocabulary metadata that the smilesTokens featurizer attaches to its blocks"
            raise ValueError(msg)
        return {
            "smiles_vocabulary_size": int(metadata["vocabulary_size"]),
            "smiles_padding_length": int(metadata["padding_length"]),
            "number_of_genes": int(batch.sample.blocks["gene_expression"].values.shape[1]),
            "n_variants": int(metadata["n_variants"]),
        }

    def _batch_loss(self, model: PaccMannNetwork, n_variants: int | None) -> Any:
        """Build the per-batch MSE loss.

        :param model: Network to evaluate.
        :param n_variants: Variants per drug to draw from for each row, or ``None`` to read plain tokens.
        :returns: Function from a loader batch to its loss tensor.
        """
        import torch
        import torch.nn.functional as functional

        def batch_loss(loader_batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            gene_inputs, drug_inputs, targets = (tensor.to(self._device) for tensor in loader_batch)
            if n_variants is None:
                smiles = drug_inputs.long()
            else:
                rows = drug_inputs.shape[0]
                variant = torch.randint(0, n_variants, (rows,)).to(self._device)
                smiles = drug_inputs.view(rows, n_variants, -1)[torch.arange(rows, device=self._device), variant].long()
            return functional.mse_loss(model(smiles, gene_inputs), targets.view(-1, 1))

        return batch_loss

    def _loaders(self, batch: ModelInputBatch) -> tuple[DataLoader, DataLoader | None]:
        """Build the training and early-stopping loaders.

        :param batch: Training batch, possibly with an early-stopping response.
        :returns: ``(train_loader, val_loader)``; the latter is ``None`` without early-stopping pairs.
        """
        gene_entity = batch.sample.blocks["gene_expression"].values
        tokens, variants = _drug_blocks(batch)
        batch_size = int(self._hyperparameters["batch_size"])
        n_pairs = batch.n_pairs
        # The batch norm layers cannot process a batch of one sample. Drop a trailing batch of one, and
        # only then, so that training sets smaller than a batch stay usable.
        drop_last = n_pairs > batch_size and n_pairs % batch_size == 1
        train_loader = make_pair_loader(
            (gene_entity, batch.sample_pair_idx),
            (variants.values, require_drug_pair_idx(batch.drug_pair_idx)),
            response=np.asarray(batch.response, dtype=np.float32),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        stopping = batch.early_stopping_response
        if stopping is None or len(stopping) == 0:
            return train_loader, None
        stop_sample_idx = batch.sample.rows_for(stopping.sample_ids)
        stop_drug_idx = None if batch.drug is None else batch.drug.rows_for(stopping.drug_ids)
        val_loader = make_pair_loader(
            (gene_entity, stop_sample_idx),
            (tokens.values, require_drug_pair_idx(stop_drug_idx)),
            response=np.asarray(stopping.response, dtype=np.float32),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the network, keeping the weights of the epoch with the lowest early-stopping loss.

        Each row is seen through one of the equivalent SMILES of its drug, drawn anew for every batch.
        Without early-stopping pairs the full epoch budget runs and the last weights are kept.

        :param batch: Training batch with gene_expression, smiles_tokens and smiles_token_variants blocks.
        """
        import torch.optim as optim

        hp = self._hyperparameters
        self._architecture = self._architecture_from(batch)
        model = _build_network(hp, self._architecture, self._device)
        self._model = model
        train_loader, val_loader = self._loaders(batch)
        optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        train_epoch = self._batch_loss(model, self._architecture["n_variants"])
        if val_loader is None:
            for _ in range(int(hp["epochs"])):
                run_epoch(model, train_loader, train_epoch, optimizer)
            return
        val_epoch = self._batch_loss(model, None)
        train_with_early_stopping(
            model,
            EarlyStoppingRun(
                epochs=int(hp["epochs"]),
                patience=int(hp["patience"]),
                checkpoint_dir=batch.training_context.checkpoint_dir,
                model_name="PaccMann",
            ),
            train_epoch=lambda: run_epoch(model, train_loader, train_epoch, optimizer),
            val_epoch=lambda: run_epoch(model, val_loader, val_epoch, None),
            device=self._device,
        )

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict with each drug's own SMILES.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "PaccMann model not initialized."
            raise ValueError(msg)
        model = self._model
        tokens = _drug_blocks(batch)[0]
        return predict_pairs(
            model,
            batch,
            PairEvalSpec(
                sample_blocks=(batch.sample.blocks["gene_expression"].values,),
                drug_blocks=(tokens.values,),
                batch_size=int(self._hyperparameters["batch_size"]),
                device=self._device,
            ),
            forward=lambda gene_inputs, drug_inputs: model(drug_inputs.long(), gene_inputs),
        )

    def is_fitted(self) -> bool:
        """Return whether the model has been trained.

        :returns: True when the network is initialized.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob when fitted, else empty.
        """
        if self._model is None:
            return {}
        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "architecture": dict(self._architecture),
            "model_state": save_state_dict(self._model.state_dict()),
        }
        return {"payload": save_trusted_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        payload = load_payload_state(state, "PaccMann")
        architecture, model_state = payload.get("architecture"), payload.get("model_state")
        if (
            not isinstance(architecture, dict)
            or not set(_ARCHITECTURE_KEYS) <= architecture.keys()
            or not isinstance(model_state, (bytes, bytearray))
        ):
            msg = "PaccMann payload is missing architecture or model_state"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._architecture = {key: int(architecture[key]) for key in _ARCHITECTURE_KEYS}
        self._model = _build_network(self._hyperparameters, self._architecture, self._device)
        self._model.load_state_dict(load_state_dict(bytes(model_state), map_location=self._device))
        self._model.eval()
