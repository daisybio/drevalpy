"""SparseGO literature predictor -- direct BlockPredictor implementation.

``torch``, the ``SparseGONetwork`` and the ontology helpers in ``.utils`` (which
pull in ``networkx``) are imported inside the methods that need them.
``drevalpy.registry`` imports this module to register the ``sparsego`` predictor on
``import drevalpy``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict, save_trusted_mapping
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors._pair_loader import make_pair_loader
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import SPARSEGO_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork


def _parse_ontology_metadata(metadata: dict[str, object]) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    """Extract layer_connections, gene2id_mapping, and gene_order from block metadata.

    The metadata dictionary is produced by the SparseGO featurizer and contains:
    - ``ontology_file``: Path to the GO ontology file.
    - ``gene2ind_file``: Path to the gene-to-index mapping file.
    - ``layer_connections``: Pre-computed list of per-layer connection arrays.
    - ``gene2id_mapping_ont``: Pre-computed gene-to-id mapping dict.
    - ``ontology_gene_order``: Ordered list of gene names matching expression columns.

    Raises ``ValueError`` (via ``resolve_ontology_metadata``) if required ontology info is missing.

    :param metadata: Metadata mapping from the active sample block.
    :returns: Tuple of (layer_connections, gene2id_mapping, gene_order).
    """
    from .utils import resolve_ontology_metadata

    return resolve_ontology_metadata(metadata)


def _resolve_active_view(batch: ModelInputBatch) -> str:
    """Determine which sample block is the active ontology view.

    SparseGO supports either 'gene_expression' or 'mutations' (exactly one).

    :param batch: Input batch with sample blocks.
    :returns: Name of the active sample view.
    :raises ValueError: If zero or multiple valid views are present.
    """
    candidates = {"gene_expression", "mutations"}
    active = [name for name in candidates if name in batch.sample.blocks]
    if len(active) != 1:
        raise ValueError("SparseGOPredictor requires exactly one sample block from ['gene_expression', 'mutations']")
    return active[0]


@register(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    sample_input_format=FeatureFormat.NUMERIC_MATRIX,
    drug_input_format=FeatureFormat.NUMERIC_MATRIX,
    reference=SPARSEGO_REFERENCE,
)
class SparseGOPredictor(BlockPredictor):
    """SparseGO predictor consuming ModelInputBatch directly."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ()
    required_sample_block_alternatives: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),
    )
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize SparseGOPredictor.

        :param hyperparameters: Optional overrides for algorithm defaults.
        """
        import torch

        super().__init__(hyperparameters)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SparseGONetwork | None = None
        self._layer_connections: list | None = None
        self._gene2id_mapping_ont: dict[str, int] | None = None
        self._ontology_gene_order: list[str] | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default SparseGO hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "num_neurons_per_GO": 6,
            "num_neurons_per_final_GO": 30,
            "num_neurons_drug": [100, 50, 6],
            "num_neurons_final": 12,
            "drug_dim": 2048,
            "learning_rate": 0.1,
            "momentum": 0.9,
            "decay_rate": 0.002,
            "p_drop_genes": 0.15,
            "p_drop_terms": 0.15,
            "p_drop_drugs": 0.15,
            "p_drop_final": 0.0,
            "epochs": 400,
            "batch_size": 20000,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Hyperparameter search space specs.
        """
        return {}

    def _build_network(self) -> SparseGONetwork:
        """Construct the SparseGONetwork from stored ontology metadata and hyperparameters.

        :returns: The new network, also stored on ``self._model``.
        :raises ValueError: If ontology metadata is not available.
        """
        from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork

        if self._layer_connections is None or self._gene2id_mapping_ont is None:
            raise ValueError("SparseGO ontology metadata must be provided before building the network.")
        hp = self._hyperparameters
        model = SparseGONetwork(
            layer_connections=self._layer_connections,
            num_neurons_per_go=hp["num_neurons_per_GO"],
            num_neurons_per_final_go=hp["num_neurons_per_final_GO"],
            num_neurons_drug=hp["num_neurons_drug"],
            num_neurons_final=hp["num_neurons_final"],
            drug_dim=hp["drug_dim"],
            gene2id_mapping=self._gene2id_mapping_ont,
            p_drop_final=hp["p_drop_final"],
            p_drop_genes=hp["p_drop_genes"],
            p_drop_terms=hp["p_drop_terms"],
            p_drop_drugs=hp["p_drop_drugs"],
        ).to(self._device)
        self._model = model
        return model

    def _entity_matrices(self, batch: ModelInputBatch, active_view: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the entity-level sample and fingerprint matrices as float32.

        :param batch: Batch to read the blocks from.
        :param active_view: Name of the sample block holding the ontology view.
        :returns: Tuple of ``(sample_matrix, fingerprint_matrix)``.
        """
        return (
            np.asarray(batch.sample.blocks[active_view].values, dtype=np.float32),
            np.asarray(batch.drug.blocks["fingerprints"].values, dtype=np.float32),
        )

    def _build_train_loader(self, batch: ModelInputBatch, blocks: tuple[np.ndarray, np.ndarray]) -> DataLoader:
        """Build the shuffled training loader over the entity matrices in *blocks*.

        :param batch: Training batch supplying the pair indices and response.
        :param blocks: Tuple of ``(sample_matrix, fingerprint_matrix)``.
        :returns: Training loader yielding sample, drug and response tensors.
        """
        cell_entity, drug_entity = blocks
        return make_pair_loader(
            (cell_entity, batch.sample_pair_idx),
            (drug_entity, require_drug_pair_idx(batch.drug_pair_idx)),
            response=np.asarray(batch.response, dtype=np.float32).reshape(-1, 1),
            batch_size=self._hyperparameters["batch_size"],
            shuffle=True,
        )

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train SparseGO on the batch.

        :param batch: Training batch with responses and sample/drug blocks.
        :raises ValueError: If ontology metadata is missing.
        """
        active_view = _resolve_active_view(batch)
        block = batch.sample.blocks[active_view]
        if block.annotations is None:
            raise ValueError("SparseGOPredictor requires ontology metadata on its active sample block")

        self._layer_connections, self._gene2id_mapping_ont, self._ontology_gene_order = _parse_ontology_metadata(
            dict(block.annotations)
        )

        blocks = self._entity_matrices(batch, active_view)
        self._hyperparameters["drug_dim"] = int(blocks[1].shape[1])
        model = self._build_network()
        self._run_training(model, self._build_train_loader(batch, blocks))

    def _run_training(self, model: SparseGONetwork, loader: DataLoader) -> None:
        """Run the SGD training loop with the paper's per-epoch learning-rate decay.

        :param model: The network to train.
        :param loader: Training loader over sample, drug and response tensors.
        """
        import torch
        import torch.nn as nn

        lr = self._hyperparameters["learning_rate"]
        decay_rate = self._hyperparameters["decay_rate"]
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._hyperparameters["momentum"],
        )

        model.train()
        for epoch in range(self._hyperparameters["epochs"]):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (1 / (1 + decay_rate * epoch))

            for cell_feats, drug_feats, batch_labels in loader:
                batch_features = torch.cat([cell_feats, drug_feats], dim=1).to(self._device)
                optimizer.zero_grad()
                loss = criterion(model(batch_features), batch_labels.to(self._device))
                loss.backward()
                optimizer.step()

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict drug response for pairs in the batch.

        :param batch: Featurized pairs to score.
        :returns: Predicted response values as a 1D numpy array.
        :raises ValueError: If the model is not fitted.
        """
        if self._model is None:
            raise ValueError("SparseGOPredictor must be fitted before predict().")

        cell_entity, drug_entity = self._entity_matrices(batch, _resolve_active_view(batch))
        return predict_pairs(
            self._model,
            batch,
            PairEvalSpec(
                sample_blocks=(cell_entity,),
                drug_blocks=(drug_entity,),
                batch_size=self._hyperparameters["batch_size"],
                device=self._device,
            ),
            forward=concatenated_forward(self._model),
        )

    def is_fitted(self) -> bool:
        """Return whether the predictor has been trained.

        :returns: True if the model is built and trained.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with a binary payload blob when fitted, else empty.
        """
        if self._model is None:
            return {}

        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "preload": {
                "layer_connections": self._layer_connections,
                "gene2id_mapping_ont": self._gene2id_mapping_ont,
                "ontology_gene_order": self._ontology_gene_order,
            },
            "model_state": save_state_dict(self._model.state_dict()),
        }
        return {"payload": save_trusted_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore a predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If the payload is missing or invalid.
        """
        payload = load_payload_state(state, "SparseGO")
        model_state = payload.get("model_state")
        preload = payload.get("preload")
        if not isinstance(model_state, (bytes, bytearray)):
            msg = "SparseGO payload is missing model_state"
            raise PredictorStateError(msg)
        if not isinstance(preload, dict):
            msg = "SparseGO payload is missing preload"
            raise PredictorStateError(msg)

        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._layer_connections = preload["layer_connections"]
        self._gene2id_mapping_ont = preload["gene2id_mapping_ont"]
        self._ontology_gene_order = preload["ontology_gene_order"]
        self._build_network().load_state_dict(load_state_dict(bytes(model_state)))
