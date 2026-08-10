"""SparseGO literature predictor -- direct BlockPredictor implementation."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from drevalpy.components.core.batch.feature_block import BlockSpec
from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._tensor_data import make_pair_loader
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import SPARSEGO_REFERENCE
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork
from drevalpy.components.predictors.literature.sparsego.utils import load_ontology, pairs_in_layers, sort_pairs
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


def _parse_ontology_metadata(metadata: dict[str, object]) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    """Extract layer_connections, gene2id_mapping, and gene_order from block metadata.

    The metadata dictionary is produced by the SparseGO featurizer and contains:
    - ``ontology_file``: Path to the GO ontology file.
    - ``gene2ind_file``: Path to the gene-to-index mapping file.
    - ``layer_connections``: Pre-computed list of per-layer connection arrays.
    - ``gene2id_mapping_ont``: Pre-computed gene-to-id mapping dict.
    - ``ontology_gene_order``: Ordered list of gene names matching expression columns.

    :param metadata: Metadata mapping from the active cell-line block.
    :returns: Tuple of (layer_connections, gene2id_mapping, gene_order).
    :raises ValueError: If required ontology info is missing from metadata.
    """
    layer_connections = metadata.get("layer_connections")
    gene2id_mapping = metadata.get("gene2id_mapping_ont")
    gene_order = metadata.get("ontology_gene_order")

    if layer_connections is None or gene2id_mapping is None:
        from drevalpy.components.predictors.literature.sparsego.utils import load_mapping

        ontology_file = metadata.get("ontology_file")
        gene2ind_file = metadata.get("gene2ind_file")
        if ontology_file is None or gene2ind_file is None:
            raise ValueError(
                "SparseGO block metadata must provide either pre-computed ontology structures "
                "(layer_connections, gene2id_mapping_ont) or file paths (ontology_file, gene2ind_file)."
            )
        gene2id_mapping = load_mapping(str(gene2ind_file))
        _, terms_pairs, genes_terms_pairs = load_ontology(str(ontology_file), gene2id_mapping)
        sorted_pairs, level_list, level_number = sort_pairs(genes_terms_pairs, terms_pairs, _, gene2id_mapping)
        layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)
        gene_order = list(gene2id_mapping.keys())

    return (
        list(layer_connections),  # type: ignore[call-overload]
        dict(gene2id_mapping),  # type: ignore[call-overload]
        list(gene_order) if gene_order else list(gene2id_mapping.keys()),  # type: ignore[call-overload, attr-defined]
    )


def _resolve_active_view(batch: ModelInputBatch) -> str:
    """Determine which cell-line block is the active ontology view.

    SparseGO supports either 'gene_expression' or 'mutations' (exactly one).

    :param batch: Input batch with cell-line blocks.
    :returns: Name of the active cell-line view.
    :raises ValueError: If zero or multiple valid views are present.
    """
    candidates = {"gene_expression", "mutations"}
    active = [name for name in candidates if name in batch.cell_line_blocks]
    if len(active) != 1:
        raise ValueError("SparseGOPredictor requires exactly one cell-line block from ['gene_expression', 'mutations']")
    return active[0]


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SPARSEGO_REFERENCE,
)
class SparseGOPredictor(BlockPredictor):
    """SparseGO predictor consuming ModelInputBatch directly."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_cell_line_block_alternatives: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX, metadata=True),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX, metadata=True),
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
            "num_neurons_per_final_GO": 6,
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

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {}

    def _build_network(self) -> None:
        """Construct the SparseGONetwork from stored ontology metadata and hyperparameters.

        :raises ValueError: If ontology metadata is not available.
        """
        if self._layer_connections is None or self._gene2id_mapping_ont is None:
            raise ValueError("SparseGO ontology metadata must be provided before building the network.")
        self._model = SparseGONetwork(
            layer_connections=self._layer_connections,
            num_neurons_per_go=self._hyperparameters.get("num_neurons_per_GO", 6),
            num_neurons_per_final_go=self._hyperparameters.get("num_neurons_per_final_GO", 6),
            num_neurons_drug=self._hyperparameters.get("num_neurons_drug", [200, 100, 50]),
            num_neurons_final=self._hyperparameters.get("num_neurons_final", 12),
            drug_dim=self._hyperparameters.get("drug_dim", 2048),
            gene2id_mapping=self._gene2id_mapping_ont,
            p_drop_final=self._hyperparameters.get("p_drop_final", 0.0),
            p_drop_genes=self._hyperparameters.get("p_drop_genes", 0.1),
            p_drop_terms=self._hyperparameters.get("p_drop_terms", 0.1),
            p_drop_drugs=self._hyperparameters.get("p_drop_drugs", 0.1),
        ).to(self._device)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train SparseGO on the batch.

        :param batch: Training batch with responses and cell-line/drug blocks.
        :raises ValueError: If ontology metadata is missing or network build fails.
        :raises RuntimeError: If drug_pair_idx is None.
        """
        active_view = _resolve_active_view(batch)
        block = batch.cell_line_blocks[active_view]
        if block.metadata is None:
            raise ValueError("SparseGOPredictor requires ontology metadata on its active cell-line block")

        self._layer_connections, self._gene2id_mapping_ont, self._ontology_gene_order = _parse_ontology_metadata(
            dict(block.metadata)
        )

        cell_entity = np.asarray(block.values, dtype=np.float32)
        drug_entity = np.asarray(batch.drug_blocks["fingerprints"].values, dtype=np.float32)
        cell_pair_idx = batch.cell_line_pair_idx
        drug_pair_idx = batch.drug_pair_idx
        if drug_pair_idx is None:
            raise RuntimeError("drug_pair_idx is required for this predictor")

        self._hyperparameters["drug_dim"] = int(drug_entity.shape[1])
        self._build_network()
        if self._model is None:
            msg = "SparseGO network build failed"
            raise ValueError(msg)

        response = np.asarray(batch.response, dtype=np.float32).reshape(-1, 1)

        loader = make_pair_loader(
            (cell_entity, cell_pair_idx),
            (drug_entity, drug_pair_idx),
            response=response,
            batch_size=self._hyperparameters.get("batch_size", 10000),
            shuffle=True,
        )

        lr = self._hyperparameters.get("learning_rate", 0.1)
        decay_rate = self._hyperparameters.get("decay_rate", 0.002)
        momentum = self._hyperparameters.get("momentum", 0.9)
        epochs = self._hyperparameters.get("epochs", 100)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)

        self._model.train()
        for epoch in range(epochs):
            current_lr = lr * (1 / (1 + decay_rate * epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            for cell_feats, drug_feats, batch_labels in loader:
                batch_features = torch.cat([cell_feats, drug_feats], dim=1).to(self._device)
                batch_labels = batch_labels.to(self._device)
                optimizer.zero_grad()
                outputs = self._model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict drug response for pairs in the batch.

        :param batch: Featurized pairs to score.
        :returns: Predicted response values as a 1D numpy array.
        :raises ValueError: If the model is not fitted.
        :raises RuntimeError: If drug_pair_idx is None.
        """
        if self._model is None:
            raise ValueError("SparseGOPredictor must be fitted before predict().")

        active_view = _resolve_active_view(batch)
        block = batch.cell_line_blocks[active_view]

        cell_entity = np.asarray(block.values, dtype=np.float32)
        drug_entity = np.asarray(batch.drug_blocks["fingerprints"].values, dtype=np.float32)
        cell_pair_idx = batch.cell_line_pair_idx
        drug_pair_idx = batch.drug_pair_idx
        if drug_pair_idx is None:
            raise RuntimeError("drug_pair_idx is required for this predictor")

        loader = make_pair_loader(
            (cell_entity, cell_pair_idx),
            (drug_entity, drug_pair_idx),
            batch_size=self._hyperparameters.get("batch_size", 10000),
            shuffle=False,
        )

        self._model.eval()
        predictions = []
        with torch.no_grad():
            for cell_feats, drug_feats in loader:
                batch_features = torch.cat([cell_feats, drug_feats], dim=1).to(self._device)
                outputs = self._model(batch_features)
                predictions.append(outputs.squeeze().cpu().numpy())

        return np.concatenate(predictions)

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
        from drevalpy.components.predictors.literature._torch_state import save_state_dict

        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "preload": {
                "layer_connections": self._layer_connections,
                "gene2id_mapping_ont": self._gene2id_mapping_ont,
                "ontology_gene_order": self._ontology_gene_order,
            },
            "model_state": save_state_dict(self._model.state_dict()),
        }
        return {"payload": save_object_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore a predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If the payload is missing or invalid.
        """
        blob = state.get("payload")
        if not isinstance(blob, (bytes, bytearray)):
            msg = "SparseGOPredictor state requires a payload byte blob"
            raise PredictorStateError(msg)
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = "SparseGOPredictor payload could not be deserialized"
            raise PredictorStateError(msg) from exc

        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = "SparseGOPredictor payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)

        preload = payload.get("preload")
        if isinstance(preload, dict):
            self._layer_connections = preload.get("layer_connections")
            self._gene2id_mapping_ont = preload.get("gene2id_mapping_ont")
            self._ontology_gene_order = preload.get("ontology_gene_order")

        self._build_network()

        model_state = payload.get("model_state")
        if isinstance(model_state, (bytes, bytearray)) and self._model is not None:
            from drevalpy.components.predictors.literature._torch_state import load_state_dict

            self._model.load_state_dict(load_state_dict(bytes(model_state)))
