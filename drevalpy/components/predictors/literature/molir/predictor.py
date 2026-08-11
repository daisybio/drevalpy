"""MOLIR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from upath import UPath as Path

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import MOLIR_REFERENCE
from drevalpy.components.predictors.literature.molir.utils import MOLIModel, _realign_omic_matrix
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.utils.torch_io import load_state_dict as _load_torch_state_dict
from drevalpy.utils.torch_io import (
    load_trusted_mapping,
    save_state_dict,
    save_trusted_mapping,
)


def _checkpoint_dir_for_drug(base_dir: Path, drug_id: str) -> Path:
    """Return a unique checkpoint directory path for a given drug.

    :param base_dir: Base directory for checkpoints.
    :param drug_id: Drug identifier to hash.
    :returns: Path to the drug-specific checkpoint directory.
    """
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return base_dir / f"drug_{digest}"


@register_predictor(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=MOLIR_REFERENCE,
)
class MOLIRPredictor(BlockPredictor):
    """MOLIR predictor: per-drug multi-omics late integration model."""

    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = (
        "gene_expression",
        "mutations",
        "copy_number_variation_gistic",
    )
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("identity", FeatureFormat.NUMERIC_MATRIX),)
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize predictor with optional hyperparameter overrides.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._models: dict[str, MOLIModel] = {}
        self._feature_names: dict[str, _OmicFeatureNames] = {}

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train per-drug MOLIR models.

        :param batch: Training batch with all required cell-line omics blocks.
        """
        keys = routing_keys(batch)
        require_known_training_keys(keys)
        self._models = {}
        self._feature_names = {}

        for drug_id, mask in iter_drug_masks(batch):
            context = TrainingContext(
                checkpoint_dir=_checkpoint_dir_for_drug(batch.training_context.checkpoint_dir, drug_id),
            )
            sub = replace(batch.subset_pairs(mask), training_context=context)
            self._fit_single_drug(drug_id, sub)

    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> None:
        """Fit a single-drug MOLIR model.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        """
        pair_idx = batch.cell_line_pair_idx
        gex_block = batch.cell_line_blocks["gene_expression"]
        mut_block = batch.cell_line_blocks["mutations"]
        cnv_block = batch.cell_line_blocks["copy_number_variation_gistic"]
        n_samples = batch.n_pairs

        feature_names = _OmicFeatureNames(
            gene_expression=gex_block.feature_names,
            mutations=mut_block.feature_names,
            copy_number_variation=cnv_block.feature_names,
        )
        self._feature_names[drug_id] = feature_names

        if n_samples == 0:
            return

        dim_gex = gex_block.values.shape[1]
        dim_mut = mut_block.values.shape[1]
        dim_cnv = cnv_block.values.shape[1]
        model = MOLIModel(
            hpams=dict(self._hyperparameters),
            input_dim_expr=dim_gex,
            input_dim_mut=dim_mut,
            input_dim_cnv=dim_cnv,
        )

        if n_samples >= self._hyperparameters["mini_batch"]:
            response = np.asarray(batch.response, dtype=np.float32)

            val_pair_idx, val_response = self._build_early_stopping_indices(batch)

            model.fit(
                gene_expression=np.asarray(gex_block.values, dtype=np.float32),
                mutations=np.asarray(mut_block.values, dtype=np.float32),
                copy_number=np.asarray(cnv_block.values, dtype=np.float32),
                response=response,
                pair_idx=pair_idx,
                val_gene_expression=(
                    np.asarray(gex_block.values, dtype=np.float32) if val_pair_idx is not None else None
                ),
                val_mutations=np.asarray(mut_block.values, dtype=np.float32) if val_pair_idx is not None else None,
                val_copy_number=np.asarray(cnv_block.values, dtype=np.float32) if val_pair_idx is not None else None,
                val_response=val_response,
                val_pair_idx=val_pair_idx,
                model_checkpoint_dir=str(batch.training_context.checkpoint_dir),
            )

        self._models[drug_id] = model

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict drug responses for pairs, routing to per-drug models.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        """
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            if drug_id == "":
                continue
            model = self._models.get(str(drug_id))
            if model is None:
                continue
            mask = keys == drug_id
            sub = batch.subset_pairs(mask)
            preds = self._predict_single_drug(str(drug_id), model, sub)
            predictions[mask] = np.asarray(preds, dtype=np.float64).ravel()
        return predictions

    def _predict_single_drug(self, drug_id: str, model: MOLIModel, batch: ModelInputBatch) -> np.ndarray:
        """Predict for a single drug model.

        :param drug_id: Drug identifier for feature alignment.
        :param model: Trained MOLI model instance.
        :param batch: Subset batch for this drug.
        :returns: Predicted responses.
        """
        feature_names = self._feature_names.get(drug_id)
        if feature_names is None:
            return np.full(batch.n_pairs, np.nan)

        pair_idx = batch.cell_line_pair_idx
        gex = np.asarray(batch.cell_line_blocks["gene_expression"].values[pair_idx], dtype=np.float32)
        mut = np.asarray(batch.cell_line_blocks["mutations"].values[pair_idx], dtype=np.float32)
        cnv = np.asarray(batch.cell_line_blocks["copy_number_variation_gistic"].values[pair_idx], dtype=np.float32)

        gex = self._align_omic(
            gex, feature_names.gene_expression, batch.cell_line_blocks["gene_expression"].feature_names
        )
        mut = self._align_omic(mut, feature_names.mutations, batch.cell_line_blocks["mutations"].feature_names)
        cnv = self._align_omic(
            cnv,
            feature_names.copy_number_variation,
            batch.cell_line_blocks["copy_number_variation_gistic"].feature_names,
        )

        return np.atleast_1d(model.predict(gex, mut, cnv))

    def _build_early_stopping_indices(self, batch: ModelInputBatch) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Build validation pair indices from the batch early stopping data.

        :param batch: Training batch.
        :returns: Tuple of (val_pair_idx, val_response), both None if unavailable.
        """
        es_resp = batch.early_stopping_response
        if es_resp is None or len(es_resp) < 2:
            return None, None

        entity_map = {str(eid): row for row, eid in enumerate(batch.cell_line_entity_ids)}
        val_idx = np.array([entity_map[str(cl_id)] for cl_id in es_resp.cell_line_ids], dtype=np.intp)
        val_response = np.asarray(es_resp.response, dtype=np.float32)
        return val_idx, val_response

    @staticmethod
    def _align_omic(
        values: np.ndarray,
        model_features: tuple[str, ...] | None,
        current_features: tuple[str, ...] | None,
    ) -> np.ndarray:
        """Align omic matrix columns to match training feature order.

        :param values: Input matrix to realign.
        :param model_features: Feature names expected by the model.
        :param current_features: Feature names in the current batch.
        :returns: Realigned matrix.
        """
        if model_features is None or current_features is None:
            return values
        if len(model_features) == values.shape[1] and model_features == current_features:
            return values
        return _realign_omic_matrix(values, model_features, current_features)

    def is_fitted(self) -> bool:
        """Report whether trained models exist.

        :returns: True when at least one per-drug model has been fitted.
        """
        return bool(self._models)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted state for all per-drug models.

        :returns: Mapping with algorithm blobs and hyperparameters.
        """
        if not self._models:
            return {}
        algorithms: dict[str, bytes] = {}
        for drug_id, model in self._models.items():
            fn = self._feature_names.get(drug_id)
            payload: dict[str, Any] = {
                "hyperparameters": dict(self._hyperparameters),
                "gene_expression_features": list(fn.gene_expression) if fn and fn.gene_expression else None,
                "mutations_features": list(fn.mutations) if fn and fn.mutations else None,
                "copy_number_variation_features": (
                    list(fn.copy_number_variation) if fn and fn.copy_number_variation else None
                ),
                "model_state": save_state_dict(model.state_dict()),
                "input_dims": {
                    "expr": model.expression_encoder.encode[0].in_features,
                    "mut": model.mutation_encoder.encode[0].in_features,
                    "cnv": model.cna_encoder.encode[0].in_features,
                },
            }
            algorithms[drug_id] = save_trusted_mapping(payload)
        return {
            "algorithms": algorithms,
            "predictor_hyperparameters": dict(self._hyperparameters),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state from serialized algorithm blobs.

        :param state: State mapping from ``get_state``.
        :raises PredictorStateError: If state is malformed.
        """
        algorithms_blob = state.get("algorithms")
        if not isinstance(algorithms_blob, dict):
            msg = "MOLIRPredictor state requires an 'algorithms' mapping"
            raise PredictorStateError(msg)
        hyperparameters = state.get("predictor_hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)

        self._models = {}
        self._feature_names = {}
        for drug_id, blob in algorithms_blob.items():
            if not isinstance(blob, (bytes, bytearray)):
                msg = f"MOLIRPredictor algorithm payload for {drug_id!r} must be bytes"
                raise PredictorStateError(msg)
            payload = load_trusted_mapping(bytes(blob))
            fn = _OmicFeatureNames(
                gene_expression=(
                    tuple(payload["gene_expression_features"]) if payload.get("gene_expression_features") else None
                ),
                mutations=tuple(payload["mutations_features"]) if payload.get("mutations_features") else None,
                copy_number_variation=(
                    tuple(payload["copy_number_variation_features"])
                    if payload.get("copy_number_variation_features")
                    else None
                ),
            )
            self._feature_names[str(drug_id)] = fn
            hpams = payload.get("hyperparameters", dict(self._hyperparameters))
            input_dims = payload.get("input_dims", {})
            model = MOLIModel(
                hpams=hpams,
                input_dim_expr=input_dims["expr"],
                input_dim_mut=input_dims["mut"],
                input_dim_cnv=input_dims["cnv"],
            )
            model_state_bytes = payload.get("model_state")
            if isinstance(model_state_bytes, (bytes, bytearray)):
                model.load_state_dict(_load_torch_state_dict(bytes(model_state_bytes)))
            self._models[str(drug_id)] = model

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "mini_batch": 32,
            "h_dim1": 64,
            "h_dim2": 64,
            "h_dim3": 64,
            "learning_rate": 0.01,
            "dropout_rate": 0.5,
            "weight_decay": 0.0001,
            "gamma": 0.5,
            "epochs": 30,
            "margin": 1.5,
        }


class _OmicFeatureNames:
    """Store feature name tuples for the three omics views."""

    __slots__ = ("gene_expression", "mutations", "copy_number_variation")

    def __init__(
        self,
        gene_expression: tuple[str, ...] | None,
        mutations: tuple[str, ...] | None,
        copy_number_variation: tuple[str, ...] | None,
    ) -> None:
        """Initialize feature name storage.

        :param gene_expression: Gene expression feature names.
        :param mutations: Mutation feature names.
        :param copy_number_variation: Copy number variation feature names.
        """
        self.gene_expression = gene_expression
        self.mutations = mutations
        self.copy_number_variation = copy_number_variation
