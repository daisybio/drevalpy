"""SuperFELTR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import SUPERFELTR_REFERENCE
from drevalpy.components.predictors.literature._torch_state import (
    load_object_mapping,
)
from drevalpy.components.predictors.literature._torch_state import load_state_dict as _load_torch_state_dict
from drevalpy.components.predictors.literature._torch_state import (
    save_object_mapping,
    save_state_dict,
)
from drevalpy.components.predictors.literature.molir.utils import _realign_omic_matrix
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.config import PredictionMode
from drevalpy.types.model_scope import ModelScope

from .utils import SuperFELTEncoder, SuperFELTRegressor

if TYPE_CHECKING:
    from drevalpy.datasets.dataset import FeatureDataset


def _checkpoint_dir_for_drug(base_dir: Path, drug_id: str) -> Path:
    """Return a unique checkpoint directory path for a given drug.

    :param base_dir: Base directory for checkpoints.
    :param drug_id: Drug identifier to hash.
    :returns: Path to the drug-specific checkpoint directory.
    """
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return base_dir / f"drug_{digest}"


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


class _DrugModel:
    """Per-drug fitted model state for SuperFELTR."""

    __slots__ = ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor", "ranges", "best_checkpoint")

    def __init__(
        self,
        expr_encoder: SuperFELTEncoder | None = None,
        mut_encoder: SuperFELTEncoder | None = None,
        cnv_encoder: SuperFELTEncoder | None = None,
        regressor: SuperFELTRegressor | None = None,
        ranges: tuple[float, float] = (0.0, 1.0),
        best_checkpoint: object = None,
    ) -> None:
        """Initialize per-drug model components.

        :param expr_encoder: Expression encoder module.
        :param mut_encoder: Mutation encoder module.
        :param cnv_encoder: CNV encoder module.
        :param regressor: Final regressor module.
        :param ranges: Response normalization range.
        :param best_checkpoint: Best training checkpoint reference.
        """
        self.expr_encoder = expr_encoder
        self.mut_encoder = mut_encoder
        self.cnv_encoder = cnv_encoder
        self.regressor = regressor
        self.ranges = ranges
        self.best_checkpoint = best_checkpoint


@register_predictor(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SUPERFELTR_REFERENCE,
)
class SuperFELTRPredictor(BlockPredictor):
    """SuperFELTR predictor: per-drug multi-omics with independent encoders."""

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
        self._drug_models: dict[str, _DrugModel] = {}
        self._feature_names: dict[str, _OmicFeatureNames] = {}

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train per-drug SuperFELTR models.

        :param batch: Training batch with all required cell-line omics blocks.
        """
        keys = routing_keys(batch)
        require_known_training_keys(keys)
        self._drug_models = {}
        self._feature_names = {}

        for drug_id, mask in iter_drug_masks(batch):
            context = TrainingContext(
                checkpoint_dir=_checkpoint_dir_for_drug(batch.training_context.checkpoint_dir, drug_id),
            )
            sub = replace(batch.subset_pairs(mask), training_context=context)
            self._fit_single_drug(drug_id, sub)

    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> None:
        """Fit a single-drug SuperFELTR model including encoders and regressor.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        """
        gex, mut, cnv = self._extract_pair_omics(batch)
        n_samples = gex.shape[0]

        feature_names = _OmicFeatureNames(
            gene_expression=batch.cell_line_blocks["gene_expression"].feature_names,
            mutations=batch.cell_line_blocks["mutations"].feature_names,
            copy_number_variation=batch.cell_line_blocks["copy_number_variation_gistic"].feature_names,
        )
        self._feature_names[drug_id] = feature_names

        if n_samples == 0:
            self._drug_models[drug_id] = _DrugModel()
            return

        dim_gex, dim_mut, dim_cnv = gex.shape[1], mut.shape[1], cnv.shape[1]

        response = np.asarray(batch.response, dtype=np.float64)
        from drevalpy.components.predictors.literature.molir.utils import make_ranges

        ranges = make_ranges(
            DrugResponseDataset(response=response, cell_line_ids=batch.cell_line_ids, drug_ids=batch.drug_ids)
        )

        output_train = DrugResponseDataset(
            response=response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        output_earlystopping = batch.early_stopping_response
        if output_earlystopping is not None and len(output_earlystopping) < 2:
            output_earlystopping = None

        cell_line_input = self._build_feature_dataset(batch)

        from .utils import train_superfeltr_model

        class _AlgoProxy:
            """Minimal proxy so _train_encoder_for_omic can access hyperparameters and ranges."""

            hyperparameters = dict(self._hyperparameters)
            early_stopping = True
            wandb_project = None

        proxy = _AlgoProxy()
        proxy.ranges = ranges

        encoder_dims = {
            "expression": dim_gex,
            "mutation": dim_mut,
            "copy_number_variation_gistic": dim_cnv,
        }

        encoders = {}
        for omic_type, dim in encoder_dims.items():
            encoder = SuperFELTEncoder(input_size=dim, hpams=proxy.hyperparameters, omic_type=omic_type, ranges=ranges)
            if n_samples >= self._hyperparameters["mini_batch"]:
                best_ckpt = train_superfeltr_model(
                    model=encoder,
                    hpams=proxy.hyperparameters,
                    output_train=output_train,
                    cell_line_input=cell_line_input,
                    output_earlystopping=output_earlystopping,
                    patience=5,
                    model_checkpoint_dir=str(batch.training_context.checkpoint_dir),
                )
                encoder = SuperFELTEncoder.load_from_checkpoint(best_ckpt.best_model_path)
            encoders[omic_type] = encoder

        expr_encoder = encoders["expression"]
        mut_encoder = encoders["mutation"]
        cnv_encoder = encoders["copy_number_variation_gistic"]

        regressor_input_size = (
            int(self._hyperparameters["out_dim_expr_encoder"])
            + int(self._hyperparameters["out_dim_mutation_encoder"])
            + int(self._hyperparameters["out_dim_cnv_encoder"])
        )
        regressor = SuperFELTRegressor(
            input_size=regressor_input_size,
            hpams=proxy.hyperparameters,
            encoders=(expr_encoder, mut_encoder, cnv_encoder),
        )

        best_checkpoint = None
        if n_samples >= self._hyperparameters["mini_batch"]:
            best_checkpoint = train_superfeltr_model(
                model=regressor,
                hpams=proxy.hyperparameters,
                output_train=output_train,
                cell_line_input=cell_line_input,
                output_earlystopping=output_earlystopping,
                patience=5,
                model_checkpoint_dir=str(batch.training_context.checkpoint_dir),
            )
            if best_checkpoint is not None:
                regressor = SuperFELTRegressor.load_from_checkpoint(
                    best_checkpoint.best_model_path,
                    input_size=regressor_input_size,
                    hpams=proxy.hyperparameters,
                    encoders=(expr_encoder, mut_encoder, cnv_encoder),
                )

        self._drug_models[drug_id] = _DrugModel(
            expr_encoder=expr_encoder,
            mut_encoder=mut_encoder,
            cnv_encoder=cnv_encoder,
            regressor=regressor,
            ranges=ranges,
            best_checkpoint=best_checkpoint,
        )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict drug responses for pairs, routing to per-drug models.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        """
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            if drug_id == "":
                continue
            dm = self._drug_models.get(str(drug_id))
            if dm is None or dm.regressor is None:
                continue
            mask = keys == drug_id
            sub = batch.subset_pairs(mask)
            preds = self._predict_single_drug(str(drug_id), dm, sub)
            predictions[mask] = np.asarray(preds, dtype=np.float64).ravel()
        return predictions

    def _predict_single_drug(self, drug_id: str, dm: _DrugModel, batch: ModelInputBatch) -> np.ndarray:
        """Predict for a single drug model.

        :param drug_id: Drug identifier for feature alignment.
        :param dm: Per-drug model container.
        :param batch: Subset batch for this drug.
        :returns: Predicted responses.
        """
        feature_names = self._feature_names.get(drug_id)
        if feature_names is None or dm.regressor is None:
            return np.full(batch.n_pairs, np.nan)

        gex, mut, cnv = self._extract_pair_omics(batch)
        gex = self._align_omic(
            gex, feature_names.gene_expression, batch.cell_line_blocks["gene_expression"].feature_names
        )
        mut = self._align_omic(mut, feature_names.mutations, batch.cell_line_blocks["mutations"].feature_names)
        cnv = self._align_omic(
            cnv,
            feature_names.copy_number_variation,
            batch.cell_line_blocks["copy_number_variation_gistic"].feature_names,
        )

        return np.atleast_1d(dm.regressor.predict(gex, mut, cnv))

    def _extract_pair_omics(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract pair-level omics matrices from batch blocks.

        :param batch: Input batch with cell-line blocks.
        :returns: Tuple of (gene_expression, mutations, cnv) matrices.
        """
        idx = batch.cell_line_pair_idx
        gex = np.asarray(batch.cell_line_blocks["gene_expression"].values[idx], dtype=np.float32)
        mut = np.asarray(batch.cell_line_blocks["mutations"].values[idx], dtype=np.float32)
        cnv = np.asarray(batch.cell_line_blocks["copy_number_variation_gistic"].values[idx], dtype=np.float32)
        return gex, mut, cnv

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

    def _build_feature_dataset(self, batch: ModelInputBatch) -> FeatureDataset:
        """Build a FeatureDataset from batch blocks for encoder training compatibility.

        :param batch: Input batch with cell-line blocks.
        :returns: FeatureDataset suitable for encoder training.
        """
        from drevalpy.datasets.dataset import FeatureDataset

        gex_block = batch.cell_line_blocks["gene_expression"]
        mut_block = batch.cell_line_blocks["mutations"]
        cnv_block = batch.cell_line_blocks["copy_number_variation_gistic"]

        unique_entity_ids = np.unique(batch.cell_line_ids)
        features: dict[str, dict[str, np.ndarray]] = {}
        entity_id_to_row = {str(eid): i for i, eid in enumerate(batch.cell_line_entity_ids)}

        for cl_id in unique_entity_ids:
            row = entity_id_to_row[str(cl_id)]
            features[str(cl_id)] = {
                "gene_expression": np.asarray(gex_block.values[row], dtype=np.float32),
                "mutations": np.asarray(mut_block.values[row], dtype=np.float32),
                "copy_number_variation_gistic": np.asarray(cnv_block.values[row], dtype=np.float32),
            }

        meta_info: dict[str, list[str]] = {
            "gene_expression": list(gex_block.feature_names or []),
            "mutations": list(mut_block.feature_names or []),
            "copy_number_variation_gistic": list(cnv_block.feature_names or []),
        }
        return FeatureDataset(features=features, meta_info=meta_info)

    def is_fitted(self) -> bool:
        """Report whether trained models exist.

        :returns: True when at least one per-drug model has been fitted.
        """
        return bool(self._drug_models)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted state for all per-drug models.

        :returns: Mapping with algorithm blobs and hyperparameters.
        """
        if not self._drug_models:
            return {}
        algorithms: dict[str, bytes] = {}
        for drug_id, dm in self._drug_models.items():
            fn = self._feature_names.get(drug_id)
            algorithms[drug_id] = self._serialize_drug_model(dm, fn)
        return {
            "algorithms": algorithms,
            "predictor_hyperparameters": dict(self._hyperparameters),
        }

    def _serialize_drug_model(self, dm: _DrugModel, fn: _OmicFeatureNames | None) -> bytes:
        """Serialize a single per-drug model to bytes.

        :param dm: Per-drug model container.
        :param fn: Associated feature name record.
        :returns: Serialized payload bytes.
        """
        payload: dict[str, Any] = {
            "hyperparameters": dict(self._hyperparameters),
            "ranges": dm.ranges,
            "gene_expression_features": list(fn.gene_expression) if fn and fn.gene_expression else None,
            "mutations_features": list(fn.mutations) if fn and fn.mutations else None,
            "copy_number_variation_features": (
                list(fn.copy_number_variation) if fn and fn.copy_number_variation else None
            ),
        }
        payload["input_dims"] = self._compute_input_dims(fn)

        for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
            module = getattr(dm, attr, None)
            if module is not None and hasattr(module, "state_dict"):
                payload[f"{attr}_state"] = save_state_dict(module.state_dict())

        return save_object_mapping(payload)

    @staticmethod
    def _compute_input_dims(fn: _OmicFeatureNames | None) -> dict[str, int | None]:
        """Compute input dimension mapping from feature names.

        :param fn: Feature name record.
        :returns: Mapping of omic type to dimension.
        """
        dims: dict[str, int | None] = {"expression": None, "mutation": None, "cnv": None}
        if fn and fn.gene_expression:
            dims["expression"] = len(fn.gene_expression)
        if fn and fn.mutations:
            dims["mutation"] = len(fn.mutations)
        if fn and fn.copy_number_variation:
            dims["cnv"] = len(fn.copy_number_variation)
        return dims

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state from serialized algorithm blobs.

        :param state: State mapping from ``get_state``.
        :raises PredictorStateError: If state is malformed.
        """
        algorithms_blob = state.get("algorithms")
        if not isinstance(algorithms_blob, dict):
            msg = "SuperFELTRPredictor state requires an 'algorithms' mapping"
            raise PredictorStateError(msg)
        hyperparameters = state.get("predictor_hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)

        self._drug_models = {}
        self._feature_names = {}
        for drug_id, blob in algorithms_blob.items():
            if not isinstance(blob, (bytes, bytearray)):
                msg = f"SuperFELTRPredictor payload for {drug_id!r} must be bytes"
                raise PredictorStateError(msg)
            payload = load_object_mapping(bytes(blob))
            fn = self._deserialize_feature_names(payload)
            self._feature_names[str(drug_id)] = fn
            dm = self._deserialize_drug_model(payload)
            self._drug_models[str(drug_id)] = dm

    @staticmethod
    def _deserialize_feature_names(payload: dict[str, Any]) -> _OmicFeatureNames:
        """Extract feature names from a deserialized payload.

        :param payload: Deserialized drug model payload.
        :returns: Omic feature names record.
        """
        return _OmicFeatureNames(
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

    def _deserialize_drug_model(self, payload: dict[str, Any]) -> _DrugModel:
        """Reconstruct a _DrugModel from a deserialized payload.

        :param payload: Deserialized drug model payload.
        :returns: Reconstructed per-drug model.
        """
        hpams = payload.get("hyperparameters", dict(self._hyperparameters))
        ranges = payload.get("ranges", (0.0, 1.0))
        if isinstance(ranges, list):
            ranges = tuple(ranges)

        input_dims = payload.get("input_dims", {})
        expr_dim = input_dims.get("expression")
        mut_dim = input_dims.get("mutation")
        cnv_dim = input_dims.get("cnv")

        dm = _DrugModel(ranges=ranges)
        if isinstance(expr_dim, int) and isinstance(mut_dim, int) and isinstance(cnv_dim, int):
            self._rebuild_encoders_and_regressor(dm, hpams, ranges, expr_dim, mut_dim, cnv_dim, payload)

        return dm

    @staticmethod
    def _rebuild_encoders_and_regressor(
        dm: _DrugModel,
        hpams: dict[str, Any],
        ranges: tuple[float, float],
        expr_dim: int,
        mut_dim: int,
        cnv_dim: int,
        payload: dict[str, Any],
    ) -> None:
        """Rebuild encoder and regressor modules and load their state dicts.

        :param dm: Drug model to populate.
        :param hpams: Hyperparameters for the modules.
        :param ranges: Response normalization range.
        :param expr_dim: Expression encoder input size.
        :param mut_dim: Mutation encoder input size.
        :param cnv_dim: CNV encoder input size.
        :param payload: Deserialized payload with state blobs.
        """
        dm.expr_encoder = SuperFELTEncoder(input_size=expr_dim, hpams=hpams, omic_type="expression", ranges=ranges)
        dm.mut_encoder = SuperFELTEncoder(input_size=mut_dim, hpams=hpams, omic_type="mutation", ranges=ranges)
        dm.cnv_encoder = SuperFELTEncoder(
            input_size=cnv_dim, hpams=hpams, omic_type="copy_number_variation_gistic", ranges=ranges
        )
        regressor_input_size = (
            int(hpams["out_dim_expr_encoder"])
            + int(hpams["out_dim_mutation_encoder"])
            + int(hpams["out_dim_cnv_encoder"])
        )
        dm.regressor = SuperFELTRegressor(
            input_size=regressor_input_size,
            hpams=hpams,
            encoders=(dm.expr_encoder, dm.mut_encoder, dm.cnv_encoder),
        )

        for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
            module = getattr(dm, attr, None)
            state_blob = payload.get(f"{attr}_state")
            if module is not None and isinstance(state_blob, (bytes, bytearray)):
                module.load_state_dict(_load_torch_state_dict(bytes(state_blob)))

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "mini_batch": 55,
            "dropout_rate": 0.5,
            "weight_decay": 0.01,
            "out_dim_expr_encoder": 256,
            "out_dim_mutation_encoder": 32,
            "out_dim_cnv_encoder": 64,
            "epochs": 30,
            "margin": 1.0,
            "learning_rate": 0.01,
        }
