"""SuperFELTR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import SUPERFELTR_REFERENCE
from drevalpy.components.predictors.literature._single_drug_omics import (
    OmicFeatureNames,
    SingleDrugOmicsPredictor,
    aligned_pair_matrices,
    feature_names_payload,
    omic_feature_names,
    omic_matrices,
    validation_split,
)
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import load_state_dict as _load_torch_state_dict
from drevalpy.utils.torch_io import save_state_dict, save_trusted_mapping

# .utils imports pytorch_lightning at module scope, and this module is imported
# eagerly by register_builtin_components(), so the encoder/regressor imports are
# deferred to the methods that construct them. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit

    from .utils import SuperFELTEncoder, SuperFELTRegressor

#: Encoder omic types, in the order the regressor concatenates their outputs. Paired
#: with the ``out_dim_*`` hyperparameter naming each one's width.
_ENCODER_OMIC_TYPES = ("expression", "mutation", "copy_number_variation_gistic")
_ENCODER_DIM_KEYS = ("out_dim_expr_encoder", "out_dim_mutation_encoder", "out_dim_cnv_encoder")


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


@register(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SUPERFELTR_REFERENCE,
)
class SuperFELTRPredictor(SingleDrugOmicsPredictor):
    """SuperFELTR predictor: per-drug multi-omics with independent encoders."""

    _models: dict[str, _DrugModel]

    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> None:
        """Fit a single-drug SuperFELTR model including encoders and regressor.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        """
        self._feature_names[drug_id] = omic_feature_names(batch)
        if batch.n_pairs == 0:
            self._models[drug_id] = _DrugModel()
            return

        matrices = omic_matrices(batch)
        response = np.asarray(batch.response, dtype=np.float32)
        std = float(np.std(response))
        ranges = (std * 0.1, std)

        train = matrices.split(batch.sample_pair_idx, response)
        val = validation_split(matrices, batch)
        checkpoint_dir = str(batch.training_context.checkpoint_dir)
        #: Below one mini-batch the training loader's ``drop_last`` empties it, so the
        #: encoders and regressor stay randomly initialized rather than being fitted.
        trainable = batch.n_pairs >= self._hyperparameters["mini_batch"]

        encoders = self._fit_encoders(matrices.widths(), ranges, train, val, checkpoint_dir, trainable)
        regressor, best_checkpoint = self._fit_regressor(encoders, train, val, checkpoint_dir, trainable)

        self._models[drug_id] = _DrugModel(
            expr_encoder=encoders[0],
            mut_encoder=encoders[1],
            cnv_encoder=encoders[2],
            regressor=regressor,
            ranges=ranges,
            best_checkpoint=best_checkpoint,
        )

    def _fit_encoders(
        self,
        widths: tuple[int, int, int],
        ranges: tuple[float, float],
        train: OmicsSplit,
        val: OmicsSplit | None,
        checkpoint_dir: str,
        trainable: bool,
    ) -> tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder]:
        """Train one independent encoder per omic view.

        Each encoder sees all three views in every batch and selects its own, so they
        share the loaders even though they are fitted separately.

        :param widths: Input width of each omic view.
        :param ranges: Positive and negative triplet-loss ranges.
        :param train: Training split.
        :param val: Validation split, or None.
        :param checkpoint_dir: Directory the fits checkpoint into.
        :param trainable: Whether there are enough pairs to fit at all.
        :returns: The three encoders, in ``_ENCODER_OMIC_TYPES`` order.
        """
        from .utils import SuperFELTEncoder, train_superfeltr_model

        encoders = []
        for omic_type, width in zip(_ENCODER_OMIC_TYPES, widths, strict=True):
            encoder = SuperFELTEncoder(
                input_size=width,
                hpams=dict(self._hyperparameters),
                omic_type=omic_type,
                ranges=ranges,
            )
            if trainable:
                best_ckpt = train_superfeltr_model(
                    model=encoder,
                    hpams=dict(self._hyperparameters),
                    train=train,
                    val=val,
                    patience=5,
                    model_checkpoint_dir=checkpoint_dir,
                )
                encoder = SuperFELTEncoder.load_from_checkpoint(best_ckpt.best_model_path)
            encoders.append(encoder)
        return encoders[0], encoders[1], encoders[2]

    def _fit_regressor(
        self,
        encoders: tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder],
        train: OmicsSplit,
        val: OmicsSplit | None,
        checkpoint_dir: str,
        trainable: bool,
    ) -> tuple[SuperFELTRegressor, object | None]:
        """Train the regression head on top of the frozen encoders.

        :param encoders: The fitted encoders.
        :param train: Training split.
        :param val: Validation split, or None.
        :param checkpoint_dir: Directory the fit checkpoints into.
        :param trainable: Whether there are enough pairs to fit at all.
        :returns: Tuple of the regressor and its best checkpoint, the latter ``None``
            when the fit was skipped.
        """
        from .utils import SuperFELTRegressor, train_superfeltr_model

        input_size = self._regressor_input_size(self._hyperparameters)
        regressor = SuperFELTRegressor(
            input_size=input_size,
            hpams=dict(self._hyperparameters),
            encoders=encoders,
        )
        if not trainable:
            return regressor, None

        best_checkpoint = train_superfeltr_model(
            model=regressor,
            hpams=dict(self._hyperparameters),
            train=train,
            val=val,
            patience=5,
            model_checkpoint_dir=checkpoint_dir,
        )
        if best_checkpoint is not None:
            regressor = SuperFELTRegressor.load_from_checkpoint(
                best_checkpoint.best_model_path,
                input_size=input_size,
                hpams=dict(self._hyperparameters),
                encoders=encoders,
            )
        return regressor, best_checkpoint

    @staticmethod
    def _regressor_input_size(hpams: dict[str, Any]) -> int:
        """Sum the three encoder output widths.

        :param hpams: Hyperparameters carrying the ``out_dim_*`` widths.
        :returns: The regressor's input width.
        """
        return sum(int(hpams[key]) for key in _ENCODER_DIM_KEYS)

    def _predict_single_drug(self, drug_id: str, model: _DrugModel, batch: ModelInputBatch) -> np.ndarray:
        """Predict for a single drug model.

        :param drug_id: Drug identifier for feature alignment.
        :param model: Per-drug model container.
        :param batch: Subset batch for this drug.
        :returns: Predicted responses, ``NaN`` when the drug's regressor was never fitted.
        """
        feature_names = self._feature_names.get(drug_id)
        if feature_names is None or model.regressor is None:
            return np.full(batch.n_pairs, np.nan)

        return np.atleast_1d(model.regressor.predict(*aligned_pair_matrices(batch, feature_names)))

    def _serialize_model(self, drug_id: str, model: _DrugModel) -> bytes:
        """Serialize a single per-drug model to bytes.

        :param drug_id: Drug identifier, used to look up the recorded feature names.
        :param model: Per-drug model container.
        :returns: Serialized payload bytes.
        """
        fn = self._feature_names.get(drug_id)
        payload: dict[str, Any] = {
            "hyperparameters": dict(self._hyperparameters),
            "ranges": model.ranges,
            "input_dims": self._compute_input_dims(model, fn),
            **feature_names_payload(fn),
        }

        for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
            module = getattr(model, attr, None)
            if module is not None and hasattr(module, "state_dict"):
                payload[f"{attr}_state"] = save_state_dict(module.state_dict())

        return save_trusted_mapping(payload)

    @staticmethod
    def _compute_input_dims(dm: _DrugModel, fn: OmicFeatureNames | None) -> dict[str, int | None]:
        """Persist actual encoder widths even when feature names are absent.

        :param dm: Fitted per-drug model.
        :param fn: Feature name record.
        :returns: Mapping of omic type to dimension.
        """
        names = fn.as_tuple() if fn is not None else (None, None, None)
        keys = ("expression", "mutation", "cnv")
        return {
            key: int(encoder.encode[0].in_features) if encoder is not None else len(value) if value else None
            for key, value, encoder in zip(keys, names, (dm.expr_encoder, dm.mut_encoder, dm.cnv_encoder), strict=True)
        }

    def _restore_model(self, payload: dict[str, Any]) -> _DrugModel:
        """Reconstruct a _DrugModel from a deserialized payload.

        :param payload: Deserialized drug model payload.
        :returns: Reconstructed per-drug model; its modules stay ``None`` for a drug that was never fitted.
        """
        hpams = payload["hyperparameters"]
        ranges = tuple(payload["ranges"])
        input_dims = payload["input_dims"]
        widths = tuple(input_dims[key] for key in ("expression", "mutation", "cnv"))

        dm = _DrugModel(ranges=ranges)
        if all(isinstance(width, int) for width in widths):
            self._rebuild_encoders_and_regressor(dm, hpams, ranges, widths, payload)

        return dm

    @classmethod
    def _rebuild_encoders_and_regressor(
        cls,
        dm: _DrugModel,
        hpams: dict[str, Any],
        ranges: tuple[float, float],
        widths: tuple[int, ...],
        payload: dict[str, Any],
    ) -> None:
        """Rebuild encoder and regressor modules and load their state dicts.

        :param dm: Drug model to populate.
        :param hpams: Hyperparameters for the modules.
        :param ranges: Response normalization range.
        :param widths: Input widths of the expression, mutation and CNV encoders.
        :param payload: Deserialized payload with state blobs.
        """
        from .utils import SuperFELTEncoder, SuperFELTRegressor

        encoders = tuple(
            SuperFELTEncoder(input_size=width, hpams=hpams, omic_type=omic_type, ranges=ranges)
            for omic_type, width in zip(_ENCODER_OMIC_TYPES, widths, strict=True)
        )
        dm.expr_encoder, dm.mut_encoder, dm.cnv_encoder = encoders
        dm.regressor = SuperFELTRegressor(
            input_size=cls._regressor_input_size(hpams),
            hpams=hpams,
            encoders=encoders,
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
