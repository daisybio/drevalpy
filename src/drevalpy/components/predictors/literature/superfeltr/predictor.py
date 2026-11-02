"""SuperFELTR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.hyperparameter_space import fixed_entries
from drevalpy.components.predictors.literature._metadata import SUPERFELTR_REFERENCE
from drevalpy.components.predictors.literature._omic_helpers import (
    aligned_pair_matrices,
    omic_matrices,
    validation_split,
)
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.accelerator import Accelerator

# .utils imports pytorch_lightning at module scope, and this module is imported
# eagerly by register_builtins(), so the encoder/regressor imports are
# deferred to the methods that construct them. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit

    from .utils import SuperFELTEncoder, SuperFELTRegressor

#: Encoder omic types, in the order the regressor concatenates their outputs. Paired
#: with the ``out_dim_*`` hyperparameter naming each one's width.
_ENCODER_OMIC_TYPES = ("expression", "mutation", "copy_number_variation_gistic")
_ENCODER_DIM_KEYS = ("out_dim_expr_encoder", "out_dim_mutation_encoder", "out_dim_cnv_encoder")


@register(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    reference=SUPERFELTR_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class SuperFELTRPredictor(SingleDrugOmicsPredictor):
    """SuperFELTR predictor: per-drug multi-omics with independent encoders."""

    _models: dict[str, SuperFELTRegressor]

    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> SuperFELTRegressor | None:
        """Fit a single-drug SuperFELTR model including encoders and regressor.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        :returns: The fitted regressor, which holds the fitted encoders, or ``None`` when the drug has no pairs.
        """
        if batch.n_pairs == 0:
            return None

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
        return self._fit_regressor(encoders, train, val, checkpoint_dir, trainable)

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
    ) -> SuperFELTRegressor:
        """Train the regression head on top of the frozen encoders.

        :param encoders: The fitted encoders.
        :param train: Training split.
        :param val: Validation split, or None.
        :param checkpoint_dir: Directory the fit checkpoints into.
        :param trainable: Whether there are enough pairs to fit at all.
        :returns: The regressor, loaded from its best checkpoint when it was fitted.
        """
        from .utils import SuperFELTRegressor, train_superfeltr_model

        input_size = sum(int(self._hyperparameters[key]) for key in _ENCODER_DIM_KEYS)
        regressor = SuperFELTRegressor(
            input_size=input_size,
            hpams=dict(self._hyperparameters),
            encoders=encoders,
        )
        if not trainable:
            return regressor

        best_checkpoint = train_superfeltr_model(
            model=regressor,
            hpams=dict(self._hyperparameters),
            train=train,
            val=val,
            patience=5,
            model_checkpoint_dir=checkpoint_dir,
        )
        return SuperFELTRegressor.load_from_checkpoint(
            best_checkpoint.best_model_path,
            input_size=input_size,
            hpams=dict(self._hyperparameters),
            encoders=encoders,
        )

    def _predict_single_drug(self, drug_id: str, model: SuperFELTRegressor, batch: ModelInputBatch) -> np.ndarray:
        """Predict for a single drug model.

        :param drug_id: Drug identifier for feature alignment.
        :param model: The drug's regressor.
        :param batch: Subset batch for this drug.
        :returns: Predicted responses.
        """
        feature_names = self._feature_names.get(drug_id)
        if feature_names is None:
            return np.full(batch.n_pairs, np.nan)

        return np.atleast_1d(model.predict(*aligned_pair_matrices(batch, feature_names)))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space; SuperFELTR tunes nothing.

        :returns: Fixed specs for every SuperFELTR hyperparameter.
        """
        return fixed_entries(
            {
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
        )
