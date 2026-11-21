"""SuperFELTR literature predictor consuming PredictorInput directly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.components.predictors.literature._omics import (
    OmicsSplit,
    SingleDrugOmicsPredictor,
    fit_omics_model,
    omic_matrices,
    validation_split,
)
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

# .utils imports pytorch_lightning at module scope, and this module is imported
# eagerly by register_builtins(), so the encoder/regressor imports are
# deferred to the methods that construct them. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    import pytorch_lightning as pl

    from .utils import SuperFELTEncoder, SuperFELTRegressor

_REFERENCE = LiteratureReference(
    repo_url="https://github.com/DMCB-GIST/Super.FELT",
    citation_doi="10.1186/s12859-023-05166-7",
    citation_text=(
        "SuperFELTR is a regression adaptation of Super.FELT (Park et al., BMC Bioinformatics 2021, "
        "doi:10.1186/s12859-021-04146-z); the DOI above is the Hauptmann and Kramer benchmark that "
        "compares MOLI and Super.FELT."
    ),
    deviations=(
        "Regression adaptation: preserve supervised omics encoders and triplet training; use continuous "
        "response triplets and MSE regression without the binary classifier sigmoid/BCE."
    ),
)


@register(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    reference=_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class SuperFELTRPredictor(SingleDrugOmicsPredictor):
    """SuperFELTR predictor: per-drug multi-omics with independent encoders."""

    _models: dict[str, SuperFELTRegressor]

    fixed_hyperparameters = {
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

    def _fit_drug(self, drug_id: str, data: PredictorInput) -> SuperFELTRegressor:
        """Fit a single-drug SuperFELTR model including encoders and regressor.

        :param drug_id: Identifier of the drug to train on.
        :param data: Subset batch for this drug.
        :returns: The fitted regressor, which holds the fitted encoders.
        """
        matrices = omic_matrices(data)
        response = np.asarray(data.response, dtype=np.float32)
        std = float(np.std(response))
        ranges = (std * 0.1, std)

        train = OmicsSplit(*matrices, response=response, pair_idx=data.sample_pair_idx)
        val = validation_split(matrices, data)
        checkpoint_dir = str(data.training_context.checkpoint_dir)
        #: Below one mini-batch the training loader's ``drop_last`` empties it, so the
        #: encoders and regressor stay randomly initialized rather than being fitted.
        trainable = len(data) >= self._hyperparameters["mini_batch"]

        widths = tuple(matrix.shape[1] for matrix in matrices)
        encoders = self._fit_encoders(widths, ranges, train, val, checkpoint_dir, trainable)
        return self._fit_regressor(encoders, train, val, checkpoint_dir, trainable)

    def _fit_encoders(
        self,
        widths: tuple[int, ...],
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
        :returns: The three encoders, in ``ENCODER_DIM_KEYS`` order.
        """
        from .utils import ENCODER_DIM_KEYS, SuperFELTEncoder

        encoders = []
        for omic_type, width in zip(ENCODER_DIM_KEYS, widths, strict=True):
            encoder = SuperFELTEncoder(
                input_size=width,
                hpams=dict(self._hyperparameters),
                omic_type=omic_type,
                ranges=ranges,
            )
            if trainable:
                best_ckpt = self._fit_module(encoder, train, val, checkpoint_dir)
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
        from .utils import ENCODER_DIM_KEYS, SuperFELTRegressor

        input_size = sum(int(self._hyperparameters[key]) for key in ENCODER_DIM_KEYS.values())
        regressor = SuperFELTRegressor(
            input_size=input_size,
            hpams=dict(self._hyperparameters),
            encoders=encoders,
        )
        if not trainable:
            return regressor

        best_checkpoint = self._fit_module(regressor, train, val, checkpoint_dir)
        return SuperFELTRegressor.load_from_checkpoint(
            best_checkpoint.best_model_path,
            input_size=input_size,
            hpams=dict(self._hyperparameters),
            encoders=encoders,
        )

    def _fit_module(
        self, model: pl.LightningModule, train: OmicsSplit, val: OmicsSplit | None, checkpoint_dir: str
    ) -> pl.callbacks.ModelCheckpoint:
        """Fit one encoder or the regressor on the CPU with early stopping.

        :param model: Encoder or regressor to fit.
        :param train: Training split.
        :param val: Validation split, or None to monitor the training loss.
        :param checkpoint_dir: Directory the fit checkpoints into.
        :returns: The checkpoint callback holding the best epoch.
        :raises ValueError: If ``epochs`` or ``mini_batch`` is not an integer.
        """
        epochs, mini_batch = self._hyperparameters["epochs"], self._hyperparameters["mini_batch"]
        if not isinstance(epochs, int) or not isinstance(mini_batch, int):
            raise ValueError("epochs and mini_batch must be integers!")
        return fit_omics_model(
            model, train, val, batch_size=mini_batch, epochs=epochs, patience=5, checkpoint_dir=checkpoint_dir
        )
