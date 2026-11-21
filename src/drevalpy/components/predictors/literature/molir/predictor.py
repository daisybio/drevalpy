"""MOLIR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.hyperparameter_space import fixed_entries
from drevalpy.components.predictors.literature._metadata import MOLIR_REFERENCE
from drevalpy.components.predictors.literature._omic_helpers import (
    omic_matrices,
    validation_split,
)
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.accelerator import Accelerator

# MOLIModel lives in .utils, which imports pytorch_lightning at module scope. This
# module is imported eagerly by register_builtins(), so the import is
# deferred to the methods that construct a model - keeping the training stack off
# the critical path of `import drevalpy`. Guarded by tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.molir.utils import MOLIModel


@register(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    reference=MOLIR_REFERENCE,
    license="GPL-3.0-only",
    accelerator=Accelerator.OPTIONAL,
)
class MOLIRPredictor(SingleDrugOmicsPredictor):
    """MOLIR predictor: per-drug multi-omics late integration model."""

    _models: dict[str, MOLIModel]

    def _fit_drug(self, drug_id: str, batch: ModelInputBatch) -> MOLIModel:
        """Fit a single-drug MOLIR model.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        :returns: The model; left untrained when the drug has fewer pairs than one mini-batch.
        """
        matrices = omic_matrices(batch)
        model = self._build_model(matrices.widths())

        if batch.n_pairs >= self._hyperparameters["mini_batch"]:
            model.fit(
                train=matrices.split(batch.sample_pair_idx, np.asarray(batch.response, dtype=np.float32)),
                val=validation_split(matrices, batch),
                model_checkpoint_dir=str(batch.training_context.checkpoint_dir),
            )

        return model

    def _build_model(self, widths: tuple[int, int, int]) -> MOLIModel:
        """Construct an untrained MOLIR model sized to the three omic views.

        :param widths: Feature widths of the expression, mutation and CNV views.
        :returns: The initialized model.
        """
        from drevalpy.components.predictors.literature.molir.utils import MOLIModel

        return MOLIModel(
            hpams=dict(self._hyperparameters),
            input_dim_expr=widths[0],
            input_dim_mut=widths[1],
            input_dim_cnv=widths[2],
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space; MOLIR tunes nothing.

        :returns: Fixed specs for every MOLIR hyperparameter.
        """
        return fixed_entries(
            {
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
        )
