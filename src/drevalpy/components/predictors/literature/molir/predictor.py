"""MOLIR literature predictor consuming ModelInputBatch directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import MOLIR_REFERENCE
from drevalpy.components.predictors.literature._single_drug_omics import (
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

# MOLIModel lives in .utils, which imports pytorch_lightning at module scope. This
# module is imported eagerly by register_builtin_components(), so the import is
# deferred to the methods that construct a model - keeping the training stack off
# the critical path of `import drevalpy`. Guarded by tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.molir.utils import MOLIModel


@register(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=MOLIR_REFERENCE,
)
class MOLIRPredictor(SingleDrugOmicsPredictor):
    """MOLIR predictor: per-drug multi-omics late integration model."""

    _models: dict[str, MOLIModel]

    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> None:
        """Fit a single-drug MOLIR model.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        """
        self._feature_names[drug_id] = omic_feature_names(batch)
        if batch.n_pairs == 0:
            return

        matrices = omic_matrices(batch)
        model = self._build_model(matrices.widths())

        if batch.n_pairs >= self._hyperparameters["mini_batch"]:
            model.fit(
                train=matrices.split(batch.sample_pair_idx, np.asarray(batch.response, dtype=np.float32)),
                val=validation_split(matrices, batch),
                model_checkpoint_dir=str(batch.training_context.checkpoint_dir),
            )

        self._models[drug_id] = model

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

        return np.atleast_1d(model.predict(*aligned_pair_matrices(batch, feature_names)))

    def _serialize_model(self, drug_id: str, model: MOLIModel) -> bytes:
        """Serialize one drug's model with the feature names it was trained on.

        :param drug_id: Drug identifier.
        :param model: The drug's fitted model.
        :returns: Serialized payload bytes.
        """
        payload: dict[str, Any] = {
            "hyperparameters": dict(self._hyperparameters),
            "model_state": save_state_dict(model.state_dict()),
            "input_dims": {
                "expr": model.expression_encoder.encode[0].in_features,
                "mut": model.mutation_encoder.encode[0].in_features,
                "cnv": model.cna_encoder.encode[0].in_features,
            },
            **feature_names_payload(self._feature_names.get(drug_id)),
        }
        return save_trusted_mapping(payload)

    def _restore_model(self, payload: dict[str, Any]) -> MOLIModel:
        """Rebuild one per-drug model from its deserialized payload.

        :param payload: Deserialized per-drug payload.
        :returns: The restored model.
        """
        from drevalpy.components.predictors.literature.molir.utils import MOLIModel

        input_dims = payload["input_dims"]
        model = MOLIModel(
            hpams=payload["hyperparameters"],
            input_dim_expr=input_dims["expr"],
            input_dim_mut=input_dims["mut"],
            input_dim_cnv=input_dims["cnv"],
        )
        model.load_state_dict(_load_torch_state_dict(bytes(payload["model_state"])))
        return model

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
