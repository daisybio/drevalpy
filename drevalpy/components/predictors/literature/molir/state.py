"""Persistence mapping for the molir algorithm."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._protocols import feature_count
from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.molir.algorithm import MOLIR
from drevalpy.components.predictors.literature.molir.utils import MOLIModel


def export_state(algorithm: MOLIR) -> dict[str, Any]:
    """Serialize a fitted algorithm for predictor persistence.

    :param algorithm: Fitted algorithm instance.

    :returns: JSON-serializable state mapping.
    """
    payload: dict[str, Any] = {
        "hyperparameters": dict(algorithm.hyperparameters),
        "gene_expression_features": algorithm.gene_expression_features,
        "mutations_features": algorithm.mutations_features,
        "copy_number_variation_features": algorithm.copy_number_variation_features,
    }
    model = getattr(algorithm, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        payload["model_state"] = save_state_dict(model.state_dict())
        payload["input_dims"] = {
            "expr": feature_count(algorithm.gene_expression_features),
            "mut": feature_count(algorithm.mutations_features),
            "cnv": feature_count(algorithm.copy_number_variation_features),
        }
    return payload


def apply_state(payload: dict[str, Any]) -> MOLIR:
    """Restore an algorithm from a persisted payload.

    :param payload: Serialized state produced by ``export_state``.

    :returns: Configured algorithm instance.

    :raises ValueError: If required payload fields are missing.
    """
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = MOLIR()
    algorithm.configure(hyperparameters)
    algorithm.gene_expression_features = payload.get("gene_expression_features")
    algorithm.mutations_features = payload.get("mutations_features")
    algorithm.copy_number_variation_features = payload.get("copy_number_variation_features")
    model_state = payload.get("model_state")
    input_dims = payload.get("input_dims")
    if isinstance(model_state, (bytes, bytearray)) and isinstance(input_dims, dict):
        expr = input_dims.get("expr")
        mut = input_dims.get("mut")
        cnv = input_dims.get("cnv")
        if isinstance(expr, int) and isinstance(mut, int) and isinstance(cnv, int):
            algorithm.model = MOLIModel(
                hpams=hyperparameters,
                input_dim_expr=expr,
                input_dim_mut=mut,
                input_dim_cnv=cnv,
            )
            algorithm.model.load_state_dict(load_state_dict(bytes(model_state)))
    return algorithm
