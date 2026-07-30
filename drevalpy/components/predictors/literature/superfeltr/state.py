"""Persistence mapping for SuperFELTR."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.superfeltr.algorithm import SuperFELTR
from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder, SuperFELTRegressor


def _feature_count(features: object) -> int | None:
    if features is None:
        return None
    try:
        return len(features)  # type: ignore[arg-type]
    except TypeError:
        return None


def _save_module_state(algorithm: SuperFELTR, payload: dict[str, Any], attr: str) -> None:
    module = getattr(algorithm, attr, None)
    if module is not None and hasattr(module, "state_dict"):
        payload[f"{attr}_state"] = save_state_dict(module.state_dict())


def export_state(algorithm: SuperFELTR) -> dict[str, Any]:
    """Serialize a fitted SuperFELTR algorithm for predictor persistence."""
    payload: dict[str, Any] = {
        "hyperparameters": dict(algorithm.hyperparameters),
        "ranges": algorithm.ranges,
        "gene_expression_features": algorithm.gene_expression_features,
        "mutations_features": algorithm.mutations_features,
        "copy_number_variation_features": algorithm.copy_number_variation_features,
        "input_dims": {
            "expression": _feature_count(algorithm.gene_expression_features),
            "mutation": _feature_count(algorithm.mutations_features),
            "cnv": _feature_count(algorithm.copy_number_variation_features),
        },
    }
    for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
        _save_module_state(algorithm, payload, attr)
    return payload


def apply_state(payload: dict[str, Any]) -> SuperFELTR:
    """Restore a SuperFELTR algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = SuperFELTR()
    algorithm.configure(hyperparameters)
    ranges = payload.get("ranges")
    if isinstance(ranges, tuple):
        algorithm.ranges = ranges
    algorithm.gene_expression_features = payload.get("gene_expression_features")
    algorithm.mutations_features = payload.get("mutations_features")
    algorithm.copy_number_variation_features = payload.get("copy_number_variation_features")

    input_dims = payload.get("input_dims")
    if isinstance(input_dims, dict):
        expr_dim = input_dims.get("expression")
        mut_dim = input_dims.get("mutation")
        cnv_dim = input_dims.get("cnv")
        if isinstance(expr_dim, int) and isinstance(mut_dim, int) and isinstance(cnv_dim, int):
            algorithm.expr_encoder = SuperFELTEncoder(
                input_size=expr_dim,
                hpams=hyperparameters,
                omic_type="expression",
                ranges=algorithm.ranges,
            )
            algorithm.mut_encoder = SuperFELTEncoder(
                input_size=mut_dim,
                hpams=hyperparameters,
                omic_type="mutation",
                ranges=algorithm.ranges,
            )
            algorithm.cnv_encoder = SuperFELTEncoder(
                input_size=cnv_dim,
                hpams=hyperparameters,
                omic_type="copy_number_variation_gistic",
                ranges=algorithm.ranges,
            )
            algorithm.regressor = SuperFELTRegressor(
                input_size=(
                    int(hyperparameters["out_dim_expr_encoder"])
                    + int(hyperparameters["out_dim_mutation_encoder"])
                    + int(hyperparameters["out_dim_cnv_encoder"])
                ),
                hpams=hyperparameters,
                encoders=(algorithm.expr_encoder, algorithm.mut_encoder, algorithm.cnv_encoder),
            )

    for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
        module = getattr(algorithm, attr, None)
        state_blob = payload.get(f"{attr}_state")
        if module is not None and isinstance(state_blob, (bytes, bytearray)):
            module.load_state_dict(load_state_dict(bytes(state_blob)))
    return algorithm
