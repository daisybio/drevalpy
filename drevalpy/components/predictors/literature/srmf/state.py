"""Persistence mapping for SRMF."""

from __future__ import annotations

from typing import Any

import pandas as pd

from drevalpy.components.predictors.literature.srmf.algorithm import SRMF


def export_state(algorithm: SRMF) -> dict[str, Any]:
    """Serialize a fitted SRMF algorithm for predictor persistence."""
    return {
        "hyperparameters": dict(algorithm.hyperparameters),
        "best_u": algorithm.best_u.to_dict(orient="split"),
        "best_v": algorithm.best_v.to_dict(orient="split"),
        "w": algorithm.w.to_dict(orient="split"),
        "training_mean": float(getattr(algorithm, "training_mean", 0.0)),
        "k": int(algorithm.k),
        "lambda_l": float(algorithm.lambda_l),
        "lambda_d": float(algorithm.lambda_d),
        "lambda_c": float(algorithm.lambda_c),
        "max_iter": int(algorithm.max_iter),
        "seed": int(algorithm.seed),
    }


def apply_state(payload: dict[str, Any]) -> SRMF:
    """Restore an SRMF algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = SRMF()
    algorithm.configure(hyperparameters)
    for key in ("best_u", "best_v", "w"):
        table = payload.get(key)
        if isinstance(table, dict):
            setattr(algorithm, key, pd.DataFrame(**table))
    algorithm.training_mean = float(payload.get("training_mean", 0.0))
    for attr in ("k", "lambda_l", "lambda_d", "lambda_c", "max_iter", "seed"):
        if attr in payload:
            setattr(algorithm, attr, payload[attr])
    return algorithm
