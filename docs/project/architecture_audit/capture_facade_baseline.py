"""Capture deterministic facade predictions from the archived starting checkout.

The runner is deliberately kept with the audit results: it imports the
synthetic fixture builders from ``DREVALPY_BASELINE_ROOT`` and therefore cannot
accidentally record a working-tree implementation after the architecture work
starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any

import numpy as np
from upath import UPath

ARCHIVE = UPath(os.environ["DREVALPY_BASELINE_ROOT"]).resolve()
sys.path.insert(0, str(ARCHIVE))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _checksum(value: Any) -> str:
    """Return a stable checksum for a fitted component state."""
    import pickle

    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _run(seed: int, preset: str, dataset: Any, split: Any) -> dict[str, Any]:
    from drevalpy.models import construct_model

    _seed(seed)
    model_class = construct_model(preset)
    model = model_class(model_class.get_default_hyperparameters())
    model.train(dataset, split)
    predictions = np.asarray(model.predict(dataset, split), dtype=np.float64)
    if model._stack is None:
        raise RuntimeError(f"{preset} did not initialize a component stack")
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "predictions": predictions.tolist(),
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "component_state_sha256": _checksum(model._stack.component_state()),
        "resolved_config": model._resolved_model_config.model_dump(mode="json"),
    }


def main() -> None:
    """Write frozen facade outputs from the archived implementation only."""
    from tests.models.synthetic_fixtures import (
        lco_split_masks,
        synthetic_mudataset_gene_expression_fingerprints,
        synthetic_mudataset_identity,
    )

    output = UPath(os.environ["FACADE_BASELINE_OUTPUT"])
    split = lco_split_masks()
    records: dict[str, Any] = {
        "commit": "e4ad11766bc53554eb0dd53fd29ba47fa41000c2",
        "protocol": "facade train/predict on LCO synthetic data; seeds 0, 1, 2",
        "models": {},
    }
    for name, preset, dataset_factory in (
        ("naive", "NaivePredictor", synthetic_mudataset_identity),
        ("scaled_gene_expression_elastic_net", "ElasticNet", synthetic_mudataset_gene_expression_fingerprints),
    ):
        records["models"][name] = {str(seed): _run(seed, preset, dataset_factory(), split) for seed in (0, 1, 2)}
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
