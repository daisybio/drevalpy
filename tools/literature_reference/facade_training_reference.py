"""Record deterministic raw-dataset facade evidence for literature models.

Run with ``PYTHONPATH`` pointed at either the frozen source archive or the
working tree.  The script deliberately uses the public ``DRPModel`` lifecycle;
the private pipeline call only exposes the featurized entity IDs and projection
that the public call produced, so their identity and ordering can be audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from typing import Any

import numpy as np
from upath import UPath

ROOT = UPath(os.environ.get("DREVALPY_FACADE_ROOT", UPath(__file__).parents[2]))
# The current tree is a src layout; older archives keep the package at the root.
sys.path.insert(0, str(ROOT / "src" if (ROOT / "src").exists() else ROOT))

try:
    from drevalpy.models import DRPModel, ModelSpec

    MODERN_FACADE = True
except ImportError:  # v1 archive exposed generated DRPModel subclasses instead.
    from drevalpy.models import DRPModel, construct_model, load_model

    ModelSpec = Any
    MODERN_FACADE = False
from drevalpy.models.config import from_spec  # noqa: E402
from drevalpy.utils.seed import seed_everything  # noqa: E402
from tests.synthetic.builders import build_synthetic_dataset  # noqa: E402

MODELS = ("DrugGNN", "DIPK", "MOLIR", "SuperFELTR", "PharmaFormer", "Precily", "SRMF", "SparseGO")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _spec(name: str) -> Any:
    """Resolve a public model name, including the single-drug predictor recipes."""
    if name == "MOLIR":
        if not MODERN_FACADE:
            return construct_model("MOLIR", from_spec("molirOmics:molir"))
        return ModelSpec("MOLIR", from_spec("molirOmics:molir"))
    if name == "SuperFELTR":
        if not MODERN_FACADE:
            return construct_model("SuperFELTR", from_spec("superfeltrOmics:superfeltr"))
        return ModelSpec("SuperFELTR", from_spec("superfeltrOmics:superfeltr"))
    if not MODERN_FACADE:
        return construct_model(name)
    return ModelSpec.from_name(name)


def _hyperparameters(name: str, spec: Any) -> dict[str, Any]:
    """Keep all fits real while making the CPU evidence practical."""
    hp = dict(spec.default_hyperparameters if MODERN_FACADE else spec.get_default_hyperparameters())
    if name == "DIPK":
        hp.update(epochs=1, epochs_autoencoder=1, batch_size=8, heads=1)
    elif name in {"MOLIR", "SuperFELTR"}:
        hp["epochs"] = 1
    elif name == "PharmaFormer":
        hp.update(epochs=1, patience=2)
    elif name in {"DrugGNN", "Precily", "SparseGO"}:
        hp.update(epochs=1, batch_size=32)
    elif name == "SRMF":
        hp["max_iter"] = 2
    return hp


def _split(dataset: Any) -> Any:
    from drevalpy.registry.splitter import get as get_splitter

    return get_splitter("LTO")(dataset, n_splits=2, validation_ratio=0.4)[0]


def _result_identity(result: Any, *, entity_ids: Any = None, matrix: Any = None) -> dict[str, Any] | None:
    """Record ordering and numeric bytes for one featurizer result representation."""
    if result is None:
        return None
    entity_ids = result.entity_ids if entity_ids is None else entity_ids
    matrix = result.matrix if matrix is None else matrix
    blocks = {}
    for name, block in result.blocks.items():
        values = np.asarray(block.values)
        blocks[name] = {
            "format": str(block.format),
            "feature_names": list(block.feature_names) if block.feature_names is not None else None,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "numeric_sha256": _digest(values.tobytes()) if values.dtype.kind in "biufc" else None,
        }
    matrix = np.asarray(matrix)
    return {
        "entity_ids": np.asarray(entity_ids).tolist(),
        "projection": list(getattr(result, "projection", ())),
        "blocks": blocks,
        "matrix_shape": list(matrix.shape),
        "matrix_numeric_sha256": _digest(matrix.tobytes()) if matrix.dtype.kind in "biufc" else None,
    }


def _feature_evidence(model: DRPModel, dataset: Any, scope: Any) -> dict[str, Any]:
    """Build a prediction batch again without fitting, retaining canonical identities."""
    if not MODERN_FACADE:
        stack = model._stack
        raw = stack._extract_response_pairs(dataset, scope)
        cell_ids = np.unique(raw.cell_line_ids)
        drug_ids = np.unique(raw.drug_ids)
        cell_source, drug_source = stack._build_features_from_mudataset(dataset, cell_ids, drug_ids)
        batch = stack._batch(raw, cell_source, drug_source)
        return {
            "cell_line": _result_identity(
                type("LegacyResult", (), {"blocks": batch.cell_line_blocks})(),
                entity_ids=batch.cell_line_entity_ids,
                matrix=batch.cell_line_features,
            ),
            "drug": _result_identity(
                type("LegacyResult", (), {"blocks": batch.drug_blocks})(),
                entity_ids=batch.drug_entity_ids,
                matrix=batch.drug_features,
            )
            if batch.drug_blocks is not None
            else None,
        }
    pipeline = model._pipeline
    raw = pipeline._extract_response_pairs(dataset, scope)
    cell_ids = np.unique(raw.cell_line_ids)
    drug_ids = np.unique(raw.drug_ids)
    cell_source, drug_source = pipeline._build_features_from_mudataset(dataset, cell_ids, drug_ids)
    batch = pipeline._batch(raw, cell_source, drug_source)
    return {"cell_line": _result_identity(batch.cell_line), "drug": _result_identity(batch.drug)}


def _one(name: str, seed: int) -> dict[str, Any]:
    import torch

    _seed(seed)
    dataset = build_synthetic_dataset(seed=seed)
    split = _split(dataset)
    spec: Any = _spec(name)
    hp = _hyperparameters(name, spec)
    before_init = _digest(bytes(torch.get_rng_state().tolist()))
    model = DRPModel(spec, hp) if MODERN_FACADE else spec(hp)
    after_init = _digest(bytes(torch.get_rng_state().tolist()))
    supports_early_stopping = spec.supports_early_stopping if MODERN_FACADE else spec.supports_early_stopping()
    stopping = split.early_stopping_mask()[0] if supports_early_stopping else None
    with tempfile.TemporaryDirectory() as directory:
        model.train(dataset, split.train, early_stopping_scope=stopping, model_checkpoint_dir=directory)
        predictions = np.asarray(model.predict(dataset, split.test), dtype=np.float64)
        features = _feature_evidence(model, dataset, split.test)
        checkpoint = f"{directory}/model"
        model.save(checkpoint)
        restored = DRPModel.load(checkpoint) if MODERN_FACADE else load_model(checkpoint)
        reloaded = np.asarray(restored.predict(dataset, split.test), dtype=np.float64)
    return {
        "status": "passed",
        "seed": seed,
        "hyperparameters": hp,
        "initialization_rng": {"before": before_init, "after": after_init, "consumed": before_init != after_init},
        "feature_results": features,
        "predictions": predictions.tolist(),
        "prediction_sha256": _digest(predictions.tobytes()),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "save_load_max_error": float(np.max(np.abs(predictions - reloaded), initial=0.0)),
    }


def main() -> None:
    """Run selected facade fits and write structured evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    output: dict[str, Any] = {"models": {}, "seeds": args.seeds, "dataset": "frozen synthetic raw dataset"}
    for name in args.models:
        output["models"][name] = {}
        for seed in args.seeds:
            try:
                output["models"][name][str(seed)] = _one(name, seed)
            except Exception as exc:  # preserve an exact facade blocker without losing other models
                output["models"][name][str(seed)] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    with open(args.output, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
