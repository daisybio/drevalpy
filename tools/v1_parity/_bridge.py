"""Shared case format, seeding and synthetic inputs for the 1.6.1 parity harness.

This module is imported by *both* sides, in two different interpreters (the v2 venv and the
1.6.1 venv), so it may only import the standard library and numpy at module scope.

A *case* is the complete, seeded input of one comparison: named numpy arrays plus a
JSON-serializable metadata mapping. It is written to a directory as ``case.npz`` (arrays,
never pickled) and ``case.json`` (metadata). The two runners load the same directory,
build their implementation's native objects from it and write an *outputs* file with the
same layout (``outputs_<side>.npz`` and ``outputs_<side>.json``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import random
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HARNESS_DIR = Path(__file__).resolve().parent
MODELS_DIR = HARNESS_DIR / "models"
EVIDENCE_DIR = HARNESS_DIR / "evidence"
REPO_ROOT = HARNESS_DIR.parents[1]
V1_COMMIT = "f5ff3c70"
DEFAULT_V1_DIR = Path("/tmp/opencode/v161")  # noqa: S108 - the documented worktree location
CASE_ARRAYS = "case.npz"
CASE_META = "case.json"
MODELS_PACKAGE = "tools.v1_parity.models"


@dataclass
class Case:
    """Seeded input of one comparison, shared byte-for-byte by both implementations.

    :param model: Harness model name, for example ``"ElasticNet"``.
    :param seed: Seed every random draw of both sides derives from.
    :param arrays: Named numpy arrays. Strings are stored as unicode, never as ``object``.
    :param meta: JSON-serializable settings such as hyperparameters and variants.
    """

    model: str
    seed: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        """Write ``case.npz`` and ``case.json`` into *directory*.

        :param directory: Target directory, created when missing.
        :returns: The directory.
        :raises ValueError: If an array has ``object`` dtype (it would need pickling).
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name, array in self.arrays.items():
            if np.asarray(array).dtype == object:
                msg = f"case array {name!r} has object dtype; store strings as unicode arrays"
                raise ValueError(msg)
        np.savez(target / CASE_ARRAYS, **{name: np.asarray(a) for name, a in self.arrays.items()})
        header = {"model": self.model, "seed": self.seed, "meta": self.meta}
        (target / CASE_META).write_text(json.dumps(header, indent=2, sort_keys=True))
        return target

    @classmethod
    def load(cls, directory: str | Path) -> Case:
        """Read a case written by :meth:`save`.

        :param directory: Directory holding ``case.npz`` and ``case.json``.
        :returns: The restored case.
        """
        source = Path(directory)
        header = json.loads((source / CASE_META).read_text())
        with np.load(source / CASE_ARRAYS, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return cls(model=header["model"], seed=int(header["seed"]), arrays=arrays, meta=header["meta"])

    def input_hash(self) -> str:
        """Return a sha256 over the seed, metadata and every array (dtype, shape and bytes).

        :returns: Hex digest identifying the exact inputs of this case.
        """
        digest = hashlib.sha256()
        digest.update(json.dumps({"model": self.model, "seed": self.seed, "meta": self.meta}, sort_keys=True).encode())
        for name in sorted(self.arrays):
            array = np.ascontiguousarray(self.arrays[name])
            digest.update(f"{name}|{array.dtype.str}|{array.shape}".encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def save_outputs(directory: str | Path, side: str, arrays: dict[str, np.ndarray], info: dict[str, Any]) -> Path:
    """Write one side's outputs as ``outputs_<side>.npz`` and ``outputs_<side>.json``.

    :param directory: Output directory, created when missing.
    :param side: ``"v1"`` or ``"v2"``.
    :param arrays: Named output arrays, keyed by the step names the model module declares.
    :param info: Environment description from :func:`environment_info` plus run details.
    :returns: Path of the ``.npz`` file.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    npz = target / f"outputs_{side}.npz"
    np.savez(npz, **{name: np.asarray(a) for name, a in arrays.items()})
    (target / f"outputs_{side}.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    return npz


def load_outputs(directory: str | Path, side: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the outputs of one side.

    :param directory: Directory holding the outputs files.
    :param side: ``"v1"`` or ``"v2"``.
    :returns: The output arrays and the info mapping.
    """
    source = Path(directory)
    with np.load(source / f"outputs_{side}.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    info = json.loads((source / f"outputs_{side}.json").read_text())
    return arrays, info


def environment_info() -> dict[str, Any]:
    """Describe the running interpreter and the versions that matter for numerics.

    :returns: Python, numpy, torch, scikit-learn, xgboost, lightgbm and drevalpy versions (``None`` when absent).
    """
    from importlib import metadata

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "scikit-learn": version("scikit-learn"),
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "xgboost": version("xgboost"),
        "lightgbm": version("lightgbm"),
        "drevalpy": version("drevalpy"),
        "executable": sys.executable,
    }


def seed_all(seed: int) -> None:
    """Seed ``random``, numpy and (when installed) torch deterministically on CPU.

    :param seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - 1.6.1 draws from the legacy global state
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def tabular_case_arrays(
    seed: int,
    *,
    n_samples: int = 30,
    n_drugs: int = 6,
    n_genes: int = 40,
    n_bits: int = 32,
    test_only_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Draw the standard sample x drug regression inputs most models share.

    Expression and fingerprint values are rounded to float32 so that the float32 world of v2
    and the float64 world of 1.6.1 start from the identical numbers. The last
    ``test_only_fraction`` of the samples never occur in training, so per-sample scalers are
    exercised on unseen samples.

    :param seed: Seed of the generator.
    :param n_samples: Number of samples (cell lines).
    :param n_drugs: Number of drugs.
    :param n_genes: Width of the gene-expression matrix.
    :param n_bits: Width of the binary drug fingerprint.
    :param test_only_fraction: Fraction of samples held out of training entirely.
    :returns: ``sample_ids``, ``drug_ids``, ``gene_names``, ``tissues``, ``expression``
        (raw, positive, TPM-like), ``fingerprints``, ``response`` (float32 sample x drug,
        NaN where unobserved), ``train_pairs`` and ``test_pairs`` (row-major sorted
        ``(n, 2)`` index arrays into the response matrix).
    """
    rng = np.random.default_rng(seed)
    sample_ids = np.array([f"S{i:03d}" for i in range(n_samples)])
    drug_ids = np.array([f"D{i:02d}" for i in range(n_drugs)])
    gene_names = np.array([f"GENE{i:03d}" for i in range(n_genes)])
    tissues = np.array(["Lung", "Breast", "Blood"])[np.arange(n_samples) % 3]
    expression = rng.gamma(shape=2.0, scale=30.0, size=(n_samples, n_genes)).astype(np.float32)
    fingerprints = rng.integers(0, 2, size=(n_drugs, n_bits)).astype(np.float32)
    weights = rng.normal(size=n_genes) / np.sqrt(n_genes)
    drug_effect = rng.normal(size=n_drugs)
    signal = np.arcsinh(expression) @ weights
    response = (signal[:, None] * (1.0 + 0.3 * drug_effect[None, :]) + drug_effect[None, :]).astype(np.float64)
    response += rng.normal(scale=0.1, size=response.shape)
    observed = rng.random(response.shape) < 0.85
    response32 = np.where(observed, response, np.nan).astype(np.float32)

    n_test_only = max(1, round(n_samples * test_only_fraction))
    train_rows = np.arange(n_samples - n_test_only)
    held_out = rng.random(response.shape) < 0.2
    pairs = np.argwhere(observed)
    is_train_row = np.isin(pairs[:, 0], train_rows)
    is_held = held_out[pairs[:, 0], pairs[:, 1]]
    train_pairs = pairs[is_train_row & ~is_held]
    test_pairs = pairs[~(is_train_row & ~is_held)]
    return {
        "sample_ids": sample_ids,
        "drug_ids": drug_ids,
        "gene_names": gene_names,
        "tissues": tissues,
        "expression": expression,
        "fingerprints": fingerprints,
        "response": response32,
        "train_pairs": train_pairs.astype(np.int64),
        "test_pairs": test_pairs.astype(np.int64),
    }


@contextmanager
def record_calls(
    owner: Any,
    method: str,
    sink: dict[str, Any],
    key: str,
    after: Callable[[Any, dict[str, Any]], None] | None = None,
) -> Iterator[None]:
    """Patch ``owner.method`` to store a copy of its first data argument in ``sink[key]``.

    Used to capture what reaches an estimator without touching either implementation. The
    patch is undone on exit. A call repeated several times keeps only the last argument.
    When *owner* is a class the first data argument is the one after ``self``.

    :param owner: Class or instance whose method is wrapped.
    :param method: Method name.
    :param sink: Dictionary receiving the captured copy.
    :param key: Key in *sink*.
    :param after: Optional hook called as ``after(instance, sink)`` once the call returned.
    :yields: Nothing; the patch is active inside the ``with`` block.
    """
    original = getattr(owner, method)
    is_class = isinstance(owner, type)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sink[key] = np.array(args[1] if is_class else args[0], copy=True)
        result = original(*args, **kwargs)
        if after is not None:
            after(args[0] if is_class else owner, sink)
        return result

    setattr(owner, method, wrapper)
    try:
        yield
    finally:
        if is_class:
            setattr(owner, method, original)
        else:
            delattr(owner, method)


def model_stems() -> list[str]:
    """List the model case modules in ``models/`` (files not starting with ``_``).

    :returns: Sorted module stems.
    """
    return sorted(p.stem for p in MODELS_DIR.glob("*.py") if not p.name.startswith("_"))


def load_model_module(name: str) -> Any:
    """Import a model case module by stem or by its ``NAME``.

    :param name: Module stem (``elastic_net``) or harness name (``ElasticNet``).
    :returns: The imported module.
    :raises KeyError: If no case module matches.
    """
    if name in model_stems():
        return importlib.import_module(f"{MODELS_PACKAGE}.{name}")
    for stem in model_stems():
        module = importlib.import_module(f"{MODELS_PACKAGE}.{stem}")
        if getattr(module, "NAME", None) == name:
            return module
    msg = f"no parity case for {name!r}; available: {', '.join(model_stems()) or '(none)'}"
    raise KeyError(msg)
