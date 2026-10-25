"""On-disk cache of trained gene-expression autoencoders (opt-in).

Models that share a training fold, a gene list and the autoencoder settings train the same encoder.
With a cache directory they train it once and the others read it back. The directory is the
``autoencoder_cache_dir`` option of ``aeGeneExpression`` or, when that is unset, the environment
variable :data:`CACHE_ENV_VAR`.

The file name is a digest of everything that defines the encoder: the autoencoder settings, the
training and early-stopping sample ids, the gene *order* and the expression values. The order is
part of the key because the same genes in another column order give another encoder, and feeding a
cached encoder permuted genes would not raise but return wrong numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

#: Environment variable naming the cache directory when the featurizer is given none.
CACHE_ENV_VAR = "DREVAL_AE_CACHE"


def resolve_cache_dir(explicit: str | None) -> Path | None:
    """Pick the cache directory: the explicit option, else the environment variable.

    :param explicit: Value of the ``autoencoder_cache_dir`` option.
    :returns: The directory, or ``None`` when caching is not configured.
    """
    chosen = explicit or os.environ.get(CACHE_ENV_VAR)
    return Path(chosen).expanduser() if chosen else None


def cache_file(
    directory: Path,
    config: Mapping[str, object],
    genes: Sequence[str],
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    matrices: Sequence[np.ndarray],
) -> Path:
    """Return the cache file of one encoder.

    :param directory: Cache directory.
    :param config: Autoencoder settings that change the trained weights.
    :param genes: Gene symbols in the column order of the matrices.
    :param train_ids: Sample ids of the training matrix, in row order.
    :param validation_ids: Sample ids of the early-stopping matrix, in row order.
    :param matrices: Training and early-stopping matrices as the autoencoder sees them.
    :returns: Path ``gene_ae_<digest>.pt`` inside *directory*; it need not exist.
    """
    data = hashlib.sha256()
    for matrix in matrices:
        contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
        data.update(f"{contiguous.shape}".encode())
        data.update(contiguous.tobytes())
    payload = json.dumps(
        {
            "config": dict(config),
            "input_dim": len(genes),
            "samples": [str(i) for i in train_ids],
            "validation_samples": [str(i) for i in validation_ids],
            "gene_order": hashlib.sha256("\n".join(map(str, genes)).encode()).hexdigest(),
            "data": data.hexdigest(),
        },
        sort_keys=True,
    )
    return directory / f"gene_ae_{hashlib.sha256(payload.encode()).hexdigest()[:16]}.pt"


def write_atomically(path: Path, payload: bytes) -> None:
    """Write *payload* so that a reader never sees a half-written file.

    :param path: Target file; its directory is created.
    :param payload: Bytes to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
