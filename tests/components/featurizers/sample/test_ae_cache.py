"""Tests for the autoencoder cache key and file helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample._ae_cache import (
    CACHE_ENV_VAR,
    cache_file,
    resolve_cache_dir,
    write_atomically,
)

_CONFIG = {"gene_list": "mine", "epochs": 5, "batch_size": 8, "patience": 2, "lr": 1e-4}
_TRAIN = np.arange(12, dtype=np.float32).reshape(3, 4)
_VALIDATION = np.arange(8, dtype=np.float32).reshape(2, 4)


def _file(tmp_path: Path, **overrides) -> Path:
    arguments = {
        "config": _CONFIG,
        "genes": ["a", "b", "c", "d"],
        "train_ids": np.array(["s0", "s1", "s2"]),
        "validation_ids": np.array(["s3", "s4"]),
        "matrices": (_TRAIN, _VALIDATION),
    }
    arguments.update(overrides)
    return cache_file(tmp_path, **arguments)


def test_cache_file_is_deterministic_and_named_by_digest(tmp_path: Path) -> None:
    path = _file(tmp_path)

    assert path == _file(tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("gene_ae_") and path.suffix == ".pt"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"config": {**_CONFIG, "epochs": 6}}, id="config"),
        pytest.param({"genes": ["a", "b", "d", "c"]}, id="gene-order"),
        pytest.param({"train_ids": np.array(["s0", "s1", "s9"])}, id="train-samples"),
        pytest.param({"validation_ids": np.array(["s3", "s9"])}, id="validation-samples"),
        pytest.param({"matrices": (_TRAIN + 1, _VALIDATION)}, id="train-values"),
        pytest.param({"matrices": (_TRAIN, _VALIDATION[:1])}, id="validation-rows"),
    ],
)
def test_every_defining_input_changes_the_file(tmp_path: Path, change: dict) -> None:
    assert _file(tmp_path, **change) != _file(tmp_path)


def test_resolve_cache_dir_prefers_the_option_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))

    assert resolve_cache_dir(str(tmp_path / "option")) == tmp_path / "option"
    assert resolve_cache_dir(None) == tmp_path / "env"


def test_resolve_cache_dir_is_none_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    assert resolve_cache_dir(None) is None
    assert resolve_cache_dir("") is None


def test_write_atomically_creates_the_directory_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "gene_ae_x.pt"

    write_atomically(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert [p.name for p in target.parent.iterdir()] == ["gene_ae_x.pt"]
