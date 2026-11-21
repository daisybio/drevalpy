"""Tests for the autoencoder featurizers (AE and DIPK gene expression) and the AE cache helpers."""

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
from drevalpy.components.featurizers.sample.ae_gene_expression import (
    AEGeneExpressionFeaturizer,
    train_or_load_autoencoder,
)
from drevalpy.components.featurizers.sample.dipk_gene_expression import (
    DIPKGeneExpressionFeaturizer,
    dipk_preprocess,
)
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
)
from drevalpy.errors import InvalidInputError
from drevalpy.types.data.feature_source import FeatureSource
from tests.components.featurizers.sample._helpers import pickled, use_gene_lists
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_model_dataset

_ENCODER_MODULE = "drevalpy.components.predictors.literature.dipk.gene_expression_encoder"
_GENES = ["a", "b", "c", "d"]


class _UnitSource(MockFeatureSource):
    def __init__(self, unit: str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._unit = unit

    def expression_unit(self, name: str) -> str | None:
        return self._unit if name == "gene_expression" else None


def _source(n: int = 6, unit: str | None = None) -> MockFeatureSource:
    rng = np.random.default_rng(0)
    return _UnitSource(
        unit,
        features={f"cl{i}": {"gene_expression": rng.uniform(1.0, 9.0, size=len(_GENES))} for i in range(n)},
        meta_info={"gene_expression": _GENES},
    )


def _ae(**kwargs) -> AEGeneExpressionFeaturizer:
    return AEGeneExpressionFeaturizer(gene_list=None, **kwargs)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([(row - np.mean(row)) / np.std(row, ddof=1) for row in matrix])


def _fit(featurizer, source: MockFeatureSource, train=("cl0", "cl1"), es=("cl2",)):
    return featurizer.fit(source, np.array(train), es_pair_ids=np.array(es))


@pytest.fixture
def recording_trainer(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the autoencoder trainer, which both featurizers import inside their methods, by a recorder."""
    seen: dict = {"calls": 0}

    def _train(train, validation, epochs, **kwargs):
        seen.update(train=train, validation=validation, epochs=epochs, **kwargs)
        seen["calls"] += 1
        return GeneExpressionEncoder(train.shape[1])

    monkeypatch.setattr(f"{_ENCODER_MODULE}.train_gene_expession_autoencoder", _train)
    return seen


@pytest.fixture(autouse=True)
def _no_ambient_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)


_VARIANTS = [
    pytest.param(lambda **kwargs: _ae(**kwargs), lambda matrix: matrix, id="ae"),
    pytest.param(lambda **kwargs: DIPKGeneExpressionFeaturizer(**kwargs), _zscore, id="dipk"),
]


def test_defaults_and_hyperparameter_space() -> None:
    assert AEGeneExpressionFeaturizer.default_hyperparameters() == {
        "epochs_autoencoder": 2000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 50,
        "batch_size_autoencoder": 1024,
    }
    assert AEGeneExpressionFeaturizer()._gene_list == "tcga_autoencoder_genes_available"
    assert DIPKGeneExpressionFeaturizer.default_hyperparameters() == {
        "epochs_autoencoder": 1000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 3,
        "batch_size_autoencoder": 1024,
    }
    assert DIPKGeneExpressionFeaturizer.get_hyperparameter_space()["epochs_autoencoder"]["high"] >= 2000
    assert AEGeneExpressionFeaturizer.requires_early_stopping
    assert DIPKGeneExpressionFeaturizer.requires_early_stopping


@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_fits_on_unique_training_samples_and_forwards_the_settings(recording_trainer, make, preprocess) -> None:
    source = _source(n=4)

    make(epochs_autoencoder=7, lr_autoencoder=5e-3, patience_autoencoder=11, batch_size_autoencoder=2).fit(
        source,
        np.array(["cl1", "cl0"]),
        train_pair_ids=np.array(["cl1", "cl0", "cl1", "cl1", "cl0"]),
        es_pair_ids=np.array(["cl3", "cl3", "cl2"]),
    )

    seen = recording_trainer
    assert (seen["epochs"], seen["lr"], seen["patience"], seen["batch_size"]) == (7, 5e-3, 11, 2)
    np.testing.assert_allclose(seen["train"], preprocess(source.view("gene_expression", np.array(["cl0", "cl1"]))))
    np.testing.assert_allclose(seen["validation"], preprocess(source.view("gene_expression", np.array(["cl2", "cl3"]))))


def test_dipk_zscores_every_sample_row_and_ae_leaves_the_rows_raw(recording_trainer) -> None:
    source = _source(n=4)

    _fit(DIPKGeneExpressionFeaturizer(), source)
    np.testing.assert_allclose(recording_trainer["train"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(recording_trainer["train"].std(axis=1, ddof=1), 1.0)

    _fit(_ae(), source)
    assert not np.allclose(recording_trainer["train"].mean(axis=1), 0.0)


@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_a_trailing_single_row_moves_to_validation(recording_trainer, make, preprocess) -> None:
    _fit(make(batch_size_autoencoder=4), _source(n=6), train=("cl0", "cl1", "cl2", "cl3", "cl4"), es=("cl5",))

    assert (recording_trainer["train"].shape[0], recording_trainer["validation"].shape[0]) == (4, 2)


@pytest.mark.parametrize(("unit", "log_applied"), [("log2_tpm_plus_1", False), (None, False), ("TPM", True)])
@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_log2_applies_only_to_linear_units(recording_trainer, make, preprocess, unit, log_applied) -> None:
    source = _source(unit=unit)

    _fit(make(), source)

    raw = source.view("gene_expression", np.array(["cl0", "cl1"]))
    np.testing.assert_allclose(recording_trainer["train"], preprocess(np.log2(raw + 1) if log_applied else raw))


def test_dipk_reads_the_expression_unit_from_a_real_dataset(recording_trainer) -> None:
    dataset = synthetic_model_dataset(n_features_per_view=4)
    dataset.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"
    ids = np.array(["cl1", "cl2"])
    source = FeatureSource(dataset, "sample")

    DIPKGeneExpressionFeaturizer().fit(source, ids, es_pair_ids=ids)

    raw = source.view("gene_expression", ids)
    np.testing.assert_allclose(recording_trainer["train"], _zscore(np.log2(raw + 1)), rtol=1e-5)


@pytest.mark.parametrize(("unit", "log_applied"), [("tpm", True), (None, False)])
def test_dipk_preprocess_logs_linear_units_and_zscores_each_row(unit, log_applied) -> None:
    source = _source(unit=unit)
    raw = source.view("gene_expression", np.array(["cl0", "cl1", "cl2"]))

    actual = dipk_preprocess(source, raw)

    np.testing.assert_allclose(actual, _zscore(np.log2(raw + 1) if log_applied else raw))
    np.testing.assert_allclose(dipk_preprocess(source, raw, view="absent"), _zscore(raw))


def test_ae_transform_encodes_with_the_fitted_encoder_and_survives_pickling(recording_trainer) -> None:
    source = _source(unit="tpm")
    featurizer = _fit(_ae(epochs_autoencoder=3), source)
    ids = np.array(["cl3", "cl4"])

    result = featurizer.transform(source, ids)

    expected = encode_gene_expression(np.log2(source.view("gene_expression", ids) + 1), featurizer._encoder)
    assert result.matrix.shape == (2, 512)
    assert set(result.blocks) == {"gene_expression"}
    np.testing.assert_allclose(result.matrix, expected, rtol=1e-5, atol=1e-6)
    assert featurizer.transform(source, np.array([], dtype=str)).matrix.shape == (0, 512)
    restored = pickled(featurizer)
    assert (restored._gene_list, restored._epochs) == (None, 3)
    np.testing.assert_allclose(restored.transform(source, ids).matrix, result.matrix)


def test_dipk_transform_encodes_the_preprocessed_matrix_and_survives_pickling(recording_trainer) -> None:
    source = _source(n=3, unit="tpm")
    featurizer = _fit(DIPKGeneExpressionFeaturizer(epochs_autoencoder=1, patience_autoencoder=7), source)
    ids = np.array(["cl2"])

    matrix = featurizer.transform(source, ids).matrix

    expected = encode_gene_expression(_zscore(np.log2(source.view("gene_expression", ids) + 1)), featurizer._encoder)
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    assert featurizer.transform(source, np.array([], dtype=str)).matrix.shape == (0, 512)
    restored = pickled(featurizer)
    assert (restored._epochs, restored._patience) == (1, 7)
    np.testing.assert_allclose(restored.transform(source, ids).matrix, matrix)


@pytest.mark.parametrize("make", [lambda: _ae(), DIPKGeneExpressionFeaturizer], ids=["ae", "dipk"])
def test_unfitted_transform_and_missing_holdout_raise(make) -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        make().transform(_source(), np.array(["cl0"]))
    with pytest.raises(InvalidInputError, match="requires a usable early-stopping holdout"):
        make().fit(_source(), np.array(["cl0"]))
    with pytest.raises(InvalidInputError, match="requires a usable early-stopping holdout"):
        make().fit(_source(), np.array(["cl0"]), es_pair_ids=np.array([], dtype=str))
    with pytest.raises(InvalidInputError, match="requires a usable early-stopping holdout"):
        make().fit(_source(), np.array([], dtype=str), es_pair_ids=np.array(["cl2"]))


def test_ae_gene_list_selects_genes_in_list_order_and_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_trainer
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"], "bad": ["a", "absent"]})
    source = _source()

    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list="mine"), source)

    expected = source.view("gene_expression", np.array(["cl0", "cl1"]))[:, [2, 0]]
    np.testing.assert_allclose(recording_trainer["train"], expected)
    assert featurizer._names == ("c", "a")
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'bad'"):
        _fit(AEGeneExpressionFeaturizer(gene_list="bad"), source)


def test_dipk_gene_list_selects_the_autoencoder_input_and_survives_pickling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_trainer
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"]})
    source = _source(n=4)

    featurizer = _fit(DIPKGeneExpressionFeaturizer(gene_list="mine", epochs_autoencoder=1), source, es=("cl2", "cl3"))

    assert recording_trainer["train"].shape == (2, 2)
    restored = pickled(featurizer)
    assert restored._gene_list == "mine"
    ids = np.array(["cl0", "cl1"])
    np.testing.assert_allclose(restored.transform(source, ids).matrix, featurizer.transform(source, ids).matrix)


def test_train_or_load_autoencoder_trains_and_caches_by_gene_names(recording_trainer, tmp_path: Path) -> None:
    train = np.arange(12, dtype=np.float32).reshape(3, 4)
    validation = np.arange(8, dtype=np.float32).reshape(2, 4)
    train_ids, validation_ids = np.array(["s0", "s1", "s2"]), np.array(["s3", "s4"])
    settings = {"epochs": 4, "lr": 1e-3, "patience": 2, "batch_size": 8}

    def run(**kwargs) -> GeneExpressionEncoder:
        return train_or_load_autoencoder(train, validation, train_ids, validation_ids, **settings, **kwargs)

    encoder = run()
    assert isinstance(encoder, GeneExpressionEncoder)
    assert (recording_trainer["epochs"], recording_trainer["lr"], recording_trainer["patience"]) == (4, 1e-3, 2)
    np.testing.assert_allclose(recording_trainer["train"], train)
    run()
    assert recording_trainer["calls"] == 2

    run(cache_dir=str(tmp_path), genes=tuple(_GENES))
    run(cache_dir=str(tmp_path), genes=tuple(_GENES))
    assert recording_trainer["calls"] == 3
    run(cache_dir=str(tmp_path), genes=tuple(_GENES[::-1]))
    run(cache_dir=str(tmp_path))
    run(cache_dir=str(tmp_path), gene_list="mine", genes=tuple(_GENES))
    assert recording_trainer["calls"] == 6
    assert len(list(tmp_path.glob("gene_ae_*.pt"))) == 4


def test_cache_trains_once_and_reuses_the_encoder(recording_trainer, tmp_path: Path) -> None:
    source = _source(n=8)
    cache = tmp_path / "ae"
    ids = np.array(["cl4", "cl5"])

    first = _fit(_ae(autoencoder_cache_dir=str(cache)), source)
    second = _fit(_ae(autoencoder_cache_dir=str(cache)), source)

    assert recording_trainer["calls"] == 1
    assert len(list(cache.glob("gene_ae_*.pt"))) == 1
    np.testing.assert_allclose(second.transform(source, ids).matrix, first.transform(source, ids).matrix)


def test_cache_directory_comes_from_the_environment(
    recording_trainer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))

    _fit(_ae(), _source())
    _fit(_ae(), _source())

    assert recording_trainer["calls"] == 1
    assert len(list((tmp_path / "env").glob("gene_ae_*.pt"))) == 1


def test_cache_is_keyed_by_settings_samples_and_gene_order(recording_trainer, tmp_path: Path) -> None:
    source = _source()
    reverse = MockFeatureSource(
        {key: {"gene_expression": value["gene_expression"][::-1]} for key, value in source.features.items()},
        meta_info={"gene_expression": _GENES[::-1]},
    )

    def fit(src=source, train=("cl0", "cl1"), es=("cl2",), **kwargs) -> None:
        _fit(_ae(autoencoder_cache_dir=str(tmp_path), **kwargs), src, train, es)

    variations = [
        {},
        {"epochs_autoencoder": 3},
        {"patience_autoencoder": 3},
        {"lr_autoencoder": 1e-3},
        {"train": ("cl0", "cl1", "cl3")},
        {"es": ("cl4",)},
        {"src": reverse},
    ]
    for variation in variations:
        fit(**variation)
    assert recording_trainer["calls"] == len(variations)
    fit()
    assert recording_trainer["calls"] == len(variations)


_CONFIG = {"gene_list": "mine", "epochs": 5, "batch_size": 8, "patience": 2, "lr": 1e-4}
_TRAIN = np.arange(12, dtype=np.float32).reshape(3, 4)
_VALIDATION = np.arange(8, dtype=np.float32).reshape(2, 4)


def _cache_path(tmp_path: Path, **overrides) -> Path:
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
    path = _cache_path(tmp_path)

    assert path == _cache_path(tmp_path)
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
def test_every_defining_input_changes_the_cache_file(tmp_path: Path, change: dict) -> None:
    assert _cache_path(tmp_path, **change) != _cache_path(tmp_path)


def test_resolve_cache_dir_prefers_the_option_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))

    assert resolve_cache_dir(str(tmp_path / "option")) == tmp_path / "option"
    assert resolve_cache_dir(None) == tmp_path / "env"
    monkeypatch.delenv(CACHE_ENV_VAR)
    assert resolve_cache_dir(None) is None
    assert resolve_cache_dir("") is None


def test_write_atomically_creates_the_directory_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "gene_ae_x.pt"

    write_atomically(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert [p.name for p in target.parent.iterdir()] == ["gene_ae_x.pt"]
