"""Tests for the autoencoder featurizers (AE and DIPK gene expression) and the AE cache helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drevalpy.components.featurizers.sample._ae_cache import (
    CACHE_ENV_VAR,
    cache_file,
    resolve_cache_dir,
    write_atomically,
)
from drevalpy.components.featurizers.sample.ae_gene_expression import AEGeneExpressionFeaturizer
from drevalpy.components.featurizers.sample.dipk_gene_expression import DIPKGeneExpressionFeaturizer
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
)
from drevalpy.errors import InvalidInputError
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.components.featurizers.sample._helpers import use_gene_lists
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_model_dataset

_ENCODER_MODULE = "drevalpy.components.predictors.literature.dipk.gene_expression_encoder"
_GENES = ["a", "b", "c", "d"]


class _UnitSource(MockFeatureSource):
    def __init__(self, unit: str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._unit = unit

    @property
    def mdata(self) -> SimpleNamespace:
        uns = {} if self._unit is None else {"expression_unit": self._unit}
        return SimpleNamespace(uns={}, mod={"gene_expression": SimpleNamespace(uns=uns)})


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
    return featurizer.fit(source, pair_expanded_ids=np.array(train), pair_expanded_es_ids=np.array(es))


@pytest.fixture
def recording_trainer(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the autoencoder trainer, which both featurizers import inside ``_fit_state``, by a recorder."""
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
    assert AEGeneExpressionFeaturizer.get_default_hyperparameters() == {
        "epochs_autoencoder": 2000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 50,
        "batch_size_autoencoder": 1024,
    }
    assert AEGeneExpressionFeaturizer()._gene_list == "tcga_autoencoder_genes_available"
    assert DIPKGeneExpressionFeaturizer.get_default_hyperparameters() == {
        "epochs_autoencoder": 1000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 3,
        "batch_size_autoencoder": 1024,
    }
    assert DIPKGeneExpressionFeaturizer.get_hyperparameter_space()["epochs_autoencoder"]["high"] >= 2000


@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_fits_on_unique_training_samples_and_forwards_the_settings(recording_trainer, make, preprocess) -> None:
    source = _source(n=4)

    make(epochs_autoencoder=7, lr_autoencoder=5e-3, patience_autoencoder=11, batch_size_autoencoder=2).fit(
        source,
        pair_expanded_ids=np.array(["cl1", "cl0", "cl1", "cl1", "cl0"]),
        pair_expanded_es_ids=np.array(["cl3", "cl3", "cl2"]),
    )

    seen = recording_trainer
    assert (seen["epochs"], seen["lr"], seen["patience"], seen["batch_size"]) == (7, 5e-3, 11, 2)
    np.testing.assert_allclose(
        seen["train"], preprocess(source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"])))
    )
    np.testing.assert_allclose(
        seen["validation"], preprocess(source.get_view_matrix("gene_expression", np.array(["cl2", "cl3"])))
    )


def test_dipk_zscores_every_sample_row_and_ae_leaves_the_rows_raw(recording_trainer) -> None:
    source = _source(n=4)
    ids = np.array(["cl0", "cl1"])

    DIPKGeneExpressionFeaturizer().fit(source, pair_expanded_ids=ids, pair_expanded_es_ids=np.array(["cl2"]))
    np.testing.assert_allclose(recording_trainer["train"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(recording_trainer["train"].std(axis=1, ddof=1), 1.0)

    _fit(_ae(), source)
    assert not np.allclose(recording_trainer["train"].mean(axis=1), 0.0)


@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_a_trailing_single_row_moves_to_validation(recording_trainer, make, preprocess) -> None:
    make(batch_size_autoencoder=4).fit(
        _source(n=6),
        pair_expanded_ids=np.array(["cl0", "cl1", "cl2", "cl3", "cl4"]),
        pair_expanded_es_ids=np.array(["cl5"]),
    )

    assert (recording_trainer["train"].shape[0], recording_trainer["validation"].shape[0]) == (4, 2)


@pytest.mark.parametrize(("unit", "log_applied"), [("log2_tpm_plus_1", False), (None, False), ("TPM", True)])
@pytest.mark.parametrize(("make", "preprocess"), _VARIANTS)
def test_log2_applies_only_to_linear_units(recording_trainer, make, preprocess, unit, log_applied) -> None:
    source = _source(unit=unit)

    _fit(make(), source)

    raw = source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"]))
    np.testing.assert_allclose(recording_trainer["train"], preprocess(np.log2(raw + 1) if log_applied else raw))


def test_dipk_reads_the_expression_unit_from_a_real_dataset(recording_trainer) -> None:
    dataset = synthetic_model_dataset(n_features_per_view=4)
    dataset.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"
    source = SampleFeatureSource(dataset, dataset.sample_ids)
    ids = np.array(["cl1", "cl2"])

    DIPKGeneExpressionFeaturizer().fit(source, pair_expanded_ids=ids, pair_expanded_es_ids=ids)

    raw = source.get_view_matrix("gene_expression", ids)
    np.testing.assert_allclose(recording_trainer["train"], _zscore(np.log2(raw + 1)), rtol=1e-5)


def test_ae_transform_encodes_with_the_fitted_encoder_and_state_round_trips(recording_trainer) -> None:
    source = _source(unit="tpm")
    featurizer = _fit(_ae(epochs_autoencoder=3), source)
    ids = np.array(["cl3", "cl4"])

    matrix = featurizer.transform(source, ids)

    expected = encode_gene_expression(np.log2(source.get_view_matrix("gene_expression", ids) + 1), featurizer._encoder)
    assert matrix.shape == (2, 512)
    assert featurizer.output_dim == 512
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    restored = AEGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert (restored._gene_list, restored._epochs) == (None, 3)
    np.testing.assert_allclose(restored.transform(source, ids), matrix)


def test_dipk_transform_encodes_the_preprocessed_matrix_and_state_round_trips(recording_trainer) -> None:
    source = _source(n=3, unit="tpm")
    featurizer = DIPKGeneExpressionFeaturizer(epochs_autoencoder=1, patience_autoencoder=7).fit(
        source, pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2"])
    )

    matrix = featurizer.transform(source, np.array(["cl2"]))

    expected = encode_gene_expression(
        _zscore(np.log2(source.get_view_matrix("gene_expression", np.array(["cl2"])) + 1)), featurizer._encoder
    )
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert (restored._epochs, restored._patience) == (1, 7)
    np.testing.assert_allclose(restored.transform(source, np.array(["cl2"])), matrix)


@pytest.mark.parametrize(
    ("make", "missing_ids_error"),
    [(lambda: _ae(), "needs the training sample ids"), (DIPKGeneExpressionFeaturizer, "requires pair_expanded_ids")],
    ids=["ae", "dipk"],
)
def test_unfitted_and_malformed_state_and_missing_fit_ids(make, missing_ids_error) -> None:
    featurizer = make()
    assert featurizer.output_dim == 0
    assert featurizer.get_state() == {}
    featurizer.set_state({"encoder_state": "not-bytes", "input_dim": 3})
    assert featurizer.output_dim == 0
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(_source(), np.array(["cl0"]))
    with pytest.raises(ValueError, match=missing_ids_error):
        make().fit(_source(), pair_expanded_es_ids=np.array(["cl2"]))
    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        make().fit(_source(), pair_expanded_ids=np.array(["cl0"]))


def test_ae_gene_list_selects_genes_in_list_order_and_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_trainer
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"], "bad": ["a", "absent"]})
    source = _source()

    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list="mine"), source)

    expected = source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"]))[:, [2, 0]]
    np.testing.assert_allclose(recording_trainer["train"], expected)
    assert featurizer._view_names == ("c", "a")
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'bad'"):
        _fit(AEGeneExpressionFeaturizer(gene_list="bad"), source)


def test_dipk_gene_list_selects_the_autoencoder_input_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_trainer
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"]})
    source = _source(n=4)

    featurizer = DIPKGeneExpressionFeaturizer(gene_list="mine", epochs_autoencoder=1).fit(
        source, pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2", "cl3"])
    )

    assert recording_trainer["train"].shape == (2, 2)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored._gene_list == "mine"
    ids = np.array(["cl0", "cl1"])
    np.testing.assert_allclose(restored.transform(source, ids), featurizer.transform(source, ids))
    assert DIPKGeneExpressionFeaturizer()._fetch_hyperparameters() is None
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}


def test_cache_trains_once_and_reuses_the_encoder(recording_trainer, tmp_path: Path) -> None:
    source = _source(n=8)
    cache = tmp_path / "ae"
    ids = np.array(["cl4", "cl5"])

    first = _fit(_ae(autoencoder_cache_dir=str(cache)), source)
    second = _fit(_ae(autoencoder_cache_dir=str(cache)), source)

    assert recording_trainer["calls"] == 1
    assert len(list(cache.glob("gene_ae_*.pt"))) == 1
    np.testing.assert_allclose(second.transform(source, ids), first.transform(source, ids))


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
