"""Tests for the gene-expression autoencoder featurizer of RandomForest-AE-MolGNet."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.ae_gene_expression import AEGeneExpressionFeaturizer
from drevalpy.errors import InvalidInputError
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
from tests.conftest import MockFeatureSource

_ENCODER_MODULE = "drevalpy.components.predictors.literature.dipk.gene_expression_encoder"
_GENES = ["a", "b", "c", "d"]


class _UnitSource(MockFeatureSource):
    """Mock source whose ``gene_expression`` modality records an ``expression_unit``."""

    def __init__(self, unit: str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._unit = unit

    @property
    def mdata(self) -> SimpleNamespace:
        uns = {} if self._unit is None else {"expression_unit": self._unit}
        return SimpleNamespace(uns={}, mod={"gene_expression": SimpleNamespace(uns=uns)})


def _source(n: int = 6, unit: str | None = None, genes: list[str] | None = None) -> MockFeatureSource:
    names = genes or _GENES
    rng = np.random.default_rng(0)
    return _UnitSource(
        unit,
        features={f"cl{i}": {"gene_expression": rng.uniform(1.0, 9.0, size=len(names))} for i in range(n)},
        meta_info={"gene_expression": names},
    )


def _fit(featurizer: AEGeneExpressionFeaturizer, source: MockFeatureSource, train=("cl0", "cl1"), es=("cl2",)):
    return featurizer.fit(source, pair_expanded_ids=np.array(train), pair_expanded_es_ids=np.array(es))


def _recording_trainer(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the trainer by a stub that records its arguments and returns a real encoder."""
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

    seen: dict = {"calls": 0}

    def _train(train, validation, epochs, **kwargs):
        seen.update(train=train, validation=validation, epochs=epochs, **kwargs)
        seen["calls"] += 1
        return GeneExpressionEncoder(train.shape[1])

    monkeypatch.setattr(f"{_ENCODER_MODULE}.train_gene_expession_autoencoder", _train)
    return seen


@pytest.fixture(autouse=True)
def _no_ambient_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DREVAL_AE_CACHE", raising=False)


def test_defaults_are_the_random_forest_ae_molgnet_values_of_1_6_1() -> None:
    assert AEGeneExpressionFeaturizer.get_default_hyperparameters() == {
        "epochs_autoencoder": 2000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 50,
        "batch_size_autoencoder": 1024,
    }
    featurizer = AEGeneExpressionFeaturizer()
    assert featurizer._gene_list == "tcga_autoencoder_genes_available"
    assert (featurizer._epochs, featurizer._lr, featurizer._patience, featurizer._batch_size) == (2000, 1e-4, 50, 1024)


def test_fits_on_unique_training_samples_and_forwards_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(n=4)
    seen = _recording_trainer(monkeypatch)

    AEGeneExpressionFeaturizer(
        gene_list=None, epochs_autoencoder=7, lr_autoencoder=5e-3, patience_autoencoder=11, batch_size_autoencoder=2
    ).fit(
        source,
        pair_expanded_ids=np.array(["cl1", "cl0", "cl1", "cl1", "cl0"]),
        pair_expanded_es_ids=np.array(["cl3", "cl3", "cl2"]),
    )

    assert (seen["epochs"], seen["lr"], seen["patience"], seen["batch_size"]) == (7, 5e-3, 11, 2)
    np.testing.assert_allclose(seen["train"], source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"])))
    np.testing.assert_allclose(seen["validation"], source.get_view_matrix("gene_expression", np.array(["cl2", "cl3"])))


def test_no_per_sample_zscore_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    seen = _recording_trainer(monkeypatch)

    _fit(AEGeneExpressionFeaturizer(gene_list=None), source)

    assert not np.allclose(seen["train"].mean(axis=1), 0.0)


def test_moves_a_trailing_single_row_to_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _recording_trainer(monkeypatch)

    AEGeneExpressionFeaturizer(gene_list=None, batch_size_autoencoder=4).fit(
        _source(n=6),
        pair_expanded_ids=np.array(["cl0", "cl1", "cl2", "cl3", "cl4"]),
        pair_expanded_es_ids=np.array(["cl5"]),
    )

    assert (seen["train"].shape[0], seen["validation"].shape[0]) == (4, 2)


@pytest.mark.parametrize(
    ("unit", "log_applied"),
    [("log2_tpm_plus_1", False), (None, False), ("tpm", True), ("TPM", True)],
)
def test_log2_applies_only_to_linear_units(
    monkeypatch: pytest.MonkeyPatch, unit: str | None, log_applied: bool
) -> None:
    source = _source(unit=unit)
    seen = _recording_trainer(monkeypatch)
    ids = np.array(["cl0", "cl1"])

    _fit(AEGeneExpressionFeaturizer(gene_list=None), source)

    raw = source.get_view_matrix("gene_expression", ids)
    np.testing.assert_allclose(seen["train"], np.log2(raw + 1) if log_applied else raw)


def test_gene_list_selects_genes_in_list_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"]})
    source = _source()
    seen = _recording_trainer(monkeypatch)

    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list="mine"), source)

    expected = source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"]))[:, [2, 0]]
    np.testing.assert_allclose(seen["train"], expected)
    assert featurizer._view_names == ("c", "a")
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}


def test_missing_list_genes_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["a", "absent"]})
    _recording_trainer(monkeypatch)

    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'mine'"):
        _fit(AEGeneExpressionFeaturizer(gene_list="mine"), _source())


def test_transform_encodes_with_the_fitted_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

    source = _source(unit="tpm")
    _recording_trainer(monkeypatch)
    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list=None), source)

    matrix = featurizer.transform(source, np.array(["cl3", "cl4"]))

    expected = encode_gene_expression(
        np.log2(source.get_view_matrix("gene_expression", np.array(["cl3", "cl4"])) + 1), featurizer._encoder
    )
    assert matrix.shape == (2, 512)
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    assert featurizer.output_dim == 512


def test_state_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    _recording_trainer(monkeypatch)
    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list=None, epochs_autoencoder=3), source)
    ids = np.array(["cl0", "cl5"])

    restored = AEGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())

    assert restored._gene_list is None
    assert (restored._epochs, restored._batch_size) == (3, 1024)
    np.testing.assert_allclose(restored.transform(source, ids), featurizer.transform(source, ids))


def test_state_is_empty_before_fit_and_a_malformed_state_is_ignored() -> None:
    featurizer = AEGeneExpressionFeaturizer()

    assert featurizer.get_state() == {}
    featurizer.set_state({"encoder_state": "not-bytes", "input_dim": 3})
    assert featurizer.output_dim == 0


def test_fit_requires_the_training_ids_and_an_early_stopping_holdout() -> None:
    with pytest.raises(InvalidInputError, match="needs the training sample ids"):
        AEGeneExpressionFeaturizer(gene_list=None).fit(_source(), pair_expanded_es_ids=np.array(["cl2"]))
    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        AEGeneExpressionFeaturizer(gene_list=None).fit(_source(), pair_expanded_ids=np.array(["cl0"]))


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        AEGeneExpressionFeaturizer()._transform_features(_source(), np.array(["cl0"]))


def test_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(
        AEGeneExpressionFeaturizer(gene_list=None),
        ids_kwarg="pair_expanded_ids",
        expect_output_dim=False,
        expected_blocks=("gene_expression",),
    )


def test_cache_trains_once_and_reuses_the_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _source(n=8)
    seen = _recording_trainer(monkeypatch)
    cache = tmp_path / "ae"
    ids = np.array(["cl4", "cl5"])

    first = _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(cache)), source)
    second = _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(cache)), source)

    assert seen["calls"] == 1
    assert len(list(cache.glob("gene_ae_*.pt"))) == 1
    np.testing.assert_allclose(second.transform(source, ids), first.transform(source, ids))


def test_cache_directory_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _recording_trainer(monkeypatch)
    monkeypatch.setenv("DREVAL_AE_CACHE", str(tmp_path / "env"))

    _fit(AEGeneExpressionFeaturizer(gene_list=None), _source())
    _fit(AEGeneExpressionFeaturizer(gene_list=None), _source())

    assert seen["calls"] == 1
    assert len(list((tmp_path / "env").glob("gene_ae_*.pt"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"epochs_autoencoder": 3}, id="epochs"),
        pytest.param({"patience_autoencoder": 3}, id="patience"),
        pytest.param({"lr_autoencoder": 1e-3}, id="learning-rate"),
    ],
)
def test_cache_is_keyed_by_the_autoencoder_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: dict
) -> None:
    seen = _recording_trainer(monkeypatch)
    source = _source()

    _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(tmp_path)), source)
    _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(tmp_path), **change), source)

    assert seen["calls"] == 2


def test_cache_is_keyed_by_the_training_and_early_stopping_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _recording_trainer(monkeypatch)
    source = _source()

    def fit(train, es) -> None:
        _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(tmp_path)), source, train, es)

    fit(("cl0", "cl1"), ("cl2",))
    fit(("cl0", "cl1", "cl3"), ("cl2",))
    fit(("cl0", "cl1"), ("cl4",))

    assert seen["calls"] == 3


def test_cache_is_keyed_by_the_gene_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _recording_trainer(monkeypatch)
    cache = tmp_path / "cache"
    forward = _source()
    reverse = MockFeatureSource(
        {key: {"gene_expression": value["gene_expression"][::-1]} for key, value in forward.features.items()},
        meta_info={"gene_expression": _GENES[::-1]},
    )

    _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(cache)), forward)
    _fit(AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(cache)), reverse)

    assert seen["calls"] == 2
