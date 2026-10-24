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


def test_defaults_and_hyperparameter_space() -> None:
    defaults = AEGeneExpressionFeaturizer.get_default_hyperparameters()
    assert defaults == {
        "epochs_autoencoder": 2000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 50,
        "batch_size_autoencoder": 1024,
    }
    assert AEGeneExpressionFeaturizer()._gene_list == "tcga_autoencoder_genes_available"


def test_fits_on_unique_raw_training_samples_and_forwards_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert not np.allclose(seen["train"].mean(axis=1), 0.0)


def test_moves_a_trailing_single_row_to_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _recording_trainer(monkeypatch)

    AEGeneExpressionFeaturizer(gene_list=None, batch_size_autoencoder=4).fit(
        _source(n=6),
        pair_expanded_ids=np.array(["cl0", "cl1", "cl2", "cl3", "cl4"]),
        pair_expanded_es_ids=np.array(["cl5"]),
    )

    assert (seen["train"].shape[0], seen["validation"].shape[0]) == (4, 2)


@pytest.mark.parametrize(("unit", "log_applied"), [("log2_tpm_plus_1", False), (None, False), ("TPM", True)])
def test_log2_applies_only_to_linear_units(
    monkeypatch: pytest.MonkeyPatch, unit: str | None, log_applied: bool
) -> None:
    source = _source(unit=unit)
    seen = _recording_trainer(monkeypatch)

    _fit(AEGeneExpressionFeaturizer(gene_list=None), source)

    raw = source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"]))
    np.testing.assert_allclose(seen["train"], np.log2(raw + 1) if log_applied else raw)


def test_gene_list_selects_genes_in_list_order_and_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["c", "a"], "bad": ["a", "absent"]})
    source = _source()
    seen = _recording_trainer(monkeypatch)

    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list="mine"), source)

    expected = source.get_view_matrix("gene_expression", np.array(["cl0", "cl1"]))[:, [2, 0]]
    np.testing.assert_allclose(seen["train"], expected)
    assert featurizer._view_names == ("c", "a")
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'bad'"):
        _fit(AEGeneExpressionFeaturizer(gene_list="bad"), source)


def test_transform_encodes_with_the_fitted_encoder_and_state_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

    source = _source(unit="tpm")
    _recording_trainer(monkeypatch)
    featurizer = _fit(AEGeneExpressionFeaturizer(gene_list=None, epochs_autoencoder=3), source)
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


def test_unfitted_and_malformed_state_and_missing_fit_ids() -> None:
    featurizer = AEGeneExpressionFeaturizer()
    assert featurizer.get_state() == {}
    featurizer.set_state({"encoder_state": "not-bytes", "input_dim": 3})
    assert featurizer.output_dim == 0
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(_source(), np.array(["cl0"]))
    with pytest.raises(InvalidInputError, match="needs the training sample ids"):
        AEGeneExpressionFeaturizer(gene_list=None).fit(_source(), pair_expanded_es_ids=np.array(["cl2"]))
    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        AEGeneExpressionFeaturizer(gene_list=None).fit(_source(), pair_expanded_ids=np.array(["cl0"]))


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


def test_cache_is_keyed_by_settings_samples_and_gene_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _recording_trainer(monkeypatch)
    source = _source()
    reverse = MockFeatureSource(
        {key: {"gene_expression": value["gene_expression"][::-1]} for key, value in source.features.items()},
        meta_info={"gene_expression": _GENES[::-1]},
    )

    def fit(src=source, train=("cl0", "cl1"), es=("cl2",), **kwargs) -> None:
        featurizer = AEGeneExpressionFeaturizer(gene_list=None, autoencoder_cache_dir=str(tmp_path), **kwargs)
        _fit(featurizer, src, train, es)

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
    assert seen["calls"] == len(variations)
    fit()
    assert seen["calls"] == len(variations)
