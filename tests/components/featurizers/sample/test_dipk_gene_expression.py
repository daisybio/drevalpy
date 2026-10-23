"""Tests for the DIPK gene-expression featurizer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.dipk_gene_expression import (
    DIPKGeneExpressionFeaturizer,
    GeneExpressionEncoder,
)
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.arange(3, dtype=np.float32) + i} for i in range(3)},
        meta_info={"gene_expression": ["a", "b", "c"]},
    )


_TRAINER = "drevalpy.components.predictors.literature.dipk.gene_expression_encoder.train_gene_expession_autoencoder"


class _UnitSource(MockFeatureSource):
    """Mock source whose ``gene_expression`` modality records an ``expression_unit``."""

    def __init__(self, unit: str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._unit = unit

    @property
    def mdata(self) -> SimpleNamespace:
        uns = {} if self._unit is None else {"expression_unit": self._unit}
        return SimpleNamespace(uns={}, mod={"gene_expression": SimpleNamespace(uns=uns)})


def _wide_features(n: int = 5, genes: int = 4, unit: str | None = None) -> MockFeatureSource:
    rng = np.random.default_rng(0)
    return _UnitSource(
        unit,
        features={f"cl{i}": {"gene_expression": rng.uniform(1.0, 9.0, size=genes)} for i in range(n)},
        meta_info={"gene_expression": [f"g{j}" for j in range(genes)]},
    )


def _recording_trainer(monkeypatch) -> dict:
    """Replace the trainer with a stub that records its arguments and returns a real encoder."""
    seen: dict = {}

    def _train(train, validation, epochs, **kwargs):
        seen.update(train=train, validation=validation, epochs=epochs, **kwargs)
        return GeneExpressionEncoder(train.shape[1])

    # The featurizer imports the trainer from its defining module inside ``_fit_state``
    # (to keep ``torch`` off the ``import drevalpy`` path), so the patch has to
    # land there rather than on the featurizer module's re-export.
    monkeypatch.setattr(_TRAINER, _train)
    return seen


def _zscore(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([(row - np.mean(row)) / np.std(row, ddof=1) for row in matrix])


def test_dipk_gene_expression_round_trips_state(monkeypatch) -> None:
    features = _features()
    _recording_trainer(monkeypatch)
    pair_expanded_ids = np.array(["cl0", "cl1", "cl0"])
    pair_expanded_es_ids = np.array(["cl2", "cl2"])
    featurizer = DIPKGeneExpressionFeaturizer(
        epochs_autoencoder=1, lr_autoencoder=5e-3, patience_autoencoder=7, batch_size_autoencoder=32
    ).fit(features, pair_expanded_ids=pair_expanded_ids, pair_expanded_es_ids=pair_expanded_es_ids)
    matrix = featurizer.transform(features, np.array(["cl0", "cl2"]))
    assert matrix.shape == (2, 512)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), matrix)
    assert (restored._epochs, restored._lr, restored._patience, restored._batch_size) == (1, 5e-3, 7, 32)


def test_dipk_gene_expression_fits_on_unique_sample_ids(monkeypatch) -> None:
    features = _wide_features(n=4)
    seen = _recording_trainer(monkeypatch)

    DIPKGeneExpressionFeaturizer(epochs_autoencoder=1).fit(
        features,
        pair_expanded_ids=np.array(["cl1", "cl0", "cl1", "cl1", "cl0"]),
        pair_expanded_es_ids=np.array(["cl3", "cl3", "cl2"]),
    )

    assert seen["train"].shape == (2, 4)
    assert seen["validation"].shape == (2, 4)
    np.testing.assert_allclose(
        seen["train"], _zscore(features.get_view_matrix("gene_expression", np.array(["cl0", "cl1"])))
    )
    np.testing.assert_allclose(
        seen["validation"], _zscore(features.get_view_matrix("gene_expression", np.array(["cl2", "cl3"])))
    )


def test_dipk_gene_expression_forwards_the_autoencoder_hyperparameters(monkeypatch) -> None:
    seen = _recording_trainer(monkeypatch)

    DIPKGeneExpressionFeaturizer(
        epochs_autoencoder=7, lr_autoencoder=5e-3, patience_autoencoder=11, batch_size_autoencoder=2
    ).fit(_wide_features(), pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2"]))

    assert (seen["epochs"], seen["lr"], seen["patience"], seen["batch_size"]) == (7, 5e-3, 11, 2)


def test_dipk_gene_expression_defaults_are_the_1_6_1_values() -> None:
    defaults = DIPKGeneExpressionFeaturizer.get_default_hyperparameters()

    assert defaults == {
        "epochs_autoencoder": 1000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 3,
        "batch_size_autoencoder": 1024,
    }
    featurizer = DIPKGeneExpressionFeaturizer()
    assert (featurizer._epochs, featurizer._lr, featurizer._patience, featurizer._batch_size) == (1000, 1e-4, 3, 1024)


def test_dipk_gene_expression_epoch_space_reaches_2000() -> None:
    assert DIPKGeneExpressionFeaturizer.get_hyperparameter_space()["epochs_autoencoder"]["high"] >= 2000


def test_dipk_gene_expression_moves_a_trailing_single_row_to_validation(monkeypatch) -> None:
    seen = _recording_trainer(monkeypatch)

    DIPKGeneExpressionFeaturizer(epochs_autoencoder=1, batch_size_autoencoder=4).fit(
        _wide_features(n=6),
        pair_expanded_ids=np.array(["cl0", "cl1", "cl2", "cl3", "cl4"]),
        pair_expanded_es_ids=np.array(["cl5"]),
    )

    assert seen["train"].shape[0] == 4
    assert seen["validation"].shape[0] == 2


def test_dipk_gene_expression_zscore_uses_the_sample_standard_deviation(monkeypatch) -> None:
    features = _wide_features(n=3)
    seen = _recording_trainer(monkeypatch)

    DIPKGeneExpressionFeaturizer().fit(
        features, pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2"])
    )

    np.testing.assert_allclose(seen["train"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(seen["train"].std(axis=1, ddof=1), 1.0)


@pytest.mark.parametrize(
    ("unit", "log_applied"),
    [
        pytest.param("log2_tpm_plus_1", False, id="log-unit"),
        pytest.param("log2_cpm", False, id="other-log-unit"),
        pytest.param(None, False, id="no-unit-is-taken-as-log-scale"),
        pytest.param("tpm", True, id="linear-tpm"),
        pytest.param("TPM", True, id="linear-case-insensitive"),
    ],
)
def test_dipk_gene_expression_applies_log2_only_to_linear_units(monkeypatch, unit, log_applied) -> None:
    features = _wide_features(n=3, unit=unit)
    seen = _recording_trainer(monkeypatch)
    ids = np.array(["cl0", "cl1"])

    DIPKGeneExpressionFeaturizer().fit(features, pair_expanded_ids=ids, pair_expanded_es_ids=np.array(["cl2"]))

    raw = features.get_view_matrix("gene_expression", ids)
    expected = _zscore(np.log2(raw + 1) if log_applied else raw)
    np.testing.assert_allclose(seen["train"], expected)


@pytest.mark.parametrize(("unit", "log_applied"), [("log2_tpm_plus_1", False), ("tpm", True)])
def test_dipk_gene_expression_reads_the_unit_from_a_real_dataset(monkeypatch, unit, log_applied) -> None:
    from drevalpy.types.data.feature_source import SampleFeatureSource
    from tests.models.synthetic_fixtures import synthetic_mudataset

    dataset = synthetic_mudataset(n_features_per_view=4)
    dataset.mdata.mod["gene_expression"].uns["expression_unit"] = unit
    source = SampleFeatureSource(dataset, dataset.sample_ids)
    seen = _recording_trainer(monkeypatch)
    ids = np.array(["cl1", "cl2"])

    DIPKGeneExpressionFeaturizer().fit(source, pair_expanded_ids=ids, pair_expanded_es_ids=ids)

    raw = source.get_view_matrix("gene_expression", ids)
    np.testing.assert_allclose(seen["train"], _zscore(np.log2(raw + 1) if log_applied else raw), rtol=1e-5)


def test_dipk_gene_expression_transform_encodes_the_preprocessed_matrix(monkeypatch) -> None:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

    features = _wide_features(n=3, unit="tpm")
    seen = _recording_trainer(monkeypatch)
    featurizer = DIPKGeneExpressionFeaturizer().fit(
        features, pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2"])
    )

    matrix = featurizer.transform(features, np.array(["cl2"]))

    expected = encode_gene_expression(
        _zscore(np.log2(features.get_view_matrix("gene_expression", np.array(["cl2"])) + 1)), featurizer._encoder
    )
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    assert seen["train"].shape == (2, 4)


def test_dipk_gene_expression_requires_pair_expanded_ids() -> None:
    with pytest.raises(ValueError, match="requires pair_expanded_ids"):
        DIPKGeneExpressionFeaturizer().fit(_features(), pair_expanded_es_ids=np.array(["cl2"]))


def test_dipk_gene_expression_requires_non_empty_early_stopping_ids() -> None:
    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        DIPKGeneExpressionFeaturizer().fit(_features(), pair_expanded_ids=np.array(["cl0"]))


def test_dipk_gene_expression_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        DIPKGeneExpressionFeaturizer()._transform_features(_features(), np.array(["cl0"]))


def test_dipk_gene_expression_output_dim_is_zero_before_fit() -> None:
    assert DIPKGeneExpressionFeaturizer().output_dim == 0


def test_dipk_gene_expression_state_is_empty_before_fit() -> None:
    assert DIPKGeneExpressionFeaturizer().get_state() == {}


def test_dipk_gene_expression_set_state_ignores_a_malformed_payload() -> None:
    featurizer = DIPKGeneExpressionFeaturizer()

    featurizer.set_state({"encoder_state": "not-bytes", "input_dim": 3})

    assert featurizer.output_dim == 0


def test_dipk_gene_expression_hyperparameter_space_exposes_the_autoencoder_settings() -> None:
    assert set(DIPKGeneExpressionFeaturizer.get_hyperparameter_space()) == {
        "epochs_autoencoder",
        "lr_autoencoder",
        "patience_autoencoder",
        "batch_size_autoencoder",
    }


def test_dipk_gene_expression_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(
        DIPKGeneExpressionFeaturizer(),
        ids_kwarg="pair_expanded_ids",
        expect_output_dim=False,
        expected_blocks=("gene_expression",),
    )


def test_dipk_gene_expression_gene_list_selects_the_autoencoder_input_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g2", "g0"]})
    features = _wide_features(n=4, genes=4)
    seen = _recording_trainer(monkeypatch)

    featurizer = DIPKGeneExpressionFeaturizer(gene_list="mine", epochs_autoencoder=1).fit(
        features, pair_expanded_ids=np.array(["cl0", "cl1"]), pair_expanded_es_ids=np.array(["cl2", "cl3"])
    )

    assert seen["train"].shape == (2, 2)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored._gene_list == "mine"
    np.testing.assert_allclose(
        restored.transform(features, np.array(["cl0", "cl1"])), featurizer.transform(features, np.array(["cl0", "cl1"]))
    )


def test_dipk_gene_expression_matches_a_stored_variant_only_for_its_own_gene_list() -> None:
    assert DIPKGeneExpressionFeaturizer()._fetch_hyperparameters() is None
    assert DIPKGeneExpressionFeaturizer(gene_list="mine")._fetch_hyperparameters() == {"gene_list": "mine"}
