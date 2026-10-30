"""Tests for the literature omics preprocessing featurizers: MOLIR, SuperFELTR and PharmaFormer expression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.molir_omics import MOLIROmicsFeaturizer
from drevalpy.components.featurizers.sample.pharmaformer_gene_expression import (
    PharmaFormerGeneExpressionFeaturizer,
)
from drevalpy.components.featurizers.sample.superfeltr_omics import SuperFELTROmicsFeaturizer
from drevalpy.errors import InvalidInputError
from tests.components.featurizers.sample._helpers import pickled, use_gene_lists
from tests.conftest import MockFeatureSource


def test_molir_omics_selects_expression_and_round_trips_through_pickle() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            "gene_expression": ["a", "b", "c"],
            "mutations": ["m1", "m2"],
            "copy_number_variation_gistic": ["c1", "c2"],
        },
    )
    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=2).fit(features, np.array(["cl0", "cl1"]))
    blocks = featurizer.transform(features, np.array(["cl0", "cl2"])).blocks
    assert set(blocks) == {"gene_expression", "mutations", "copy_number_variation_gistic"}
    assert blocks["gene_expression"].values.shape == (2, 2)
    np.testing.assert_allclose(
        pickled(featurizer).transform(features, np.array(["cl0", "cl2"])).matrix, blocks["gene_expression"].values
    )
    (gene_expression, *_) = MOLIROmicsFeaturizer.output_blocks({})
    assert gene_expression.name == "gene_expression"


def test_molir_omics_unfitted_and_without_metadata() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        MOLIROmicsFeaturizer().transform(MockFeatureSource({}), np.array(["cl0"]))
    assert set(MOLIROmicsFeaturizer.get_hyperparameter_space()) == {"n_gene_expression_features"}
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i], dtype=np.float32),
            }
            for i in range(2)
        }
    )

    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=2).fit(features, features.identifiers)

    blocks = featurizer.transform(features, features.identifiers).blocks
    assert blocks["gene_expression"].feature_names is None
    assert blocks["gene_expression"].values.shape == (2, 2)


def test_selection_ranks_training_variance_before_standardization() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array(values, dtype=np.float64),
                "mutations": np.array([0.0]),
                "copy_number_variation_gistic": np.array([1.0]),
            }
            for i, values in enumerate([[0.0, 1.0], [100.0, 1.01], [1.0, 1e8]])
        },
        meta_info={"gene_expression": ["variable", "nearly_constant"]},
    )
    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=1).fit(features, np.array(["cl0", "cl1"]))
    result = featurizer.transform(features, np.array(["cl0", "cl1"]))
    assert result.blocks["gene_expression"].feature_names == ("variable",)
    np.testing.assert_allclose(result.matrix, [[-1.0], [1.0]])


def _molir_omics() -> MockFeatureSource:
    return MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1, 3 - i], dtype=np.float32),
                "mutations": np.array([i, 1, 0], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, 1], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            "gene_expression": ["a", "b", "c", "d"],
            "mutations": ["m1", "m2", "m3"],
            "copy_number_variation_gistic": ["c1", "c2", "c3"],
        },
    )


def test_molir_omics_gene_lists_select_each_view_and_name_the_variance_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"ge": ["d", "b", "a"], "mut": ["m3", "m1"]})
    ids = np.array(["cl0", "cl1", "cl2"])
    featurizer = MOLIROmicsFeaturizer(
        n_gene_expression_features=2, gene_lists={"gene_expression": "ge", "mutations": "mut"}
    ).fit(_molir_omics(), ids)

    blocks = featurizer.transform(_molir_omics(), ids).blocks

    assert blocks["mutations"].feature_names == ("m3", "m1")
    assert blocks["copy_number_variation_gistic"].feature_names == ("c1", "c2", "c3")
    assert set(blocks["gene_expression"].feature_names) <= {"a", "b", "d"}
    assert blocks["gene_expression"].values.shape == (3, 2)
    restored = pickled(featurizer)
    np.testing.assert_allclose(restored.transform(_molir_omics(), ids).matrix, blocks["gene_expression"].values)
    assert restored._gene_lists == {"gene_expression": "ge", "mutations": "mut"}


def test_molir_omics_gene_lists_raise_when_a_view_lacks_genes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mut": ["m1", "missing"]})
    with pytest.raises(InvalidInputError, match="view 'mutations' lacks 1 of 2 genes in 'mut'"):
        MOLIROmicsFeaturizer(gene_lists={"mutations": "mut"}).fit(_molir_omics(), np.array(["cl0"]))


def test_superfeltr_omics_selects_each_view_and_round_trips_through_pickle() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            view: [f"{view}{i}" for i in range(3)]
            for view in (
                "gene_expression",
                "mutations",
                "copy_number_variation_gistic",
            )
        },
    )
    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features, np.array(["cl0", "cl1"]))
    blocks = featurizer.transform(features, np.array(["cl0", "cl2"])).blocks
    assert all(block.values.shape == (2, 2) for block in blocks.values())
    np.testing.assert_allclose(
        pickled(featurizer).transform(features, np.array(["cl0", "cl2"])).matrix, blocks["gene_expression"].values
    )


def test_superfeltr_omics_without_metadata_and_before_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        SuperFELTROmicsFeaturizer().transform(MockFeatureSource({}), np.array(["cl0"]))
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(2)
        }
    )

    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features, features.identifiers)

    blocks = featurizer.transform(features, features.identifiers).blocks
    assert blocks["gene_expression"].feature_names == ()
    assert sum(block.values.shape[1] for block in blocks.values()) == 6


def test_expression_transform_precedes_training_only_selection() -> None:
    # Raw variance favors column 0; arcsinh variance favors column 1. A held-out
    # extreme in column 0 must not change the fitted selection.
    features = MockFeatureSource(
        {
            "train0": {
                "gene_expression": np.array([100.0, 0.0]),
                "mutations": np.array([0.0, 1.0]),
                "copy_number_variation_gistic": np.array([1.0, 2.0]),
            },
            "train1": {
                "gene_expression": np.array([110.0, 2.0]),
                "mutations": np.array([1.0, 1.0]),
                "copy_number_variation_gistic": np.array([3.0, 2.0]),
            },
            "heldout": {
                "gene_expression": np.array([1e8, 3.0]),
                "mutations": np.array([2.0, 1.0]),
                "copy_number_variation_gistic": np.array([4.0, 2.0]),
            },
        }
    )
    fitted = SuperFELTROmicsFeaturizer(n_features_per_view=1).fit(features, np.array(["train0", "train1"]))
    blocks = fitted.transform(features, np.array(["train0", "heldout"])).blocks
    assert fitted._masks is not None
    np.testing.assert_array_equal(fitted._masks["gene_expression"], [False, True])
    np.testing.assert_allclose(blocks["gene_expression"].values[:, 0], np.arcsinh([0.0, 3.0]))
    np.testing.assert_array_equal(blocks["mutations"].values[:, 0], [0.0, 2.0])
    np.testing.assert_array_equal(blocks["copy_number_variation_gistic"].values[:, 0], [1.0, 4.0])


def _superfeltr_omics() -> MockFeatureSource:
    views = ("gene_expression", "mutations", "copy_number_variation_gistic")
    return MockFeatureSource(
        {f"cl{i}": {view: np.array([i, 1, i + 1, 3 * i], dtype=np.float32) for view in views} for i in range(3)},
        meta_info={view: [f"{view}{j}" for j in range(4)] for view in views},
    )


def test_superfeltr_omics_gene_lists_restrict_the_variance_selection_per_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"two": ["mutations3", "mutations0"]})
    ids = np.array(["cl0", "cl1", "cl2"])
    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=1, gene_lists={"mutations": "two"}).fit(
        _superfeltr_omics(), ids
    )

    blocks = featurizer.transform(_superfeltr_omics(), ids).blocks

    assert blocks["mutations"].feature_names == ("mutations3",)
    assert blocks["gene_expression"].feature_names == ("gene_expression3",)
    restored = pickled(featurizer)
    assert restored._gene_lists == {"mutations": "two"}
    np.testing.assert_allclose(restored.transform(_superfeltr_omics(), ids).matrix, blocks["gene_expression"].values)


def test_superfeltr_omics_gene_lists_raise_when_a_view_lacks_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"bad": ["gene_expression0", "nope"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'bad'"):
        SuperFELTROmicsFeaturizer(gene_lists={"gene_expression": "bad"}).fit(_superfeltr_omics(), np.array(["cl0"]))


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.array([i, i + 2], dtype=np.float32)} for i in range(3)},
        meta_info={"gene_expression": ["a", "b"]},
    )


def test_pharmaformer_gene_expression_round_trips_through_pickle() -> None:
    features = _features()
    featurizer = PharmaFormerGeneExpressionFeaturizer()
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer.transform(features, np.array(["cl0"]))
    featurizer.fit(features, np.array(["cl0", "cl1"]), train_pair_ids=np.array(["cl0", "cl1", "cl0"]))
    blocks = featurizer.transform(features, np.array(["cl0", "cl2"])).blocks
    assert blocks["gene_expression"].feature_names == ("a", "b")
    np.testing.assert_allclose(
        pickled(featurizer).transform(features, np.array(["cl0", "cl2"])).matrix, blocks["gene_expression"].values
    )
    assert featurizer.transform(features, np.array([], dtype=str)).matrix.shape == (0, 2)


def _scaler_mean(featurizer: PharmaFormerGeneExpressionFeaturizer) -> np.ndarray:
    assert featurizer._scalers is not None
    return featurizer._scalers[0].mean_


def test_pharmaformer_gene_expression_fits_the_scaler_once_per_unique_sample() -> None:
    featurizer = PharmaFormerGeneExpressionFeaturizer().fit(
        _features(), np.array(["cl0", "cl1"]), train_pair_ids=np.array(["cl0", "cl0", "cl0", "cl1"])
    )
    np.testing.assert_allclose(_scaler_mean(featurizer), [0.5, 2.5])
    every = PharmaFormerGeneExpressionFeaturizer().fit(_features(), np.array(["cl1", "cl0", "cl2", "cl0"]))
    np.testing.assert_allclose(_scaler_mean(every), [1.0, 3.0])


def test_pharmaformer_gene_expression_gene_list_selects_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["b"]})
    features = _features()
    ids = np.array(["cl0", "cl1", "cl0"])
    featurizer = PharmaFormerGeneExpressionFeaturizer(gene_list="mine").fit(features, ids)

    blocks = featurizer.transform(features, np.array(["cl0", "cl2"])).blocks

    assert blocks["gene_expression"].feature_names == ("b",)
    assert blocks["gene_expression"].values.shape == (2, 1)
    restored = pickled(featurizer)
    assert restored._gene_list == "mine"
    assert PharmaFormerGeneExpressionFeaturizer().stored_params() == {}
    assert featurizer.stored_params() == {"gene_list": "mine"}
    np.testing.assert_allclose(
        restored.transform(features, np.array(["cl0", "cl2"])).matrix, blocks["gene_expression"].values
    )


def test_pharmaformer_gene_expression_gene_list_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["a", "z"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'mine'"):
        PharmaFormerGeneExpressionFeaturizer(gene_list="mine").fit(_features(), np.array(["cl0"]))


@pytest.mark.parametrize(
    ("featurizer_cls", "unknown_view"),
    [(MOLIROmicsFeaturizer, "methylation"), (SuperFELTROmicsFeaturizer, "proteomics")],
)
def test_gene_lists_reject_unknown_views(featurizer_cls: type, unknown_view: str) -> None:
    with pytest.raises(InvalidInputError, match="unknown views"):
        featurizer_cls(gene_lists={unknown_view: "x"})
