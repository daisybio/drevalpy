"""Tests for the single-view dense sample featurizers (raw, PCA, proteomics, tissue) and the sweeps over all of them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers.featurizer import all_nan_rows_mask
from drevalpy.components.featurizers.sample._proteomics_transformer import (
    ProteomicsMedianCenterAndImputeTransformer,
)
from drevalpy.components.featurizers.sample.molir_omics import MOLIROmicsFeaturizer
from drevalpy.components.featurizers.sample.normalized_proteomics import (
    NormalizedProteomicsSampleFeaturizer,
    log10_and_set_na,
)
from drevalpy.components.featurizers.sample.pca import PCASampleFeaturizer
from drevalpy.components.featurizers.sample.pharmaformer_gene_expression import PharmaFormerGeneExpressionFeaturizer
from drevalpy.components.featurizers.sample.raw import RawSampleFeaturizer
from drevalpy.components.featurizers.sample.scaled_gene_expression import ScaledGeneExpressionFeaturizer
from drevalpy.components.featurizers.sample.superfeltr_omics import SuperFELTROmicsFeaturizer
from drevalpy.components.featurizers.sample.tissue import TissueFeaturizer
from drevalpy.errors import InvalidInputError
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.components.featurizers._helpers import mock_source, random_views
from tests.components.featurizers.sample._helpers import (
    assert_reads_by_name,
    pickled,
    use_gene_lists,
)
from tests.conftest import MockFeatureSource


def _raw_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "mutations": np.array([0.0, 1.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "mutations": np.array([1.0, 0.0], dtype=np.float32),
            },
        },
        meta_info={"gene_expression": ["a", "b"], "mutations": ["a", "b"]},
    )


def test_raw_passes_through_view_and_requires_one() -> None:
    features = _raw_features()
    featurizer = RawSampleFeaturizer(view="gene_expression")
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids)
    np.testing.assert_allclose(featurizer.transform(features, entity_ids).matrix, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="requires an explicit view"):
        RawSampleFeaturizer(view="")


def test_view_bound_featurizers_check_usability_on_their_own_view() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0]), "methylation": np.array([np.nan, np.nan])},
            "cl2": {"gene_expression": np.array([np.nan, np.nan]), "methylation": np.array([1.0, 2.0])},
        }
    )
    ids = np.array(["cl1", "cl2"])

    for featurizer in (RawSampleFeaturizer(view="methylation"), PCASampleFeaturizer(view="methylation")):
        np.testing.assert_array_equal(featurizer.valid_mask(features, ids), [False, True])
    np.testing.assert_array_equal(all_nan_rows_mask(features, "gene_expression", ids), [True, False])


def test_raw_gene_list_keeps_the_listed_genes_in_list_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["b"]})
    features = _raw_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = RawSampleFeaturizer(view="mutations", gene_list="mine").fit(features, entity_ids)

    blocks = featurizer.transform(features, entity_ids).blocks

    assert blocks["mutations"].feature_names == ("b",)
    np.testing.assert_allclose(blocks["mutations"].values, [[1.0], [0.0]])


def test_raw_gene_list_raises_when_genes_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["a", "z"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'mine'"):
        RawSampleFeaturizer(view="mutations", gene_list="mine").fit(_raw_features(), np.array(["cl1"], dtype=str))


def _pca_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.arange(6, dtype=np.float32)},
            "cl2": {"gene_expression": np.arange(6, 12, dtype=np.float32)},
            "cl3": {"gene_expression": np.arange(12, 18, dtype=np.float32)},
        }
    )


_PCA_IDS = np.array(["cl1", "cl2", "cl3"], dtype=str)


def test_pca_reduces_view_dimension_caps_components_and_requires_a_view() -> None:
    features = _pca_features()
    featurizer = PCASampleFeaturizer(view="gene_expression", n_components=2)
    featurizer.fit(features, _PCA_IDS)
    result = featurizer.transform(features, _PCA_IDS)
    assert result.matrix.shape == (3, 2)
    assert result.blocks["gene_expression"].feature_names is None
    assert featurizer.transform(features, np.array([], dtype=str)).matrix.shape == (0, 2)

    capped = PCASampleFeaturizer(view="gene_expression", n_components=128).fit(features, _PCA_IDS)
    assert capped.transform(features, _PCA_IDS).matrix.shape == (3, 3)
    with pytest.raises(ValueError, match="requires an explicit view"):
        PCASampleFeaturizer(view="")
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        PCASampleFeaturizer(view="gene_expression").transform(features, _PCA_IDS)


def test_pca_pickle_round_trip_transforms_identically() -> None:
    features = _pca_features()
    featurizer = PCASampleFeaturizer(view="gene_expression", n_components=2).fit(features, _PCA_IDS)
    np.testing.assert_allclose(
        featurizer.transform(features, _PCA_IDS).matrix, pickled(featurizer).transform(features, _PCA_IDS).matrix
    )


def test_pca_no_longer_zero_fills_missing_features() -> None:
    training = random_views({"methylation": ["a", "b", "c"]}, n_samples=3)
    featurizer = PCASampleFeaturizer(view="methylation", n_components=2)
    source = mock_source(training)
    featurizer.fit(source, source.identifiers)
    cross_study = mock_source(random_views({"methylation": ["b", "c", "d"]}, n_samples=1))

    with pytest.raises(FeatureMismatchError, match=r"view 'methylation' is missing 1 of the 3 features.*'a'"):
        featurizer.transform(cross_study, cross_study.identifiers)


def _proteomics_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
            "cl3": {"proteomics": np.array([7.0, 8.0, 9.0], dtype=np.float32)},
        }
    )


def test_normalized_proteomics_fit_transform_and_state_round_trip() -> None:
    features = _proteomics_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer()
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer.transform(features, entity_ids)
    featurizer.fit(features, entity_ids)

    matrix = featurizer.transform(features, entity_ids).matrix

    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(pickled(featurizer).transform(features, entity_ids).matrix, matrix)


def test_log10_replaces_infinities_with_nan() -> None:
    transformed = log10_and_set_na(np.array([[1.0, 0.0, 100.0]]))

    np.testing.assert_allclose(transformed[0, [0, 2]], [0.0, 2.0])
    assert np.isnan(transformed[0, 1])


def test_normalized_proteomics_selects_features_and_imputes_missing_values() -> None:
    thresholded = NormalizedProteomicsSampleFeaturizer(proteomics_n_features=1)
    ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    thresholded.fit(_proteomics_features(), ids)
    assert thresholded.transform(_proteomics_features(), ids).matrix.shape == (3, 3)

    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 0.0, 100.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([10.0, 100.0, 1000.0], dtype=np.float32)},
        }
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer().fit(features, entity_ids)

    assert not np.isnan(featurizer.transform(features, entity_ids).matrix).any()


def test_normalized_proteomics_blocks_carry_feature_names() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        },
        meta_info={"proteomics": ["p1", "p2", "p3"]},
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer().fit(features, entity_ids)

    blocks = featurizer.transform(features, entity_ids).blocks

    assert set(blocks) == {"proteomics"}
    retained = featurizer._transformer.protein_indices
    assert blocks["proteomics"].feature_names == tuple(np.array(["p1", "p2", "p3"])[retained])
    assert sorted(blocks["proteomics"].feature_names) == ["p1", "p2", "p3"]


def test_normalized_proteomics_names_only_the_retained_proteins() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, np.nan], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, np.nan], dtype=np.float32)},
        },
        meta_info={"proteomics": ["p1", "p2", "p3"]},
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer(proteomics_feature_threshold=1.0, proteomics_n_features=1)

    result = featurizer.fit(features, entity_ids).transform(features, entity_ids)

    assert result.blocks["proteomics"].feature_names == ("p1", "p2")
    assert result.matrix.shape == (2, 2)


_MATRIX = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.mark.parametrize(("n_features", "kept"), [(1, [0, 1]), (3, [0, 1, 2])])
def test_fit_keeps_complete_features_then_falls_back_to_the_n_most_complete(n_features: int, kept: list[int]) -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=n_features)

    assert transformer.fit(_MATRIX) is transformer
    assert sorted(transformer.protein_indices.tolist()) == kept


def test_median_centering_uses_the_mean_of_the_per_row_medians() -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=1).fit(_MATRIX)

    np.testing.assert_allclose(transformer.mean_median, 4.5)  # medians over columns 0 and 1: 1.5, 4.5, 7.5
    (row,) = transformer.transform(np.array([[4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(row, [4.0, 5.0])


def test_imputation_fills_missing_values_reproducibly() -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.0, n_features=3).fit(_MATRIX)

    (first,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))
    (second,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))

    assert not np.isnan(first).any()
    np.testing.assert_allclose(first, second)


def test_the_transformer_satisfies_the_sklearn_estimator_protocol() -> None:
    from sklearn.base import clone

    params = ProteomicsMedianCenterAndImputeTransformer(n_features=7).get_params()
    assert params["n_features"] == 7
    assert set(params) == {
        "feature_threshold",
        "imputation_seed",
        "n_features",
        "normalization_downshift",
        "normalization_width",
    }
    cloned = clone(ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.5))
    assert cloned.feature_threshold == 0.5
    assert cloned.protein_indices.size == 0


def _tissue(**per_sample) -> MockFeatureSource:
    return MockFeatureSource(
        features={cl: {key: np.array([value]) for key, value in view.items()} for cl, view in per_sample.items()}
    )


_TISSUE_IDS = np.array(["cl1", "cl2"], dtype=str)


def test_tissue_one_hot_blocks_and_pickle_round_trip() -> None:
    features = _tissue(cl1={"tissue": "lung"}, cl2={"tissue": "skin"})
    featurizer = TissueFeaturizer()
    featurizer.fit(features, _TISSUE_IDS)

    result = featurizer.transform(features, _TISSUE_IDS)

    assert result.matrix.shape == (2, 2)
    assert list(result.blocks["tissue_categories"].values) == ["lung", "skin"]
    assert TissueFeaturizer.output_blocks({}) == (BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX),)
    np.testing.assert_allclose(pickled(featurizer).transform(features, _TISSUE_IDS).matrix, result.matrix)
    assert featurizer.transform(features, np.array([], dtype=str)).matrix.shape == (0, 2)


def test_tissue_strict_mode_raises_on_missing_annotations() -> None:
    features = _tissue(cl1={"tissue": "lung"}, cl2={"gene_expression": 1.0})
    with pytest.raises(ValueError, match="requires tissue"):
        TissueFeaturizer(allow_missing=False).fit(features, _TISSUE_IDS)
    fitted = TissueFeaturizer(allow_missing=False).fit(features, _TISSUE_IDS[:1])
    with pytest.raises(ValueError, match="requires tissue"):
        fitted.transform(features, _TISSUE_IDS)
    with pytest.raises(ValueError, match="requires tissue"):
        TissueFeaturizer(allow_missing=False).fit(MockFeatureSource(features={}), np.array([], dtype=str))


@pytest.mark.parametrize(
    ("per_sample", "width"),
    [
        ({"cl1": {"tissue": "lung"}, "cl2": {"gene_expression": 1.0}}, 1),
        ({"cl1": {"tissue": "lung"}, "cl2": {"tissue": np.nan}}, 1),
        ({"cl1": {"tissue": "lung"}, "cl2": {"tissue": None}}, 1),
        ({"cl1": {"tissue": None}, "cl2": {"tissue": None}}, 0),
        ({"cl1": {"gene_expression": 1.0}, "cl2": {"gene_expression": 2.0}}, 0),
    ],
    ids=["partial", "nan-label", "none-label", "all-none", "fully-absent"],
)
def test_tissue_allow_missing_gives_zero_rows(per_sample, width) -> None:
    features = _tissue(**per_sample)
    featurizer = TissueFeaturizer(allow_missing=True)
    featurizer.fit(features, _TISSUE_IDS)

    matrix = featurizer.transform(features, _TISSUE_IDS).matrix

    assert matrix.shape == (2, width)
    if width:
        np.testing.assert_array_equal(matrix[:, 0], [1.0, 0.0])


_GENES = list("abcdef")
_OMICS_VIEWS = {
    "gene_expression": _GENES,
    "mutations": ["m1", "m2", "m3", "m4"],
    "copy_number_variation_gistic": ["c1", "c2", "c3", "c4"],
}
_ALIGNMENT_CASES = [
    pytest.param(
        lambda: PCASampleFeaturizer(view="methylation", n_components=2),
        {"methylation": _GENES},
        {"drop_views": ("methylation",)},
        id="pca",
    ),
    pytest.param(
        lambda: NormalizedProteomicsSampleFeaturizer(proteomics_n_features=4),
        {"proteomics": _GENES},
        {"drop_views": ("proteomics",)},
        id="normalized_proteomics",
    ),
    pytest.param(ScaledGeneExpressionFeaturizer, {"gene_expression": _GENES}, {}, id="scaled_expression"),
    pytest.param(PharmaFormerGeneExpressionFeaturizer, {"gene_expression": _GENES}, {}, id="pharmaformer"),
    pytest.param(
        lambda: MOLIROmicsFeaturizer(n_gene_expression_features=3),
        {**_OMICS_VIEWS, "mutations": ["m1", "m2", "m3"], "copy_number_variation_gistic": ["c1", "c2", "c3"]},
        {"drop_views": tuple(_OMICS_VIEWS)},
        id="molir",
    ),
    pytest.param(
        lambda: SuperFELTROmicsFeaturizer(n_features_per_view=3),
        _OMICS_VIEWS,
        {"drop_views": tuple(_OMICS_VIEWS)},
        id="superfeltr",
    ),
]


@pytest.mark.parametrize(("make", "names_by_view", "options"), _ALIGNMENT_CASES)
def test_dense_featurizers_read_every_view_by_feature_name(make, names_by_view, options) -> None:
    views = random_views(names_by_view)
    for drop_view in options.get("drop_views", ("gene_expression",)):
        assert_reads_by_name(make, views, drop_view=drop_view)
