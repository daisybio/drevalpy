"""Tests for the shared concat, constant and identity featurizers, each registered once for both sides."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers.sample.view import ViewSampleFeaturizer
from drevalpy.components.featurizers.shared.concat import ConcatFeaturizer
from drevalpy.components.featurizers.shared.constant import ConstantFeaturizer
from drevalpy.components.featurizers.shared.one_hot import OneHotFeaturizer
from drevalpy.models._feature_preparation import FitIds, prepare_features
from drevalpy.models.config import FeaturizerConfig
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, ragged_feature_block
from tests.conftest import MockFeatureSource

_IDS = np.array(["cl1", "cl2"], dtype=str)
_DRUG_IDS = np.array(["drug1", "drug2"], dtype=str)


def _sample_features(second_view: str = "mutations", width: int = 2) -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                second_view: np.arange(width, dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                second_view: np.arange(width, dtype=np.float32)[::-1].copy(),
            },
            "cl3": {
                "gene_expression": np.array([np.nan, np.nan], dtype=np.float32),
                second_view: np.ones(width, dtype=np.float32),
            },
        },
        meta_info={
            "gene_expression": ["g1", "g2"],
            "mutations": ["m1", "m2"],
            second_view: [f"f{i}" for i in range(width)],
        },
    )


def _drug_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "drug1": {"morgan_fingerprint": np.array([1.0, 0.0, 1.0], dtype=np.float32)},
            "drug2": {"morgan_fingerprint": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2", "fp3"]},
        side="drug",
    )


def _run(featurizer, source, ids) -> FeaturizerResult:
    return prepare_features(featurizer, source, ids, fit=FitIds(ids))


def _raw_configs() -> list[FeaturizerConfig]:
    return [FeaturizerConfig(name="view", side="sample", view=view) for view in ("gene_expression", "mutations")]


@pytest.mark.parametrize(
    ("name", "cls"),
    [("concatFeaturizers", ConcatFeaturizer), ("constant", ConstantFeaturizer), ("oneHot", OneHotFeaturizer)],
)
def test_each_shared_featurizer_is_registered_once_per_side(name: str, cls: type) -> None:
    sample, drug = get_featurizer(name, "sample"), get_featurizer(name, "drug")
    assert sample is not drug
    assert issubclass(sample, cls) and issubclass(drug, cls)
    assert (sample.side, drug.side) == ("sample", "drug")


def test_sample_concat_merges_the_children_blocks() -> None:
    result = _run(ConcatFeaturizer(featurizers=_raw_configs()), _sample_features(), _IDS)

    assert result.matrix.shape == (2, 4)
    assert set(result.blocks) == {"gene_expression", "mutations"}
    assert result.blocks["gene_expression"].feature_names == ("g1", "g2")
    np.testing.assert_allclose(
        result.matrix, np.concatenate([result.blocks["gene_expression"].values, result.blocks["mutations"].values], 1)
    )


def test_drug_concat_builds_its_children_for_the_drug_side() -> None:
    featurizer = ConcatFeaturizer(
        featurizers=[FeaturizerConfig(name="fingerprints", side="drug"), FeaturizerConfig(name="oneHot", side="drug")]
    )

    result = _run(featurizer, _drug_features(), _DRUG_IDS)

    assert result.matrix.shape == (2, 5)
    assert set(result.blocks) == {"fingerprints", "oneHot", "oneHot_categories"}


def test_concat_pads_entities_unusable_for_any_child() -> None:
    featurizer = ConcatFeaturizer(featurizers=_raw_configs())
    _run(featurizer, _sample_features(), np.array(["cl1"], dtype=str))

    result = prepare_features(featurizer, _sample_features(), np.array(["cl1", "cl3"], dtype=str), fit=None)

    assert np.isnan(result.matrix[1]).all()
    np.testing.assert_allclose(result.matrix[0], [1.0, 2.0, 0.0, 1.0])


def test_concat_of_the_same_featurizer_on_two_views_names_blocks_by_view() -> None:
    featurizer = ConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="pca", side="sample", view="gene_expression").create_instance({"n_components": 1}),
            FeaturizerConfig(name="pca", side="sample", view="proteomics").create_instance({"n_components": 1}),
        ],
    )

    result = _run(featurizer, _sample_features("proteomics", 3), _IDS)

    assert set(result.blocks) == {"gene_expression", "proteomics"}
    assert result.matrix.shape == (2, 2)


def test_concat_rejects_duplicate_block_names_and_selectors() -> None:
    featurizer = ConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="view", side="sample", view="gene_expression"),
            FeaturizerConfig(name="scaledGeneExpression", side="sample"),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate featurizer block name 'gene_expression'"):
        _run(featurizer, _sample_features(), _IDS)

    duplicate = FeaturizerConfig(name="view", side="sample", view="gene_expression")
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'view\\[gene_expression\\]'"):
        ConcatFeaturizer(featurizers=[duplicate, duplicate])


def test_concat_rejects_non_numeric_children_and_empty_lists() -> None:
    class _Tokens(ConstantFeaturizer):
        def transform(self, source, ids) -> FeaturizerResult:
            values = np.empty(len(ids), dtype=object)
            values[:] = [np.arange(2) for _ in ids]
            return FeaturizerResult(ids, {"tokens": ragged_feature_block(values)}, ())

    featurizer = ConcatFeaturizer(featurizers=[OneHotFeaturizer(), _Tokens()])
    with pytest.raises(ValueError, match="child _Tokens emits non-numeric blocks \\['tokens'\\]"):
        _run(featurizer, _drug_features(), _DRUG_IDS)
    with pytest.raises(ValueError, match="non-empty list"):
        ConcatFeaturizer(featurizers=[])


def test_concat_accepts_pre_built_featurizer_instances() -> None:
    featurizer = ConcatFeaturizer(
        featurizers=[ViewSampleFeaturizer(view="gene_expression"), ViewSampleFeaturizer(view="mutations")]
    )

    assert _run(featurizer, _sample_features(), _IDS).matrix.shape == (2, 4)


def test_concat_requires_early_stopping_when_a_child_does() -> None:
    child = type("OneHotFeaturizer", (OneHotFeaturizer,), {"requires_early_stopping": True})()
    featurizer = ConcatFeaturizer(featurizers=[child])

    with pytest.raises(ValueError, match="OneHotFeaturizer requires a usable early-stopping holdout"):
        featurizer.fit(_sample_features(), _IDS)


def test_a_fitted_concat_pickles() -> None:
    features = _sample_features()
    featurizer = ConcatFeaturizer(featurizers=_raw_configs())
    expected = _run(featurizer, features, _IDS)

    restored = pickle.loads(pickle.dumps(featurizer))  # noqa: S301 -- pickled in the same expression

    np.testing.assert_allclose(prepare_features(restored, features, _IDS, fit=None).matrix, expected.matrix)


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_constant_is_a_ones_column(side: str) -> None:
    ids = np.array(["e1", "e2", "e3"], dtype=str)

    result = _run(ConstantFeaturizer(), MockFeatureSource({}, side=side), ids)

    assert result.matrix.shape == (3, 1)
    assert result.matrix.dtype == np.float32
    assert np.allclose(result.matrix, 1.0)
    assert list(result.blocks) == ["constant"]
    assert {spec.name for spec in ConstantFeaturizer.output_blocks({})} == {"constant"}


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_identity_one_hot_blocks_and_pickling(side: str) -> None:
    source = MockFeatureSource({}, side=side)
    featurizer = OneHotFeaturizer()
    featurizer.fit(source, np.array(["e2", "e1"], dtype=str))
    ids = np.array(["e2", "e1", "e3"], dtype=str)

    result = prepare_features(featurizer, source, ids, fit=None)
    restored = pickle.loads(pickle.dumps(featurizer))  # noqa: S301 -- pickled in the same expression

    np.testing.assert_array_equal(result.matrix, [[0, 1], [1, 0], [0, 0]])
    assert set(result.blocks) == {"oneHot", "oneHot_categories"}
    assert list(result.blocks["oneHot_categories"].values) == ["e1", "e2"]
    np.testing.assert_array_equal(prepare_features(restored, source, ids, fit=None).matrix, result.matrix)
