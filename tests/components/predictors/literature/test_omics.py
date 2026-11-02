"""Tests for the three-omic helpers, per-drug predictor shell and loaders shared by MOLIR and SuperFELTR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from drevalpy.components.predictors.literature._omic_helpers import (
    OMIC_BLOCK_NAMES,
    OmicFeatureNames,
    OmicMatrices,
    aligned_pair_matrices,
    omic_feature_names,
    omic_matrices,
    validation_split,
)
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors.literature._helpers import two_by_two_batch

EXPRESSION = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
MUTATIONS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
CNV = np.array([[0.5], [1.5]], dtype=np.float32)
EXPRESSION_FEATURES = ("geneA", "geneB", "geneC")
MUTATION_FEATURES = ("mutA", "mutB")
CNV_FEATURES = ("cnvA",)


def _batch(
    *,
    early_stopping_response: ResponseBatch | None = None,
    expression: np.ndarray = EXPRESSION,
    expression_features: tuple[str, ...] = EXPRESSION_FEATURES,
    checkpoint_dir: str = "/base",
) -> ModelInputBatch:
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    return two_by_two_batch(
        response=response,
        sample_blocks={
            "gene_expression": numeric_feature_block(expression, feature_names=expression_features),
            "mutations": numeric_feature_block(MUTATIONS, feature_names=MUTATION_FEATURES),
            "copy_number_variation_gistic": numeric_feature_block(CNV, feature_names=CNV_FEATURES),
        },
        drug_blocks={
            "oneHot": numeric_feature_block(np.eye(2, dtype=np.float32)),
            "oneHot_categories": numeric_feature_block(np.array([["d1", "d2"]], dtype=object)),
        },
        early_stopping_response=early_stopping_response,
        checkpoint_dir=checkpoint_dir,
    )


def _matrices() -> OmicMatrices:
    return OmicMatrices(gene_expression=EXPRESSION, mutations=MUTATIONS, copy_number_variation=CNV)


def _trained_names() -> OmicFeatureNames:
    return OmicFeatureNames(
        gene_expression=EXPRESSION_FEATURES, mutations=MUTATION_FEATURES, copy_number_variation=CNV_FEATURES
    )


def _validation(sample_ids: list[str], values: list[float]) -> ResponseBatch:
    return ResponseBatch(
        response=np.array(values), sample_ids=np.array(sample_ids), drug_ids=np.repeat("d1", len(values))
    )


def test_matrices_are_read_from_the_batch_and_split_by_pair() -> None:
    assert OMIC_BLOCK_NAMES == ("gene_expression", "mutations", "copy_number_variation_gistic")
    matrices = omic_matrices(_batch())
    assert matrices.gene_expression.dtype == np.float32
    np.testing.assert_allclose(matrices.gene_expression, EXPRESSION)
    np.testing.assert_allclose(matrices.mutations, MUTATIONS)
    np.testing.assert_allclose(matrices.copy_number_variation, CNV)
    assert _matrices().widths() == (3, 2, 1)

    pair_idx = np.array([0, 1, 1])
    response = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    split = _matrices().split(pair_idx, response)
    assert split.gene_expression is EXPRESSION
    assert split.copy_number is CNV
    np.testing.assert_array_equal(split.pair_idx, pair_idx)
    np.testing.assert_allclose(split.response, response)


def test_feature_names_are_recorded_and_frozen() -> None:
    assert omic_feature_names(_batch()).as_tuple() == (EXPRESSION_FEATURES, MUTATION_FEATURES, CNV_FEATURES)
    with pytest.raises(AttributeError):
        _trained_names().gene_expression = ("other",)  # type: ignore[misc]


def test_early_stopping_needs_at_least_two_validation_pairs() -> None:
    assert validation_split(_matrices(), _batch()) is None
    assert validation_split(_matrices(), _batch(early_stopping_response=_validation(["cl1"], [1.0]))) is None


def test_validation_samples_map_onto_entity_rows_and_share_matrices() -> None:
    batch = _batch(early_stopping_response=_validation(["cl2", "cl1"], [7.0, 8.0]))
    matrices = _matrices()
    split = validation_split(matrices, batch)
    assert split is not None
    assert split.gene_expression is matrices.gene_expression
    np.testing.assert_array_equal(split.pair_idx, [1, 0])
    np.testing.assert_allclose(split.response, [7.0, 8.0])
    assert split.response.dtype == np.float32


def test_matching_features_are_expanded_to_pair_level_unchanged() -> None:
    matrices = aligned_pair_matrices(_batch(), _trained_names())

    for matrix, source in zip(matrices, (EXPRESSION, MUTATIONS, CNV), strict=True):
        assert matrix.dtype == np.float32
        np.testing.assert_allclose(matrix, source[[0, 0, 1, 1]])
    unknown, _, _ = aligned_pair_matrices(_batch(), OmicFeatureNames(None, None, None))
    np.testing.assert_allclose(unknown, EXPRESSION[[0, 0, 1, 1]])


def test_a_narrower_batch_is_realigned_onto_the_trained_columns() -> None:
    narrow = np.array([[1.0, 3.0], [4.0, 6.0]], dtype=np.float32)
    batch = _batch(expression=narrow, expression_features=("geneA", "geneC"))

    expression, _, _ = aligned_pair_matrices(batch, _trained_names())

    assert expression.shape == (4, 3)
    np.testing.assert_allclose(expression[0], [1.0, 0.0, 3.0])
    np.testing.assert_allclose(expression[2], [4.0, 0.0, 6.0])


def test_a_same_width_reordering_is_aligned_to_the_training_columns() -> None:
    batch = _batch(expression=EXPRESSION[:, [2, 0, 1]], expression_features=("geneC", "geneA", "geneB"))

    expression, _, _ = aligned_pair_matrices(batch, _trained_names())

    np.testing.assert_allclose(expression, EXPRESSION[[0, 0, 1, 1]])


@pytest.mark.parametrize("predictor_cls", [MOLIRPredictor, SuperFELTRPredictor])
def test_the_shell_is_shared_and_routes_by_drug_id(predictor_cls: type[SingleDrugOmicsPredictor]) -> None:
    assert issubclass(predictor_cls, SingleDrugOmicsPredictor)
    assert tuple(spec.name for spec in predictor_cls.sample_inputs.specs) == OMIC_BLOCK_NAMES
    assert predictor_cls.drug_inputs.kind == "none"
    assert predictor_cls().is_fitted() is False


class _RecordingPredictor(SingleDrugOmicsPredictor):
    def _fit_drug(self, drug_id: str, batch: Any) -> Any:
        return None if drug_id == "d2" else batch


def test_feature_names_are_recorded_once_and_drugs_without_a_model_are_skipped() -> None:
    predictor = _RecordingPredictor()

    predictor.fit(_batch())

    assert list(predictor._models) == ["d1"]
    assert predictor._feature_names.as_tuple() == (EXPRESSION_FEATURES, MUTATION_FEATURES, CNV_FEATURES)


def test_a_drug_batch_keeps_its_early_stopping_pairs_and_their_samples() -> None:
    stopping = ResponseBatch(np.array([7.0, 8.0, 9.0]), np.array(["cl2", "cl1", "cl2"]), np.array(["d1", "d1", "d2"]))
    predictor = _RecordingPredictor()

    predictor.fit(_batch(early_stopping_response=stopping))

    drug_batch = predictor._models["d1"]
    split = validation_split(omic_matrices(drug_batch), drug_batch)
    assert split is not None
    np.testing.assert_allclose(split.response, [7.0, 8.0])
    np.testing.assert_allclose(split.gene_expression[split.pair_idx], EXPRESSION[[1, 0]])


class _ExpressionSum:
    def predict(self, expression: np.ndarray, mutations: np.ndarray, copy_number: np.ndarray) -> np.ndarray:
        return expression.sum(axis=1)


class _ExpressionSumPredictor(SingleDrugOmicsPredictor):
    def _fit_drug(self, drug_id: str, batch: Any) -> _ExpressionSum:
        return _ExpressionSum()


def test_drugs_are_scored_on_columns_realigned_to_the_training_order() -> None:
    predictor = _ExpressionSumPredictor()
    predictor.fit(_batch())

    narrow = _batch(expression=EXPRESSION[:, [2, 0]], expression_features=("geneC", "geneA"))

    np.testing.assert_allclose(predictor.predict(narrow), EXPRESSION[[0, 0, 1, 1]][:, [0, 2]].sum(axis=1))


LOADER_ENTITIES = 4
LOADER_EXPR_DIM = 3
LOADER_MUT_DIM = 2
LOADER_CNV_DIM = 1


def _loader_split(n_pairs: int = 5, *, two_dimensional_response: bool = False) -> OmicsSplit:
    response = np.linspace(0.0, 1.0, n_pairs, dtype=np.float32)
    return OmicsSplit(
        gene_expression=np.arange(LOADER_ENTITIES * LOADER_EXPR_DIM, dtype=np.float32).reshape(
            LOADER_ENTITIES, LOADER_EXPR_DIM
        ),
        mutations=np.ones((LOADER_ENTITIES, LOADER_MUT_DIM), dtype=np.float32),
        copy_number=np.zeros((LOADER_ENTITIES, LOADER_CNV_DIM), dtype=np.float32),
        response=response.reshape(-1, 1) if two_dimensional_response else response,
        pair_idx=np.arange(n_pairs, dtype=np.int64) % LOADER_ENTITIES,
    )


@pytest.mark.slow
def test_each_batch_carries_the_three_views_and_a_response_column() -> None:
    split = _loader_split(n_pairs=4)
    train_loader, val_loader = make_omics_loaders(split, None, batch_size=4)

    assert val_loader is None
    expression, mutations, copy_number, response = next(iter(train_loader))
    assert (expression.shape, mutations.shape, copy_number.shape) == (
        (4, LOADER_EXPR_DIM),
        (4, LOADER_MUT_DIM),
        (4, LOADER_CNV_DIM),
    )
    assert response.shape == (4, 1)
    np.testing.assert_allclose(expression.numpy(), split.gene_expression[split.pair_idx])


@pytest.mark.slow
def test_training_drops_the_trailing_incomplete_batch_but_validation_keeps_it() -> None:
    train_loader, val_loader = make_omics_loaders(_loader_split(n_pairs=5), _loader_split(n_pairs=5), batch_size=2)

    assert sum(batch[0].shape[0] for batch in train_loader) == 4
    assert val_loader is not None
    assert sum(batch[0].shape[0] for batch in val_loader) == 5


@pytest.mark.slow
def test_an_already_two_dimensional_response_is_passed_through() -> None:
    train_loader, _ = make_omics_loaders(_loader_split(two_dimensional_response=True), None, batch_size=2)

    assert next(iter(train_loader))[3].shape == (2, 1)


@pytest.mark.slow
def test_neither_loader_shuffles_and_the_split_is_frozen() -> None:
    split = _loader_split(n_pairs=4)
    train_loader, _ = make_omics_loaders(split, None, batch_size=1)

    assert [float(batch[3].item()) for batch in train_loader] == pytest.approx(split.response.tolist())
    with pytest.raises(AttributeError):
        split.pair_idx = np.zeros(1, dtype=np.int64)  # type: ignore[misc]
