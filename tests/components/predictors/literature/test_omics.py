"""Tests for the three-omic helpers, per-drug predictor shell and loaders shared by MOLIR and SuperFELTR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from drevalpy.components.predictors.literature._omics import (
    OMIC_BLOCK_NAMES,
    SingleDrugOmicsPredictor,
    aligned_pair_matrices,
    make_omics_loaders,
    omic_feature_names,
)
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from tests.components.predictors.literature._helpers import two_by_two_batch

EXPRESSION = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
MUTATIONS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
CNV = np.array([[0.5], [1.5]], dtype=np.float32)
EXPRESSION_FEATURES = ("geneA", "geneB", "geneC")
MUTATION_FEATURES = ("mutA", "mutB")
CNV_FEATURES = ("cnvA",)


def _batch(
    *,
    early_stopping: ResponseTable | None = None,
    expression: np.ndarray = EXPRESSION,
    expression_features: tuple[str, ...] = EXPRESSION_FEATURES,
    checkpoint_dir: str = "/base",
) -> PredictorInput:
    response = ResponseTable(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    return two_by_two_batch(
        response=response,
        sample_blocks={
            "gene_expression": FeatureBlock(expression, feature_names=expression_features),
            "mutations": FeatureBlock(MUTATIONS, feature_names=MUTATION_FEATURES),
            "copy_number_variation_gistic": FeatureBlock(CNV, feature_names=CNV_FEATURES),
        },
        drug_blocks={
            "oneHot": FeatureBlock(np.eye(2, dtype=np.float32)),
            "oneHot_categories": FeatureBlock(np.array([["d1", "d2"]], dtype=object)),
        },
        early_stopping=early_stopping,
        checkpoint_dir=checkpoint_dir,
    )


def _trained_names() -> dict[str, tuple[str, ...] | None]:
    return dict(zip(OMIC_BLOCK_NAMES, (EXPRESSION_FEATURES, MUTATION_FEATURES, CNV_FEATURES), strict=True))


def test_feature_names_are_recorded_by_block_name() -> None:
    assert OMIC_BLOCK_NAMES == ("gene_expression", "mutations", "copy_number_variation_gistic")
    assert omic_feature_names(_batch()) == _trained_names()


def test_matching_features_are_expanded_to_pair_level_unchanged() -> None:
    matrices = aligned_pair_matrices(_batch(), _trained_names())

    for matrix, source in zip(matrices, (EXPRESSION, MUTATIONS, CNV), strict=True):
        assert matrix.dtype == np.float32
        np.testing.assert_allclose(matrix, source[[0, 0, 1, 1]])
    unknown, _, _ = aligned_pair_matrices(_batch(), {})
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
    assert predictor._feature_names == _trained_names()


def test_a_drug_batch_keeps_its_early_stopping_pairs_and_their_samples() -> None:
    stopping = ResponseTable(np.array([7.0, 8.0, 9.0]), np.array(["cl2", "cl1", "cl2"]), np.array(["d1", "d1", "d2"]))
    predictor = _RecordingPredictor()

    predictor.fit(_batch(early_stopping=stopping))

    drug_batch = predictor._models["d1"]
    assert drug_batch.stopping is not None
    np.testing.assert_allclose(drug_batch.stopping.response, [7.0, 8.0])
    np.testing.assert_allclose(
        drug_batch.stopping.sample.blocks["gene_expression"].values[drug_batch.stopping.sample_pair_idx],
        EXPRESSION[[1, 0]],
    )


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


def _loader_batch(n_pairs: int = 5, *, val_pairs: int | None = None) -> PredictorInput:
    entity_ids = np.array([f"s{i}" for i in range(LOADER_ENTITIES)])
    expression = np.arange(LOADER_ENTITIES * LOADER_EXPR_DIM, dtype=np.float32).reshape(
        LOADER_ENTITIES, LOADER_EXPR_DIM
    )
    sample = FeaturizerOutput(
        entity_ids=entity_ids,
        blocks={
            "gene_expression": FeatureBlock(expression),
            "mutations": FeatureBlock(np.ones((LOADER_ENTITIES, LOADER_MUT_DIM), dtype=np.float32)),
            "copy_number_variation_gistic": FeatureBlock(np.zeros((LOADER_ENTITIES, LOADER_CNV_DIM), dtype=np.float32)),
        },
    )
    pairs = ResponseTable(
        response=np.linspace(0.0, 1.0, n_pairs, dtype=np.float32),
        sample_ids=entity_ids[np.arange(n_pairs) % LOADER_ENTITIES],
        drug_ids=np.repeat("d1", n_pairs),
    )
    early_stopping = (
        None
        if val_pairs is None
        else ResponseTable(
            response=np.linspace(1.0, 2.0, val_pairs, dtype=np.float32),
            sample_ids=entity_ids[np.arange(val_pairs) % LOADER_ENTITIES],
            drug_ids=np.repeat("d1", val_pairs),
        )
    )
    return PredictorInput(pairs=pairs, sample=sample, early_stopping=early_stopping)


@pytest.mark.slow
def test_each_batch_carries_the_three_views_and_response() -> None:
    batch = _loader_batch(n_pairs=4)
    train_loader, val_loader = make_omics_loaders(batch, batch_size=4)

    assert val_loader is None
    expression, mutations, copy_number, response = next(iter(train_loader))
    assert (expression.shape, mutations.shape, copy_number.shape) == (
        (4, LOADER_EXPR_DIM),
        (4, LOADER_MUT_DIM),
        (4, LOADER_CNV_DIM),
    )
    assert response.shape == (4,)
    np.testing.assert_allclose(expression.numpy(), batch.sample.blocks["gene_expression"].values[batch.sample_pair_idx])


@pytest.mark.slow
def test_early_stopping_needs_at_least_two_validation_pairs() -> None:
    _, without = make_omics_loaders(_loader_batch(n_pairs=4), batch_size=2)
    _, single = make_omics_loaders(_loader_batch(n_pairs=4, val_pairs=1), batch_size=2)

    assert without is None
    assert single is None


@pytest.mark.slow
def test_training_drops_the_trailing_incomplete_batch_but_validation_keeps_it() -> None:
    train_loader, val_loader = make_omics_loaders(_loader_batch(n_pairs=5, val_pairs=5), batch_size=2)

    assert sum(batch[0].shape[0] for batch in train_loader) == 4
    assert val_loader is not None
    assert sum(batch[0].shape[0] for batch in val_loader) == 5


@pytest.mark.slow
def test_neither_loader_shuffles() -> None:
    batch = _loader_batch(n_pairs=4, val_pairs=3)
    train_loader, val_loader = make_omics_loaders(batch, batch_size=1)

    assert [float(item[3].item()) for item in train_loader] == pytest.approx(batch.response.tolist())
    assert val_loader is not None
    assert batch.stopping is not None
    assert [float(item[3].item()) for item in val_loader] == pytest.approx(batch.stopping.response.tolist())
