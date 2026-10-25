"""Tests for the three-omic helpers, per-drug predictor shell and loaders shared by MOLIR and SuperFELTR."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from upath import UPath

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature._omic_helpers import (
    OMIC_BLOCK_NAMES,
    OmicFeatureNames,
    OmicMatrices,
    aligned_pair_matrices,
    checkpoint_dir_for_drug,
    early_stopping_indices,
    feature_names_from_payload,
    feature_names_payload,
    iter_drug_subsets,
    omic_feature_names,
    omic_matrices,
    restore_algorithms,
    validation_split,
)
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
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
            "identity": numeric_feature_block(np.eye(2, dtype=np.float32)),
            "identity_categories": numeric_feature_block(np.array([["d1", "d2"]], dtype=object)),
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


def test_feature_names_are_recorded_and_round_trip_through_the_payload() -> None:
    assert omic_feature_names(_batch()).as_tuple() == (EXPRESSION_FEATURES, MUTATION_FEATURES, CNV_FEATURES)
    payload = feature_names_payload(_trained_names())
    assert payload == {
        "gene_expression_features": list(EXPRESSION_FEATURES),
        "mutations_features": list(MUTATION_FEATURES),
        "copy_number_variation_features": list(CNV_FEATURES),
    }
    assert feature_names_from_payload(payload) == _trained_names()
    assert set(feature_names_payload(None).values()) == {None}
    assert feature_names_from_payload({}) == OmicFeatureNames(None, None, None)
    with pytest.raises(AttributeError):
        _trained_names().gene_expression = ("other",)  # type: ignore[misc]


def test_checkpoint_dir_is_derived_per_drug() -> None:
    directory = checkpoint_dir_for_drug(UPath("/base"), "drug-1")

    assert directory.parent == UPath("/base")
    assert directory.name.startswith("drug_")
    assert directory == checkpoint_dir_for_drug(UPath("/base"), "drug-1")
    assert directory != checkpoint_dir_for_drug(UPath("/base"), "drug-2")
    assert checkpoint_dir_for_drug(UPath("/base"), "a/b c:*?").parent == UPath("/base")


def test_iter_drug_subsets_yields_one_subset_and_checkpoint_dir_per_drug() -> None:
    subsets = dict(iter_drug_subsets(_batch()))

    assert sorted(subsets) == ["d1", "d2"]
    assert subsets["d1"].n_pairs == 2
    np.testing.assert_array_equal(subsets["d1"].drug_ids, ["d1", "d1"])
    directories = {UPath(sub.training_context.checkpoint_dir) for sub in subsets.values()}
    assert len(directories) == 2
    assert all(d.parent == UPath("/base") and d.name.startswith("drug_") for d in directories)


def test_early_stopping_needs_at_least_two_validation_pairs() -> None:
    assert early_stopping_indices(_batch()) == (None, None)
    assert validation_split(_matrices(), _batch()) is None
    assert early_stopping_indices(_batch(early_stopping_response=_validation(["cl1"], [1.0]))) == (None, None)


def test_validation_samples_map_onto_entity_rows_and_share_matrices() -> None:
    batch = _batch(early_stopping_response=_validation(["cl2", "cl1"], [7.0, 8.0]))
    pair_idx, values = early_stopping_indices(batch)
    np.testing.assert_array_equal(pair_idx, [1, 0])
    np.testing.assert_allclose(values, [7.0, 8.0])

    matrices = _matrices()
    split = validation_split(matrices, batch)
    assert split is not None
    assert split.gene_expression is matrices.gene_expression
    np.testing.assert_array_equal(split.pair_idx, [1, 0])


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


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "algorithms": {"d1": save_trusted_mapping({"ranges": (0.0, 1.0)})},
        "predictor_hyperparameters": {"epochs": 3},
    }
    state.update(overrides)
    return state


def test_restore_algorithms_decodes_payloads_and_hyperparameters_per_drug() -> None:
    hyperparameters, payloads = restore_algorithms(_state(), "Toy")

    assert hyperparameters == {"epochs": 3}
    assert payloads == {"d1": {"ranges": (0.0, 1.0)}}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithms": None}, "requires an 'algorithms' mapping"),
        ({"predictor_hyperparameters": None}, "requires predictor_hyperparameters"),
        ({"algorithms": {"d1": "text"}}, "'d1' must be bytes"),
        ({"algorithms": {"d1": b"junk"}}, "'d1' could not be deserialized"),
    ],
)
def test_malformed_state_is_a_predictor_state_error(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(PredictorStateError, match=message):
        restore_algorithms(_state(**overrides), "Toy")


@pytest.mark.parametrize("predictor_cls", [MOLIRPredictor, SuperFELTRPredictor])
def test_the_shell_is_shared_and_a_failed_restore_leaves_the_predictor_untouched(
    predictor_cls: type[SingleDrugOmicsPredictor],
) -> None:
    assert issubclass(predictor_cls, SingleDrugOmicsPredictor)
    assert predictor_cls.required_sample_blocks == OMIC_BLOCK_NAMES
    assert predictor_cls().is_fitted() is False
    assert predictor_cls().get_state() == {}

    predictor = predictor_cls({"epochs": 7})
    state = {"algorithms": {"d1": save_trusted_mapping({})}, "predictor_hyperparameters": {"epochs": 1}}
    with pytest.raises(PredictorStateError, match="payload for 'd1' is missing"):
        predictor.set_state(state)

    assert predictor._hyperparameters["epochs"] == 7
    assert predictor.is_fitted() is False


class _RecordingPredictor(SingleDrugOmicsPredictor):
    def _fit_single_drug(self, drug_id: str, batch: Any) -> str | None:
        return None if drug_id == "d2" else f"model-{drug_id}"

    def _predict_single_drug(self, drug_id: str, model: Any, batch: Any) -> Any:
        raise NotImplementedError

    def _serialize_model(self, drug_id: str, model: Any) -> bytes:
        raise NotImplementedError

    def _restore_model(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError


def test_models_are_collected_per_drug_and_every_drug_records_feature_names() -> None:
    predictor = _RecordingPredictor()

    predictor._fit(_batch())

    assert predictor._models == {"d1": "model-d1"}
    assert set(predictor._feature_names) == {"d1", "d2"}


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_drugs_are_fitted_through_the_shared_fan_out(monkeypatch: pytest.MonkeyPatch, n_jobs: int) -> None:
    from drevalpy.components.predictors.literature import _single_drug_omics

    seen: list[int] = []
    real = _single_drug_omics.run_per_drug
    monkeypatch.setattr(
        _single_drug_omics, "run_per_drug", lambda fit, tasks, jobs: seen.append(jobs) or real(fit, tasks, jobs)
    )

    _RecordingPredictor()._fit(replace(_batch(), training_context=TrainingContext(n_jobs=n_jobs)))

    assert seen == [n_jobs]


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
