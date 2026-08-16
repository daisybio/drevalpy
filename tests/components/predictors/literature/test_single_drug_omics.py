"""Tests for the per-drug three-omic plumbing shared by MOLIR and SuperFELTR.

``literature/_single_drug_omics.py`` collects what both single-drug predictors used to
duplicate: the per-drug checkpoint directory, the record of which feature names a model
was trained on, the predict-time column realignment, the early-stopping index lookup,
and the feature-name (de)serialization.

The interesting cases are the degenerate ones - a missing or too-short early-stopping
response, and a predict-time batch whose omic columns are reordered or narrower than
what the model was trained on.
"""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.components.predictors.literature._single_drug_omics import (
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
    validation_split,
)
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


def _identity_blocks() -> dict:
    """Build the drug identity blocks per-drug routing needs.

    :returns: Identity and identity-categories blocks for drugs ``d1``/``d2``.
    """
    return {
        "identity": numeric_feature_block(np.eye(2, dtype=np.float32)),
        "identity_categories": numeric_feature_block(np.array([["d1", "d2"]], dtype=object)),
    }


def _batch(
    *,
    early_stopping_response: ResponseBatch | None = None,
    expression: np.ndarray = EXPRESSION,
    expression_features: tuple[str, ...] = EXPRESSION_FEATURES,
    checkpoint_dir: str = "checkpoints",
) -> ModelInputBatch:
    """Build a four-pair batch carrying all three omic views.

    :param early_stopping_response: Optional validation pairs.
    :param expression: Gene-expression matrix to use.
    :param expression_features: Feature names for *expression*.
    :param checkpoint_dir: Base checkpoint directory.
    :returns: Featurized ``ModelInputBatch``.
    """
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    return two_by_two_batch(
        response=response,
        cell_line_blocks={
            "gene_expression": numeric_feature_block(expression, feature_names=expression_features),
            "mutations": numeric_feature_block(MUTATIONS, feature_names=MUTATION_FEATURES),
            "copy_number_variation_gistic": numeric_feature_block(CNV, feature_names=CNV_FEATURES),
        },
        drug_blocks=_identity_blocks(),
        early_stopping_response=early_stopping_response,
        checkpoint_dir=checkpoint_dir,
    )


def _matrices() -> OmicMatrices:
    return OmicMatrices(gene_expression=EXPRESSION, mutations=MUTATIONS, copy_number_variation=CNV)


def _trained_names() -> OmicFeatureNames:
    return OmicFeatureNames(
        gene_expression=EXPRESSION_FEATURES,
        mutations=MUTATION_FEATURES,
        copy_number_variation=CNV_FEATURES,
    )


class TestBlockNames:
    def test_the_three_views_are_in_encoder_order(self) -> None:
        assert OMIC_BLOCK_NAMES == ("gene_expression", "mutations", "copy_number_variation_gistic")


class TestOmicMatrices:
    def test_matrices_are_read_out_of_the_batch_as_float32(self) -> None:
        matrices = omic_matrices(_batch())

        assert matrices.gene_expression.dtype == np.float32
        np.testing.assert_allclose(matrices.gene_expression, EXPRESSION)
        np.testing.assert_allclose(matrices.mutations, MUTATIONS)
        np.testing.assert_allclose(matrices.copy_number_variation, CNV)

    def test_widths_report_one_feature_count_per_view(self) -> None:
        assert _matrices().widths() == (3, 2, 1)

    def test_a_split_reuses_the_entity_matrices_and_takes_pair_level_data(self) -> None:
        pair_idx = np.array([0, 1, 1])
        response = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        split = _matrices().split(pair_idx, response)

        assert split.gene_expression is EXPRESSION
        assert split.copy_number is CNV
        np.testing.assert_array_equal(split.pair_idx, pair_idx)
        np.testing.assert_allclose(split.response, response)


class TestFeatureNames:
    def test_names_are_recorded_per_view(self) -> None:
        assert omic_feature_names(_batch()).as_tuple() == (
            EXPRESSION_FEATURES,
            MUTATION_FEATURES,
            CNV_FEATURES,
        )

    def test_the_payload_round_trips(self) -> None:
        restored = feature_names_from_payload(feature_names_payload(_trained_names()))

        assert restored == _trained_names()

    def test_the_payload_is_json_shaped_lists(self) -> None:
        payload = feature_names_payload(_trained_names())

        assert payload == {
            "gene_expression_features": list(EXPRESSION_FEATURES),
            "mutations_features": list(MUTATION_FEATURES),
            "copy_number_variation_features": list(CNV_FEATURES),
        }

    def test_unknown_names_serialize_as_none(self) -> None:
        assert feature_names_payload(None) == {
            "gene_expression_features": None,
            "mutations_features": None,
            "copy_number_variation_features": None,
        }

    def test_a_payload_from_an_older_model_deserializes_to_none(self) -> None:
        assert feature_names_from_payload({}) == OmicFeatureNames(None, None, None)


class TestCheckpointDir:
    def test_the_directory_is_derived_from_the_drug_id(self) -> None:
        directory = checkpoint_dir_for_drug(UPath("/base"), "drug-1")

        assert directory.parent == UPath("/base")
        assert directory.name.startswith("drug_")

    def test_the_same_drug_always_maps_to_the_same_directory(self) -> None:
        assert checkpoint_dir_for_drug(UPath("/base"), "d") == checkpoint_dir_for_drug(UPath("/base"), "d")

    def test_different_drugs_do_not_share_a_directory(self) -> None:
        assert checkpoint_dir_for_drug(UPath("/base"), "d1") != checkpoint_dir_for_drug(UPath("/base"), "d2")

    def test_a_filesystem_hostile_drug_id_still_yields_one_path_segment(self) -> None:
        directory = checkpoint_dir_for_drug(UPath("/base"), "a/b c:*?")

        assert directory.parent == UPath("/base")


class TestIterDrugSubsets:
    def test_one_subset_per_drug_with_only_that_drugs_pairs(self) -> None:
        subsets = dict(iter_drug_subsets(_batch()))

        assert sorted(subsets) == ["d1", "d2"]
        assert subsets["d1"].n_pairs == 2
        np.testing.assert_array_equal(subsets["d1"].drug_ids, ["d1", "d1"])

    def test_each_subset_gets_its_own_checkpoint_directory(self) -> None:
        subsets = dict(iter_drug_subsets(_batch(checkpoint_dir="/base")))

        directories = {str(sub.training_context.checkpoint_dir) for sub in subsets.values()}
        assert len(directories) == 2
        assert all(directory.startswith("/base/drug_") for directory in directories)


class TestEarlyStopping:
    def test_no_early_stopping_response_yields_no_split(self) -> None:
        assert early_stopping_indices(_batch()) == (None, None)
        assert validation_split(_matrices(), _batch()) is None

    def test_a_single_validation_pair_is_refused(self) -> None:
        """One point cannot support an early-stopping decision."""
        response = ResponseBatch(
            response=np.array([1.0]),
            cell_line_ids=np.array(["cl1"]),
            drug_ids=np.array(["d1"]),
        )

        assert early_stopping_indices(_batch(early_stopping_response=response)) == (None, None)

    def test_validation_cell_lines_map_onto_entity_rows(self) -> None:
        response = ResponseBatch(
            response=np.array([7.0, 8.0]),
            cell_line_ids=np.array(["cl2", "cl1"]),
            drug_ids=np.array(["d1", "d1"]),
        )

        pair_idx, values = early_stopping_indices(_batch(early_stopping_response=response))

        np.testing.assert_array_equal(pair_idx, [1, 0])
        np.testing.assert_allclose(values, [7.0, 8.0])

    def test_the_validation_split_shares_the_training_entity_matrices(self) -> None:
        response = ResponseBatch(
            response=np.array([7.0, 8.0]),
            cell_line_ids=np.array(["cl2", "cl1"]),
            drug_ids=np.array(["d1", "d1"]),
        )
        matrices = _matrices()

        split = validation_split(matrices, _batch(early_stopping_response=response))

        assert split is not None
        assert split.gene_expression is matrices.gene_expression
        np.testing.assert_array_equal(split.pair_idx, [1, 0])


class TestAlignedPairMatrices:
    def test_matching_features_are_expanded_to_pair_level_unchanged(self) -> None:
        expression, mutations, cnv = aligned_pair_matrices(_batch(), _trained_names())

        np.testing.assert_allclose(expression, EXPRESSION[[0, 0, 1, 1]])
        np.testing.assert_allclose(mutations, MUTATIONS[[0, 0, 1, 1]])
        np.testing.assert_allclose(cnv, CNV[[0, 0, 1, 1]])

    def test_unknown_training_features_leave_the_matrix_alone(self) -> None:
        expression, _, _ = aligned_pair_matrices(_batch(), OmicFeatureNames(None, None, None))

        np.testing.assert_allclose(expression, EXPRESSION[[0, 0, 1, 1]])

    def test_a_narrower_batch_is_realigned_onto_the_trained_columns(self) -> None:
        # The batch only carries geneA and geneC; geneB must come back as zeros.
        narrow = np.array([[1.0, 3.0], [4.0, 6.0]], dtype=np.float32)
        batch = _batch(expression=narrow, expression_features=("geneA", "geneC"))

        expression, _, _ = aligned_pair_matrices(batch, _trained_names())

        assert expression.shape == (4, 3)
        np.testing.assert_allclose(expression[0], [1.0, 0.0, 3.0])
        np.testing.assert_allclose(expression[2], [4.0, 0.0, 6.0])

    def test_a_same_width_reordering_is_left_alone(self) -> None:
        """Pre-existing semantics: realignment triggers on a width change, not a rename.

        ``_realign_omic_matrix`` short-circuits when the incoming width already matches,
        so equally wide but differently ordered columns pass through. Pinned here because
        the extraction had to preserve it, not because it is desirable.
        """
        reordered = EXPRESSION[:, [2, 0, 1]]
        batch = _batch(expression=reordered, expression_features=("geneC", "geneA", "geneB"))

        expression, _, _ = aligned_pair_matrices(batch, _trained_names())

        np.testing.assert_allclose(expression[0], reordered[0])

    def test_matching_views_keep_their_float32_dtype(self) -> None:
        for matrix in aligned_pair_matrices(_batch(), _trained_names()):
            assert matrix.dtype == np.float32


def test_the_recorded_feature_names_are_frozen() -> None:
    """A trained model's feature record must not drift after the fit."""
    names = _trained_names()

    with pytest.raises(AttributeError):
        names.gene_expression = ("other",)  # type: ignore[misc]
