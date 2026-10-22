"""Tests for the leave-study-out splitter."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter import splitter_registry
from drevalpy.registry.splitter._validation import validate_folds
from tests.data.splitters._helpers import MockMuDataset

STUDIES = ("GDSC2", "CTRPv2", "PRISM")


class StudyMuDataset(MockMuDataset):
    """A mock whose samples cycle over *studies*, served as the ``study`` column."""

    def __init__(self, studies: tuple[str, ...] = STUDIES, **kwargs) -> None:
        """Build the mock with one study per sample, cycling through *studies*."""
        super().__init__(n_cl=12, n_dr=8, **kwargs)
        self._studies = np.array([studies[i % len(studies)] for i in range(len(self._cl_ids))])

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Serve the study column on top of the tissue column."""
        if column == "study":
            return self._studies
        return super().sample_column(column, ids)


@pytest.fixture
def study_dataset() -> StudyMuDataset:
    return StudyMuDataset()


@pytest.fixture
def splitter():
    return splitter_registry.get("LStO")


class TestLeaveStudyOut:
    def test_is_registered_with_its_validation(self) -> None:
        metadata = splitter_registry.get_metadata("LStO")

        assert metadata["validation"] == "LStO"
        assert metadata["description"] == "Leave-Study-Out: test folds contain one unseen study"

    def test_produces_one_fold_per_study(self, splitter, study_dataset: StudyMuDataset) -> None:
        folds = splitter(study_dataset)

        assert len(folds) == len(STUDIES)

    def test_n_splits_is_ignored_and_the_real_count_is_recorded(self, splitter, study_dataset: StudyMuDataset) -> None:
        folds = splitter(study_dataset, n_splits=7)

        assert len(folds) == 3
        assert {fold.metadata["n_splits"] for fold in folds} == {3}
        assert [fold.metadata["fold_index"] for fold in folds] == [0, 1, 2]
        assert {fold.metadata["mode"] for fold in folds} == {"LStO"}

    def test_each_fold_records_the_study_it_tests(self, splitter, study_dataset: StudyMuDataset) -> None:
        folds = splitter(study_dataset)

        assert [fold.metadata["test_study"] for fold in folds] == sorted(STUDIES)

    def test_the_test_set_is_every_observed_pair_of_the_study(self, splitter, study_dataset: StudyMuDataset) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        observed = ~np.isnan(study_dataset.response_matrix)

        for fold, study in zip(splitter(study_dataset), sorted(STUDIES), strict=True):
            expected = observed & (studies == study)[:, None]
            np.testing.assert_array_equal(fold.test.mask, expected)

    def test_train_and_validation_are_the_other_studies_observed_pairs(
        self, splitter, study_dataset: StudyMuDataset
    ) -> None:
        observed = ~np.isnan(study_dataset.response_matrix)
        for fold in splitter(study_dataset):
            np.testing.assert_array_equal(fold.train.mask | fold.val.mask | fold.test.mask, observed)

    def test_the_masks_are_disjoint(self, splitter, study_dataset: StudyMuDataset) -> None:
        for fold in splitter(study_dataset):
            assert not (fold.train.mask & fold.test.mask).any()
            assert not (fold.val.mask & fold.test.mask).any()
            assert not (fold.train.mask & fold.val.mask).any()

    def test_no_study_is_in_both_train_and_test(self, splitter, study_dataset: StudyMuDataset) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        for fold in splitter(study_dataset):
            test_studies = set(studies[fold.test.mask.any(axis=1)].tolist())
            train_val_studies = set(studies[(fold.train.mask | fold.val.mask).any(axis=1)].tolist())
            assert len(test_studies) == 1
            assert test_studies & train_val_studies == set()

    @pytest.mark.parametrize("ratio", [0.1, 0.3])
    def test_validation_is_that_share_of_the_training_pairs(
        self, splitter, study_dataset: StudyMuDataset, ratio: float
    ) -> None:
        for fold in splitter(study_dataset, validation_ratio=ratio):
            n_rest = int(fold.train.mask.sum() + fold.val.mask.sum())
            assert int(fold.val.mask.sum()) == int(n_rest * ratio)

    def test_a_zero_ratio_leaves_the_validation_set_empty(self, splitter, study_dataset: StudyMuDataset) -> None:
        for fold in splitter(study_dataset, validation_ratio=0.0):
            assert not fold.val.mask.any()

    def test_the_validation_pairs_span_the_training_studies(self, splitter, study_dataset: StudyMuDataset) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        fold = splitter(study_dataset, validation_ratio=0.5)[0]

        assert len(set(studies[fold.val.mask.any(axis=1)].tolist())) == 2

    def test_the_folds_are_reproducible_and_depend_on_the_seed(self, splitter, study_dataset: StudyMuDataset) -> None:
        first = splitter(study_dataset, random_state=1)
        again = splitter(study_dataset, random_state=1)
        other = splitter(study_dataset, random_state=2)

        np.testing.assert_array_equal(first[0].val.mask, again[0].val.mask)
        assert not np.array_equal(first[0].val.mask, other[0].val.mask)
        np.testing.assert_array_equal(first[0].test.mask, other[0].test.mask)

    def test_the_folds_pass_study_validation(self, splitter, study_dataset: StudyMuDataset) -> None:
        validate_folds(splitter(study_dataset), "LStO", study_dataset)

    def test_unmeasured_pairs_appear_in_no_fold(self, splitter, study_dataset: StudyMuDataset) -> None:
        unmeasured = np.isnan(study_dataset.response_matrix)
        for fold in splitter(study_dataset):
            assert not (fold.train.mask | fold.val.mask | fold.test.mask)[unmeasured].any()

    def test_a_single_study_is_rejected(self, splitter) -> None:
        with pytest.raises(ValueError, match="at least two studies, found 1"):
            splitter(StudyMuDataset(studies=("GDSC2",)))

    def test_a_dataset_without_study_points_to_concat(self, splitter, mock_dataset: MockMuDataset) -> None:
        with pytest.raises(KeyError, match=r"drevalpy\.data\.concat"):
            splitter(mock_dataset)
