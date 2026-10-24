"""Tests for the built-in splitter modes: LDO, LTO, LPO, LSO (with its LCO alias) and LStO."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.registry.splitter import splitter_registry
from drevalpy.registry.splitter._validation import validate_folds
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs

QUALITY = CurveQuality.default()


def test_no_drug_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LDO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for fold in folds:
        train_cols = set(np.where(fold.train.mask.any(axis=0))[0].tolist())
        test_cols = set(np.where(fold.test.mask.any(axis=0))[0].tolist())
        assert train_cols & test_cols == set()


@pytest.mark.parametrize("mode", ["LDO", "LTO", "LPO", "LSO"])
def test_low_quality_pairs_appear_in_no_fold(mock_dataset: MockMuDataset, mode: str) -> None:
    failing = first_measured_pairs(mock_dataset, 3)
    splitter = splitter_registry.get(mode)

    without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3, quality=QUALITY))
    with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3, quality=QUALITY))

    for row, column in failing:
        assert without_filtering[row, column]
        assert not with_filtering[row, column]


def test_no_tissue_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LTO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    tissues = mock_dataset.get_tissue(mock_dataset.sample_ids)
    for fold in folds:
        train_rows = np.where(fold.train.mask.any(axis=1))[0]
        test_rows = np.where(fold.test.mask.any(axis=1))[0]
        assert set(tissues[train_rows].tolist()) & set(tissues[test_rows].tolist()) == set()


def test_folds_are_disjoint_boolean_masks_with_metadata(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LPO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for i, fold in enumerate(folds):
        for part in (fold.train, fold.val, fold.test):
            assert part.shape == mock_dataset.response_matrix.shape
            assert part.mask.dtype == bool
        assert not (fold.train & fold.test).any()
        assert fold.metadata["mode"] == "LPO"
        assert fold.metadata["fold_index"] == i
        assert fold.metadata["n_splits"] == 3


def test_no_sample_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LSO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for fold in folds:
        train_rows = set(np.where(fold.train.mask.any(axis=1))[0].tolist())
        test_rows = set(np.where(fold.test.mask.any(axis=1))[0].tolist())
        assert train_rows & test_rows == set()
        for part in (fold.train, fold.val, fold.test):
            assert part.shape == mock_dataset.response_matrix.shape


class TestLcoAlias:
    """``LCO`` is the pre-rename name of ``LSO`` and must keep working."""

    def test_the_alias_returns_the_same_splitter(self) -> None:
        assert splitter_registry.get("LCO") is splitter_registry.get("LSO")

    def test_the_alias_gives_the_same_folds_under_the_canonical_mode(self, mock_dataset: MockMuDataset) -> None:
        via_alias = splitter_registry.get("LCO")(mock_dataset, n_splits=3, quality=QUALITY)
        via_mode = splitter_registry.get("LSO")(mock_dataset, n_splits=3, quality=QUALITY)

        assert len(via_alias) == len(via_mode)
        for old, new in zip(via_alias, via_mode, strict=True):
            for name in ("train", "val", "test"):
                np.testing.assert_array_equal(getattr(old, name).mask, getattr(new, name).mask)
        assert {fold.metadata["mode"] for fold in via_alias} == {"LSO"}

    def test_the_alias_is_not_a_separate_mode(self) -> None:
        assert "LCO" not in splitter_registry.modes
        assert splitter_registry.get_metadata("LCO")["aliases"] == ["LCO"]

    def test_re_registering_the_alias_as_a_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="conflicts with an alias"):
            splitter_registry.register("LCO", "clash", validation="LSO")(lambda *args, **kwargs: [])


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

    def test_one_fold_per_study_ignoring_n_splits(self, splitter, study_dataset: StudyMuDataset) -> None:
        folds = splitter(study_dataset, n_splits=7, quality=QUALITY)

        assert len(folds) == len(STUDIES)
        assert {fold.metadata["n_splits"] for fold in folds} == {3}
        assert [fold.metadata["fold_index"] for fold in folds] == [0, 1, 2]
        assert {fold.metadata["mode"] for fold in folds} == {"LStO"}
        assert [fold.metadata["test_study"] for fold in folds] == sorted(STUDIES)

    def test_the_test_set_is_every_observed_pair_of_the_study(self, splitter, study_dataset: StudyMuDataset) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        observed = ~np.isnan(study_dataset.response_matrix)

        for fold, study in zip(splitter(study_dataset, quality=QUALITY), sorted(STUDIES), strict=True):
            np.testing.assert_array_equal(fold.test.mask, observed & (studies == study)[:, None])

    def test_the_masks_partition_the_observed_pairs_and_no_study_is_shared(
        self, splitter, study_dataset: StudyMuDataset
    ) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        observed = ~np.isnan(study_dataset.response_matrix)
        for fold in splitter(study_dataset, quality=QUALITY):
            train, val, test = fold.train.mask, fold.val.mask, fold.test.mask
            np.testing.assert_array_equal(train | val | test, observed)
            assert not (train & test).any()
            assert not (val & test).any()
            assert not (train & val).any()
            test_studies = set(studies[test.any(axis=1)].tolist())
            assert len(test_studies) == 1
            assert test_studies & set(studies[(train | val).any(axis=1)].tolist()) == set()

    @pytest.mark.parametrize("ratio", [0.0, 0.1, 0.3])
    def test_validation_is_that_share_of_the_training_pairs(
        self, splitter, study_dataset: StudyMuDataset, ratio: float
    ) -> None:
        for fold in splitter(study_dataset, validation_ratio=ratio, quality=QUALITY):
            n_rest = int(fold.train.mask.sum() + fold.val.mask.sum())
            assert int(fold.val.mask.sum()) == int(n_rest * ratio)

    def test_the_validation_pairs_span_the_training_studies(self, splitter, study_dataset: StudyMuDataset) -> None:
        studies = study_dataset.sample_column("study", study_dataset.sample_ids)
        fold = splitter(study_dataset, validation_ratio=0.5, quality=QUALITY)[0]

        assert len(set(studies[fold.val.mask.any(axis=1)].tolist())) == 2

    def test_the_folds_are_reproducible_and_depend_on_the_seed(self, splitter, study_dataset: StudyMuDataset) -> None:
        first = splitter(study_dataset, random_state=1, quality=QUALITY)
        again = splitter(study_dataset, random_state=1, quality=QUALITY)
        other = splitter(study_dataset, random_state=2, quality=QUALITY)

        np.testing.assert_array_equal(first[0].val.mask, again[0].val.mask)
        assert not np.array_equal(first[0].val.mask, other[0].val.mask)
        np.testing.assert_array_equal(first[0].test.mask, other[0].test.mask)

    def test_the_folds_pass_study_validation(self, splitter, study_dataset: StudyMuDataset) -> None:
        validate_folds(splitter(study_dataset, quality=QUALITY), "LStO", study_dataset)

    def test_a_single_study_is_rejected(self, splitter) -> None:
        with pytest.raises(ValueError, match="at least two studies, found 1"):
            splitter(StudyMuDataset(studies=("GDSC2",)), quality=QUALITY)

    def test_a_dataset_without_study_points_to_concat(self, splitter, mock_dataset: MockMuDataset) -> None:
        with pytest.raises(KeyError, match=r"drevalpy\.data\.concat"):
            splitter(mock_dataset, quality=QUALITY)

    def test_low_quality_pairs_appear_in_no_fold(self, splitter) -> None:
        failing = first_measured_pairs(StudyMuDataset(), 3)

        covered = covered_pairs(splitter(StudyMuDataset(failing_pairs=failing), quality=QUALITY))
        uncovered = covered_pairs(splitter(StudyMuDataset(failing_pairs=failing), quality=CurveQuality.off()))

        for row, column in failing:
            assert not covered[row, column]
            assert uncovered[row, column]
