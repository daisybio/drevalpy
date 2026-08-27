"""Tests for the group-aware response transformations (drug_mean, drug_tissue_mean)."""

from argparse import Namespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.response_transformation import GroupMeanCenterer
from drevalpy.utils import check_arguments, get_response_transformation

_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 5.0])
_DRUGS = np.array(["A", "A", "B", "B", "B", "C"])
#: means of _RESPONSE per group in _DRUGS
_GROUP_MEANS = {"A": 2.0, "B": 20.0, "C": 5.0}


def test_requires_groups_flag() -> None:
    """The dataset uses this flag to decide whether it has to supply drug ids."""
    assert GroupMeanCenterer.requires_groups is True
    assert getattr(StandardScaler, "requires_groups", False) is False


def test_fit_estimates_group_means() -> None:
    """Fitting has to record the per-group means and the global mean."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)

    assert np.isclose(centerer.global_mean_, _RESPONSE.mean())
    assert len(centerer.level_keys_) == 1
    assert list(centerer.level_keys_[0]) == ["A", "B", "C"]
    assert np.allclose(centerer.level_means_[0], [_GROUP_MEANS[key] for key in ["A", "B", "C"]])


def test_transform_subtracts_the_group_mean() -> None:
    """Every group of the transformed response has to be centered on zero."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
    residuals = centerer.transform(_RESPONSE.reshape(-1, 1), groups=_DRUGS).squeeze()

    expected = _RESPONSE - np.array([_GROUP_MEANS[drug] for drug in _DRUGS])
    assert np.allclose(residuals, expected)
    for drug in np.unique(_DRUGS):
        assert np.isclose(residuals[_DRUGS == drug].mean(), 0.0)


def test_inverse_transform_round_trips() -> None:
    """inverse_transform has to return values on the original response scale."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
    residuals = centerer.transform(_RESPONSE.reshape(-1, 1), groups=_DRUGS)

    assert np.allclose(centerer.inverse_transform(residuals, groups=_DRUGS).squeeze(), _RESPONSE)


def test_transform_preserves_shape() -> None:
    """The transformer is called with column vectors and has to return them unchanged in shape."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
    column = _RESPONSE.reshape(-1, 1)

    assert centerer.transform(column, groups=_DRUGS).shape == column.shape
    assert centerer.transform(_RESPONSE, groups=_DRUGS).shape == _RESPONSE.shape


def test_unseen_group_falls_back_to_the_global_mean() -> None:
    """Unseen drugs (LDO, cross study) must not raise, they fall back to the global mean."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
    # "0" sorts before and "Z" after every fitted key, which is where searchsorted needs clipping.
    unseen = np.array(["Z", "0", "A"])
    values = np.array([1.0, 1.0, 1.0])

    residuals = centerer.transform(values, groups=unseen)
    expected = values - np.array([centerer.global_mean_, centerer.global_mean_, _GROUP_MEANS["A"]])
    assert np.allclose(residuals, expected)
    assert np.allclose(centerer.inverse_transform(residuals, groups=unseen), values)


def test_without_groups_it_is_plain_mean_centering() -> None:
    """Without groups the transformer degenerates to subtracting the global mean."""
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1))

    assert centerer.level_keys_ == []
    assert np.allclose(centerer.transform(_RESPONSE), _RESPONSE - _RESPONSE.mean())


def test_fit_on_empty_input() -> None:
    """An empty training fold must not blow up with a numpy error."""
    centerer = GroupMeanCenterer().fit(np.array([]).reshape(-1, 1), groups=np.array([]))

    assert centerer.global_mean_ == 0.0
    assert centerer.transform(np.array([])).size == 0


def _toy_dataset() -> DrugResponseDataset:
    """
    Build a small dataset whose drug means differ strongly.

    :returns: dataset with six rows over three drugs
    """
    return DrugResponseDataset(
        response=_RESPONSE.copy(),
        cell_line_ids=np.array(["CL0", "CL1", "CL0", "CL1", "CL2", "CL0"]),
        drug_ids=_DRUGS.copy(),
        dataset_name="TOY_GM",
    )


def test_dataset_fit_transform_uses_drug_ids_as_groups() -> None:
    """The dataset has to hand the drug ids to a group-aware transformation."""
    dataset = _toy_dataset()
    centerer = GroupMeanCenterer()
    dataset.fit_transform(centerer)

    expected = _RESPONSE - np.array([_GROUP_MEANS[drug] for drug in _DRUGS])
    assert np.allclose(dataset.response, expected)

    dataset.inverse_transform(centerer)
    assert np.allclose(dataset.response, _RESPONSE)


def test_dataset_transforms_predictions_with_the_same_groups() -> None:
    """Predictions are transformed row-wise as well and have to use the same drug ids."""
    dataset = _toy_dataset()
    dataset._predictions = _RESPONSE.copy()
    centerer = GroupMeanCenterer()
    dataset.fit_transform(centerer)

    assert dataset.predictions is not None
    assert np.allclose(dataset.predictions, dataset.response)

    dataset.inverse_transform(centerer)
    assert np.allclose(dataset.predictions, _RESPONSE)


def test_dataset_still_works_with_a_group_unaware_transformation() -> None:
    """A plain sklearn scaler must not be called with a groups argument."""
    dataset = _toy_dataset()
    scaler = StandardScaler()
    dataset.fit_transform(scaler)

    assert np.isclose(dataset.response.mean(), 0.0)
    dataset.inverse_transform(scaler)
    assert np.allclose(dataset.response, _RESPONSE)


def test_get_response_transformation_drug_mean() -> None:
    """The CLI name "drug_mean" has to map to the GroupMeanCenterer."""
    transformation = get_response_transformation("drug_mean")

    assert isinstance(transformation, GroupMeanCenterer)
    assert transformation.group_fields == ("drug_ids",)


def test_get_response_transformation_drug_tissue_mean() -> None:
    """The CLI name "drug_tissue_mean" has to group by drug and tissue."""
    transformation = get_response_transformation("drug_tissue_mean")

    assert isinstance(transformation, GroupMeanCenterer)
    assert transformation.group_fields == ("drug_ids", "tissue")


def _valid_args(response_transformation: str, path_data: str) -> Namespace:
    """
    Build a minimal, otherwise valid argument namespace.

    :param response_transformation: value under test
    :param path_data: data directory, check_arguments creates it
    :returns: namespace for check_arguments
    """
    return Namespace(
        run_id="test_run",
        dataset_name="TOYv1",
        models=["ElasticNet"],
        baselines=["NaivePredictor"],
        test_mode=["LPO"],
        randomization_mode=["None"],
        randomization_type="permutation",
        n_trials_robustness=0,
        cross_study_datasets=[],
        no_refitting=True,
        curve_curator_cores=1,
        measure="LN_IC50",
        overwrite=False,
        optim_metric="RMSE",
        n_cv_splits=2,
        response_transformation=response_transformation,
        multiprocessing=False,
        model_checkpoint_dir="TEMPORARY",
        no_hyperparameter_tuning=True,
        final_model_on_full_data=False,
        wandb_project=None,
        path_data=path_data,
    )


def test_check_arguments_accepts_drug_mean(tmp_path) -> None:
    """check_arguments has to allow the new response transformations.

    :param tmp_path: pytest tmp_path fixture, used as data directory
    """
    check_arguments(_valid_args("drug_mean", str(tmp_path)))
    check_arguments(_valid_args("drug_tissue_mean", str(tmp_path)))

    with pytest.raises(AssertionError, match="drug_mean"):
        check_arguments(_valid_args("nonsense", str(tmp_path)))


_TT_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 40.0, 7.0, 9.0])
_TT_DRUGS = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
_TT_TISSUES = np.array(["lung", "lung", "skin", "skin", "lung", "lung", "skin", "skin"])
#: means of _TT_RESPONSE per (drug, tissue) combination
_TT_MEANS = {("A", "lung"): 2.0, ("A", "skin"): 15.0, ("B", "lung"): 35.0, ("B", "skin"): 8.0}
#: means of _TT_RESPONSE per drug, i.e. the next level of the fallback
_TT_DRUG_MEANS = {"A": 8.5, "B": 21.5}


def _tissue_centerer() -> GroupMeanCenterer:
    """
    Fit a drug+tissue centerer on the toy data above.

    :returns: the fitted transformer
    """
    groups = np.stack([_TT_DRUGS, _TT_TISSUES], axis=1)
    return GroupMeanCenterer(group_fields=("drug_ids", "tissue")).fit(_TT_RESPONSE.reshape(-1, 1), groups=groups)


def test_drug_tissue_transform_centers_every_combination() -> None:
    """Grouping by drug and tissue has to center each (drug, tissue) combination on zero."""
    centerer = _tissue_centerer()
    groups = np.stack([_TT_DRUGS, _TT_TISSUES], axis=1)
    residuals = centerer.transform(_TT_RESPONSE, groups=groups)

    expected = _TT_RESPONSE - np.array([_TT_MEANS[(d, t)] for d, t in zip(_TT_DRUGS, _TT_TISSUES)])
    assert np.allclose(residuals, expected)
    for combination in _TT_MEANS:
        rows = (_TT_DRUGS == combination[0]) & (_TT_TISSUES == combination[1])
        assert np.isclose(residuals[rows].mean(), 0.0)
    assert np.allclose(centerer.inverse_transform(residuals, groups=groups), _TT_RESPONSE)


def test_drug_tissue_fit_estimates_both_levels() -> None:
    """Both nesting levels are needed for the fallback, the drug level is the coarser one."""
    centerer = _tissue_centerer()

    assert len(centerer.level_keys_) == 2
    assert list(centerer.level_keys_[1]) == ["A", "B"]
    assert np.allclose(centerer.level_means_[1], [_TT_DRUG_MEANS["A"], _TT_DRUG_MEANS["B"]])
    assert np.isclose(centerer.global_mean_, _TT_RESPONSE.mean())


def test_unseen_tissue_falls_back_to_the_drug_mean() -> None:
    """An unseen tissue (LTO, cross study) must fall back to the drug mean, not to the global mean."""
    centerer = _tissue_centerer()
    # "liver" is unknown for both drugs, "Z" is an unknown drug, "0" and "z" bracket the fitted keys.
    groups = np.array([["A", "liver"], ["B", "liver"], ["Z", "lung"], ["0", "0"], ["z", "z"]])
    values = np.ones(len(groups))

    offsets = values - centerer.transform(values, groups=groups)
    expected = [
        _TT_DRUG_MEANS["A"],
        _TT_DRUG_MEANS["B"],
        centerer.global_mean_,
        centerer.global_mean_,
        centerer.global_mean_,
    ]
    assert np.allclose(offsets, expected)
    assert np.allclose(centerer.inverse_transform(centerer.transform(values, groups=groups), groups=groups), values)


def test_group_fields_of_fit_and_transform_have_to_match() -> None:
    """Handing over fewer fields than at fit time would silently center on the wrong mean."""
    centerer = _tissue_centerer()

    with pytest.raises(ValueError, match="group field"):
        centerer.transform(_TT_RESPONSE, groups=_TT_DRUGS)


def test_group_keys_cannot_collide() -> None:
    """The fields are joined for the lookup, so ambiguous concatenations must stay distinct."""
    responses = np.array([2.0, 8.0])
    groups = np.array([["A", "B_C"], ["A_B", "C"]])
    centerer = GroupMeanCenterer(group_fields=("drug_ids", "tissue")).fit(responses.reshape(-1, 1), groups=groups)

    assert np.allclose(centerer.transform(responses, groups=groups), [0.0, 0.0])


def _tissue_dataset(with_tissues: bool = True) -> DrugResponseDataset:
    """
    Build a small dataset with a tissue per response.

    :param with_tissues: if False, the dataset has no tissue information
    :returns: dataset with eight rows over two drugs and two tissues
    """
    return DrugResponseDataset(
        response=_TT_RESPONSE.copy(),
        cell_line_ids=np.array([f"CL{i}" for i in range(len(_TT_RESPONSE))]),
        drug_ids=_TT_DRUGS.copy(),
        tissues=_TT_TISSUES.copy() if with_tissues else None,
        dataset_name="TOY_GM_TISSUE",
    )


def test_dataset_supplies_the_tissues_as_second_group_field() -> None:
    """The dataset has to hand over every field the transformation asks for."""
    dataset = _tissue_dataset()
    centerer = GroupMeanCenterer(group_fields=("drug_ids", "tissue"))
    dataset.fit_transform(centerer)

    expected = _TT_RESPONSE - np.array([_TT_MEANS[(d, t)] for d, t in zip(_TT_DRUGS, _TT_TISSUES)])
    assert np.allclose(dataset.response, expected)

    dataset.inverse_transform(centerer)
    assert np.allclose(dataset.response, _TT_RESPONSE)


def test_dataset_of_an_unseen_tissue_is_centered_on_the_drug_mean() -> None:
    """The LTO situation: fitted on one tissue, applied to another one, going through the dataset API."""
    train = _tissue_dataset()
    train.mask(_TT_TISSUES == "lung")
    test = _tissue_dataset()
    test.mask(_TT_TISSUES == "skin")
    test._predictions = test.response.copy()

    centerer = GroupMeanCenterer(group_fields=("drug_ids", "tissue"))
    train.fit_transform(centerer)
    test.transform(centerer)

    # "skin" was never seen, so every response is centered on the drug mean of the lung training fold.
    lung_means = {"A": 2.0, "B": 35.0}
    expected = _TT_RESPONSE[_TT_TISSUES == "skin"] - np.array(
        [lung_means[drug] for drug in _TT_DRUGS[_TT_TISSUES == "skin"]]
    )
    assert np.allclose(test.response, expected)

    test.inverse_transform(centerer)
    assert np.allclose(test.response, _TT_RESPONSE[_TT_TISSUES == "skin"])
    assert np.allclose(test.predictions, _TT_RESPONSE[_TT_TISSUES == "skin"])


def test_dataset_without_tissues_raises() -> None:
    """Without tissues the transformation would silently be a drug_mean, so it has to fail loudly."""
    dataset = _tissue_dataset(with_tissues=False)

    with pytest.raises(ValueError, match="tissue"):
        dataset.fit_transform(GroupMeanCenterer(group_fields=("drug_ids", "tissue")))
