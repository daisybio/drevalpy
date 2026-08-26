"""Tests for the group-aware response transformation (drug_mean)."""

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
    assert list(centerer.group_keys_) == ["A", "B", "C"]
    assert np.allclose(centerer.group_means_, [_GROUP_MEANS[key] for key in ["A", "B", "C"]])


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

    assert centerer.group_keys_.size == 0
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
    assert isinstance(get_response_transformation("drug_mean"), GroupMeanCenterer)


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
    """check_arguments has to allow the new response transformation.

    :param tmp_path: pytest tmp_path fixture, used as data directory
    """
    check_arguments(_valid_args("drug_mean", str(tmp_path)))

    with pytest.raises(AssertionError, match="drug_mean"):
        check_arguments(_valid_args("nonsense", str(tmp_path)))
