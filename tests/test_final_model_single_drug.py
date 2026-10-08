"""Tests that the final model of a single drug model is fitted on the responses of that drug only."""

import os
from typing import Any

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.experiment import train_final_model
from drevalpy.models.drp_model import DRPModel

_CELL_LINES = np.array([f"CL-{i}" for i in range(20)])
_DRUGS = np.array(["Drug-1", "Drug-2"])


class _RecordingModel(DRPModel):
    """Model that only records which responses it was trained on."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []
    early_stopping = False
    #: Responses of the last train() call, class level so the test sees them without the instance.
    seen_drug_ids: np.ndarray = np.array([])
    #: Cell lines the in memory features are built for, class level so a test can point it at real data.
    cell_lines: np.ndarray = _CELL_LINES

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: RecordingModel
        """
        return "RecordingModel"

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        """
        Avoids reading a hyperparameters.yaml next to this test file.

        :returns: a single empty hyperparameter set
        """
        return [{}]

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Nothing to build.

        :param hyperparameters: unused
        """

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Records the drug ids of the training data.

        :param output: training data
        :param cell_line_input: cell line features
        :param drug_input: drug features
        :param output_earlystopping: early stopping data
        :param model_checkpoint_dir: unused
        """
        type(self).seen_drug_ids = np.array(output.drug_ids)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts zeros.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell line features
        :param drug_input: drug features
        :returns: zeros
        """
        return np.zeros(len(cell_line_ids))

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Builds the features in memory, no data directory is read.

        :param data_path: unused
        :param dataset_name: unused
        :returns: one feature per cell line
        """
        return FeatureDataset(
            features={cl: {"gene_expression": np.array([float(i)])} for i, cl in enumerate(type(self).cell_lines)}
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Single drug models do not use drug features.

        :param data_path: unused
        :param dataset_name: unused
        :returns: None
        """
        return None

    def save(self, directory: str) -> None:
        """
        Writes a marker file so the test can see that the final model was saved.

        :param directory: target directory
        """
        with open(os.path.join(directory, "saved.txt"), "w", encoding="utf-8") as handle:
            handle.write(str(len(type(self).seen_drug_ids)))


class _SingleDrugRecordingModel(_RecordingModel):
    """Single drug variant, i.e., the pipeline fits one model per drug."""

    is_single_drug_model = True

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugRecordingModel
        """
        return "SingleDrugRecordingModel"


@pytest.fixture
def full_dataset() -> DrugResponseDataset:
    """
    Responses of two drugs on the same cell lines.

    :returns: dataset with 40 responses, 20 per drug
    """
    cell_line_ids = np.tile(_CELL_LINES, len(_DRUGS))
    drug_ids = np.repeat(_DRUGS, len(_CELL_LINES))
    return DrugResponseDataset(
        response=np.arange(len(cell_line_ids), dtype=float),
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        dataset_name="Toy_Data",
    )


def _train(model_class: type[DRPModel], dataset: DrugResponseDataset, path: str, drug_id: str | None) -> None:
    """
    Run train_final_model with the arguments the pipeline uses.

    :param model_class: model to train
    :param dataset: full dataset
    :param path: final model directory
    :param drug_id: drug id handed over by the pipeline for single drug models
    """
    train_final_model(
        model_class=model_class,
        full_dataset=dataset,
        drug_id=drug_id,
        # The pipeline may also hand over None here, but that contradicts the annotation of train_final_model.
        response_transformation=StandardScaler(),
        path_data="",
        model_checkpoint_dir="",
        metric="RMSE",
        final_model_path=path,
        test_mode="LCO",
        hyperparameter_tuning=False,
    )


def test_final_model_of_a_single_drug_model_only_sees_its_own_drug(full_dataset, tmp_path) -> None:
    """
    The final model has to be fitted per drug, like every other step of a single drug model.

    :param full_dataset: responses of two drugs
    :param tmp_path: pytest temporary path fixture
    """
    for drug_id in _DRUGS:
        _train(_SingleDrugRecordingModel, full_dataset.copy(), str(tmp_path / str(drug_id)), str(drug_id))
        seen = _SingleDrugRecordingModel.seen_drug_ids
        assert len(seen) > 0
        assert set(seen) == {drug_id}, f"final model of {drug_id} was trained on {set(seen)}"
        assert os.path.isfile(tmp_path / str(drug_id) / "saved.txt")


def test_final_model_of_a_single_drug_model_requires_a_drug_id(full_dataset, tmp_path) -> None:
    """
    Without a drug id the model would silently be fitted on all drugs, so it has to fail loudly.

    :param full_dataset: responses of two drugs
    :param tmp_path: pytest temporary path fixture
    """
    with pytest.raises(ValueError, match="single drug model"):
        _train(_SingleDrugRecordingModel, full_dataset.copy(), str(tmp_path / "no_drug"), None)


def test_final_model_of_a_multi_drug_model_still_sees_all_drugs(full_dataset, tmp_path) -> None:
    """
    Multi drug models are unaffected by the fix.

    :param full_dataset: responses of two drugs
    :param tmp_path: pytest temporary path fixture
    """
    _train(_RecordingModel, full_dataset.copy(), str(tmp_path / "multi"), None)
    assert set(_RecordingModel.seen_drug_ids) == set(_DRUGS)


def test_pipeline_hands_the_drug_id_to_the_final_model(monkeypatch, sample_dataset, data_dir, tmp_path) -> None:
    """
    The call site in drug_response_experiment has to pass the drug id of the current model.

    :param monkeypatch: pytest monkeypatch fixture
    :param sample_dataset: TOYv1, needed because the pipeline always adds NaiveMeanEffectsPredictor
    :param data_dir: path to the data directory
    :param tmp_path: pytest temporary path fixture
    """
    from drevalpy import experiment as experiment_module
    from drevalpy.models import MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY

    response_data = sample_dataset.copy()
    two_drugs = sorted(np.unique(response_data.drug_ids))[:2]
    response_data.mask(np.isin(response_data.drug_ids, two_drugs))

    monkeypatch.setitem(MODEL_FACTORY, "SingleDrugRecordingModel", _SingleDrugRecordingModel)
    monkeypatch.setitem(SINGLE_DRUG_MODEL_FACTORY, "SingleDrugRecordingModel", _SingleDrugRecordingModel)
    monkeypatch.setattr(_RecordingModel, "cell_lines", np.unique(response_data.cell_line_ids))

    calls: list[dict] = []

    def _record(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(experiment_module, "train_final_model", _record)

    experiment_module.drug_response_experiment(
        models=[_SingleDrugRecordingModel],
        baselines=[],
        response_data=response_data,
        run_id="single_drug_final_model",
        path_data=str(data_dir),
        path_out=str(tmp_path),
        test_mode="LCO",
        n_cv_splits=2,
        hyperparameter_tuning=False,
        final_model_on_full_data=True,
        overwrite=True,
    )

    # One final model per drug, each with its own drug id and the full, unmasked dataset.
    assert [call["drug_id"] for call in calls] == list(two_drugs)
    for call in calls:
        assert set(call["full_dataset"].drug_ids) == set(two_drugs)
