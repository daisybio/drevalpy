"""Tests that train_final_model persists the fitted response transformation next to the model."""

import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.experiment import RESPONSE_TRANSFORMATION_FILE, train_final_model
from drevalpy.models.drp_model import DRPModel

_N_CELL_LINES = 20
_N_DRUGS = 5


def _toy_response() -> DrugResponseDataset:
    """
    Build a fully crossed toy response dataset with a non-trivial mean and scale.

    :returns: dataset with _N_CELL_LINES x _N_DRUGS rows
    """
    cell_line_ids = np.repeat([f"CL{i}" for i in range(_N_CELL_LINES)], _N_DRUGS)
    drug_ids = np.tile([f"D{j}" for j in range(_N_DRUGS)], _N_CELL_LINES)
    rng = np.random.default_rng(0)
    return DrugResponseDataset(
        response=rng.normal(loc=5.0, scale=2.0, size=len(cell_line_ids)),
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        dataset_name="TOY_RT",
    )


def _toy_features() -> FeatureDataset:
    """
    Build cell line features covering every cell line of the toy dataset.

    :returns: FeatureDataset with a gene_expression view
    """
    rng = np.random.default_rng(1)
    return FeatureDataset(features={f"CL{i}": {"gene_expression": rng.normal(size=5)} for i in range(_N_CELL_LINES)})


class _SavingModel(DRPModel):
    """Minimal model whose save() only writes a marker file."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: name of this test model
        """
        return "SavingModel"

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict]:
        """
        Returns a single empty hyperparameter set, there is no hyperparameters.yaml for a test model.

        :returns: list with one empty hyperparameter dict
        """
        return [{}]

    def build_model(self, hyperparameters: dict) -> None:
        """
        Stores the hyperparameters.

        :param hyperparameters: hyperparameters to use
        """
        self.hyperparameters = hyperparameters

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Returns the toy features, ignoring path and dataset name.

        :param data_path: unused
        :param dataset_name: unused
        :returns: toy cell line features
        """
        return _toy_features()

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        This model does not use drug features.

        :param data_path: unused
        :param dataset_name: unused
        :returns: None
        """
        return None

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Does nothing, the test only cares about what is written to disk afterwards.

        :param output: training dataset
        :param cell_line_input: cell line features
        :param drug_input: unused
        :param output_earlystopping: unused
        :param model_checkpoint_dir: unused
        """

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Returns zeros.

        :param cell_line_ids: cell line ids to predict for
        :param drug_ids: unused
        :param cell_line_input: unused
        :param drug_input: unused
        :returns: zero predictions
        """
        return np.zeros(len(cell_line_ids))

    def save(self, directory: str) -> None:
        """
        Writes a marker file so the saved model directory is not empty.

        :param directory: target directory
        """
        with open(os.path.join(directory, "model.txt"), "w", encoding="utf-8") as handle:
            handle.write("saved")


def _train(final_model_path: str, checkpoint_dir: str, response_transformation) -> DrugResponseDataset:
    """
    Run train_final_model with the toy model.

    :param final_model_path: directory the final model is written to
    :param checkpoint_dir: checkpoint directory for tuning
    :param response_transformation: transformation to use, may be None
    :returns: the dataset that was handed to train_final_model
    """
    full_dataset = _toy_response()
    train_final_model(
        model_class=_SavingModel,
        full_dataset=full_dataset.copy(),
        response_transformation=response_transformation,
        path_data="unused",
        model_checkpoint_dir=checkpoint_dir,
        metric="RMSE",
        final_model_path=final_model_path,
        test_mode="LPO",
        val_ratio=0.1,
        hyperparameter_tuning=False,
    )
    return full_dataset


def test_final_model_saves_fitted_response_transformation(tmp_path):
    """
    The fitted transformation has to be stored next to the model and be usable after reloading.

    :param tmp_path: pytest temporary path fixture
    """
    final_model_path = str(tmp_path / "final_model")
    scaler = StandardScaler()
    full_dataset = _train(final_model_path, str(tmp_path / "checkpoints"), scaler)

    dump_path = os.path.join(final_model_path, RESPONSE_TRANSFORMATION_FILE)
    assert os.path.isfile(dump_path)

    reloaded = joblib.load(dump_path)
    assert isinstance(reloaded, StandardScaler)
    # The reloaded object is fitted, and fitted on the response of this run.
    assert np.allclose(reloaded.mean_, scaler.mean_)
    assert np.allclose(reloaded.scale_, scaler.scale_)

    # It really maps predictions back to the original response scale.
    residuals = np.array([0.0, 1.0, -1.0]).reshape(-1, 1)
    back = reloaded.inverse_transform(residuals).squeeze()
    assert np.isclose(back[0], full_dataset.response.mean(), atol=0.5)


def test_final_model_without_transformation_writes_no_dump(tmp_path):
    """
    Without a response transformation there is nothing to persist.

    :param tmp_path: pytest temporary path fixture
    """
    final_model_path = str(tmp_path / "final_model")
    _train(final_model_path, str(tmp_path / "checkpoints"), None)

    assert os.path.isfile(os.path.join(final_model_path, "model.txt"))
    assert not os.path.exists(os.path.join(final_model_path, RESPONSE_TRANSFORMATION_FILE))
