"""Regression test: train_final_model must reduce the early stopping set to available features."""

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset, split_early_stopping_data
from drevalpy.experiment import make_train_val_split, train_final_model
from drevalpy.models.drp_model import DRPModel

_N_CELL_LINES = 40
_N_DRUGS = 10
#: Only the first half of the cell lines has a feature row, the rest has to be dropped from every split.
_N_WITH_FEATURES = _N_CELL_LINES // 2


def _toy_response() -> DrugResponseDataset:
    """
    Build a fully crossed toy response dataset.

    :returns: dataset with _N_CELL_LINES x _N_DRUGS rows
    """
    cell_line_ids = np.repeat([f"CL{i}" for i in range(_N_CELL_LINES)], _N_DRUGS)
    drug_ids = np.tile([f"D{j}" for j in range(_N_DRUGS)], _N_CELL_LINES)
    rng = np.random.default_rng(0)
    return DrugResponseDataset(
        response=rng.normal(size=len(cell_line_ids)),
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        dataset_name="TOY_ES",
    )


def _toy_features() -> FeatureDataset:
    """
    Build cell line features that cover only the first half of the cell lines.

    :returns: FeatureDataset with a gene_expression view
    """
    rng = np.random.default_rng(1)
    return FeatureDataset(features={f"CL{i}": {"gene_expression": rng.normal(size=5)} for i in range(_N_WITH_FEATURES)})


class _EarlyStoppingModel(DRPModel):
    """Minimal early stopping model that only records what train() was handed."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []
    early_stopping = True

    #: unique early stopping cell line ids seen by train(), read by the test
    seen_early_stopping_ids: np.ndarray | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: name of this test model
        """
        return "EarlyStoppingModel"

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
        Fetches the feature matrices exactly like a real model does, which asserts on missing ids.

        :param output: training dataset
        :param cell_line_input: cell line features
        :param drug_input: unused
        :param output_earlystopping: early stopping dataset
        :param model_checkpoint_dir: unused
        """
        cell_line_input.get_feature_matrix(view="gene_expression", identifiers=output.cell_line_ids)
        if output_earlystopping is not None:
            cell_line_input.get_feature_matrix(view="gene_expression", identifiers=output_earlystopping.cell_line_ids)
            type(self).seen_early_stopping_ids = np.unique(output_earlystopping.cell_line_ids)

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
        Nothing to persist for a test model.

        :param directory: target directory
        """


def test_final_model_reduces_early_stopping_set(tmp_path):
    """
    Without a response transformation the early stopping set must still be reduced to available features.

    :param tmp_path: pytest temporary path fixture
    """
    with_features = set(_toy_features().identifiers)

    # Precondition: with the splitting parameters train_final_model uses, the raw early stopping set
    # really does contain cell lines without features, otherwise the test would be vacuous.
    _, validation = make_train_val_split(_toy_response(), test_mode="LPO", val_ratio=0.1)
    _, raw_early_stopping = split_early_stopping_data(validation, test_mode="LPO")
    assert not set(raw_early_stopping.cell_line_ids).issubset(with_features)

    _EarlyStoppingModel.seen_early_stopping_ids = None
    train_final_model(
        model_class=_EarlyStoppingModel,
        full_dataset=_toy_response(),
        response_transformation=None,
        path_data="unused",
        model_checkpoint_dir=str(tmp_path / "checkpoints"),
        metric="RMSE",
        final_model_path=str(tmp_path / "final_model"),
        test_mode="LPO",
        val_ratio=0.1,
        hyperparameter_tuning=False,
    )

    seen = _EarlyStoppingModel.seen_early_stopping_ids
    assert seen is not None and len(seen) > 0
    assert set(seen).issubset(with_features)
