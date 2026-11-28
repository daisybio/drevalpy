"""
RidgeMLPEnsemble: the plain average of PerDrugRidge and MultiTaskMLP.

Both parts are configured from the ensemble's own hyperparameters.yaml entry, each part is trained on its own, and
the ensemble's prediction is their unweighted mean. No weights are fitted, so nothing beyond the training data is used.
"""

import json
import os
from typing import Any

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

from .multitask_mlp import MultiTaskMLP
from .per_drug_ridge import PerDrugRidge
from .utils import load_landmark_expression

COMPONENTS: list[type[DRPModel]] = [PerDrugRidge, MultiTaskMLP]
# The ensemble hyperparameter "<prefix><name>" is the component's setting "<name>", e.g. mlp_dropout -> dropout.
COMPONENT_PREFIXES = {"PerDrugRidge": "ridge_", "MultiTaskMLP": "mlp_"}


class RidgeMLPEnsemble(DRPModel):
    """Unweighted average of the per-drug ridge and the multi-task MLP."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []
    early_stopping = True  # so DrEval passes the early-stopping cell lines the MLP needs

    def __init__(self) -> None:
        """Create the component models."""
        super().__init__()
        self.parts: list[DRPModel] = [c() for c in COMPONENTS]
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "RidgeMLPEnsemble"."""
        return "RidgeMLPEnsemble"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Build every component from the ensemble's hyperparameters.

        Each component gets the ensemble hyperparameters that carry its prefix, without the prefix
        (``mlp_dropout`` becomes the MLP's ``dropout``), plus the ensemble's ``seed``. Settings that are not given
        fall back to the component's code defaults.

        :param hyperparameters: ``gene_list`` (the expression features of both components, landmark genes by default),
            ``seed`` and the prefixed component settings
        """
        self.hyperparameters = dict(hyperparameters)
        for part in self.parts:
            prefix = COMPONENT_PREFIXES[part.get_model_name()]
            component_hpams = {
                key.removeprefix(prefix): value for key, value in self.hyperparameters.items() if key.startswith(prefix)
            }
            if "seed" in self.hyperparameters:
                component_hpams.setdefault("seed", self.hyperparameters["seed"])
            part.build_model(component_hpams)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load gene expression, the input of every component.

        :param data_path: path to the data directory
        :param dataset_name: dataset name
        :returns: FeatureDataset with the "gene_expression" view
        """
        return load_landmark_expression(
            data_path, dataset_name, self.hyperparameters.get("gene_list", "landmark_genes")
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        No drug features are used.

        :param data_path: path to the data directory
        :param dataset_name: dataset name
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
        Train every component on exactly the data it gets when DrEval runs it on its own.

        A component with early stopping trains on ``output`` and stops on ``output_earlystopping``. A component
        without it gets both together, as DrEval gives it train + the whole validation set.

        :param output: training responses (train + validation minus the early-stopping cell lines)
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :param output_earlystopping: responses of the early-stopping cell lines
        :param model_checkpoint_dir: not used
        """
        for part in self.parts:
            if part.early_stopping:
                part.train(output, cell_line_input, output_earlystopping=output_earlystopping)
            else:
                full = output.copy()
                if output_earlystopping is not None:
                    full.add_rows(output_earlystopping)
                part.train(full, cell_line_input)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Average the components' predictions.

        :param cell_line_ids: cell-line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :returns: (n,) predicted ln(IC50)
        """
        return np.mean([p.predict(cell_line_ids, drug_ids, cell_line_input) for p in self.parts], axis=0)

    def save(self, directory: str) -> None:
        """
        Save the ensemble's hyperparameters and every component in its own subfolder.

        :param directory: target directory
        """
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "hyperparameters.json"), "w", encoding="utf-8") as f:
            json.dump(self.hyperparameters, f)
        for part in self.parts:
            part.save(os.path.join(directory, part.get_model_name()))

    @classmethod
    def load(cls, directory: str) -> "RidgeMLPEnsemble":
        """
        Restore a model saved with ``save``.

        :param directory: directory with the hyperparameters and the component subfolders
        :returns: the restored model
        """
        instance = cls()
        path = os.path.join(directory, "hyperparameters.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                instance.hyperparameters = json.load(f)
        instance.parts = [c.load(os.path.join(directory, c.get_model_name())) for c in COMPONENTS]
        return instance
