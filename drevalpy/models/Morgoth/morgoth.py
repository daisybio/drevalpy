"""
MORGOTH as a single-drug model, wrapping the ``morgoth`` package (https://github.com/volkamerlab/MORGOTH).

MORGOTH is a random forest whose trees learn the response value and a class (response below / above a per-drug
threshold) simultaneously; ``impact_classification`` (lambda in the paper) weights the classification in the split
criterion. drevalpy uses the regression output as the prediction.

Rolli, L.-M. et al. Increasing trustworthiness of machine learning-based drug sensitivity prediction with a
multivariate random forest approach. Digital Discovery 5 (4), 1746-1764 (2026).
"""

import contextlib
import json
import multiprocessing
import os
import tempfile
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import ThreadPool

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..utils import MRMRSelector, load_single_cell_line_view, scale_gene_expression
from .fast_split import fast_split


@contextlib.contextmanager
def _pool(n_jobs: int) -> Iterator[None]:
    """
    Set the number of processes of the ``Pool()`` hard-wired in the package and restore it afterwards.

    ``joblib.effective_n_jobs`` resolves -1 to all cores, but to 1 inside the workers of ``run_per_drug``, where the
    drugs already run in parallel. With one process, the trees are grown in this process instead of in a child.

    :param n_jobs: number of processes, -1 = all cores
    :yields: nothing
    """
    import morgoth.morgoth as morgoth_module

    processes = joblib.effective_n_jobs(n_jobs)
    original = morgoth_module.Pool
    morgoth_module.Pool = partial(ThreadPool, 1) if processes == 1 else partial(multiprocessing.Pool, processes)
    try:
        yield
    finally:
        morgoth_module.Pool = original


class MorgothRegressor(RegressorMixin, BaseEstimator):
    """sklearn interface to the MORGOTH forest of the ``morgoth`` package."""

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 10,
        max_features: str | float = "sqrt",
        max_depth: int = 20,
        criterion_class: str = "gini",
        criterion_reg: str = "mse",
        impact_classification: float = 0.5,
        sample_weights: str = "simple",
        tree_weights: bool = True,
        threshold_quantile: float = 0.5,
        random_state: int = 42,
        n_jobs: int = -1,
        fast_split: bool = True,
    ):
        """
        Initializes the regressor; the parameters are passed on to ``morgoth.MORGOTH``.

        :param n_estimators: number of trees
        :param min_samples_leaf: minimal number of samples per leaf
        :param max_features: features per split, 'sqrt', 'log2' or a fraction in [0, 1]
        :param max_depth: maximal depth of the trees
        :param criterion_class: split criterion of the classification, 'gini', 'entropy' or 'hellinger'
        :param criterion_reg: split criterion of the regression, 'mse'
        :param impact_classification: share of the classification in the split criterion (lambda in the paper)
        :param sample_weights: 'simple', 'linear' or 'none'
        :param tree_weights: whether to use the class-aware tree weights of SAURON-RF
        :param threshold_quantile: quantile of the training responses that separates the two classes
        :param random_state: seed of the forest
        :param n_jobs: number of processes, -1 = all cores
        :param fast_split: replace the package's split search by the equivalent vectorized one, False = original
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.impact_classification = impact_classification
        self.sample_weights = sample_weights
        self.tree_weights = tree_weights
        self.threshold_quantile = threshold_quantile
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fast_split = fast_split

    def fit(self, X, y):  # noqa: N803
        """
        Fits the forest; the class threshold is the ``threshold_quantile`` of the given responses.

        :param X: feature matrix (rows x features)
        :param y: response vector
        :returns: the fitted regressor
        """
        from morgoth.morgoth import MORGOTH

        y = np.asarray(y, dtype=float)
        self.threshold_ = float(np.quantile(y, self.threshold_quantile))
        # same rule as the package's sample weights: values below the threshold are class 0
        y_class = (y >= self.threshold_).astype(int)
        # the package writes these files during the fit, they are not needed
        with tempfile.TemporaryDirectory() as tmp, _pool(self.n_jobs), fast_split(self.fast_split):
            self.forest_ = MORGOTH(
                X_train=pd.DataFrame(X),
                y_train=np.column_stack([y, y_class]),
                criterion_class=self.criterion_class,
                criterion_reg=self.criterion_reg,
                sample_names_train=np.arange(len(y)).astype(str),
                min_number_of_samples_per_leaf=self.min_samples_leaf,
                number_of_trees_in_forest=self.n_estimators,
                number_of_features_per_split=self.max_features,
                class_names=[0, 1],
                output_format="multioutput",
                threshold=[self.threshold_],
                time_file=os.path.join(tmp, "time.txt"),
                sample_weights_included=self.sample_weights,
                random_state=self.random_state,
                max_depth=self.max_depth,
                impact_classification=self.impact_classification,
                sample_info_file=os.path.join(tmp, "sample_info.txt"),
                analysis_name="drevalpy",
                leaf_assignment_file_train=os.path.join(tmp, "leaf_assignment.txt"),
                feature_imp_output_file=os.path.join(tmp, "feature_importance.txt"),
                tree_weights=self.tree_weights,
            )
            self.forest_.fit()
        return self

    def predict(self, X):  # noqa: N803
        """
        Predicts the response value.

        :param X: feature matrix (rows x features)
        :returns: predicted responses
        """
        with _pool(self.n_jobs):
            prediction = self.forest_.predict(X_test=pd.DataFrame(X))
        return prediction[:, 0].astype(float)


class SingleDrugMorgoth(DRPModel):
    """MORGOTH, fitted for each drug separately on the mRMR-selected genes (selection on the training rows only)."""

    cell_line_views = ["gene_expression"]
    drug_views = []
    is_single_drug_model = True
    early_stopping = False

    def __init__(self):
        """Initializes the model; the pipeline is built in build_model."""
        super().__init__()
        self.model: Pipeline | None = None
        self.gene_expression_scaler = StandardScaler()
        self.hyperparameters: dict = {}

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugMorgoth
        """
        return "SingleDrugMorgoth"

    def build_model(self, hyperparameters: dict):
        """
        Builds the mRMR selection followed by the MORGOTH forest.

        :param hyperparameters: hyperparameters, ``gene_list`` is the gene list the features are loaded from,
            ``mrmr_k`` the number of selected genes, the others are passed on to MorgothRegressor
        :raises ImportError: if morgoth is not installed
        """
        try:
            import morgoth  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SingleDrugMorgoth requires the optional 'morgoth' extra. "
                "Install it with: pip install drevalpy[morgoth] (or `poetry install -E morgoth`)."
            ) from e
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        self.model = Pipeline(
            [
                ("mrmr", MRMRSelector(k=int(hyperparameters["mrmr_k"]))),
                (
                    "morgoth",
                    MorgothRegressor(
                        n_estimators=hyperparameters["n_estimators"],
                        min_samples_leaf=hyperparameters["min_samples_leaf"],
                        max_features=hyperparameters["max_features"],
                        max_depth=hyperparameters["max_depth"],
                        criterion_class=hyperparameters["criterion_class"],
                        criterion_reg=hyperparameters["criterion_reg"],
                        impact_classification=hyperparameters["impact_classification"],
                        sample_weights=hyperparameters["sample_weights"],
                        tree_weights=hyperparameters["tree_weights"],
                        threshold_quantile=hyperparameters["threshold_quantile"],
                        fast_split=hyperparameters.get("fast_split", True),
                    ),
                ),
            ]
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model on the responses of one drug.

        The gene expression is arcsinh-transformed and standardized with a scaler fitted on the training cell lines.

        :param output: training responses of one drug
        :param cell_line_input: gene expression of the cell lines
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        :raises RuntimeError: if the model has not been built yet
        """
        if len(output) == 0:
            print("No training data provided, will predict NA.")
            self.model = None
            return
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Call build_model() first.")
        cell_line_input = scale_gene_expression(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(output.cell_line_ids),
            training=True,
            gene_expression_scaler=self.gene_expression_scaler,
        )
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        self.model.fit(x, output.response)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response of the given cell lines to the drug.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids, all the same drug
        :param cell_line_input: gene expression of the cell lines
        :param drug_input: not needed
        :returns: predicted responses
        """
        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        cell_line_input = scale_gene_expression(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            gene_expression_scaler=self.gene_expression_scaler,
        )
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the gene expression of the genes in the gene list ``gene_list``.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset with the gene expression
        """
        return load_single_cell_line_view(
            self.cell_line_views, data_path, dataset_name, self.get_model_name(), self.hyperparameters["gene_list"]
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load drug features. Not needed for MORGOTH.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None

    def save(self, directory: str) -> None:
        """
        Saves the fitted pipeline, the gene expression scaler and the hyperparameters.

        Files: model.pkl, scaler.pkl and hyperparameters.json.

        :param directory: target directory
        :raises ValueError: if the model is not trained
        """
        if self.model is None:
            raise ValueError("Cannot save: model is not trained.")
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "model.pkl"))
        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

    @classmethod
    def load(cls, directory: str) -> "SingleDrugMorgoth":
        """
        Loads a model saved with save.

        :param directory: directory with model.pkl, scaler.pkl and hyperparameters.json
        :returns: the restored model
        """
        instance = cls()
        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.build_model(json.load(f))
        instance.model = joblib.load(os.path.join(directory, "model.pkl"))
        instance.gene_expression_scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        return instance
