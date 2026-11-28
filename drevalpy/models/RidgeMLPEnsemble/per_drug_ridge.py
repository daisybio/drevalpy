"""
PerDrugRidge: one ridge regression per drug on landmark-gene expression.

Each drug gets its own linear model, fitted on the cell lines measured for that drug. The penalty is chosen
separately for every drug, by exact leave-one-out error on its training cell lines, from a fixed grid.
"""

import os
from typing import Any

import joblib
import numpy as np
import torch

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

from .utils import Standardizer, expression, gather, load_landmark_expression, response_matrix

RIDGE_ALPHAS = np.logspace(-2, 2.5, 14)
MIN_CELL_LINES = 10


def ridge_path(x: np.ndarray, y: np.ndarray, alphas: np.ndarray = RIDGE_ALPHAS, device: str | None = None):
    """
    Fit a ridge regression per drug for every penalty in ``alphas`` and pick one per drug by leave-one-out error.

    Works in the dual (cell lines x cell lines) form, which is cheap because there are fewer cell lines than
    genes, and returns ordinary linear weights. Leave-one-out errors are exact (closed form).

    :param x: (n_cells, n_genes) standardized training expression
    :param y: (n_cells, n_drugs) training responses, NaN where unmeasured
    :param alphas: relative penalties
    :param device: torch device; GPU if available when None
    :returns: weights (n_alphas, n_genes, n_drugs), intercepts (n_alphas, n_drugs), chosen penalty index per drug
        (-1 for drugs with fewer than ``MIN_CELL_LINES`` measurements, which get their mean and zero weights)
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_genes = x.shape[1]
    xt = torch.as_tensor(x, dtype=torch.float64, device=dev)
    kernel = xt @ xt.T / n_genes
    lam = torch.as_tensor(alphas, dtype=torch.float64, device=dev) * torch.diagonal(kernel).mean()
    weights = np.zeros((len(alphas), n_genes, y.shape[1]))
    intercepts = np.zeros((len(alphas), y.shape[1]))
    chosen = np.full(y.shape[1], -1)
    for j in range(y.shape[1]):
        obs = ~np.isnan(y[:, j])
        n = int(obs.sum())
        if n == 0:
            continue
        yj = torch.as_tensor(y[obs, j], dtype=torch.float64, device=dev)
        if n < MIN_CELL_LINES:
            intercepts[:, j] = float(yj.mean())
            continue
        o = torch.as_tensor(obs, device=dev)
        k = kernel[o][:, o]
        col_mean = k.mean(0)
        kc = k - col_mean[None, :] - col_mean[:, None] + col_mean.mean()  # kernel of the centred cell lines
        s, u = torch.linalg.eigh((kc + kc.T) / 2)
        s = s.clamp(min=0)
        uy = u.T @ (yj - yj.mean())
        inv = 1.0 / (s[None, :] + lam[:, None])
        coef = (u[None] * inv[:, None, :]) @ uy  # dual coefficients, one row per penalty
        diag = ((u**2)[None] * inv[:, None, :]).sum(-1) - 1.0 / (n * lam[:, None])
        chosen[j] = int(((coef / diag) ** 2).mean(1).argmin())  # exact leave-one-out error
        xc = xt[o] - xt[o].mean(0)
        w = (coef @ xc / n_genes).T  # (n_genes, n_alphas)
        weights[:, :, j] = w.T.cpu().numpy()
        intercepts[:, j] = (yj.mean() - xt[o].mean(0) @ w).cpu().numpy()
    return weights, intercepts, chosen


class PerDrugRidge(DRPModel):
    """One ridge regression per drug, penalty chosen per drug by leave-one-out error."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []
    early_stopping = False

    def __init__(self) -> None:
        """Initialize; everything is fitted in ``train``."""
        super().__init__()
        self.state: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "PerDrugRidge"."""
        return "PerDrugRidge"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store the hyperparameters.

        :param hyperparameters: hyperparameter dictionary, set by RidgeMLPEnsemble from its prefixed settings
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load landmark-gene expression.

        :param data_path: path to the data directory
        :param dataset_name: dataset name
        :returns: FeatureDataset with the "gene_expression" view
        """
        return load_landmark_expression(data_path, dataset_name)

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
        Fit one ridge regression per drug.

        :param output: training responses
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :param output_earlystopping: not used
        :param model_checkpoint_dir: not used
        """
        cells, drugs, y = response_matrix(output, set(cell_line_input.identifiers))
        x_raw = expression(cell_line_input, cells)
        scaler = Standardizer(x_raw)
        alphas = np.asarray(self.hyperparameters.get("alphas", RIDGE_ALPHAS), dtype=float)
        w, b, chosen = ridge_path(scaler(x_raw), y, alphas)
        pick = np.maximum(chosen, 0)
        cols = np.arange(len(drugs))
        self.state = {
            "drugs": drugs,
            "scaler": scaler,
            "weights": w[pick, :, cols].T,  # (n_genes, n_drugs)
            "intercepts": b[pick, cols],
            "chosen": chosen,
            "alphas": alphas,
            "training_mean": float(np.nanmean(y)),
        }

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict responses; drugs unseen in training get the training mean.

        :param cell_line_ids: cell-line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :returns: (n,) predicted ln(IC50)
        """
        st = self.state
        cells = np.unique(cell_line_ids)
        full = st["scaler"](expression(cell_line_input, cells)) @ st["weights"] + st["intercepts"][None, :]
        return gather(full, cells, st["drugs"], cell_line_ids, drug_ids, st["training_mean"])

    def save(self, directory: str) -> None:
        """
        Save the fitted weights.

        :param directory: target directory
        """
        os.makedirs(directory, exist_ok=True)
        joblib.dump(
            {"hyperparameters": self.hyperparameters, "state": self.state},
            os.path.join(directory, "per_drug_ridge.pkl"),
        )

    @classmethod
    def load(cls, directory: str) -> "PerDrugRidge":
        """
        Restore a model saved with ``save``.

        :param directory: directory with the saved file
        :returns: the restored model
        """
        blob = joblib.load(os.path.join(directory, "per_drug_ridge.pkl"))
        instance = cls()
        instance.build_model(blob["hyperparameters"])
        instance.state = blob["state"]
        return instance
