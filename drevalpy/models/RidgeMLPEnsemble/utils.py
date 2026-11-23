"""Helper functions shared by the RidgeMLPEnsemble components"""

import copy

import numpy as np

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.utils import load_and_select_gene_features

_CACHE: dict[tuple[str, str, str | None], FeatureDataset] = {}


def load_landmark_expression(
    data_path: str, dataset_name: str, gene_list: str | None = "landmark_genes"
) -> FeatureDataset:
    """
    Load gene expression (by default of the L1000 landmark genes), reading the file once per process.

    :param data_path: path to the data directory
    :param dataset_name: dataset name, e.g. CTRPv2
    :param gene_list: gene list to select, None for all genes
    :returns: a copy of the FeatureDataset with the "gene_expression" view
    """
    key = (data_path, dataset_name, gene_list)
    if key not in _CACHE:
        _CACHE[key] = load_and_select_gene_features(
            feature_type="gene_expression", gene_list=gene_list, data_path=data_path, dataset_name=dataset_name
        )
    return copy.deepcopy(_CACHE[key])


def expression(cell_line_input: FeatureDataset, ids: np.ndarray) -> np.ndarray:
    """
    Arcsinh-transformed expression matrix.

    :param cell_line_input: cell-line features
    :param ids: cell-line ids
    :returns: (len(ids), n_genes) matrix
    """
    return np.arcsinh(cell_line_input.get_feature_matrix(view="gene_expression", identifiers=ids).astype(np.float64))


class Standardizer:
    """Per-gene standardization fitted on the training cell lines; genes without variance are dropped."""

    def __init__(self, x: np.ndarray) -> None:
        """
        Fit on training expression.

        :param x: (n_cells, n_genes) training matrix
        """
        mu, sd = x.mean(0), x.std(0)
        self.genes = sd > 1e-8
        self.mu, self.sd = mu[self.genes], sd[self.genes]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply to any expression matrix.

        :param x: (n, n_genes) matrix
        :returns: (n, n_kept_genes) standardized matrix
        """
        return (x[:, self.genes] - self.mu) / self.sd


def response_matrix(output, known_cells: set) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Arrange a DrEval response dataset as a cell lines x drugs matrix with NaN for unmeasured pairs.

    :param output: DrugResponseDataset
    :param known_cells: cell lines that have features; others are dropped
    :returns: (cell ids, drug ids, matrix)
    """
    keep = np.array([c in known_cells for c in output.cell_line_ids]) & ~np.isnan(output.response)
    cells, ci = np.unique(output.cell_line_ids[keep], return_inverse=True)
    drugs, di = np.unique(output.drug_ids[keep], return_inverse=True)
    y = np.full((len(cells), len(drugs)), np.nan)
    y[ci, di] = output.response[keep]
    return cells, drugs, y


def gather(
    full: np.ndarray, cells: np.ndarray, drugs: np.ndarray, cell_line_ids, drug_ids, fallback: float
) -> np.ndarray:
    """
    Pick the requested (cell line, drug) pairs out of a cells x drugs prediction matrix.

    :param full: (len(cells), len(drugs)) predictions
    :param cells: row ids of ``full``
    :param drugs: column ids of ``full``
    :param cell_line_ids: requested cell lines
    :param drug_ids: requested drugs
    :param fallback: value for drugs not in ``drugs``
    :returns: (n,) predictions
    """
    row = {c: i for i, c in enumerate(cells)}
    col = {d: i for i, d in enumerate(drugs)}
    r = np.array([row[c] for c in cell_line_ids])
    c = np.array([col.get(d, -1) for d in drug_ids])
    out = np.full(len(r), fallback, dtype=np.float64)
    ok = c >= 0
    out[ok] = full[r[ok], c[ok]]
    return out
