"""
Contains the SRMF (Similarity Regularization Matrix Factorization) model.

Original publication: Wang, L., Li, X., Zhang, L. et al. Improved anticancer drug response prediction in cell lines
using matrix factorization with similarity regularization. BMC Cancer 17, 513 (2017).
https://doi.org/10.1186/s12885-017-3500-5.
Matlab code adapted from https://github.com/linwang1982/SRMF.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_reduce_gene_features, load_drug_fingerprint_features


class SRMF(DRPModel):
    """
    SRMF model: Similarity Regularization Matrix Factorization.

    The primary idea is to map m drugs and n cell lines into a shared latent space, with a low dimensionality K,
    where K << min (m, n). The properties of a drug $d_i$ and a cell line $c_j$ are described by two latent coordinates
    $u_i$ and $v_j$ (K dimensional row vectors), respectively. The drug response matrix Y is approximated by:
    $min_{U,V} || W * (Y - U * V^T) ||^2_F + lambda_l * (||U||^2_F + ||V||^2_F) + lambda_d * ||S_d - U * U^T||^2_F +
    lambda_c * ||S_c - V * V^T||^2_F$
    where W is a weight matrix ($W_{ij} = 1 if Y_{ij}$ is a known response value, else 0). U, V contain $u_i$ ,
    $v_j$ as row vectors, respectively, $||.||_F$ is the Frobenius norm. To avoid overfitting, L2 regularization is
    used. S_d, S_c are drug/cell line similarity matrices. Differences between two drugs/cell lines are minimized in
    latent space.
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self) -> None:
        """Initalization method for SRMF Model."""
        super().__init__()
        self.best_u: pd.DataFrame = pd.DataFrame()
        self.best_v: pd.DataFrame = pd.DataFrame()
        self.w: pd.DataFrame = pd.DataFrame()
        self.k: int = 45
        self.lambda_l: float = 0.01
        self.lambda_d: float = 0.0
        self.lambda_c: float = 0.01
        self.max_iter: int = 50
        self.seed: int = 1

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SRMF
        """
        return "SRMF"

    def build_model(self, hyperparameters: dict) -> None:
        """
        Initializes hyperparameters for SRMF model.

        K is the latent dimensionality, lambda_l, lambda_d, lambda_c are regularization parameters, max_iter is the
        number of iterations, seed is the random seed.

        :param hyperparameters: dictionary containing the hyperparameters
        """
        self.k = hyperparameters.get("K", 45)
        self.lambda_l = hyperparameters.get("lambda_l", 0.01)
        self.lambda_d = hyperparameters.get("lambda_d", 0)
        self.lambda_c = hyperparameters.get("lambda_c", 0.01)
        self.max_iter = hyperparameters.get("max_iter", 50)
        self.seed = hyperparameters.get("seed", 1)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Prepares data and trains the SRMF model.

        :param output: response data
        :param cell_line_input: feature data for cell lines
        :param drug_input: feature data for drugs
        :param output_earlystopping: optional early stopping dataset
        :raises ValueError: if drug_input is None
        """
        if drug_input is None:
            raise ValueError("SRMF requires drug features.")

        drugs = np.unique(drug_input.identifiers)  # transductive approach - all drug features are used
        cell_lines = np.unique(cell_line_input.identifiers)  # transductive approach - all cell line features are used

        drug_features = drug_input.features

        drug_similarity = pd.DataFrame(index=drugs, columns=drugs, dtype=float)  # jaccard similarity

        for drug_from in drugs:
            for drug_to in drugs:
                # skip if already computed
                if not np.isnan(drug_similarity.loc[drug_from, drug_to]):
                    continue
                drug_similarity.loc[drug_from, drug_to] = 1 - jaccard(
                    drug_features[drug_from]["fingerprints"],
                    drug_features[drug_to]["fingerprints"],
                )
                drug_similarity.loc[drug_to, drug_from] = drug_similarity.loc[drug_from, drug_to]

        cell_line_features = cell_line_input.get_feature_matrix(view="gene_expression", identifiers=cell_lines)
        # pearson correlation as similarity
        cell_line_similarity = np.corrcoef(cell_line_features, rowvar=True)

        # Prepare response and weight matrices
        drug_response_matrix = output.to_dataframe()
        drug_response_matrix = drug_response_matrix.groupby(["cell_line_id", "drug_id"]).mean().reset_index()
        drug_response_matrix = drug_response_matrix.pivot(index="cell_line_id", columns="drug_id", values="response")

        drug_response_matrix = drug_response_matrix.reindex(
            index=cell_lines, columns=drugs
        )  # missing rows and columns are filled with NaN

        self.w = ~np.isnan(drug_response_matrix)
        drug_response_matrix = drug_response_matrix.copy()
        drug_response_matrix[np.isnan(drug_response_matrix)] = 0

        # Train the model
        best_u, best_v = self._cmf(
            w=self.w.T.values,
            int_mat=drug_response_matrix.values.T,
            drug_mat=drug_similarity.values,
            cell_mat=cell_line_similarity,
        )
        self.best_u = pd.DataFrame(best_u, index=drugs)
        self.best_v = pd.DataFrame(best_v, index=cell_lines)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the drug response based on the trained latent factors.

        :param drug_ids: drug identifiers
        :param cell_line_ids: cell line identifiers
        :param cell_line_input: not needed for prediction in SRMF
        :param drug_input: not needed for prediction in SRMF
        :returns: predicted response matrix
        """
        best_u = self.best_u.loc[drug_ids].values
        best_v = self.best_v.loc[cell_line_ids].values
        # calculate the diagonal of the matrix product which is the prediction,
        # faster than np.dot(best_u, best_v.T).diagonal()
        diagonal_predictions = np.einsum("ij,ji->i", best_u, best_v.T)

        return diagonal_predictions

    def _cmf(self, w, int_mat, drug_mat, cell_mat) -> tuple[np.ndarray, np.ndarray]:
        """
        Implements the SRMF model with specific update rules and regularization.

        :param w: weight matrix
        :param int_mat: interaction matrix
        :param drug_mat: drug similarity matrix
        :param cell_mat: cell line similarity matrix
        :returns: best drug and cell line latent factors
        """
        np.random.seed(self.seed)
        m, n = w.shape
        u0 = np.sqrt(1 / self.k) * np.random.randn(m, self.k)
        v0 = np.sqrt(1 / self.k) * np.random.randn(n, self.k)

        best_u, best_v = u0, v0

        last_loss = self._compute_loss(u0, v0, w, int_mat, drug_mat, cell_mat)
        best_loss = last_loss
        wr = w * int_mat

        for _ in range(self.max_iter):
            u = self._alg_update(u0, v0, w, wr, drug_mat, self.lambda_l, self.lambda_d)
            v = self._alg_update(v0, u, w.T, wr.T, cell_mat, self.lambda_l, self.lambda_c)
            curr_loss = self._compute_loss(u, v, w, int_mat, drug_mat, cell_mat)

            if curr_loss < best_loss:
                best_u, best_v = u, v
                best_loss = curr_loss

            delta_loss = (curr_loss - last_loss) / last_loss
            if abs(delta_loss) < 1e-6:
                break

            last_loss = curr_loss
            u0, v0 = u, v

        return best_u, best_v

    def _compute_loss(self, u, v, w, int_mat, drug_mat, cell_mat) -> np.float64:
        """
        Computes the loss for SRMF, including similarity regularization.

        :param u: drug latent factors
        :param v: cell line latent factors
        :param w: weight matrix
        :param int_mat: interaction matrix
        :param drug_mat: drug similarity matrix
        :param cell_mat: cell line similarity matrix
        :returns: loss value
        """
        loss = np.sum((w * (int_mat - np.dot(u, v.T))) ** 2)
        loss += self.lambda_l * (np.sum(u**2) + np.sum(v**2))
        loss += self.lambda_d * np.sum((drug_mat - np.dot(u, u.T)) ** 2)
        loss += self.lambda_c * np.sum((cell_mat - np.dot(v, v.T)) ** 2)
        return loss

    def _alg_update(self, u, v, w, r, s, lambda_l, lambda_d) -> np.ndarray:
        """
        Algorithm update rule for u or v in the SRMF model.

        :param u: drug latent factors
        :param v: cell line latent factors
        :param w: weight matrix
        :param r: weight * interaction matrix
        :param s: drug/cell line similarity matrix
        :param lambda_l: regularization parameter
        :param lambda_d: drug/cell line similarity regularization parameter
        :returns: updated u or v
        """
        x = np.dot(r, v) + 2 * lambda_d * np.dot(s, u)
        y = 2 * lambda_d * np.dot(u.T, u)
        u0 = np.zeros_like(u)
        d = np.dot(v.T, v)
        m, _ = w.shape

        for i in range(m):
            ii = np.where(w[i, :] > 0)[0]
            if ii.size == 0:
                b = y + lambda_l * np.eye(u.shape[1])
            elif ii.size == w.shape[1]:
                b = d + y + lambda_l * np.eye(u.shape[1])
            else:
                a = np.dot(v[ii, :].T, v[ii, :])
                b = a + y + lambda_l * np.eye(u.shape[1])

            u0[i, :] = np.linalg.solve(b, x[i, :])

        return u0

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the gene expression features.

        :param data_path: Path to the gene expression and landmark genes, e.g., data/
        :param dataset_name: Name of the dataset, e.g., GDSC2
        :returns: FeatureDataset containing the cell line gene expression features, filtered
            through the landmark genes
        """
        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug fingerprints.

        :param data_path: Path to the drug features, in this case the drug fingerprints, e.g., data/
        :param dataset_name: Name of the dataset, e.g., GDSC2
        :returns: FeatureDataset containing the drug fingerprint features
        """
        return load_drug_fingerprint_features(data_path, dataset_name)
