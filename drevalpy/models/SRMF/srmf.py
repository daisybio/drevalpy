import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.distance import jaccard

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_reduce_gene_features, load_drug_fingerprint_features


class SRMF(DRPModel):
    """SRMF model: Similarity Regularization Matrix Factorization."""

    model_name = "SRMF"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        """Initalization method for SRMF Model."""
        super().__init__()
        self.best_u = None
        self.best_v = None
        self.w = None

    def build_model(self, hyperparameters: dict):
        """
        Initializes hyperparameters for SRMF model.

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
        cell_line_input: FeatureDataset = None,
        drug_input: FeatureDataset = None,
        output_earlystopping=None,
    ) -> None:
        """
        Prepares data and trains the SRMF model.

        :param output: response data
        :param cell_line_input: feature data for cell lines
        :param drug_input: feature data for drugs
        """
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
        best_u, best_v = self.cmf(
            w=self.w.T.values,
            int_mat=drug_response_matrix.values.T,
            drug_mat=drug_similarity.values,
            cell_mat=cell_line_similarity,
        )
        self.best_u = pd.DataFrame(best_u, index=drugs)
        self.best_v = pd.DataFrame(best_v, index=cell_lines)

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the drug response based on the trained latent factors.

        :param drug_ids: drug identifiers
        :param cell_line_ids: cell line identifiers
        :return: predicted response matrix
        """
        best_u = self.best_u.loc[drug_ids].values
        best_v = self.best_v.loc[cell_line_ids].values
        # calculate the diagonal of the matrix product which is the prediction,
        # faster than np.dot(best_u, best_v.T).diagonal()
        diagonal_predictions = np.einsum("ij,ji->i", best_u, best_v.T)

        return diagonal_predictions

    def cmf(self, w, int_mat, drug_mat, cell_mat):
        """
        Implements the SRMF model with specific update rules and regularization.

        :param w:
        :param int_mat:
        :param drug_mat:
        :param cell_mat:
        """
        np.random.seed(self.seed)
        m, n = w.shape
        u0 = np.sqrt(1 / self.k) * np.random.randn(m, self.k)
        v0 = np.sqrt(1 / self.k) * np.random.randn(n, self.k)

        best_u, best_v = u0, v0

        last_loss = self.compute_loss(u0, v0, w, int_mat, drug_mat, cell_mat)
        best_loss = last_loss
        wr = w * int_mat

        for _ in range(self.max_iter):
            u = self.alg_update(u0, v0, w, wr, drug_mat, self.lambda_l, self.lambda_d)
            v = self.alg_update(v0, u, w.T, wr.T, cell_mat, self.lambda_l, self.lambda_c)
            curr_loss = self.compute_loss(u, v, w, int_mat, drug_mat, cell_mat)

            if curr_loss < best_loss:
                best_u, best_v = u, v
                best_loss = curr_loss

            delta_loss = (curr_loss - last_loss) / last_loss
            if abs(delta_loss) < 1e-6:
                break

            last_loss = curr_loss
            u0, v0 = u, v

        return best_u, best_v

    def compute_loss(self, u, v, w, int_mat, drug_mat, cell_mat):
        """
        Computes the loss for SRMF, including similarity regularization.

        :param u:
        :param v:
        :param w:
        :param int_mat:
        :param drug_mat:
        :param cell_mat:
        """
        loss = np.sum((w * (int_mat - np.dot(u, v.T))) ** 2)
        loss += self.lambda_l * (np.sum(u**2) + np.sum(v**2))
        loss += self.lambda_d * np.sum((drug_mat - np.dot(u, u.T)) ** 2)
        loss += self.lambda_c * np.sum((cell_mat - np.dot(v, v.T)) ** 2)
        return loss

    def alg_update(self, u, v, w, r, s, lambda_l, lambda_d):
        """
        Algorithm update rule for u or v in the SRMF model.

        :param u:
        :param v:
        :param w:
        :param r:
        :param s:
        :param lambda_l:
        :param lambda_d:
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
        Loads the cell line features.

        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered
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
        Loads the drug features.

        :param data_path:
        :param dataset_name:
        """
        return load_drug_fingerprint_features(data_path, dataset_name)

    def load(self, path):
        """
        Loads the model from a given path.

        :param path: Path to the model
        """
        raise NotImplementedError("SRMF does not support loading yet ...")

    def save(self, path):
        """
        Saves the model to a given path.

        :param path: Path to save the model
        """
        raise NotImplementedError("SRMF does not support saving yet ...")
