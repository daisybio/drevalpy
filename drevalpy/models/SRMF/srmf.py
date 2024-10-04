import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard
from numpy.typing import ArrayLike
from typing import Dict

from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


from drevalpy.models.utils import (
    load_and_reduce_gene_features,
    load_drug_fingerprint_features,
)


class SRMF(DRPModel):
    """
    SRMF model: Similarity Regularization Matrix Factorization.
    """

    model_name = "SRMF"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        super().__init__()
        self.bestU = None
        self.bestV = None
        self.W = None

    def build_model(self, hyperparameters: Dict):
        """
        Initializes hyperparameters for SRMF model.
        :param hyperparameters: dictionary containing the hyperparameters
        """
        self.K = hyperparameters.get("K", 45)
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
        drugs = np.unique(
            drug_input.identifiers
        )  # transductive approach - all drug features are used
        cell_lines = np.unique(
            cell_line_input.identifiers
        )  # transductive approach - all cell line features are used

        drug_features = drug_input.features

        drug_similarity = pd.DataFrame(
            index=drugs, columns=drugs, dtype=float
        )  # jaccard similarity

        for drug_from in drugs:
            for drug_to in drugs:
                # skip if already computed
                if not np.isnan(drug_similarity.loc[drug_from, drug_to]):
                    continue
                drug_similarity.loc[drug_from, drug_to] = 1 - jaccard(
                    drug_features[drug_from]["fingerprints"],
                    drug_features[drug_to]["fingerprints"],
                )
                drug_similarity.loc[drug_to, drug_from] = drug_similarity.loc[
                    drug_from, drug_to
                ]

        cell_line_features = cell_line_input.get_feature_matrix(
            view="gene_expression", identifiers=cell_lines
        )
        # pearson correlation as similarity
        cell_line_similarity = np.corrcoef(cell_line_features, rowvar=True)

        # Prepare response and weight matrices
        drug_response_matrix = output.to_dataframe()
        drug_response_matrix = (
            drug_response_matrix.groupby(["cell_line_id", "drug_id"])
            .mean()
            .reset_index()
        )
        drug_response_matrix = drug_response_matrix.pivot(
            index="cell_line_id", columns="drug_id", values="response"
        )

        drug_response_matrix = drug_response_matrix.reindex(
            index=cell_lines, columns=drugs
        )  # missing rows and columns are filled with NaN

        self.W = ~np.isnan(drug_response_matrix)
        drug_response_matrix = drug_response_matrix.copy()
        drug_response_matrix[np.isnan(drug_response_matrix)] = 0

        # Train the model
        bestU, bestV = self.CMF(
            W=self.W.T.values,
            intMat=drug_response_matrix.values.T,
            drugMat=drug_similarity.values,
            cellMat=cell_line_similarity,
        )
        self.bestU = pd.DataFrame(bestU, index=drugs)
        self.bestV = pd.DataFrame(bestV, index=cell_lines)

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
        bestU = self.bestU.loc[drug_ids].values
        bestV = self.bestV.loc[cell_line_ids].values
        # calculate the diagonal of the matrix product which is the prediction, faster than np.dot(bestU, bestV.T).diagonal()
        diagonal_predictions = np.einsum("ij,ji->i", bestU, bestV.T)

        return diagonal_predictions

    def CMF(self, W, intMat, drugMat, cellMat):
        """
        Implements the SRMF model with specific update rules and regularization.
        """
        np.random.seed(self.seed)
        m, n = W.shape
        U0 = np.sqrt(1 / self.K) * np.random.randn(m, self.K)
        V0 = np.sqrt(1 / self.K) * np.random.randn(n, self.K)

        bestU, bestV = U0, V0

        last_loss = self.compute_loss(U0, V0, W, intMat, drugMat, cellMat)
        bestloss = last_loss
        WR = W * intMat

        for t in range(self.max_iter):
            U = self.alg_update(U0, V0, W, WR, drugMat, self.lambda_l, self.lambda_d)
            V = self.alg_update(V0, U, W.T, WR.T, cellMat, self.lambda_l, self.lambda_c)
            curr_loss = self.compute_loss(U, V, W, intMat, drugMat, cellMat)

            if curr_loss < bestloss:
                bestU, bestV = U, V
                bestloss = curr_loss

            delta_loss = (curr_loss - last_loss) / last_loss
            if abs(delta_loss) < 1e-6:
                break

            last_loss = curr_loss
            U0, V0 = U, V

        return bestU, bestV

    def compute_loss(self, U, V, W, intMat, drugMat, cellMat):
        """
        Computes the loss for SRMF, including similarity regularization.
        """
        loss = np.sum((W * (intMat - np.dot(U, V.T))) ** 2)
        loss += self.lambda_l * (np.sum(U**2) + np.sum(V**2))
        loss += self.lambda_d * np.sum((drugMat - np.dot(U, U.T)) ** 2)
        loss += self.lambda_c * np.sum((cellMat - np.dot(V, V.T)) ** 2)
        return loss

    def alg_update(self, U, V, W, R, S, lambda_l, lambda_d):
        """
        Algorithm update rule for U or V in the SRMF model.
        """
        X = np.dot(R, V) + 2 * lambda_d * np.dot(S, U)
        Y = 2 * lambda_d * np.dot(U.T, U)
        U0 = np.zeros_like(U)
        D = np.dot(V.T, V)
        m, _ = W.shape

        for i in range(m):
            ii = np.where(W[i, :] > 0)[0]
            if ii.size == 0:
                B = Y + lambda_l * np.eye(U.shape[1])
            elif ii.size == W.shape[1]:
                B = D + Y + lambda_l * np.eye(U.shape[1])
            else:
                A = np.dot(V[ii, :].T, V[ii, :])
                B = A + Y + lambda_l * np.eye(U.shape[1])

            U0[i, :] = np.linalg.solve(B, X[i, :])

        return U0

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
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
        return load_drug_fingerprint_features(data_path, dataset_name)
