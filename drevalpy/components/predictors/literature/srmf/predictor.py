"""SRMF block predictor – Similarity Regularization Matrix Factorization.

Original publication: Wang, L., Li, X., Zhang, L. et al. Improved anticancer drug response prediction in cell lines
using matrix factorization with similarity regularization. BMC Cancer 17, 513 (2017).
https://doi.org/10.1186/s12885-017-3500-5.
Matlab code adapted from https://github.com/linwang1982/SRMF.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import SRMF_REFERENCE
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SRMF_REFERENCE,
)
class SRMFPredictor(BlockPredictor):
    """Registered SRMF predictor using similarity-regularized matrix factorization."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize SRMF predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._best_u: pd.DataFrame = pd.DataFrame()
        self._best_v: pd.DataFrame = pd.DataFrame()
        self._training_mean: float = 0.0

    @property
    def _k(self) -> int:
        return int(self._hyperparameters.get("K", 45))

    @property
    def _lambda_l(self) -> float:
        return float(self._hyperparameters.get("lambda_l", 0.01))

    @property
    def _lambda_d(self) -> float:
        return float(self._hyperparameters.get("lambda_d", 0.0))

    @property
    def _lambda_c(self) -> float:
        return float(self._hyperparameters.get("lambda_c", 0.01))

    @property
    def _max_iter(self) -> int:
        return int(self._hyperparameters.get("max_iter", 50))

    @property
    def _seed(self) -> int:
        return int(self._hyperparameters.get("seed", 1))

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default SRMF hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "K": 45,
            "lambda_l": 0.01,
            "lambda_d": 0.0,
            "lambda_c": 0.01,
            "max_iter": 50,
            "seed": 1,
            "n_features": 1036,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter space.

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit the SRMF model on training data.

        :param batch: Training batch with gene_expression and fingerprints blocks.
        :raises ValueError: If drug features or response data is missing.
        """
        cell_lines = batch.cell_line_entity_ids
        drugs = batch.drug_entity_ids
        if drugs is None:
            msg = "SRMF requires drug features"
            raise ValueError(msg)

        cell_line_features = batch.cell_line_blocks["gene_expression"].values
        drug_features = batch.drug_blocks["fingerprints"].values

        # Drug similarity: Jaccard on binary fingerprints
        n_drugs = len(drugs)
        drug_similarity = np.zeros((n_drugs, n_drugs), dtype=np.float64)
        for i in range(n_drugs):
            drug_similarity[i, i] = 1.0
            for j in range(i + 1, n_drugs):
                sim = 1.0 - jaccard(drug_features[i], drug_features[j])
                drug_similarity[i, j] = sim
                drug_similarity[j, i] = sim

        # Cell-line similarity: Pearson correlation
        cell_line_similarity = np.corrcoef(cell_line_features, rowvar=True)

        # Build response matrix (cell_lines x drugs)
        cl_id_to_idx = {str(cid): i for i, cid in enumerate(cell_lines)}
        dr_id_to_idx = {str(did): i for i, did in enumerate(drugs)}

        response_matrix = np.full((len(cell_lines), n_drugs), np.nan, dtype=np.float64)
        if batch.response is None:
            msg = "SRMF requires training response data"
            raise ValueError(msg)
        for pair_idx in range(batch.n_pairs):
            cl_idx = cl_id_to_idx.get(str(batch.cell_line_ids[pair_idx]))
            dr_idx = dr_id_to_idx.get(str(batch.drug_ids[pair_idx]))
            if cl_idx is not None and dr_idx is not None:
                # Average duplicates by accumulating then dividing
                if np.isnan(response_matrix[cl_idx, dr_idx]):
                    response_matrix[cl_idx, dr_idx] = batch.response[pair_idx]
                else:
                    # Simple running average via pandas-like approach
                    response_matrix[cl_idx, dr_idx] = (response_matrix[cl_idx, dr_idx] + batch.response[pair_idx]) / 2

        # Handle duplicate pairs more precisely with averaging
        self._build_response_matrix_averaged(batch, cl_id_to_idx, dr_id_to_idx, response_matrix)

        # Weight matrix: 1 where we have observed data
        w = ~np.isnan(response_matrix)
        response_matrix_filled = response_matrix.copy()
        response_matrix_filled[np.isnan(response_matrix_filled)] = 0.0

        # Train: note the transposition — _cmf expects (drugs x cell_lines)
        best_u, best_v = self._cmf(
            w=w.T,
            int_mat=response_matrix_filled.T,
            drug_mat=drug_similarity,
            cell_mat=cell_line_similarity,
        )

        self._best_u = pd.DataFrame(best_u, index=[str(d) for d in drugs])
        self._best_v = pd.DataFrame(best_v, index=[str(c) for c in cell_lines])
        self._training_mean = float(np.nanmean(batch.response))

    @staticmethod
    def _build_response_matrix_averaged(
        batch: ModelInputBatch,
        cl_id_to_idx: dict[str, int],
        dr_id_to_idx: dict[str, int],
        response_matrix: np.ndarray,
    ) -> None:
        """Fill response_matrix with per-pair averaged responses.

        Overwrites the naive fill with proper grouped means matching old pandas logic.

        :param batch: Training batch with response data.
        :param cl_id_to_idx: Cell-line ID to matrix row mapping.
        :param dr_id_to_idx: Drug ID to matrix column mapping.
        :param response_matrix: Matrix to fill in-place.
        :raises ValueError: If batch has no response data.
        """
        if batch.response is None:
            msg = "SRMF requires training response data"
            raise ValueError(msg)
        sums: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for pair_idx in range(batch.n_pairs):
            cl_idx = cl_id_to_idx.get(str(batch.cell_line_ids[pair_idx]))
            dr_idx = dr_id_to_idx.get(str(batch.drug_ids[pair_idx]))
            if cl_idx is not None and dr_idx is not None:
                key = (cl_idx, dr_idx)
                sums[key] = sums.get(key, 0.0) + batch.response[pair_idx]
                counts[key] = counts.get(key, 0) + 1

        for (ci, di), total in sums.items():
            response_matrix[ci, di] = total / counts[(ci, di)]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses using learned latent factors.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        """
        drug_ids = batch.drug_ids
        cell_line_ids = batch.cell_line_ids

        best_u = np.full((len(drug_ids), self._k), self._training_mean)
        for idx, drug in enumerate(drug_ids):
            key = str(drug)
            if key in self._best_u.index:
                best_u[idx, :] = self._best_u.loc[key].values

        best_v = np.full((len(cell_line_ids), self._k), self._training_mean)
        for idx, cell in enumerate(cell_line_ids):
            key = str(cell)
            if key in self._best_v.index:
                best_v[idx, :] = self._best_v.loc[key].values

        return np.einsum("ij,ji->i", best_u, best_v.T)

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit.

        :returns: True when latent factors have been learned.
        """
        return not self._best_u.empty

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob.
        """
        if not self.is_fitted():
            return {}
        payload: dict[str, Any] = {
            "best_u": self._best_u.to_dict(orient="split"),
            "best_v": self._best_v.to_dict(orient="split"),
            "training_mean": self._training_mean,
            "predictor_hyperparameters": dict(self._hyperparameters),
        }
        return {"payload": save_object_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        blob = state.get("payload")
        if not isinstance(blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} state requires a payload byte blob"
            raise PredictorStateError(msg)
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = f"{self.__class__.__name__} payload could not be deserialized"
            raise PredictorStateError(msg) from exc
        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        self._best_u = pd.DataFrame(**payload["best_u"])
        self._best_v = pd.DataFrame(**payload["best_v"])
        self._training_mean = float(payload.get("training_mean", 0.0))

    # ------------------------------------------------------------------
    # SRMF algorithm internals
    # ------------------------------------------------------------------

    def _cmf(
        self,
        w: np.ndarray,
        int_mat: np.ndarray,
        drug_mat: np.ndarray,
        cell_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collective matrix factorization with similarity regularization.

        :param w: Binary weight matrix (drugs x cell_lines).
        :param int_mat: Response interaction matrix (drugs x cell_lines).
        :param drug_mat: Drug-drug similarity matrix.
        :param cell_mat: Cell-line similarity matrix.
        :returns: Best drug and cell-line latent factors.
        """
        rng = np.random.default_rng(self._seed)
        m, n = w.shape
        u0 = np.sqrt(1 / self._k) * rng.standard_normal(size=(m, self._k))
        v0 = np.sqrt(1 / self._k) * rng.standard_normal(size=(n, self._k))

        best_u, best_v = u0, v0

        last_loss = self._compute_loss(u0, v0, w, int_mat, drug_mat, cell_mat)
        best_loss = last_loss
        wr = w * int_mat

        for _ in range(self._max_iter):
            u = self._alg_update(u0, v0, w, wr, drug_mat, self._lambda_l, self._lambda_d)
            v = self._alg_update(v0, u, w.T, wr.T, cell_mat, self._lambda_l, self._lambda_c)
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

    def _compute_loss(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        int_mat: np.ndarray,
        drug_mat: np.ndarray,
        cell_mat: np.ndarray,
    ) -> np.float64:
        """Compute SRMF loss including similarity regularization.

        :param u: Drug latent factors.
        :param v: Cell-line latent factors.
        :param w: Binary weight matrix.
        :param int_mat: Response interaction matrix.
        :param drug_mat: Drug-drug similarity matrix.
        :param cell_mat: Cell-line similarity matrix.
        :returns: Total loss value.
        """
        loss = np.sum((w * (int_mat - np.dot(u, v.T))) ** 2)
        loss += self._lambda_l * (np.sum(u**2) + np.sum(v**2))
        loss += self._lambda_d * np.sum((drug_mat - np.dot(u, u.T)) ** 2)
        loss += self._lambda_c * np.sum((cell_mat - np.dot(v, v.T)) ** 2)
        return loss

    @staticmethod
    def _alg_update(
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        lambda_l: float,
        lambda_d: float,
    ) -> np.ndarray:
        """SRMF alternating update rule for latent factor matrix.

        :param u: Current latent factor matrix to update.
        :param v: Other latent factor matrix (fixed).
        :param w: Binary weight matrix.
        :param r: Weighted response matrix.
        :param s: Similarity matrix for regularization.
        :param lambda_l: L2 regularization weight.
        :param lambda_d: Similarity regularization weight.
        :returns: Updated latent factor matrix.
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
