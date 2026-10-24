"""SRMF block predictor – Similarity Regularization Matrix Factorization.

Original publication: Wang, L., Li, X., Zhang, L. et al. Improved anticancer drug response prediction in cell lines
using matrix factorization with similarity regularization. BMC Cancer 17, 513 (2017).
https://doi.org/10.1186/s12885-017-3500-5.
Matlab code adapted from https://github.com/linwang1982/SRMF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import SRMF_REFERENCE
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import save_trusted_mapping

if TYPE_CHECKING:
    import pandas as pd


@register(
    "srmf",
    description="SRMF matrix factorization model.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SRMF_REFERENCE,
)
class SRMFPredictor(BlockPredictor):
    """Registered SRMF predictor using similarity-regularized matrix factorization."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize SRMF predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        import pandas as pd

        super().__init__(hyperparameters)
        self._best_u: pd.DataFrame = pd.DataFrame()
        self._best_v: pd.DataFrame = pd.DataFrame()
        self._training_mean: float = 0.0

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

        :returns: Hyperparameter search space specs.
        """
        return {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit the SRMF model on training data.

        :param batch: Training batch with gene_expression and fingerprints blocks.
        :raises ValueError: If drug features are missing.
        """
        import pandas as pd

        if batch.drug is None:
            msg = "SRMF requires drug features"
            raise ValueError(msg)

        # Sorted ids, as drevalpy 1.6.1 (np.unique): the seeded initial factors are drawn row by row.
        sample_order, drug_order = (
            np.argsort(ids.astype(str), kind="stable") for ids in (batch.sample.entity_ids, batch.drug.entity_ids)
        )
        samples, drugs = batch.sample.entity_ids[sample_order], batch.drug.entity_ids[drug_order]
        response_matrix = self._response_matrix(batch, samples, drugs)
        observed = ~np.isnan(response_matrix)
        response_matrix_filled = np.where(observed, response_matrix, 0.0)

        # _cmf works in (drugs x samples) orientation, hence the transpositions.
        best_u, best_v = self._cmf(
            w=observed.T,
            int_mat=response_matrix_filled.T,
            drug_mat=self._drug_similarity(batch.drug.blocks["fingerprints"].values[drug_order]),
            cell_mat=np.corrcoef(batch.sample.blocks["gene_expression"].values[sample_order], rowvar=True),
        )

        self._best_u = pd.DataFrame(best_u, index=[str(d) for d in drugs])
        self._best_v = pd.DataFrame(best_v, index=[str(c) for c in samples])
        self._training_mean = float(np.nanmean(batch.response))

    @staticmethod
    def _drug_similarity(drug_features: np.ndarray) -> np.ndarray:
        """Compute pairwise Jaccard similarity over binary fingerprints.

        :param drug_features: Binary fingerprint matrix, one row per drug.
        :returns: Symmetric similarity matrix with ones on the diagonal.
        """
        from scipy.spatial.distance import jaccard

        n_drugs = len(drug_features)
        similarity = np.eye(n_drugs, dtype=np.float64)
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                similarity[i, j] = similarity[j, i] = 1.0 - jaccard(drug_features[i], drug_features[j])
        return similarity

    @staticmethod
    def _response_matrix(
        batch: ModelInputBatch,
        samples: np.ndarray,
        drugs: np.ndarray,
    ) -> np.ndarray:
        """Pivot the pair-level response into a sample x drug matrix.

        Duplicate pairs are averaged, matching the grouped mean the original pandas
        implementation took. Unobserved cells stay ``NaN`` so the caller can derive the
        weight mask from them.

        :param batch: Training batch with response data.
        :param samples: Sample entity ids, in row order.
        :param drugs: Drug entity ids, in column order.
        :returns: Response matrix with ``NaN`` for unobserved pairs.
        :raises ValueError: If batch has no response data.
        """
        if batch.response is None:
            msg = "SRMF requires training response data"
            raise ValueError(msg)

        cl_id_to_idx = {str(cid): i for i, cid in enumerate(samples)}
        dr_id_to_idx = {str(did): i for i, did in enumerate(drugs)}
        sums: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for pair_idx in range(batch.n_pairs):
            cl_idx = cl_id_to_idx.get(str(batch.sample_ids[pair_idx]))
            dr_idx = dr_id_to_idx.get(str(batch.drug_ids[pair_idx]))
            if cl_idx is None or dr_idx is None:
                continue
            key = (cl_idx, dr_idx)
            sums[key] = sums.get(key, 0.0) + batch.response[pair_idx]
            counts[key] = counts.get(key, 0) + 1

        matrix = np.full((len(samples), len(drugs)), np.nan, dtype=np.float64)
        for (ci, di), total in sums.items():
            matrix[ci, di] = total / counts[(ci, di)]
        return matrix

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses using learned latent factors.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        """
        drug_ids, sample_ids = batch.drug_ids.astype(str), batch.sample_ids.astype(str)
        best_u = np.ascontiguousarray(self._best_u.reindex(drug_ids).to_numpy())
        best_v = np.ascontiguousarray(self._best_v.reindex(sample_ids).to_numpy())
        known = np.isin(drug_ids, self._best_u.index) & np.isin(sample_ids, self._best_v.index)
        # A mean response is a prediction fallback, not a latent coordinate:
        # multiplying mean-filled vectors would return K * mean**2.
        return np.where(known, np.einsum("ij,ji->i", best_u, best_v.T), self._training_mean)

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
        return {"payload": save_trusted_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        import pandas as pd

        payload = load_payload_state(state, "SRMF")
        try:
            best_u = pd.DataFrame(**payload["best_u"])
            best_v = pd.DataFrame(**payload["best_v"])
            training_mean = float(payload["training_mean"])
        except (KeyError, TypeError) as exc:
            msg = "SRMF payload is missing best_u, best_v or training_mean"
            raise PredictorStateError(msg) from exc
        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._best_u = best_u
        self._best_v = best_v
        self._training_mean = training_mean

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
        """Factorize the response matrix jointly with drug and sample similarity regularization.

        :param w: Binary weight matrix (drugs x samples).
        :param int_mat: Response interaction matrix (drugs x samples).
        :param drug_mat: Drug-drug similarity matrix.
        :param cell_mat: Sample similarity matrix.
        :returns: Best drug and sample latent factors.
        """
        hp = self._hyperparameters
        k = int(hp["K"])
        lambda_l, lambda_d, lambda_c = float(hp["lambda_l"]), float(hp["lambda_d"]), float(hp["lambda_c"])
        rng = np.random.default_rng(int(hp["seed"]))
        m, n = w.shape
        u0 = np.sqrt(1 / k) * rng.standard_normal(size=(m, k))
        v0 = np.sqrt(1 / k) * rng.standard_normal(size=(n, k))

        best_u, best_v = u0, v0

        last_loss = self._compute_loss(u0, v0, w, int_mat, drug_mat, cell_mat)
        best_loss = last_loss
        wr = w * int_mat

        for _ in range(int(hp["max_iter"])):
            u = self._alg_update(u0, v0, w, wr, drug_mat, lambda_l, lambda_d)
            v = self._alg_update(v0, u, w.T, wr.T, cell_mat, lambda_l, lambda_c)
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
        :param v: Sample latent factors.
        :param w: Binary weight matrix.
        :param int_mat: Response interaction matrix.
        :param drug_mat: Drug-drug similarity matrix.
        :param cell_mat: Sample similarity matrix.
        :returns: Total loss value.
        """
        loss = np.sum((w * (int_mat - np.dot(u, v.T))) ** 2)
        hp = self._hyperparameters
        loss += float(hp["lambda_l"]) * (np.sum(u**2) + np.sum(v**2))
        loss += float(hp["lambda_d"]) * np.sum((drug_mat - np.dot(u, u.T)) ** 2)
        loss += float(hp["lambda_c"]) * np.sum((cell_mat - np.dot(v, v.T)) ** 2)
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
