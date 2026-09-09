"""
EnsembleMF: an ensembled two-tower matrix factorization for drug-response prediction.

Predicts the cell-line x drug response matrix as ``R = U V^T``, where the latent factors are not
free parameters but the outputs of small residual MLPs over cell-line and drug features. That
keeps the model usable in leave-cell-line-out, where a held-out cell line has features but no
observed responses and therefore no free factor to fit.

On top of the dot product sit per-cell, per-drug, and global bias terms, a free per-drug id
embedding, and a small interaction head. The id embedding is indexed by drug identity and only
ever contributes for a drug that actually appeared in a training batch; a drug held out of
training (leave-drug-out, or a drug requested only at predict time) is scored purely from its
fingerprint instead of an untrained embedding. Predictions are averaged over an ensemble of
independently initialized members.
"""

import os
from typing import Any, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features, load_drug_fingerprint_features


def _select_device() -> torch.device:
    """
    Pick CUDA if available, else CPU.

    :returns: the selected torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ResidualEncoder(nn.Module):
    """Project features to a latent factor through residual blocks, with a direct input skip."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.skip_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into latent factors.

        :param x: (n_nodes, in_dim) feature matrix
        :returns: (n_nodes, emb_dim) latent factors
        """
        h0 = self.act(self.input_proj(x))
        h = self.dropout(h0)
        for block, norm in zip(self.blocks, self.norms):
            h = self.dropout(self.act(norm(block(h)))) + h
        return self.output_proj(h) + self.skip_proj(h0)


class _MFNet(nn.Module):
    """One ensemble member: two encoders plus a bilinear head with main effects."""

    def __init__(
        self,
        cell_in_dim: int,
        drug_in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        n_layers: int,
        dropout: float,
        n_drugs: int,
        use_drug_id_embedding: bool,
        use_mlp_head: bool,
        mlp_hidden: int,
    ):
        super().__init__()
        self.cell_encoder = _ResidualEncoder(cell_in_dim, hidden_dim, emb_dim, n_layers, dropout)
        self.drug_encoder = _ResidualEncoder(drug_in_dim, hidden_dim, emb_dim, n_layers, dropout)

        # Every drug is present in training (only cell lines are held out), so a free id-indexed
        # latent is learnable here and transfers to test pairs. Cell lines get no such table: they
        # are unseen at test time and must stay purely feature-derived.
        self.use_drug_id_embedding = use_drug_id_embedding
        if use_drug_id_embedding:
            self.drug_id_emb = nn.Embedding(n_drugs, emb_dim)
            nn.init.normal_(self.drug_id_emb.weight, std=0.01)

        self.cell_bias = nn.Linear(emb_dim, 1)
        self.drug_bias = nn.Linear(emb_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dot_scale = nn.Parameter(torch.ones(1))

        self.use_mlp_head = use_mlp_head
        if use_mlp_head:
            self.mlp = nn.Sequential(
                nn.Linear(3 * emb_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
            )

    def encode(
        self, x_cell: torch.Tensor, x_drug: torch.Tensor, drug_id_emb_rows: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute latent factors for every cell line and every drug.

        :param x_cell: (n_cells, cell_in_dim) cell-line features
        :param x_drug: (n_drugs, drug_in_dim) drug features
        :param drug_id_emb_rows: (n_drugs, emb_dim) id-embedding row to add per drug, aligned with
            ``x_drug``'s row order; defaults to this net's own embedding table, which is only
            correct when ``x_drug`` uses the training-time drug ordering. A caller encoding a
            different drug ordering (e.g. predict-time cross-study features) must gather the
            right row per drug itself, using zeros for drugs that never appeared in training.
        :returns: (cell factors, drug factors)
        """
        z_cell = self.cell_encoder(x_cell)
        z_drug = self.drug_encoder(x_drug)
        if self.use_drug_id_embedding:
            z_drug = z_drug + (self.drug_id_emb.weight if drug_id_emb_rows is None else drug_id_emb_rows)
        return z_cell, z_drug

    def score_pairs(self, z_cell: torch.Tensor, z_drug: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of (cell, drug) pairs from their gathered factors.

        :param z_cell: (batch, emb_dim) cell factors
        :param z_drug: (batch, emb_dim) drug factors
        :returns: (batch,) predicted responses
        """
        pred = (z_cell * z_drug).sum(dim=-1, keepdim=True) * self.dot_scale
        pred = pred + self.cell_bias(z_cell) + self.drug_bias(z_drug) + self.global_bias
        if self.use_mlp_head:
            pred = pred + self.mlp(torch.cat([z_cell, z_drug, z_cell * z_drug], dim=-1))
        return pred.squeeze(-1)


class EnsembleMF(DRPModel):
    """Ensembled two-tower matrix factorization over gene expression and drug fingerprints."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the model; the networks are built in ``train`` once the dimensions are known."""
        super().__init__()
        self.nets: list[_MFNet] = []
        self.hyperparameters: dict[str, Any] = {}
        self.device = _select_device()

        # filled in by train(), used internally during training; predict() re-encodes fresh
        # features instead of reusing these (see _encode_dataset).
        self._cell_id_to_idx: dict[str, int] = {}
        self._drug_id_to_idx: dict[str, int] = {}
        self._drug_seen_mask: np.ndarray = np.array([], dtype=bool)
        self._x_cell: torch.Tensor | None = None
        self._x_drug: torch.Tensor | None = None
        self._cell_in_dim: int = 0
        self._drug_in_dim: int = 0
        self._n_drugs: int = 0
        self._rank_reference: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self.training_mean: float = 0.0

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "EnsembleMF"."""
        return "EnsembleMF"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store hyperparameters.

        Per-ensemble-member seeding happens inside ``train()``, scoped to a forked RNG state so
        it cannot leak into other code sharing the process. Unlike ``experiment.seed_everything``
        (meant to be called once at the top of a run), this model never touches the global RNG.

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load gene expression, reduced to the configured gene list.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "gene_expression" view
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list=self.hyperparameters.get("gene_list", "gene_expression_intersection"),
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load Morgan fingerprints.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "fingerprints" view
        """
        return load_drug_fingerprint_features(
            data_path, dataset_name, fill_na=True, n_bits=int(self.hyperparameters.get("n_bits", 256))
        )

    def _build_cell_matrix(
        self, cell_line_input: FeatureDataset, cell_ids: np.ndarray, train_ids: np.ndarray
    ) -> np.ndarray:
        """
        Transform and standardize the cell-line features.

        ``feature_transform`` picks between ``rank`` (each gene's percentile position against a
        reference distribution) and ``arcsinh`` (a plain pointwise transform). Both the rank
        reference and the scaler are fit once, on training cell lines only, and reused as-is for
        every later call - so a single new cell line (one row) is scored against that fixed
        reference rather than against itself, and the same cell line gets the same features
        regardless of what other cell lines happen to be requested alongside it in the same call.

        :param cell_line_input: cell-line FeatureDataset
        :param cell_ids: ordered cell-line ids (all cell lines with features)
        :param train_ids: cell-line ids present in the training responses; empty to reuse the
            rank reference and scaler fitted during train()
        :returns: (n_cells, n_genes) scaled feature matrix
        :raises ValueError: if train_ids is empty and no rank reference/scaler has been fit yet
        """
        mat = cell_line_input.get_feature_matrix(view="gene_expression", identifiers=cell_ids).astype(np.float64)
        feature_transform = str(self.hyperparameters.get("feature_transform", "rank"))
        if feature_transform == "rank":
            if len(train_ids) > 0:
                self._rank_reference = np.sort(mat[np.isin(cell_ids, np.unique(train_ids))], axis=0)
            elif self._rank_reference is None:
                raise ValueError(
                    "No fitted rank reference available: train() must be called with at least "
                    "one training response whose cell line has features before predict() can "
                    "reuse it."
                )
            reference = cast(np.ndarray, self._rank_reference)
            n_reference = reference.shape[0]
            percentile = np.empty_like(mat)
            for gene in range(mat.shape[1]):
                percentile[:, gene] = np.searchsorted(reference[:, gene], mat[:, gene], side="left")
            mat = np.clip(percentile / max(1, n_reference - 1), 0.0, 1.0)
        elif feature_transform == "arcsinh":
            mat = np.arcsinh(mat)
        else:
            raise ValueError(f"Unknown feature_transform {feature_transform!r}; expected 'rank' or 'arcsinh'.")
        if len(train_ids) > 0:
            self._scaler = StandardScaler().fit(mat[np.isin(cell_ids, np.unique(train_ids))])
        elif self._scaler is None:
            raise ValueError(
                "No fitted scaler available: train() must be called with at least one training "
                "response whose cell line has features before the scaler can be reused."
            )
        return cast(StandardScaler, self._scaler).transform(mat).astype(np.float32)

    def _encode_dataset(
        self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, train_cell_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
        """
        Build the cell/drug feature tensors and id->index maps for one feature dataset pair.

        Called once in train() (fitting the scaler on train_cell_ids) and once per predict() call
        (with an empty train_cell_ids, reusing the already-fitted scaler) - so predict() always
        encodes the features it is actually handed, rather than reusing train()'s cached tensors.
        That matters for cross-study prediction, where predict() receives a different dataset's
        features and those must not be silently ignored in favor of stale training-time values.

        :param cell_line_input: cell-line FeatureDataset to encode
        :param drug_input: drug FeatureDataset to encode
        :param train_cell_ids: cell-line ids to fit the scaler on; empty to reuse the existing one
        :returns: (x_cell, x_drug, cell_ids, drug_ids, cell_id_to_idx, drug_id_to_idx)
        """
        cell_ids = np.unique(cell_line_input.identifiers)
        drug_ids = np.unique(drug_input.identifiers)
        x_cell = self._build_cell_matrix(cell_line_input, cell_ids, train_cell_ids)
        x_drug = drug_input.get_feature_matrix(view="fingerprints", identifiers=drug_ids).astype(np.float32)
        return (
            torch.tensor(x_cell, device=self.device),
            torch.tensor(x_drug, device=self.device),
            cell_ids,
            drug_ids,
            {c: i for i, c in enumerate(cell_ids)},
            {d: i for i, d in enumerate(drug_ids)},
        )

    def _pairs(self, data: DrugResponseDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map a response dataset to (cell index, drug index, response) tensors, dropping unknown ids.

        :param data: response dataset to convert
        :returns: (cell idx, drug idx, response) tensors on the model device
        """
        ci, di, y = [], [], []
        for cl, dr, resp in zip(data.cell_line_ids, data.drug_ids, data.response):
            if cl in self._cell_id_to_idx and dr in self._drug_id_to_idx:
                ci.append(self._cell_id_to_idx[cl])
                di.append(self._drug_id_to_idx[dr])
                y.append(resp)
        return (
            torch.tensor(ci, dtype=torch.long, device=self.device),
            torch.tensor(di, dtype=torch.long, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device),
        )

    def _build_net(self) -> _MFNet:
        """
        Instantiate one ensemble member from the stored dimensions and hyperparameters.

        :returns: a new ``_MFNet``
        """
        hp = self.hyperparameters
        return _MFNet(
            cell_in_dim=self._cell_in_dim,
            drug_in_dim=self._drug_in_dim,
            hidden_dim=int(hp.get("hidden_dim", 256)),
            emb_dim=int(hp.get("emb_dim", 128)),
            n_layers=int(hp.get("n_layers", 1)),
            dropout=float(hp.get("dropout", 0.2)),
            n_drugs=self._n_drugs,
            use_drug_id_embedding=bool(hp.get("use_drug_id_embedding", True)),
            use_mlp_head=bool(hp.get("use_mlp_head", True)),
            mlp_hidden=int(hp.get("mlp_hidden", 256)),
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
        Train the ensemble on the observed (cell, drug) pairs.

        :param output: training responses
        :param cell_line_input: cell-line features (all cell lines)
        :param drug_input: drug features (all drugs)
        :param output_earlystopping: responses used for early stopping; a 10% split of the
            training pairs is carved out when this is None
        :param model_checkpoint_dir: unused, kept for interface compatibility
        :raises ValueError: if drug_input is None; if no training pair matches the feature
            datasets; or if output_earlystopping is non-empty but none of its pairs do
        """
        if drug_input is None:
            raise ValueError("EnsembleMF requires drug features (fingerprints).")
        hp = self.hyperparameters
        self.training_mean = float(np.nanmean(output.response))

        x_cell, x_drug, _cell_ids, drug_ids, self._cell_id_to_idx, self._drug_id_to_idx = self._encode_dataset(
            cell_line_input, drug_input, train_cell_ids=np.asarray(output.cell_line_ids)
        )
        self._x_cell, self._x_drug = x_cell, x_drug
        self._cell_in_dim, self._drug_in_dim = int(x_cell.shape[1]), int(x_drug.shape[1])
        self._n_drugs = int(x_drug.shape[0])

        ci, di, y = self._pairs(output)
        if len(y) == 0:
            raise ValueError("No training pairs matched the cell-line/drug feature sets; there is nothing to train.")
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            val = self._pairs(output_earlystopping)
            if len(val[2]) == 0:
                raise ValueError(
                    "output_earlystopping was provided but none of its cell lines/drugs matched "
                    "the feature datasets, so there is nothing to evaluate early stopping on. "
                    "Pass a dataset that overlaps the feature sets, or omit output_earlystopping "
                    "to fall back to the automatic 10% split of the training pairs."
                )
        else:
            perm = torch.randperm(len(y), device=self.device)
            n_val = max(1, int(0.1 * len(y)))
            val = (ci[perm[:n_val]], di[perm[:n_val]], y[perm[:n_val]])
            ci, di, y = ci[perm[n_val:]], di[perm[n_val:]], y[perm[n_val:]]

        # A drug's free id-embedding only ever receives a gradient for drugs that end up in a
        # training batch here; predict() must not add that embedding for any other drug (see
        # _MFNet.encode's drug_id_emb_rows).
        self._drug_seen_mask = np.zeros(len(drug_ids), dtype=bool)
        self._drug_seen_mask[di.unique().cpu().numpy()] = True

        seed = int(hp.get("seed", 0))
        self.nets = []
        for member in range(int(hp.get("n_ensemble", 20))):
            fork_devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                if seed >= 0:
                    torch.manual_seed(seed + member)
                net = self._build_net().to(self.device)
                self._train_net(net, ci, di, y, val)
            self.nets.append(net)

    def _train_net(
        self,
        net: _MFNet,
        ci: torch.Tensor,
        di: torch.Tensor,
        y: torch.Tensor,
        val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Train one ensemble member in place, keeping the weights with the best validation MSE.

        :param net: the network to train (modified in place)
        :param ci: (n_train,) cell indices of the training pairs
        :param di: (n_train,) drug indices of the training pairs
        :param y: (n_train,) target responses
        :param val: (cell idx, drug idx, response) tensors for early stopping
        """
        hp = self.hyperparameters
        x_cell, x_drug = cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug)
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(hp.get("learning_rate", 3e-4)),
            weight_decay=float(hp.get("weight_decay", 1e-5)),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(hp.get("batch_size", 2048))
        patience = int(hp.get("patience", 25))
        best_val, best_state, stale = float("inf"), None, 0

        for _epoch in range(int(hp.get("max_epochs", 300))):
            net.train()
            perm = torch.randperm(len(y), device=self.device)
            for start in range(0, len(y), batch_size):
                end = start + batch_size
                idx = perm[start:end]
                batch_ci, batch_di = ci[idx], di[idx]
                uniq_ci, inv_ci = torch.unique(batch_ci, return_inverse=True)
                uniq_di, inv_di = torch.unique(batch_di, return_inverse=True)
                optimizer.zero_grad()
                drug_id_emb_rows = net.drug_id_emb.weight[uniq_di] if net.use_drug_id_embedding else None
                z_cell, z_drug = net.encode(x_cell[uniq_ci], x_drug[uniq_di], drug_id_emb_rows)
                loss = loss_fn(net.score_pairs(z_cell[inv_ci], z_drug[inv_di]), y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

            val_mse = self._eval_mse(net, val)
            if val_mse < best_val - 1e-6:
                best_val, stale = val_mse, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

    @torch.no_grad()
    def _eval_mse(self, net: _MFNet, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Mean squared error of one member on the early-stopping pairs.

        :param net: the network to evaluate
        :param val: (cell idx, drug idx, response) tensors
        :returns: mean squared error, or inf if there is nothing to evaluate
        """
        ci, di, y = val
        if len(y) == 0:
            return float("inf")
        net.eval()
        z_cell, z_drug = net.encode(cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug))
        return float(nn.functional.mse_loss(net.score_pairs(z_cell[ci], z_drug[di]), y).item())

    @torch.no_grad()
    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict responses for (cell, drug) pairs, averaging over the ensemble.

        Cell-line and drug factors are (re-)encoded here from ``cell_line_input``/``drug_input``
        using the scaler fitted during ``train()`` - they are not reused from train()'s cached
        tensors. That means a cross-study prediction call (a different dataset's features) is
        scored on that dataset's own features rather than silently falling back to stale
        training-study values for any id that happens to collide by name. A pair whose cell line
        or drug is absent from the supplied feature datasets falls back to the training mean. The
        free per-drug id embedding only ever contributes for drugs that actually appeared in a
        training batch; every other drug - an unseen leave-drug-out test drug, or any drug from a
        different study - is scored purely from its fingerprint.

        :param cell_line_ids: cell-line ids to predict
        :param drug_ids: drug ids to predict
        :param cell_line_input: cell-line features to encode (may differ from the training
            dataset, e.g. for cross-study prediction)
        :param drug_input: drug features to encode
        :raises ValueError: if drug_input is None
        :returns: (n,) predicted responses
        """
        if drug_input is None:
            raise ValueError("EnsembleMF requires drug features (fingerprints).")
        preds = np.full(len(cell_line_ids), self.training_mean, dtype=np.float32)
        if not self.nets:
            return preds

        x_cell, x_drug, _cell_ids, drug_ids_fresh, cell_id_to_idx, drug_id_to_idx = self._encode_dataset(
            cell_line_input, drug_input, train_cell_ids=np.array([])
        )

        rows = [
            (i, cell_id_to_idx[c], drug_id_to_idx[d])
            for i, (c, d) in enumerate(zip(cell_line_ids, drug_ids))
            if c in cell_id_to_idx and d in drug_id_to_idx
        ]
        if not rows:
            return preds
        idx, ci, di = (np.array(v) for v in zip(*rows))
        ci_t = torch.tensor(ci, dtype=torch.long, device=self.device)
        di_t = torch.tensor(di, dtype=torch.long, device=self.device)

        # Only add a drug's trained id-embedding row where it actually has one: the drug must
        # both be known from training (present in _drug_id_to_idx) and have appeared in a
        # training batch (_drug_seen_mask), not merely have had features available at train time.
        use_emb = bool(self.nets[0].use_drug_id_embedding)
        train_idx_for_drug = np.empty(0, dtype=np.int64)
        seen_mask = np.empty(0, dtype=bool)
        if use_emb:
            train_idx_for_drug = np.array([self._drug_id_to_idx.get(d, -1) for d in drug_ids_fresh], dtype=np.int64)
            seen_mask = train_idx_for_drug >= 0
            seen_mask[seen_mask] = self._drug_seen_mask[train_idx_for_drug[seen_mask]]

        member_preds = []
        for net in self.nets:
            net.eval()
            drug_id_emb_rows = None
            if use_emb:
                drug_id_emb_rows = torch.zeros(len(drug_ids_fresh), net.drug_id_emb.embedding_dim, device=self.device)
                if seen_mask.any():
                    drug_id_emb_rows[seen_mask] = net.drug_id_emb.weight[train_idx_for_drug[seen_mask]]
            z_cell, z_drug = net.encode(x_cell, x_drug, drug_id_emb_rows)
            member_preds.append(net.score_pairs(z_cell[ci_t], z_drug[di_t]).cpu().numpy())
        preds[idx] = np.mean(member_preds, axis=0)
        return preds

    def save(self, directory: str) -> None:
        """
        Persist the trained ensemble.

        :param directory: target directory
        :raises RuntimeError: if the model has not been trained
        """
        if not self.nets:
            raise RuntimeError("No trained model to save.")
        os.makedirs(directory, exist_ok=True)
        torch.save([net.state_dict() for net in self.nets], os.path.join(directory, "nets.pt"))  # noqa: S614
        joblib.dump(
            {
                "hyperparameters": self.hyperparameters,
                "drug_id_to_idx": self._drug_id_to_idx,
                "drug_seen_mask": self._drug_seen_mask,
                "cell_in_dim": self._cell_in_dim,
                "drug_in_dim": self._drug_in_dim,
                "n_drugs": self._n_drugs,
                "rank_reference": self._rank_reference,
                "scaler": self._scaler,
                "training_mean": self.training_mean,
            },
            os.path.join(directory, "state.pkl"),
        )

    @classmethod
    def load(cls, directory: str) -> "EnsembleMF":
        """
        Restore a model saved with ``save``.

        :param directory: directory containing the saved files
        :returns: the restored model
        """
        instance = cls()
        state = joblib.load(os.path.join(directory, "state.pkl"))
        instance.build_model(state["hyperparameters"])
        instance._drug_id_to_idx = state["drug_id_to_idx"]
        instance._drug_seen_mask = state["drug_seen_mask"]
        instance._cell_in_dim = state["cell_in_dim"]
        instance._drug_in_dim = state["drug_in_dim"]
        instance._n_drugs = state["n_drugs"]
        instance._rank_reference = state["rank_reference"]
        instance._scaler = state["scaler"]
        instance.training_mean = state["training_mean"]
        # map_location: a model trained on a GPU node must still load on a CPU-only machine
        state_dicts = torch.load(os.path.join(directory, "nets.pt"), map_location=instance.device)  # noqa: S614
        instance.nets = []
        for sd in state_dicts:
            net = instance._build_net().to(instance.device)
            net.load_state_dict(sd)
            net.eval()
            instance.nets.append(net)
        return instance
