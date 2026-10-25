"""EnsembleMF block literature predictor.

An ensembled two-tower matrix factorization over sample gene expression and drug
fingerprints. ``torch``, ``.network`` and ``.training`` are imported inside the methods
that need them: ``drevalpy.registry`` imports this module to register the ``ensembleMF``
predictor on ``import drevalpy``, so a module-scope ``import torch`` would put the
training stack on the startup path of every CLI invocation. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict, save_trusted_mapping
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import ENSEMBLEMF_REFERENCE
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    import torch

    from .network import MFNet
    from .training import PairTensors


@register(
    "ensembleMF",
    description="EnsembleMF: ensembled two-tower matrix factorization over expression and fingerprints.",
    sample_input_format=FeatureFormat.NUMERIC_MATRIX,
    drug_input_format=FeatureFormat.NUMERIC_MATRIX,
    reference=ENSEMBLEMF_REFERENCE,
)
class EnsembleMFPredictor(BlockPredictor):
    """Ensemble of two-tower matrix-factorization networks consuming ModelInputBatch directly."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    required_sample_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),
    )
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the EnsembleMF predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        import torch

        super().__init__(hyperparameters)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nets: list[MFNet] = []
        self._drug_ids = np.empty(0, dtype=str)
        self._drug_seen = np.empty(0, dtype=bool)
        self._input_dims = (0, 0)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return the DrEvalPy 1.6.1 defaults.

        :returns: Default hyperparameter mapping.
        """
        return {
            "hidden_dim": 256,
            "emb_dim": 128,
            "n_layers": 1,
            "dropout": 0.2,
            "use_mlp_head": True,
            "mlp_hidden": 256,
            "use_drug_id_embedding": True,
            "n_ensemble": 20,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "batch_size": 2048,
            "max_epochs": 300,
            "patience": 25,
            "seed": 0,
        }

    def _build_net(self, n_drugs: int) -> MFNet:
        """Construct one ensemble member on the target device.

        :param n_drugs: Number of rows of the drug id-embedding table.
        :returns: Freshly initialized network.
        """
        from .network import MFNet, MFNetConfig

        hp = self._hyperparameters
        config = MFNetConfig(
            sample_in_dim=self._input_dims[0],
            drug_in_dim=self._input_dims[1],
            n_drugs=n_drugs,
            hidden_dim=int(hp["hidden_dim"]),
            emb_dim=int(hp["emb_dim"]),
            n_layers=int(hp["n_layers"]),
            dropout=float(hp["dropout"]),
            use_drug_id_embedding=bool(hp["use_drug_id_embedding"]),
            use_mlp_head=bool(hp["use_mlp_head"]),
            mlp_hidden=int(hp["mlp_hidden"]),
        )
        return MFNet(config).to(self._device)

    def _pairs(self, sample_idx: np.ndarray, drug_idx: np.ndarray, response: np.ndarray) -> PairTensors:
        """Move pair indices and responses to the target device.

        :param sample_idx: Row of each pair's sample in the sorted sample matrix.
        :param drug_idx: Row of each pair's drug in the sorted drug matrix.
        :param response: Response of each pair.
        :returns: The pairs as tensors.
        """
        import torch

        from .training import PairTensors

        return PairTensors(
            torch.as_tensor(sample_idx, dtype=torch.long, device=self._device),
            torch.as_tensor(drug_idx, dtype=torch.long, device=self._device),
            torch.as_tensor(response, dtype=torch.float32, device=self._device),
        )

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the ensemble members with their own seeds.

        :param batch: Training batch with gene_expression and fingerprints blocks.
        """
        import torch

        from .data import drug_table, sorted_entities, split_pairs
        from .training import MemberData, MemberSettings, train_member

        hp = self._hyperparameters
        _, x_sample, sample_pos = sorted_entities(batch.sample, "gene_expression")
        drug_ids, x_drug, drug_pos = sorted_entities(drug_table(batch), "fingerprints")
        train = self._pairs(
            sample_pos[batch.sample_pair_idx], drug_pos[np.asarray(batch.drug_pair_idx)], np.asarray(batch.response)
        )
        validation = None
        if batch.early_stopping_response is not None:
            stop = batch.early_stopping_response
            stop_sample = batch.sample.rows_for(stop.sample_ids)
            stop_drug = None if batch.drug is None else batch.drug.rows_for(stop.drug_ids)
            validation = self._pairs(
                sample_pos[stop_sample], drug_pos[np.asarray(stop_drug)], np.asarray(stop.response)
            )
        train, validation = split_pairs(train, validation, int(hp["seed"]))

        self._input_dims = (int(x_sample.shape[1]), int(x_drug.shape[1]))
        self._drug_ids = drug_ids.astype(str)
        self._drug_seen = np.zeros(len(drug_ids), dtype=bool)
        self._drug_seen[train.drug_idx.unique().cpu().numpy()] = True

        data = MemberData(
            torch.as_tensor(x_sample, dtype=torch.float32, device=self._device),
            torch.as_tensor(x_drug, dtype=torch.float32, device=self._device),
            train,
            validation,
        )
        settings = MemberSettings(
            learning_rate=float(hp["learning_rate"]),
            weight_decay=float(hp["weight_decay"]),
            batch_size=int(hp["batch_size"]),
            max_epochs=int(hp["max_epochs"]),
            patience=int(hp["patience"]),
        )
        seed = int(hp["seed"])
        fork_devices = [self._device] if self._device.type == "cuda" else []
        self._nets = []
        for member in range(int(hp["n_ensemble"])):
            with torch.random.fork_rng(devices=fork_devices):
                if seed >= 0:
                    torch.manual_seed(seed + member)
                net = self._build_net(len(drug_ids))
                train_member(net, data, settings)
            self._nets.append(net)

    def _embedding_rows(self, net: MFNet, drug_ids: np.ndarray) -> torch.Tensor:
        """Gather each drug's trained id-embedding row, with zeros for drugs no training batch contained.

        :param net: Member whose embedding table is read.
        :param drug_ids: Identifiers of the drug rows to encode.
        :returns: ``(len(drug_ids), emb_dim)`` rows.
        """
        import torch

        position = np.searchsorted(self._drug_ids, drug_ids.astype(str)).clip(max=len(self._drug_ids) - 1)
        seen = (self._drug_ids[position] == drug_ids.astype(str)) & self._drug_seen[position]
        rows = torch.zeros(len(drug_ids), net.drug_id_emb.embedding_dim, device=self._device)
        if seen.any():
            rows[torch.as_tensor(seen, device=self._device)] = net.drug_id_emb.weight[
                torch.as_tensor(position[seen], device=self._device)
            ]
        return rows

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Average the member predictions on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        import torch

        from .data import drug_table

        if not self._nets:
            msg = "EnsembleMF model not initialized."
            raise ValueError(msg)
        drug = drug_table(batch)
        x_sample = torch.as_tensor(
            batch.sample.blocks["gene_expression"].values, dtype=torch.float32, device=self._device
        )
        x_drug = torch.as_tensor(drug.blocks["fingerprints"].values, dtype=torch.float32, device=self._device)
        sample_idx = torch.as_tensor(batch.sample_pair_idx, dtype=torch.long, device=self._device)
        drug_idx = torch.as_tensor(batch.drug_pair_idx, dtype=torch.long, device=self._device)
        member_predictions = []
        with torch.no_grad():
            for net in self._nets:
                net.eval()
                rows = self._embedding_rows(net, drug.entity_ids) if net.use_drug_id_embedding else None
                z_sample, z_drug = net.encode(x_sample, x_drug, rows)
                member_predictions.append(net.score_pairs(z_sample[sample_idx], z_drug[drug_idx]).cpu().numpy())
        return np.mean(member_predictions, axis=0).astype(np.float64)

    def is_fitted(self) -> bool:
        """Return whether the ensemble has been trained.

        :returns: True when at least one member exists.
        """
        return bool(self._nets)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob when fitted, else empty.
        """
        if not self._nets:
            return {}
        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "input_dims": list(self._input_dims),
            "drug_ids": self._drug_ids.tolist(),
            "drug_seen": self._drug_seen.tolist(),
            "member_states": [save_state_dict(net.state_dict()) for net in self._nets],
        }
        return {"payload": save_trusted_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        payload = load_payload_state(state, "EnsembleMF")
        input_dims, member_states = payload.get("input_dims"), payload.get("member_states")
        if not isinstance(input_dims, list) or not isinstance(member_states, list) or not member_states:
            msg = "EnsembleMF payload is missing input_dims or member_states"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._input_dims = (int(input_dims[0]), int(input_dims[1]))
        self._drug_ids = np.asarray(payload["drug_ids"], dtype=str)
        self._drug_seen = np.asarray(payload["drug_seen"], dtype=bool)
        self._nets = []
        for member_state in member_states:
            net = self._build_net(len(self._drug_ids))
            net.load_state_dict(load_state_dict(bytes(member_state), map_location=self._device))
            net.eval()
            self._nets.append(net)
