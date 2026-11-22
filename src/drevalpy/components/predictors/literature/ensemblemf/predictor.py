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

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._torch_training import DevicePicklingMixin
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.feature_format import BlockSpec
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

if TYPE_CHECKING:
    import torch

    from .network import MFNet
    from .training import PairTensors

_REFERENCE = LiteratureReference(
    repo_url="https://github.com/daisybio/drevalpy",
    citation_doi="10.1038/s41467-026-72903-w",
    citation_text=(
        "EnsembleMF was contributed to DrEvalPy 1.6.1 (pull request 458) and has no publication of its own; "
        "the DOI above is the DrEval paper (Bernett et al., Nature Communications 2026) that describes the "
        "framework."
    ),
    deviations=(
        "Port of the DrEvalPy 1.6.1 model with the same network and ensemble training. Features are float32 "
        "from the dataset featurizers, ids without a feature row are not scored here but get NaN like every "
        "predictor, and the 10% early-stopping fallback split uses a generator seeded from the seed "
        "hyperparameter."
    ),
)


@register(
    "ensembleMF",
    description="EnsembleMF: ensembled two-tower matrix factorization over expression and fingerprints.",
    reference=_REFERENCE,
    license="MIT",
    accelerator=Accelerator.OPTIONAL,
)
class EnsembleMFPredictor(DevicePicklingMixin, Predictor):
    """Ensemble of two-tower matrix-factorization networks consuming PredictorInput directly."""

    device_modules: ClassVar[tuple[str, ...]] = ("_nets",)
    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("gene_expression"))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("fingerprints"))
    supports_early_stopping: ClassVar[bool] = True

    fixed_hyperparameters = {
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

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the EnsembleMF predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._nets: list[MFNet] = []
        self._drug_ids = np.empty(0, dtype=str)
        self._drug_seen = np.empty(0, dtype=bool)
        self._input_dims = (0, 0)

    def _build_net(self, n_drugs: int) -> MFNet:
        """Construct one ensemble member on the target device.

        :param n_drugs: Number of rows of the drug id-embedding table.
        :returns: Freshly initialized network.
        """
        from .network import MFNet

        hp = self._hyperparameters
        return MFNet(
            self._input_dims[0],
            self._input_dims[1],
            n_drugs,
            hidden_dim=int(hp["hidden_dim"]),
            emb_dim=int(hp["emb_dim"]),
            n_layers=int(hp["n_layers"]),
            dropout=float(hp["dropout"]),
            use_drug_id_embedding=bool(hp["use_drug_id_embedding"]),
            use_mlp_head=bool(hp["use_mlp_head"]),
            mlp_hidden=int(hp["mlp_hidden"]),
        ).to(self._device)

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

    def _fit(self, data: PredictorInput) -> None:
        """Train the ensemble members with their own seeds.

        :param data: Training input with gene_expression and fingerprints blocks.
        """
        import torch

        from .data import drug_table, sorted_entities, split_pairs
        from .training import MemberData, train_member

        hp = self._hyperparameters
        _, x_sample, sample_pos = sorted_entities(data.sample, "gene_expression")
        drug_ids, x_drug, drug_pos = sorted_entities(drug_table(data), "fingerprints")
        train = self._pairs(
            sample_pos[data.sample_pair_idx], drug_pos[np.asarray(data.drug_pair_idx)], np.asarray(data.response)
        )
        validation = None
        stop = data.stopping
        if stop is not None:
            validation = self._pairs(
                sample_pos[stop.sample_pair_idx], drug_pos[np.asarray(stop.drug_pair_idx)], np.asarray(stop.response)
            )
        train, validation = split_pairs(train, validation, int(hp["seed"]))

        self._input_dims = (int(x_sample.shape[1]), int(x_drug.shape[1]))
        self._drug_ids = drug_ids.astype(str)
        self._drug_seen = np.zeros(len(drug_ids), dtype=bool)
        self._drug_seen[train.drug_idx.unique().cpu().numpy()] = True

        member_data = MemberData(
            torch.as_tensor(x_sample, dtype=torch.float32, device=self._device),
            torch.as_tensor(x_drug, dtype=torch.float32, device=self._device),
            train,
            validation,
        )
        seed = int(hp["seed"])
        on_cpu = self._device.type == "cpu"
        fork: dict[str, Any] = (
            {"devices": []} if on_cpu else {"devices": [self._device], "device_type": self._device.type}
        )
        self._nets = []
        for member in range(int(hp["n_ensemble"])):
            with torch.random.fork_rng(**fork):
                if seed >= 0:
                    torch.manual_seed(seed + member)
                net = self._build_net(len(drug_ids))
                train_member(
                    net,
                    member_data,
                    learning_rate=float(hp["learning_rate"]),
                    weight_decay=float(hp["weight_decay"]),
                    batch_size=int(hp["batch_size"]),
                    max_epochs=int(hp["max_epochs"]),
                    patience=int(hp["patience"]),
                )
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

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Average the member predictions on the given batch.

        :param data: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        import torch

        from .data import drug_table

        if not self._nets:
            msg = "EnsembleMF model not initialized."
            raise ValueError(msg)
        drug = drug_table(data)
        x_sample = torch.as_tensor(
            data.sample.blocks["gene_expression"].values, dtype=torch.float32, device=self._device
        )
        x_drug = torch.as_tensor(drug.blocks["fingerprints"].values, dtype=torch.float32, device=self._device)
        sample_idx = torch.as_tensor(data.sample_pair_idx, dtype=torch.long, device=self._device)
        drug_idx = torch.as_tensor(data.drug_pair_idx, dtype=torch.long, device=self._device)
        member_predictions = []
        with torch.no_grad():
            for net in self._nets:
                net.eval()
                rows = self._embedding_rows(net, drug.entity_ids) if net.use_drug_id_embedding else None
                z_sample, z_drug = net.encode(x_sample, x_drug, rows)
                member_predictions.append(net.score_pairs(z_sample[sample_idx], z_drug[drug_idx]).cpu().numpy())
        return np.mean(member_predictions, axis=0).astype(np.float64)
