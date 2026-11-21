"""SparseGO literature predictor -- direct Predictor implementation.

``torch``, the ``SparseGONetwork`` and the ontology helpers in ``.utils`` (which
pull in ``networkx``) are imported inside the methods that need them.
``drevalpy.registry`` imports this module to register the ``sparsego`` predictor on
``import drevalpy``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors._pair_loader import concatenated_forward, pair_loader, predict_pairs
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._torch_training import DevicePicklingMixin, run_epoch
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork


def _resolve_active_view(data: PredictorInput) -> str:
    """Determine which sample block is the active ontology view.

    SparseGO supports either 'gene_expression' or 'mutations' (exactly one).

    :param data: Input batch with sample blocks.
    :returns: Name of the active sample view.
    :raises ValueError: If zero or multiple valid views are present.
    """
    candidates = {"gene_expression", "mutations"}
    active = [name for name in candidates if name in data.sample.blocks]
    if len(active) != 1:
        raise ValueError("SparseGOPredictor requires exactly one sample block from ['gene_expression', 'mutations']")
    return active[0]


_REFERENCE = LiteratureReference(
    repo_url="https://github.com/KatynaSada/SparseGO_lightning",
    citation_doi="10.1016/j.ebiom.2023.104767",
    citation_text="SparseGO visible neural network structured by the Gene Ontology hierarchy.",
    deviations=(
        "Preserve ontology-indexed sparse connections. Align named input genes explicitly; train on "
        "training-fold features. Local fixed-epoch MSE training differs from upstream correlation-based "
        "checkpoint winners."
    ),
)


@register(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    reference=_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class SparseGOPredictor(DevicePicklingMixin, Predictor):
    """SparseGO predictor consuming PredictorInput directly."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(
        (
            BlockSpec("gene_expression"),
            BlockSpec("mutations"),
        ),
    )
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("fingerprints"))
    supports_early_stopping: ClassVar[bool] = False

    fixed_hyperparameters = {
        "num_neurons_per_GO": 6,
        "num_neurons_per_final_GO": 30,
        "num_neurons_drug": [100, 50, 6],
        "num_neurons_final": 12,
        "drug_dim": 2048,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "decay_rate": 0.002,
        "p_drop_genes": 0.15,
        "p_drop_terms": 0.15,
        "p_drop_drugs": 0.15,
        "p_drop_final": 0.0,
        "epochs": 400,
        "batch_size": 20000,
    }

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize SparseGOPredictor.

        :param hyperparameters: Optional overrides for algorithm defaults.
        """
        super().__init__(hyperparameters)
        self._model: SparseGONetwork | None = None
        self._layer_connections: list | None = None
        self._gene2id_mapping_ont: dict[str, int] | None = None

    def _build_network(self) -> SparseGONetwork:
        """Construct the SparseGONetwork from stored ontology metadata and hyperparameters.

        :returns: The new network, also stored on ``self._model``.
        :raises ValueError: If ontology metadata is not available.
        """
        from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork

        if self._layer_connections is None or self._gene2id_mapping_ont is None:
            raise ValueError("SparseGO ontology metadata must be provided before building the network.")
        hp = self._hyperparameters
        model = SparseGONetwork(
            layer_connections=self._layer_connections,
            num_neurons_per_go=hp["num_neurons_per_GO"],
            num_neurons_per_final_go=hp["num_neurons_per_final_GO"],
            num_neurons_drug=hp["num_neurons_drug"],
            num_neurons_final=hp["num_neurons_final"],
            drug_dim=hp["drug_dim"],
            gene2id_mapping=self._gene2id_mapping_ont,
            p_drop_final=hp["p_drop_final"],
            p_drop_genes=hp["p_drop_genes"],
            p_drop_terms=hp["p_drop_terms"],
            p_drop_drugs=hp["p_drop_drugs"],
        ).to(self._device)
        self._model = model
        return model

    def _entity_matrices(self, data: PredictorInput, active_view: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the entity-level sample and fingerprint matrices as float32.

        :param data: Batch to read the blocks from.
        :param active_view: Name of the sample block holding the ontology view.
        :returns: Tuple of ``(sample_matrix, fingerprint_matrix)``.
        """
        return (
            np.asarray(data.sample.blocks[active_view].values, dtype=np.float32),
            np.asarray(data.drug.blocks["fingerprints"].values, dtype=np.float32),
        )

    def _fit(self, data: PredictorInput) -> None:
        """Train SparseGO on the batch.

        :param data: Training batch with responses and sample/drug blocks.
        :raises ValueError: If ontology metadata is missing.
        """
        active_view = _resolve_active_view(data)
        block = data.sample.blocks[active_view]
        if block.annotations is None:
            raise ValueError("SparseGOPredictor requires ontology metadata on its active sample block")

        from .utils import resolve_ontology_metadata

        self._layer_connections, self._gene2id_mapping_ont, _ = resolve_ontology_metadata(dict(block.annotations))

        blocks = self._entity_matrices(data, active_view)
        self._hyperparameters["drug_dim"] = int(blocks[1].shape[1])
        model = self._build_network()
        loader = pair_loader(data, *blocks, batch_size=self._hyperparameters["batch_size"], shuffle=True)
        self._run_training(model, loader)

    def _run_training(self, model: SparseGONetwork, loader: DataLoader) -> None:
        """Run the SGD training loop with the paper's per-epoch learning-rate decay.

        :param model: The network to train.
        :param loader: Training loader over sample, drug and response tensors.
        """
        import torch
        import torch.nn as nn

        lr = self._hyperparameters["learning_rate"]
        decay_rate = self._hyperparameters["decay_rate"]
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._hyperparameters["momentum"],
        )

        def batch_loss(loader_batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            cell_feats, drug_feats, batch_labels = loader_batch
            batch_features = torch.cat([cell_feats, drug_feats], dim=1).to(self._device)
            return criterion(model(batch_features), batch_labels.reshape(-1, 1).to(self._device))

        for epoch in range(self._hyperparameters["epochs"]):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (1 / (1 + decay_rate * epoch))
            run_epoch(model, loader, batch_loss, optimizer)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict drug response for pairs in the batch.

        :param data: Featurized pairs to score.
        :returns: Predicted response values as a 1D numpy array.
        :raises ValueError: If the model is not fitted.
        """
        if self._model is None:
            raise ValueError("SparseGOPredictor must be fitted before predict().")

        cell_entity, drug_entity = self._entity_matrices(data, _resolve_active_view(data))
        return predict_pairs(
            self._model,
            data,
            cell_entity,
            drug_entity,
            batch_size=self._hyperparameters["batch_size"],
            device=self._device,
            forward=concatenated_forward(self._model),
        )
