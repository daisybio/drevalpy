"""Capture immutable starting-v2 numerical fixtures from an archived checkout."""

import json
import os
import random
import sys

import numpy as np
from upath import UPath as Path

ARCHIVE = Path(os.environ["DREVALPY_BASELINE_ROOT"]).resolve()
sys.path.insert(0, str(ARCHIVE))
OUT = Path(os.environ.get("BASELINE_OUTPUT_ROOT", "baseline-output"))
OUT.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    """Set all random streams and deterministic CPU execution."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def arr(x):
    """Record the shape, dtype and values of one tensor."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "values": x.tolist()}


def main(seed: int) -> None:
    """Execute the original sequence of eight model operations."""
    seed_all(seed)
    result = {"seed": seed, "commit": "cf5eeb6a0e020ab7e98c9829c0d81dc00a8cdd5c"}
    import torch

    with torch.no_grad():
        from torch_geometric.data import Data

        from drevalpy.components.predictors.literature.dipk.model_utils import Predictor as DIPKNetwork
        from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGraphNet
        from drevalpy.components.predictors.literature.pharmaformer.model_utils import CombinedModel
        from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork
        from drevalpy.components.predictors.literature.sparsego.algorithm import SparseLinearNew

        graph = Data(
            x=torch.randn(6, 5),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        cell = torch.randn(2, 7)
        druggnn = DrugGraphNet(num_node_features=5, num_cell_features=7, hidden_dim=8, dropout=0.0).eval()
        result["druggnn"] = arr(druggnn(graph, cell))
        pharma = CombinedModel(
            gene_input_size=6,
            gene_hidden_size=8,
            drug_hidden_size=8,
            feature_dim=4,
            nhead=2,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
        ).eval()
        result["pharmaformer"] = arr(pharma(torch.randn(2, 6), torch.randn(2, 128)))
        precily = PrecilyNetwork(input_dim=8, dropout=0.0).eval()
        result["precily"] = arr(precily(torch.randn(2, 8)))
        sparse = SparseLinearNew(
            in_features=2, out_features=2, connectivity=torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        ).eval()
        result["sparsego"] = arr(sparse(torch.tensor([[1.0, 2.0]])))
        dipk = DIPKNetwork(heads=1, fc_layer_num=3, fc_layer_dim=[8, 8, 8, 8, 8, 8], dropout_rate=0.0).eval()
        result["dipk"] = arr(dipk(torch.randn(2, 3, 768), torch.randn(2, 512), torch.randn(2, 512), torch.ones(2, 3)))
    from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix

    result["molir"] = arr(
        _realign_omic_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), ["g1", "g2"], ["g0", "g1", "g2"])
    )
    from drevalpy.components.predictors.literature.molir.utils import generate_triplets_indices

    pos, neg = generate_triplets_indices(
        np.array([0.0, 0.2, 0.8, 1.0]), positive_range=0.1, negative_range=1.0, random_seed=seed
    )
    result["molir_triplets"] = {"positive": np.asarray(pos).tolist(), "negative": np.asarray(neg).tolist()}
    from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor

    srmf = SRMFPredictor({"K": 2, "max_iter": 2, "seed": seed})
    w = np.ones((2, 3), dtype=np.float64)
    interaction = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    similarity_drug = np.eye(2, dtype=np.float64)
    similarity_cell = np.eye(3, dtype=np.float64)
    factors = srmf._cmf(w, interaction, similarity_drug, similarity_cell)
    result["srmf"] = {"u": arr(factors[0]), "v": arr(factors[1])}
    from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder

    hp = {
        "dropout_rate": 0.0,
        "margin": 1.0,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "out_dim_expr_encoder": 4,
        "out_dim_mutation_encoder": 3,
        "out_dim_cnv_encoder": 2,
        "epochs": 1,
        "mini_batch": 2,
    }
    encoder = SuperFELTEncoder(input_size=6, hpams=hp, omic_type="expression", ranges=(0.1, 1.0)).eval()
    result["superfeltr"] = arr(encoder(torch.randn(2, 6)))
    (OUT / f"seed-{seed}.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]))
