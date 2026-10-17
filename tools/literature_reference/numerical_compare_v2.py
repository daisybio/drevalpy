"""Compare selected upstream numerical kernels without executing training scripts."""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
from upath import UPath as Path

ROOT = Path(os.environ["UPSTREAM_REFERENCE_ROOT"])
OUT = Path(os.environ.get("NUMERICAL_RESULTS_PATH", "numerical-results.json"))
REPO_ROOT = Path(os.environ.get("DREVALPY_ROOT", "."))
R = {
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
    },
    "tolerances": {"float32": {"rtol": 1e-5, "atol": 1e-6}, "float64": {"rtol": 1e-7, "atol": 1e-9}},
    "comparisons": {},
}


def exact(name, fn):
    """Record one scoped comparison and retain failure evidence."""
    try:
        R["comparisons"][name] = fn()
    except Exception as e:
        R["comparisons"][name] = {
            "status": "blocked_or_failed",
            "error": repr(e),
            "traceback": traceback.format_exc(limit=4),
        }


def stat(x):
    """Summarize tensor values without storing large intermediate arrays."""
    x = x.detach().cpu()
    return {
        "shape": list(x.shape),
        "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "mean": float(x.float().mean()),
        "maxabs": float(x.abs().max()),
    }


def compare_models(a, b, args):
    """Compare mapped weights through loss, gradients and one SGD update."""
    a.eval()
    b.eval()
    st = a.state_dict()
    st = b._source_state_map(st) if hasattr(b, "_source_state_map") else st
    b.load_state_dict(st, strict=True)
    aa = [x.detach().clone().requires_grad_(True) for x in args]
    bb = [x.detach().clone().requires_grad_(True) for x in args]
    oa = a(*aa)
    ob = b(*bb)
    target = torch.linspace(-0.3, 0.3, oa.numel(), dtype=oa.dtype).reshape_as(oa)
    la = nn.MSELoss()(oa, target)
    lb = nn.MSELoss()(ob, target)
    la.backward()
    lb.backward()
    ga = {k: v.grad.detach().clone() for k, v in a.named_parameters()}
    gb = {k: v.grad.detach().clone() for k, v in b.named_parameters()}
    opta = torch.optim.SGD(a.parameters(), lr=0.01)
    optb = torch.optim.SGD(b.parameters(), lr=0.01)
    opta.step()
    optb.step()

    def key(k):
        return b._source_key(k) if hasattr(b, "_source_key") else k

    checks = {
        "output": torch.allclose(oa, ob, rtol=1e-5, atol=1e-6),
        "loss": torch.allclose(la, lb, rtol=1e-5, atol=1e-6),
        "gradients": all(torch.allclose(ga[k], gb[key(k)], rtol=1e-5, atol=1e-6) for k in ga),
        "one_update_state": all(
            torch.allclose(v, b.state_dict()[key(k)], rtol=1e-5, atol=1e-6) for k, v in a.state_dict().items()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "output": stat(oa),
        "loss": float(la),
        "n_parameters": sum(p.numel() for p in a.parameters()),
    }


def classes_from(path, names, globs):
    """Compile selected class definitions without running upstream training code."""
    tree = ast.parse(Path(path).read_text(), str(path))
    nodes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name in names]
    m = dict(globs)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), m)  # noqa: S102 - selected definitions only
    return [m[n] for n in names]


class Light(nn.Module):
    """Minimal Lightning base for independently extracted class definitions."""

    def save_hyperparameters(self, *a, **kw):
        """Accept constructor metadata without altering numerical behavior."""
        pass


class PL:
    """Namespace supplying the minimal Lightning base."""

    LightningModule = Light


# MOLIR encoder, exact source family class.
def molir():
    """Compare the MultiOmics encoder with the local MOLIR encoder."""
    (source_class,) = classes_from(
        ROOT / "MultiOmics/src/models/moli_model.py", ["MOLIEncoder"], {"torch": torch, "nn": nn}
    )
    (local_class,) = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/molir/utils.py"),
        ["MOLIEncoder"],
        {"torch": torch, "nn": nn, "pl": PL, "np": np},
    )
    torch.manual_seed(7)
    return compare_models(source_class(5, 4, 0.0), local_class(5, 4, 0.0), [torch.randn(3, 5)])


# SuperFELTR encoder shares source ordering.
def superfelt():
    """Compare SuperFELT encoders after mapping parameter names."""
    (source_class,) = classes_from(ROOT / "SuperFELT/models.py", ["Supervised_Encoder"], {"torch": torch, "nn": nn})
    (superfelt_encoder,) = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/superfeltr/utils.py"),
        ["SuperFELTEncoder"],
        {"torch": torch, "nn": nn, "pl": PL, "np": np, "cast": lambda x, y: y},
    )
    hp = {
        "dropout_rate": 0.0,
        "margin": 1.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "h_dim1": 4,
        "h_dim2": 4,
        "h_dim3": 4,
        "out_dim_expr_encoder": 4,
        "out_dim_mut_encoder": 4,
        "out_dim_cnv_encoder": 4,
        "out_dim_mutation_encoder": 4,
    }
    torch.manual_seed(8)
    loc = superfelt_encoder(5, hp, "expression", (1.0, 1.0))
    loc._source_state_map = lambda st: {
        k.replace("model.", "encode."): v for k, v in st.items() if k.replace("model.", "encode.") in loc.state_dict()
    }
    loc._source_key = lambda k: k.replace("model.", "encode.")
    return compare_models(source_class(5, 4, 0.0), loc, [torch.randn(3, 5)])


# DIPK exact DAE modules; import source class nodes without optional optuna dependency.
def dipk():
    """Compare both upstream denoising autoencoder modules."""
    source_encoder, source_decoder = classes_from(
        ROOT / "DIPK/Task1/PreTrain/Model_DAE.py",
        ["Encoder", "Decoder"],
        {
            "torch": torch,
            "nn": nn,
            "F": torch.nn.functional,
            "deepcopy": __import__("copy").deepcopy,
            "ldim": 512,
            "hdim": [2048, 1024],
        },
    )
    gene_encoder, gene_decoder = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/dipk/gene_expression_encoder.py"),
        ["GeneExpressionEncoder", "GeneExpressionDecoder"],
        {"torch": torch, "nn": nn, "functional": torch.nn.functional, "deepcopy": __import__("copy").deepcopy},
    )
    torch.manual_seed(9)
    e = compare_models(source_encoder(7, 3, [5, 4], 0.0), gene_encoder(7, 3, [5, 4], 0.0), [torch.randn(3, 7)])
    torch.manual_seed(10)
    d = compare_models(source_decoder(7, 3, [5, 4], 0.0), gene_decoder(7, 3, [5, 4], 0.0), [torch.randn(3, 3)])
    return {"status": "pass" if e["status"] == d["status"] == "pass" else "fail", "encoder": e, "decoder": d}


# graph convolution stack only: exact shared GCNConv dimensions/weights; later architecture intentionally diverges.
def graphdrp():
    """Compare the shared GCN stack before the intentional architecture changes."""
    spec = importlib.util.spec_from_file_location("graphdrp_gcn", ROOT / "GraphDRP/models/gcn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (drug_graph_net,) = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/druggnn/algorithm.py"),
        ["DrugGraphNet"],
        {
            "torch": torch,
            "nn": nn,
            "pl": PL,
            "GCNConv": __import__("torch_geometric.nn", fromlist=["GCNConv"]).GCNConv,
            "global_mean_pool": __import__("torch_geometric.nn", fromlist=["global_mean_pool"]).global_mean_pool,
            "Adam": torch.optim.Adam,
        },
    )
    src = m.GCNNet(num_features_xd=78, output_dim=128, dropout=0.0)
    loc = drug_graph_net(78, 5, hidden_dim=78, dropout=0.0)
    for i in [1, 2, 3]:
        getattr(loc, f"conv{i}").load_state_dict(getattr(src, f"conv{i}").state_dict())
    x = torch.randn(6, 78, requires_grad=True)
    edge = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2, 4], [1, 2, 3, 4, 5, 0, 2, 4, 0]])
    x2 = x.detach().clone().requires_grad_(True)

    def run(model, z):
        for i in [1, 2, 3]:
            z = torch.relu(getattr(model, f"conv{i}")(z, edge))
        return z

    a, b = run(src, x), run(loc, x2)
    ta = torch.zeros_like(a)
    la = nn.MSELoss()(a, ta)
    lb = nn.MSELoss()(b, ta)
    la.backward()
    lb.backward()
    oa = torch.optim.SGD([*src.conv1.parameters(), *src.conv2.parameters(), *src.conv3.parameters()], lr=0.01)
    ob = torch.optim.SGD([*loc.conv1.parameters(), *loc.conv2.parameters(), *loc.conv3.parameters()], lr=0.01)
    oa.step()
    ob.step()
    c = {
        "output": torch.allclose(a, b, rtol=1e-5, atol=1e-6),
        "loss": torch.allclose(la, lb, rtol=1e-5, atol=1e-6),
        "input_gradient": torch.allclose(x.grad, x2.grad, rtol=1e-5, atol=1e-6),
        "one_update_state": all(
            torch.allclose(
                getattr(src, f"conv{i}").state_dict()[k], getattr(loc, f"conv{i}").state_dict()[k], rtol=1e-5, atol=1e-6
            )
            for i in [1, 2, 3]
            for k in getattr(src, f"conv{i}").state_dict()
        ),
    }
    return {
        "status": "pass" if all(c.values()) else "fail",
        "checks": c,
        "output": stat(a),
        "loss": float(la),
        "scope": "three shared GCNConv layers only; pooling/cell/head intentionally differ",
    }


# PharmaFormer source classes loaded without executing data/dependency-heavy script.
def pharma():
    """Compare the complete PharmaFormer attention network."""
    source_feature, source_trans, source_combined = classes_from(
        ROOT / "PharmaFormer/pretrain.py",
        ["FeatureExtractor", "TransModel", "CombinedModel"],
        {"torch": torch, "nn": nn, "F": torch.nn.functional},
    )
    _, _, combined_model = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/pharmaformer/model_utils.py"),
        ["FeatureExtractor", "TransModel", "CombinedModel"],
        {"torch": torch, "nn": nn, "functional": torch.nn.functional},
    )
    torch.manual_seed(11)
    return compare_models(
        source_combined(6, 4, 4, 4, 2, num_layers=1, dim_feedforward=8, dropout=0.0),
        combined_model(6, 4, 4, 4, 2, num_layers=1, dim_feedforward=8, dropout=0.0),
        [torch.randn(3, 6), torch.randn(3, 128)],
    )


# SparseGO exact layer requires optional torch_sparse upstream dependency.
def sparsego():
    """Compare sparse layers using equivalent native COO coalescing."""
    try:
        (mtype,) = classes_from(
            ROOT / "SparseGO/code/sparse_linear_new.py",
            ["SparseLinearNew"],
            {
                "torch": torch,
                "nn": nn,
                "np": np,
                "math": __import__("math"),
                "warnings": __import__("warnings"),
                "random": __import__("random"),
                "torch_sparse": type(
                    "TS",
                    (),
                    {
                        "coalesce": staticmethod(
                            lambda indices, values, rows, cols: (
                                torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce().indices(),
                                torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce().values(),
                            )
                        )
                    },
                ),
            },
        )
        m = type("M", (), {"SparseLinearNew": mtype})
    except Exception as e:
        return {"status": "blocked", "reason": "upstream sparse_linear_new import failed", "error": repr(e)}
    (sparse_linear_new,) = classes_from(
        (REPO_ROOT / "src/drevalpy/components/predictors/literature/sparsego/algorithm.py"),
        ["SparseLinearNew"],
        {
            "torch": torch,
            "nn": nn,
            "cast": lambda x, y: y,
            "_validate_sparse_linear_dimensions": lambda *a: None,
            "_sparse_connectivity_indices": lambda inf, outf, sparsity, connectivity, device: (
                connectivity.to(device),
                connectivity.shape[1],
                connectivity.shape[1] / (inf * outf),
            ),
        },
    )
    conn = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 3]], dtype=torch.long)
    a = m.SparseLinearNew(4, 3, connectivity=conn)
    b = sparse_linear_new(4, 3, connectivity=conn)
    torch.manual_seed(12)
    return compare_models(a, b, [torch.randn(3, 4)])


for n, f in [
    ("MOLIR_encoder", molir),
    ("SuperFELTR_encoder", superfelt),
    ("DIPK_AE", dipk),
    ("GraphDRP_GCN_stack", graphdrp),
    ("PharmaFormer", pharma),
    ("SparseGO_SparseLinear", sparsego),
]:
    exact(n, f)
OUT.write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
if any(value["status"] != "pass" for value in R["comparisons"].values()):
    raise SystemExit("An upstream comparison failed; inspect the result file")
