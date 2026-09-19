"""Compare adapted MOLIR and SuperFELTR fusion compositions."""

import ast
import json

import numpy as np
import torch
from torch import nn
from upath import UPath

from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.molir.utils import MOLIModel
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder, SuperFELTRegressor

ROOT = UPath(__import__("os").environ.get("UPSTREAM_REFERENCE_ROOT", "."))


def classes(path, names, extra=None):
    """Audit helper."""
    scope = {"torch": torch, "nn": nn, **(extra or {})}
    tree = ast.parse(UPath(path).read_text(), filename=str(path))
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 - selected classes only
    return [scope[name] for name in names]


def check(left, right, args, feature_left, feature_right, loss):
    """Audit helper."""
    left.eval()
    right.eval()
    zl = feature_left(*args)
    zr = feature_right(*args)
    np.testing.assert_allclose(zl.detach(), zr.detach(), rtol=1e-5, atol=1e-6)
    a = left(*args)
    b = right(*args)
    la = loss(a, zl)
    lb = loss(b, zr)
    la.backward()
    lb.backward()
    np.testing.assert_allclose(a.detach(), b.detach(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(la.detach(), lb.detach(), rtol=1e-5, atol=1e-6)
    # Parameter lists are mapped in assembly order: three encoders then one head.
    pa = [p for p in left.parameters() if p.requires_grad]
    pb = [p for p in right.parameters() if p.requires_grad]
    for aa, bb in zip(pa, pb, strict=True):
        np.testing.assert_allclose(aa.grad, bb.grad, rtol=1e-5, atol=1e-6)
    torch.optim.SGD(pa, lr=0.01).step()
    torch.optim.SGD(pb, lr=0.01).step()
    for aa, bb in zip(pa, pb, strict=True):
        np.testing.assert_allclose(aa.detach(), bb.detach(), rtol=1e-5, atol=1e-6)
    return {
        "status": "pass",
        "checks": ["intermediate_fusion", "forward", "adapted_loss", "active_gradients", "SGD_update"],
        "predictions": a.detach().flatten().tolist(),
        "loss": float(la.detach()),
        "active_parameter_tensors": len(pa),
    }


class ReferenceFusion(nn.Module):
    """Audit helper."""

    def __init__(self, encoders, head, normalize=False):
        """Audit helper."""
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.head = head
        self.normalize = normalize

    def features(self, *args):
        """Audit helper."""
        z = torch.cat([m(x) for m, x in zip(self.encoders, args, strict=True)], dim=1)
        return nn.functional.normalize(z, p=2, dim=0) if self.normalize else z

    def forward(self, *args):
        """Audit helper."""
        return self.head(self.features(*args))


class FeltLocal(nn.Module):
    """Audit helper."""

    def __init__(self, model):
        """Audit helper."""
        super().__init__()
        self.model = model

    def forward(self, *args):
        """Audit helper."""
        return self.model(self.model._encode_and_concatenate(*args))


torch.set_num_threads(1)
torch.manual_seed(21)
inputs = [torch.randn(4, 5), torch.randn(4, 3), torch.randn(4, 2)]
target = torch.tensor([0.1, 0.15, 1.0, 1.05]).reshape(-1, 1)
(encoder,) = classes(ROOT / "MultiOmics/src/models/moli_model.py", ["MOLIEncoder"])
(head,) = classes(
    ROOT / "MOLI/Cross validation/MOLI Complete/Erlotinib_cvSoftTripletClassifierNetv16_Script.py",
    ["Classifier"],
    {"Z_in": 12, "rate4": 0.0},
)
reference = ReferenceFusion([encoder(width, 4, 0.0) for width in [5, 3, 2]], head().FC[:-1], normalize=True)
hp = {
    **MOLIRPredictor.get_default_hyperparameters(),
    "h_dim1": 4,
    "h_dim2": 4,
    "h_dim3": 4,
    "dropout_rate": 0.0,
    "gamma": 0.5,
}
local = MOLIModel(hp, 5, 3, 2)
for src, dst in zip(
    reference.encoders, [local.expression_encoder, local.mutation_encoder, local.cna_encoder], strict=True
):
    dst.load_state_dict(src.state_dict())
local.regressor.regressor.load_state_dict(reference.head.state_dict())
positive = torch.tensor([1, 0, 3, 2])
negative = torch.tensor([2, 3, 0, 1])
triplet = nn.TripletMarginLoss(margin=1.5, p=2)


def molir_loss(pred, z):
    """Audit helper."""
    return ((pred - target) ** 2).mean() + 0.5 * triplet(z, z[positive], z[negative])


results = {
    "MOLIR_regression_adaptation": check(
        reference, local, inputs, reference.features, local._encode_and_concatenate, molir_loss
    )
}
results["MOLIR_regression_adaptation"]["scope"] = (
    "MultiOmics encoder order plus original MOLI dim=0 fusion normalization and classifier layers before sigmoid; "
    "MSE replaces BCE and fixed continuous-response triplets use gamma=.5. This composed regression adaptation is "
    "not an unchanged upstream classifier."
)
encoder, head = classes(ROOT / "SuperFELT/models.py", ["Supervised_Encoder", "Classifier"])
reference = ReferenceFusion([encoder(width, 4, 0.0) for width in [5, 3, 2]], head(12, 1, 0.0).model[:-1])
hp = {
    **SuperFELTRPredictor.get_default_hyperparameters(),
    "out_dim_expr_encoder": 4,
    "out_dim_mutation_encoder": 4,
    "out_dim_cnv_encoder": 4,
    "dropout_rate": 0.0,
}
encoders = [
    SuperFELTEncoder(width, hp, omic, (0.1, 1.0))
    for width, omic in zip([5, 3, 2], ["expression", "mutation", "copy_number_variation_gistic"], strict=True)
]
for src, dst in zip(reference.encoders, encoders, strict=True):
    dst.encode.load_state_dict(src.model.state_dict())
# Encoder phase: same explicit triplets avoid changing upstream binary sampler semantics.
for src, dst, x in zip(reference.encoders, encoders, inputs, strict=True):
    src.eval()
    dst.eval()
    a = src(x)
    b = dst(x)
    la = triplet(a, a[positive], a[negative])
    lb = triplet(b, b[positive], b[negative])
    la.backward()
    lb.backward()
    np.testing.assert_allclose(la.detach(), lb.detach(), rtol=1e-5, atol=1e-6)
    for aa, bb in zip(src.parameters(), dst.parameters(), strict=True):
        np.testing.assert_allclose(aa.grad, bb.grad, rtol=1e-5, atol=1e-6)
    src.zero_grad()
    dst.zero_grad()
model = SuperFELTRegressor(12, hp, tuple(encoders))
model.regressor.load_state_dict(reference.head.state_dict())
local = FeltLocal(model)
for src in reference.encoders:
    src.requires_grad_(False)
results["SuperFELTR_regression_adaptation"] = check(
    reference,
    local,
    inputs,
    reference.features,
    model._encode_and_concatenate,
    lambda pred, z: ((pred - target) ** 2).mean(),
)
results["SuperFELTR_regression_adaptation"]["encoder_triplet_loss_and_gradients"] = "pass"
results["SuperFELTR_regression_adaptation"]["scope"] = (
    "Published three encoders and classifier before sigmoid; frozen encoders during head fitting, MSE replaces BCE. "
    "Encoder triplet checks use shared indices, not binary-vs-continuous sampler equality."
)
UPath(__import__("os").environ.get("FUSION_RESULTS_PATH", "fusion-reference.json")).write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
