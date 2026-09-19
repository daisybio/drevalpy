"""Compare the SRMF implementation with the frozen Octave output."""

import json
import os
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat
from upath import UPath

from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor

m = loadmat(os.environ["SRMF_REFERENCE_MAT"])
p = SRMFPredictor({"K": 2, "max_iter": 5, "lambda_l": 0.25, "lambda_d": 0.15, "lambda_c": 0.35, "seed": 50})
# _cmf's validated runtime fields are normally initialized by _fit.
loss = p._compute_loss(m["U"], m["V"], m["W"], m["R"], m["S"], m["T"])
u = p._alg_update(m["U"], m["V"], m["W"], m["W"] * m["R"], m["S"], 0.25, 0.15)
v = p._alg_update(m["V"], u, m["W"].T, (m["W"] * m["R"]).T, m["T"], 0.25, 0.35)
np.testing.assert_allclose(loss, m["loss"].item(), rtol=1e-7, atol=1e-9)
np.testing.assert_allclose(u, m["updated_u"], rtol=1e-7, atol=1e-9)
np.testing.assert_allclose(v, m["updated_v"], rtol=1e-7, atol=1e-9)


class FixedInitial:
    """Replay the initial factors captured by the Octave fixture."""

    def __init__(self):
        """Initialize the fixed factor iterator."""
        self.values = iter([m["initial_u"], m["initial_v"]])

    def standard_normal(self, size):
        """Return the next captured factor with NumPy RNG scaling."""
        return next(self.values) * np.sqrt(2)


with patch("numpy.random.default_rng", return_value=FixedInitial()):
    best_u, best_v = p._cmf(m["W"], m["R"], m["S"], m["T"])
np.testing.assert_allclose(best_u, m["best_u"], rtol=1e-7, atol=1e-9)
np.testing.assert_allclose(best_v, m["best_v"], rtol=1e-7, atol=1e-9)
result = {
    "status": "pass",
    "runtime": UPath(__import__("os").environ.get("SRMF_OCTAVE_VERSION", "version.txt")).read_text().strip(),
    "checks": {"loss": True, "u_update": True, "v_update": True, "five_step_best_factors": True},
    "loss": float(loss),
    "scope": "Original SRMF .m files executed by Octave; nonzero drug and cell similarity "
    "regularization; zero/full/partial observation rows. Shared initial factors "
    "isolate intentional NumPy vs MATLAB RNG difference.",
}
UPath(__import__("os").environ.get("SRMF_RESULTS_PATH", "srmf-reference.json")).write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
