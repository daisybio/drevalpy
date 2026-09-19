"""Compare the Precily published TensorFlow artifact with the local network."""

import hashlib
import json

import keras
import numpy as np
import tensorflow as tf
import torch
from upath import UPath

from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork

torch.set_num_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
checkpoint = UPath(__import__("os").environ["PRECILY_MODEL1"])
original = keras.models.load_model(str(checkpoint), compile=False)
keras.config.set_floatx("float64")
reference = keras.models.clone_model(
    original,
    input_tensors=keras.Input(shape=(1429,), dtype="float64"),
    clone_function=lambda layer: type(layer).from_config({**layer.get_config(), "dtype": "float64"}),
)
reference.set_weights(original.get_weights())
local = PrecilyNetwork(1429).double().eval()
dense = [layer for layer in reference.layers if isinstance(layer, keras.layers.Dense)]
linear = [layer for layer in local.net if isinstance(layer, torch.nn.Linear)]
for kl, pl in zip(dense, linear, strict=True):
    kernel, bias = kl.get_weights()
    pl.weight.data.copy_(torch.from_numpy(kernel.T))
    pl.bias.data.copy_(torch.from_numpy(bias))
x = np.linspace(-0.5, 0.5, 3 * 1429).reshape(3, 1429)
y = np.array([0.1, -0.2, 0.3])
with tf.GradientTape() as tape:
    a = tf.reshape(reference(x, training=False), (-1,))
    la = tf.reduce_mean((a - y) ** 2)
ga = tape.gradient(la, reference.trainable_weights)
b = local(torch.from_numpy(x))
lb = ((b - torch.from_numpy(y)) ** 2).mean()
lb.backward()
np.testing.assert_allclose(a.numpy(), b.detach(), rtol=1e-7, atol=1e-9)
np.testing.assert_allclose(la.numpy(), lb.detach(), rtol=1e-7, atol=1e-9)
for i, pl in enumerate(linear):
    np.testing.assert_allclose(ga[2 * i].numpy().T, pl.weight.grad, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ga[2 * i + 1].numpy(), pl.bias.grad, rtol=1e-7, atol=1e-9)
# A native optimizer step in each runtime, using the same configured learning rate.
keras.optimizers.SGD(learning_rate=0.001).apply_gradients(zip(ga, reference.trainable_weights, strict=True))
torch.optim.SGD(local.parameters(), lr=0.001).step()
for kl, pl in zip(dense, linear, strict=True):
    kernel, bias = kl.get_weights()
    np.testing.assert_allclose(kernel.T, pl.weight.detach(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(bias, pl.bias.detach(), rtol=1e-7, atol=1e-9)
result = {
    "status": "pass",
    "runtime": {"keras": keras.__version__, "tensorflow": tf.__version__, "torch": torch.__version__},
    "artifact": str(checkpoint),
    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "checks": {"forward": True, "loss": True, "gradients": True, "sgd_update": True},
    "dtype": "float64",
    "rtol": 1e-7,
    "atol": 1e-9,
    "input_shape": list(x.shape),
    "predictions": b.detach().tolist(),
    "loss": float(lb.detach()),
    "scope": "Published pathways Model1 trained Dense weights mapped to PyTorch; same Keras "
    "architecture promoted to float64; dropout disabled. Initial Keras torch-backend "
    "float32 check exceeded entrywise gradient atol by 1.4e-6; Keras torch dtype "
    "promotion truncates float64 to float32. TensorFlow float64 check uses the "
    "predeclared float64 tolerances.",
}
UPath(__import__("os").environ.get("PRECILY_RESULTS_PATH", "precily-reference.json")).write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
