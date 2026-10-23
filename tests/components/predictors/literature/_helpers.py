"""Shared batch factory for the literature predictor tests.

Every literature predictor is exercised against the same shape of input - two
samples, two drugs, the four pairs between them, and a checkpoint directory -
so each of those test files had written out the same twelve-keyword
``ModelInputBatch.from_response`` call. What actually differs per predictor is
which feature blocks it consumes, and that is all a caller passes here.

Plain ``_``-prefixed module: the
underscore keeps it out of collection, and the mirror policy walks ``drevalpy/``
only, so no mirrored test is demanded for it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.models.synthetic_fixtures import multi_drug_response

SAMPLE_IDS = np.array(["cl1", "cl2"])
DRUG_IDS = np.array(["d1", "d2"])

#: The four (sample, drug) pairs of the 2x2 grid, in row-major order.
SAMPLE_PAIR_IDX = np.array([0, 0, 1, 1])
DRUG_PAIR_IDX = np.array([0, 1, 0, 1])

#: Tiny fits run many float32 updates, so CPU architectures drift apart by up to
#: ~2e-6 relative; the references mix macOS arm64 and x86_64 captures. Single
#: forward passes keep the strict 1e-6 bound in ``_assert_array_payload_matches``.
TRAINING_TOLERANCE = {"rtol": 1e-4, "atol": 1e-6}


def seed_literature(seed: int) -> None:
    """Seed NumPy, Torch, and Python for a repeatable tiny model fit."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def trainable_parameter_vector(module: Any) -> np.ndarray:
    """Flatten the finite trainable parameters of a fitted Torch module."""
    values = [parameter.detach().cpu().numpy().ravel() for parameter in module.parameters() if parameter.requires_grad]
    vector = np.concatenate(values)
    assert vector.size
    assert np.isfinite(vector).all()
    return vector


def assert_trainable_parameters_changed(initial: np.ndarray, fitted_module: Any) -> None:
    """Require an optimizer-backed fit to update at least one trainable value."""
    fitted = trainable_parameter_vector(fitted_module)
    assert fitted.shape == initial.shape
    assert not np.array_equal(initial, fitted)


def assert_training_predictions(model: str, seed: int, actual: Any) -> None:
    """Compare tiny-fit predictions to the same fit at the frozen v2 commit."""
    from tests.components.predictors.literature._reference.loader import load_training_seed

    expected = load_training_seed(seed)[model]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), **TRAINING_TOLERANCE)


def assert_training_checkpoint_score(model: str, seed: int, actual: Any) -> None:
    """Compare the selected tiny-fit checkpoint score to the frozen v2 fit."""
    from tests.components.predictors.literature._reference.loader import load_training_seed

    expected = load_training_seed(seed)[f"{model}_checkpoint_score"]
    np.testing.assert_allclose(float(actual), float(expected), **TRAINING_TOLERANCE)


def assert_corrected_training_result(
    correction: str,
    seed: int,
    predictions: Any,
    checkpoint_score: Any,
) -> None:
    """Compare a fit against evidence kept separate from the immutable v2 baseline."""
    from tests.components.predictors.literature._reference.loader import load_corrected_training_seed

    expected = load_corrected_training_seed(correction, seed)
    np.testing.assert_allclose(np.asarray(predictions), np.asarray(expected["predictions"]), **TRAINING_TOLERANCE)
    np.testing.assert_allclose(float(checkpoint_score), float(expected["checkpoint_score"]), **TRAINING_TOLERANCE)


def identity_drug_blocks() -> dict[str, FeatureBlock]:
    """Return identity blocks used by per-drug literature predictors."""
    return {
        "identity": numeric_feature_block(np.eye(2, dtype=np.float32)),
        "identity_categories": numeric_feature_block(np.array([["d1"], ["d2"]], dtype=object)),
    }


def single_drug_omics_batch(checkpoint_dir: Any) -> ModelInputBatch:
    """Build a varied, trainable three-omic batch with disjoint validation rows."""
    rng = np.random.default_rng(2025)
    sample_ids = np.array([*(f"train-{idx}" for idx in range(6)), "val-0", "val-1"])
    training = ResponseBatch(
        response=np.array([0.0, 0.02, 0.3, 0.7, 0.98, 1.0], dtype=np.float32),
        sample_ids=sample_ids[:6],
        drug_ids=np.repeat("d1", 6),
    )
    validation = ResponseBatch(
        response=np.array([0.15, 0.85], dtype=np.float32),
        sample_ids=sample_ids[6:],
        drug_ids=np.repeat("d1", 2),
    )
    widths = {"gene_expression": 6, "mutations": 5, "copy_number_variation_gistic": 4}
    blocks = {
        name: numeric_feature_block(rng.normal(size=(8, width)).astype(np.float32)) for name, width in widths.items()
    }
    return ModelInputBatch.from_response(
        training,
        sample=FeaturizerResult.from_blocks(blocks, sample_ids),
        drug=FeaturizerResult(
            np.array(["d1"]),
            {
                "identity": numeric_feature_block(np.ones((1, 1), dtype=np.float32)),
                "identity_categories": numeric_feature_block(np.array([["d1"]], dtype=object)),
            },
            ("identity",),
        ),
        sample_pair_idx=np.arange(6),
        drug_pair_idx=np.zeros(6, dtype=np.int64),
        early_stopping_response=validation,
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


@contextmanager
def _isolated_reference_rng(seed: int) -> Iterator[None]:
    """Reproduce the capture process without leaking process-global RNG settings."""
    import random

    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    thread_count = torch.get_num_threads()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        torch.set_num_threads(thread_count)


def _array_payload(value: Any) -> dict[str, Any]:
    """Serialize an array exactly as the frozen capture harness did."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "values": array.tolist()}


@lru_cache(maxsize=3)
def _current_reference_sequence(seed: int) -> dict[str, Any]:
    """Replay all eight baseline operations in their original RNG order."""
    import torch
    from torch_geometric.data import Data

    from drevalpy.components.predictors.literature.dipk.model_utils import Predictor as DIPKNetwork
    from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGraphNet
    from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix
    from drevalpy.components.predictors.literature.molir.utils import generate_triplets_indices
    from drevalpy.components.predictors.literature.pharmaformer.model_utils import CombinedModel
    from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork
    from drevalpy.components.predictors.literature.sparsego.algorithm import SparseLinearNew
    from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
    from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder

    result: dict[str, Any] = {"seed": seed}
    with _isolated_reference_rng(seed), torch.no_grad():
        graph = Data(
            x=torch.randn(6, 5),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        cell = torch.randn(2, 7)
        druggnn = DrugGraphNet(num_node_features=5, num_cell_features=7, hidden_dim=8, dropout=0.0).eval()
        result["druggnn"] = _array_payload(druggnn(graph, cell))

        pharmaformer = CombinedModel(
            gene_input_size=6,
            gene_hidden_size=8,
            drug_hidden_size=8,
            feature_dim=4,
            nhead=2,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
        ).eval()
        result["pharmaformer"] = _array_payload(pharmaformer(torch.randn(2, 6), torch.randn(2, 128)))

        precily = PrecilyNetwork(input_dim=8, dropout=0.0).eval()
        result["precily"] = _array_payload(precily(torch.randn(2, 8)))

        sparsego = SparseLinearNew(
            in_features=2,
            out_features=2,
            connectivity=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        ).eval()
        result["sparsego"] = _array_payload(sparsego(torch.tensor([[1.0, 2.0]])))

        dipk = DIPKNetwork(
            heads=1,
            fc_layer_num=3,
            fc_layer_dim=[8, 8, 8, 8, 8, 8],
            dropout_rate=0.0,
        ).eval()
        result["dipk"] = _array_payload(
            dipk(torch.randn(2, 3, 768), torch.randn(2, 512), torch.randn(2, 512), torch.ones(2, 3))
        )

        result["molir"] = _array_payload(
            _realign_omic_matrix(
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                ["g1", "g2"],
                ["g0", "g1", "g2"],
            )
        )
        positive, negative = generate_triplets_indices(
            np.array([0.0, 0.2, 0.8, 1.0]),
            positive_range=0.1,
            negative_range=1.0,
            random_seed=seed,
        )
        result["molir_triplets"] = {
            "positive": np.asarray(positive).tolist(),
            "negative": np.asarray(negative).tolist(),
        }

        srmf = SRMFPredictor({"K": 2, "max_iter": 2, "seed": seed})
        factors = srmf._cmf(
            np.ones((2, 3), dtype=np.float64),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64),
            np.eye(2, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )
        result["srmf"] = {"u": _array_payload(factors[0]), "v": _array_payload(factors[1])}

        encoder = SuperFELTEncoder(
            input_size=6,
            hpams={
                "dropout_rate": 0.0,
                "margin": 1.0,
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "out_dim_expr_encoder": 4,
                "out_dim_mutation_encoder": 3,
                "out_dim_cnv_encoder": 2,
                "epochs": 1,
                "mini_batch": 2,
            },
            omic_type="expression",
            ranges=(0.1, 1.0),
        ).eval()
        result["superfeltr"] = _array_payload(encoder(torch.randn(2, 6)))
    return result


def _assert_array_payload_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare one captured tensor including its dimensional and dtype contract."""
    assert actual["shape"] == expected["shape"]
    assert actual["dtype"] == expected["dtype"]
    np.testing.assert_allclose(actual["values"], expected["values"], rtol=1e-6, atol=1e-7)


def assert_frozen_reference(model: str, seed: int) -> None:
    """Run the current implementation and compare it to the frozen v2 output."""
    from tests.components.predictors.literature._reference.loader import load_seed

    expected = load_seed(seed)
    actual = _current_reference_sequence(seed)
    if model == "srmf":
        for factor in ("u", "v"):
            _assert_array_payload_matches(actual[model][factor], expected[model][factor])
    else:
        _assert_array_payload_matches(actual[model], expected[model])
    if model == "molir":
        assert actual["molir_triplets"] == expected["molir_triplets"]


#: Distinguishes "use :data:`DRUG_PAIR_IDX`" from an explicit ``drug_pair_idx=None``,
#: which is how a single-drug-side predictor says it has no per-drug axis.
_DEFAULT = object()


def two_by_two_batch(
    *,
    sample_blocks: dict[str, FeatureBlock],
    drug_blocks: dict[str, FeatureBlock],
    response: ResponseBatch | None = None,
    sample_pair_idx: np.ndarray | None = None,
    drug_pair_idx: Any = _DEFAULT,
    early_stopping_response: ResponseBatch | None = None,
    checkpoint_dir: Any = ".",
) -> ModelInputBatch:
    """Build a featurized batch over two samples and two drugs.

    :param sample_blocks: Sample blocks the predictor under test consumes.
    :param drug_blocks: Drug blocks the predictor under test consumes.
    :param response: Training responses; defaults to ``multi_drug_response()``.
    :param sample_pair_idx: Per-pair sample row index; defaults to the 2x2 grid.
    :param drug_pair_idx: Per-pair drug row index; defaults to the 2x2 grid. Pass
        ``None`` explicitly for a predictor with no per-drug axis.
    :param early_stopping_response: Optional early-stopping responses.
    :param checkpoint_dir: Directory recorded on the ``TrainingContext``.
    :returns: Featurized ``ModelInputBatch``.
    """
    return ModelInputBatch.from_response(
        multi_drug_response() if response is None else response,
        sample=FeaturizerResult.from_blocks(sample_blocks, SAMPLE_IDS),
        drug=FeaturizerResult(
            DRUG_IDS,
            {
                name: (
                    FeatureBlock(block.values, block.format, block.feature_names, block.metadata, entity_aligned=False)
                    if block.entity_aligned and block.values.ndim > 0 and len(block.values) != len(DRUG_IDS)
                    else block
                )
                for name, block in drug_blocks.items()
            },
            tuple(name for name, block in drug_blocks.items() if block.values.dtype.kind in "biufc"),
        ),
        sample_pair_idx=SAMPLE_PAIR_IDX if sample_pair_idx is None else sample_pair_idx,
        drug_pair_idx=DRUG_PAIR_IDX if drug_pair_idx is _DEFAULT else drug_pair_idx,
        early_stopping_response=early_stopping_response,
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


def early_stopping_response() -> ResponseBatch:
    """Return a two-pair early-stopping response over the same entities.

    :returns: ``ResponseBatch`` for the early-stopping split.
    """
    return ResponseBatch(
        response=np.array([1.5, 2.5]),
        sample_ids=SAMPLE_IDS,
        drug_ids=DRUG_IDS,
    )
