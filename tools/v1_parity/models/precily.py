"""Precily parity case: 1.6.1 pathway scores and network against the v2 ``pathways`` featurizer and network.

The case holds a log-scale expression matrix, a small synthetic GMT (twelve gene sets of 6-30 genes
plus two sets below GSVA's ``min_size=5``, which both sides drop) and a drug embedding matrix standing
in for SMILESVec. Network weights are drawn in ``build_case`` (numpy) and loaded into both networks,
so no training is compared.

Steps:

``pathways_all``
    1.6.1 ``create_precily_pathway_features._run_gsva`` over every sample (the 1.6.1 flow: one pass,
    before any split) against the v2 ``PathwaysSampleFeaturizer`` fitted on every sample.
``pathway_names``
    The pathway columns of both sides.
``net_forward``
    The two ``PrecilyNetwork`` classes (eval mode) with the copied weights on the same random matrix.
``predictions``
    GSVA scores of each side, concatenated with the drug embedding, through that side's network.
``pathways_fold`` (deviation)
    1.6.1 once-over-all scores of the training samples against v2 scores fitted on the training
    samples only. This is plan section 4.4's "GSVA per fold vs once".
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, StepSpec

NAME = "Precily"

N_SETS = 12
N_SMALL_SETS = 2
DRUG_DIM = 16
DROPOUT = 0.1
LAYER_SHAPES = ((1429, None), (512, 1429), (140, 512), (200, 140), (1, 200))
LAYER_INDEX = (0, 2, 5, 8, 11)

FINDINGS = [
    "1.6.1 computes the GSVA pathway scores once, over every sample of the dataset, in "
    "`create_precily_pathway_features` and the model reads the CSV; v2 fits the `pathways` featurizer on "
    "the training samples of each fold and scores unseen samples together with them. The scores of one "
    "sample therefore depend on the cohort (deviation `pathways_fold`).",
    "Both sides call `gseapy.gsva` with the same options (Gaussian kernel, min_size=5, max_size=2000, "
    "mx_diff, threads=4, seed=42) and pivot the long result the same way, so a fit over the same samples "
    "gives the same scores (`pathways_all`). The 1.6.1 flow log2(TPM+1)-transforms the expression for "
    "TPM datasets before this call; v2 reads the `gene_expression` view as it is, so the case feeds both "
    "the same log-scale matrix.",
    "On this random case the per-fold scores of the training samples differ from the once-over-all scores "
    "by up to 0.2 (Pearson 0.975, 12 pathways, 24 of 30 samples); the size depends on the cohort and is "
    "not bounded by a tolerance class, so the step is a deviation and misses `statistical`.",
    "The v2 featurizer downloads the MSigDB C2 CP v6.1 file when the dataset has no `pathways_gmt`; this "
    "case supplies a small GMT in `uns` so that it stays offline.",
]


def _gmt_text(rng: np.random.Generator, genes: np.ndarray) -> str:
    """Build a GMT with gene sets of 6-30 genes and two below GSVA's minimum size.

    :param rng: Generator of the case.
    :param genes: Gene names of the expression matrix.
    :returns: GMT file content.
    """
    sizes = [*rng.integers(6, 31, size=N_SETS).tolist(), *([3] * N_SMALL_SETS)]
    lines = []
    for index, size in enumerate(sizes):
        members = rng.choice(genes, size=size, replace=False)
        lines.append("\t".join([f"SET_{index:02d}", "parity", *members]))
    return "\n".join(lines) + "\n"


def build_case(seed: int) -> Case:
    """Draw the expression, gene sets, drug embeddings, network weights and a probe matrix.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    arrays = tabular_case_arrays(seed, n_samples=30, n_drugs=6, n_genes=120, n_bits=8)
    arrays["expression"] = np.log2(arrays["expression"].astype(np.float64) + 1.0).astype(np.float32)
    rng = np.random.default_rng(seed + 7)
    gmt = _gmt_text(rng, arrays["gene_names"])
    arrays["smilesvec"] = rng.normal(size=(len(arrays["drug_ids"]), DRUG_DIM)).astype(np.float32)
    input_dim = N_SETS + DRUG_DIM
    for (out_dim, in_dim), index in zip(LAYER_SHAPES, LAYER_INDEX, strict=True):
        fan_in = in_dim or input_dim
        arrays[f"weight_{index}"] = (rng.normal(size=(out_dim, fan_in)) / np.sqrt(fan_in)).astype(np.float32)
        arrays[f"bias_{index}"] = (0.1 * rng.normal(size=out_dim)).astype(np.float32)
    arrays["probe"] = rng.normal(size=(64, input_dim)).astype(np.float32)
    return Case(
        model=NAME,
        seed=seed,
        arrays=arrays,
        meta={"gmt": gmt, "n_pathways": N_SETS, "dropout": DROPOUT, "gsva": {"min_size": 5, "max_size": 2000}},
    )


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the scores and the network, and the per-fold deviation.
    """
    return [
        StepSpec("pathway_names", "exact", description="pathway columns, in order"),
        StepSpec("pathways_all", "exact", description="GSVA scores over every sample, same samples both sides"),
        StepSpec("net_forward", "forward", description="PrecilyNetwork with copied weights on a random matrix"),
        StepSpec("predictions", "forward", description="GSVA scores + drug embeddings through the network"),
        StepSpec(
            "pathways_fold",
            "statistical",
            v1_key="pathways_train_once",
            v2_key="pathways_train_fold",
            scope=DEVIATION,
            description="scores of the training samples: 1.6.1 over all samples vs v2 over the training samples",
            explanation="1.6.1 runs GSVA once over every sample before splitting; v2 fits the featurizer on the "
            "training samples of each fold, so a score depends on the cohort it was computed with (plan 4.4).",
        ),
    ]


def _load_weights(network: Any, case: Case) -> None:
    """Copy the case's weights into a ``PrecilyNetwork``.

    :param network: The network.
    :param case: The case holding ``weight_<i>`` / ``bias_<i>``.
    """
    import torch

    state = {
        f"net.{index}.{kind}": torch.from_numpy(case.arrays[f"{kind}_{index}"].astype(np.float32))
        for index in LAYER_INDEX
        for kind in ("weight", "bias")
    }
    network.load_state_dict(state)


def _network_outputs(network_class: Any, case: Case, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Run the network on the probe matrix and on the test pairs' scores plus drug embeddings.

    :param network_class: ``PrecilyNetwork`` of either side.
    :param case: The case.
    :param scores: Pathway scores for every sample, in ``sample_ids`` order.
    :returns: ``net_forward`` and ``predictions``.
    """
    import torch

    network = network_class(input_dim=scores.shape[1] + DRUG_DIM, dropout=case.meta["dropout"])
    _load_weights(network, case)
    network.eval()
    pairs = case.arrays["test_pairs"]
    features = np.concatenate([scores[pairs[:, 0]], case.arrays["smilesvec"][pairs[:, 1]]], axis=1)
    with torch.no_grad():
        probe = network(torch.from_numpy(case.arrays["probe"])).numpy()
        predictions = network(torch.from_numpy(features.astype(np.float32))).numpy()
    return {"net_forward": probe, "predictions": predictions}


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Score every sample with the 1.6.1 GSVA flow and run the 1.6.1 network.

    :param case: The shared case.
    :returns: ``pathways_all``, ``pathway_names``, ``pathways_train_once``, ``net_forward``, ``predictions``.
    """
    import pandas as pd
    from drevalpy.datasets.featurizer.create_precily_pathway_features import _run_gsva
    from drevalpy.models.Precily.model_utils import PrecilyNetwork

    sample_ids = case.arrays["sample_ids"]
    expression = pd.DataFrame(
        case.arrays["expression"].astype(np.float64), index=sample_ids, columns=list(case.arrays["gene_names"])
    )
    with tempfile.TemporaryDirectory() as directory:
        gmt_path = os.path.join(directory, "sets.gmt")
        with open(gmt_path, "w", encoding="utf-8") as handle:
            handle.write(case.meta["gmt"])
        scores = _run_gsva(
            expression.T,
            gene_sets=gmt_path,
            min_size=case.meta["gsva"]["min_size"],
            max_size=case.meta["gsva"]["max_size"],
            kcdf="Gaussian",
            mx_diff=True,
            threads=4,
            seed=42,
        )
    scores = scores.loc[list(sample_ids)]
    values = scores.to_numpy(dtype=np.float32)
    train_rows = np.unique(case.arrays["train_pairs"][:, 0])
    out = {
        "pathways_all": values,
        "pathway_names": np.array(list(scores.columns)),
        "pathways_train_once": values[train_rows],
    }
    out.update(_network_outputs(PrecilyNetwork, case, values))
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Score with the v2 ``pathways`` featurizer (all samples, then the training samples) and run its network.

    :param case: The shared case.
    :returns: ``pathways_all``, ``pathway_names``, ``pathways_train_fold``, ``net_forward``, ``predictions``.
    """
    from drevalpy.components.featurizers.sample.pathways import PathwaysSampleFeaturizer
    from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork
    from drevalpy.types.data.feature_source import SampleFeatureSource
    from tools.v1_parity._v2_runner import tabular_inputs

    data = tabular_inputs(case)
    data.dataset.mdata.uns["pathways_gmt"] = case.meta["gmt"]
    sample_ids = case.arrays["sample_ids"]
    source = SampleFeatureSource(data.dataset, sample_ids)

    everyone = PathwaysSampleFeaturizer().fit(source, entity_ids=sample_ids)
    scores = everyone.transform(source, sample_ids)
    names = everyone.transform_blocks(source, sample_ids)["pathways"].feature_names

    train_rows = np.unique(case.arrays["train_pairs"][:, 0])
    per_fold = PathwaysSampleFeaturizer().fit(source, entity_ids=sample_ids[train_rows])
    out = {
        "pathways_all": scores,
        "pathway_names": np.array(list(names)),
        "pathways_train_fold": per_fold.transform(source, sample_ids[train_rows]),
    }
    out.update(_network_outputs(PrecilyNetwork, case, scores))
    return out
