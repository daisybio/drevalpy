"""Inputs shared by the three-omic single-drug cases (MOLIR and SuperFELTR); no case module of its own.

Both 1.6.1 models take gene expression, mutations and copy number variation of the cell lines, and 1.6.1 trains one
model per drug (``experiment.make_model_list``: one ``Model.<drug>`` per drug, each on the pairs of that drug).
This module draws the inputs (real gene symbols, every view wider than its gene list, so that the list selection
runs for real), builds the 1.6.1 and v2 data objects, and holds the per-drug loops of the 1.6.1 side.

The response matrix has no gaps: 1.6.1 fits scaler and feature selection on the samples of each drug, v2 fits them
once on the samples of all training pairs, and the two agree exactly when every drug has the same training samples.
The ``ragged`` pair split drops training samples of the last drug to show the difference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, StepSpec

VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")
DEFAULT_LISTS = {
    "gene_expression": "gene_expression_intersection",
    "mutations": "mutations_intersection",
    "copy_number_variation_gistic": "copy_number_variation_gistic_intersection",
}
N_EXTRA = 10
N_SAMPLES = 36
N_DRUGS = 2
DRUGS = [f"D{i:02d}" for i in range(N_DRUGS)]
#: Samples held out for early stopping (every ``STOP_EVERY``-th training sample), the same for every drug.
STOP_EVERY = 4
RAGGED_DROPPED = 6
#: Root seed of the per-drug seeds. v2 draws it from the global RNG; the case pins it so that 1.6.1 can use it too.
#: Not arbitrary for MOLIR: see the finding on training sensitivity in ``molir.FINDINGS``.
ROOT_SEED = 11
#: The variant that also records the unpatched 1.6.1 behaviour (all drugs from one shared RNG stream).
SHARED_STREAM_VARIANT = "default"
SHARED_STREAM_EXPLANATION = (
    "1.6.1 trains the per-drug models one after the other and draws all of them from one RNG stream, seeded once "
    "per run: the initial weights, dropout and loader shuffles of a drug depend on the drugs trained before it. v2 "
    "seeds every drug from `drug_seed(root seed, drug id)` (`single_drug_routing.run_per_drug`), so that a drug's "
    "model does not depend on the order or the process it is fitted in, and the results are the same for every "
    "`n_jobs`. The parity steps give the 1.6.1 side the same per-drug seed before each drug; this step compares "
    "the unpatched 1.6.1 run (one `seed_all` per run) with v2, and records how far the two streams are apart."
)
OMIC_GROUPS = {
    "expression": "expression_encoder",
    "mutation": "mutation_encoder",
    "cnv": "cna_encoder",
    "regressor": "regressor",
}


def _draw_view(view: str, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Draw one omics matrix, float32-representable.

    Mutations and copy numbers are continuous values, not 0/1 and gistic levels. Both models pass these views through
    unchanged, and a column that is constant within a batch (all zero, or all one) has a gradient that is
    exactly zero in theory behind the batch norm layer and rounding noise in practice; Adagrad turns that noise
    into a full learning-rate step.

    :param view: View name.
    :param rng: Generator of the case.
    :param shape: ``(samples, genes)``.
    :returns: The matrix.
    """
    if view == "gene_expression":
        values = rng.gamma(shape=2.0, scale=30.0, size=shape)
    else:
        values = rng.normal(size=shape)
    return values.astype(np.float32)


def build_three_omics_case(
    name: str, seed: int, variants: dict[str, dict[str, Any]], lists: dict[str, str | None], n_synthetic: int = 60
) -> Case:
    """Draw the three wide views, the gene lists, a gap-free response and the pair splits.

    :param name: Harness model name.
    :param seed: Seed of the input generator.
    :param variants: Variant settings, stored in the case metadata.
    :param lists: Gene list name per view; ``None`` keeps every column of a synthetic view of *n_synthetic* genes.
    :param n_synthetic: Width of the views without a gene list.
    :returns: The case. ``train_pairs`` / ``test_pairs`` hold every drug for the training / test samples;
        ``stop_mask`` marks the early-stopping pairs (whole samples); ``ragged_train_pairs`` lacks some samples of
        the last drug.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=1, n_bits=8)
    rng = np.random.default_rng(seed + 211)
    resolved = {
        view: np.array(resolve_gene_list(list_name))
        if list_name is not None
        else np.array([f"{view[:3].upper()}_G{i:03d}" for i in range(n_synthetic)])
        for view, list_name in lists.items()
    }
    for view in VIEWS:
        n_extra = N_EXTRA if lists[view] is not None else 0
        genes = np.array([*resolved[view], *(f"X{view[:3].upper()}{i:03d}" for i in range(n_extra))])
        genes = genes[rng.permutation(len(genes))]
        if lists[view] is None:
            resolved[view] = genes
        arrays[f"view__{view}"] = _draw_view(view, rng, (N_SAMPLES, len(genes)))
        arrays[f"view__{view}__names"] = genes
        arrays[f"genelist__{view}"] = resolved[view]
    arrays["view__mutations_binary"] = (rng.random(arrays["view__mutations"].shape) < 0.3).astype(np.float32)
    arrays["expression"], arrays["gene_names"] = arrays["view__gene_expression"], arrays["view__gene_expression__names"]

    columns = {view: [list(arrays[f"view__{view}__names"]).index(g) for g in resolved[view]] for view in VIEWS}
    expression = np.arcsinh(arrays["view__gene_expression"][:, columns["gene_expression"]].astype(np.float64))
    signal = expression[:, :60] @ (rng.normal(size=60) / 60.0)
    signal += arrays["view__mutations"][:, columns["mutations"][:40]].astype(np.float64) @ rng.normal(size=40) / 20.0
    effect = rng.normal(size=N_DRUGS)
    response = (
        signal[:, None] * (1.0 + 0.3 * effect[None, :]) + effect[None, :] + rng.normal(0.0, 0.1, (N_SAMPLES, N_DRUGS))
    )
    arrays["response"] = response.astype(np.float32)

    n_test_only = max(1, round(N_SAMPLES * 0.2))
    n_train = N_SAMPLES - n_test_only
    train_rows = np.arange(n_train)
    arrays["train_pairs"] = np.array([(i, j) for i in train_rows for j in range(N_DRUGS)], dtype=np.int64)
    arrays["test_pairs"] = np.array([(i, j) for i in range(n_train, N_SAMPLES) for j in range(N_DRUGS)], dtype=np.int64)
    arrays["stop_mask"] = (arrays["train_pairs"][:, 0] % STOP_EVERY) == STOP_EVERY - 1
    dropped = (arrays["train_pairs"][:, 1] == N_DRUGS - 1) & (arrays["train_pairs"][:, 0] < RAGGED_DROPPED)
    arrays["ragged_train_pairs"] = arrays["train_pairs"][~dropped]
    arrays["ragged_stop_mask"] = (arrays["ragged_train_pairs"][:, 0] % STOP_EVERY) == STOP_EVERY - 1
    meta = {"variants": variants, "lists": lists, "n_drugs": N_DRUGS}
    return Case(model=name, seed=seed, arrays=arrays, meta=meta)


def variant_pairs(case: Case, setting: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the training pairs of a variant and the early-stopping flag of each.

    :param case: The shared case.
    :param setting: Variant settings (``ragged`` selects the split without some samples of the last drug).
    :returns: ``(train_pairs, stop_mask)``; the mask is all ``False`` for variants without early stopping.
    """
    ragged = setting.get("ragged", False)
    pairs = case.arrays["ragged_train_pairs" if ragged else "train_pairs"]
    stop = case.arrays["ragged_stop_mask" if ragged else "stop_mask"]
    return pairs, (stop if setting["early_stopping"] else np.zeros(len(pairs), dtype=bool))


def used_sample_ids(case: Case, setting: dict[str, Any], drug_index: int) -> np.ndarray:
    """Return the sorted ids of the samples of one drug's fit and early-stopping pairs.

    :param case: The shared case.
    :param setting: Variant settings.
    :param drug_index: Column of the response matrix.
    :returns: Sorted unicode array.
    """
    pairs, _ = variant_pairs(case, setting)
    return np.unique(case.arrays["sample_ids"][pairs[pairs[:, 1] == drug_index, 0]]).astype(str)


def view_matrix(case: Case, setting: dict[str, Any], view: str) -> np.ndarray:
    """Return the wide matrix of a view, with the binary mutations when the variant asks for them.

    :param case: The shared case.
    :param setting: Variant settings (``binary`` swaps in 0/1 mutations).
    :param view: View name.
    :returns: Samples x columns, float32.
    """
    if view == "mutations" and setting.get("binary"):
        return case.arrays["view__mutations_binary"]
    return case.arrays[f"view__{view}"]


def v1_cell_lines(case: Case, setting: dict[str, Any] | None = None, *, arcsinh: bool = False) -> Any:
    """Build the 1.6.1 three-view ``FeatureDataset`` restricted to the gene lists.

    1.6.1 applies the lists while reading the CSVs, which the in-memory harness skips, so the columns are
    selected and ordered here with the same result.

    :param case: The shared case.
    :param setting: Variant settings (see :func:`view_matrix`).
    :param arcsinh: Apply ``arcsinh`` to the expression, which the 1.6.1 SuperFELTR loader does while reading.
    :returns: The ``FeatureDataset`` with float32 features.
    """
    from drevalpy.datasets.dataset import FeatureDataset

    sample_ids = case.arrays["sample_ids"]
    features: dict[str, dict[str, np.ndarray]] = {str(sid): {} for sid in sample_ids}
    meta: dict[str, Any] = {}
    for view in VIEWS:
        names = list(case.arrays[f"view__{view}__names"])
        wanted = list(case.arrays[f"genelist__{view}"])
        columns = [names.index(gene) for gene in wanted]
        matrix = view_matrix(case, setting or {}, view)[:, columns].astype(np.float32)
        if arcsinh and view == "gene_expression":
            matrix = np.arcsinh(matrix)
        for row, sid in enumerate(sample_ids):
            features[str(sid)][view] = matrix[row].copy()
        meta[view] = np.array(wanted)
    return FeatureDataset(features=features, meta_info=meta)


def v2_three_omics_dataset(case: Case, setting: dict[str, Any] | None = None) -> Any:
    """Build the v2 ``Dataset`` with the three wide views and the response matrix.

    :param case: The shared case.
    :param setting: Variant settings (see :func:`view_matrix`).
    :returns: The dataset.
    """
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset

    sample_ids, drug_ids = case.arrays["sample_ids"], case.arrays["drug_ids"]
    response = ad.AnnData(
        X=case.arrays["response"].astype(np.float32),
        obs=pd.DataFrame({"cell_line_name": sample_ids, "tissue": case.arrays["tissues"]}, index=sample_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    response.varm["morgan_fingerprint"] = case.arrays["fingerprints"].astype(np.float32)
    modalities = {"response": response}
    for view in VIEWS:
        modalities[view] = ad.AnnData(
            X=view_matrix(case, setting or {}, view).astype(np.float32),
            obs=pd.DataFrame(index=sample_ids),
            var=pd.DataFrame(index=case.arrays[f"view__{view}__names"]),
        )
    return Dataset(md.MuData(modalities), name="parity")


@contextmanager
def seeded_triplets(*modules: Any, base: int = 1000) -> Iterator[None]:
    """Give ``generate_triplets_indices`` of the given modules an explicit, call-counted seed.

    Without it 1.6.1 draws the triplets from an unseeded ``np.random.default_rng(None)`` and v2 from the global
    NumPy stream, so no two runs (and no two implementations) see the same triplets. Both implementations seed
    ``np.random.default_rng(random_seed)`` when a seed is passed, so the same sequence of seeds gives the same
    triplets. The call order (batches of the training epoch, then of the validation epoch, drug after drug) is the
    same on both sides.

    :param modules: Modules that have a module-level ``generate_triplets_indices`` (and call it from there).
    :param base: Seed of the first call.
    :yields: Nothing; the patch is undone on exit.
    """
    counter = itertools.count(base)
    originals = [module.generate_triplets_indices for module in modules]

    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def patched(y: np.ndarray, positive_range: float, negative_range: float, random_seed: int | None = None) -> Any:
            del random_seed
            return original(y, positive_range, negative_range, random_seed=next(counter))

        return patched

    for module, original in zip(modules, originals, strict=True):
        module.generate_triplets_indices = wrap(original)
    try:
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            module.generate_triplets_indices = original


def grouped_params(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten the floating-point ``state_dict`` entries into one vector per network part.

    :param state: ``state_dict`` of a MOLIR or SuperFELTR-style network.
    :returns: ``params_<group>`` vectors, entries concatenated in key order.
    """
    out: dict[str, np.ndarray] = {}
    for group, prefix in OMIC_GROUPS.items():
        parts = [
            state[key].detach().cpu().numpy().astype(np.float64).ravel()
            for key in sorted(state)
            if key.startswith(prefix + ".") and state[key].is_floating_point()
        ]
        out[f"params_{group}"] = np.concatenate(parts)
    return out


def run_v1_drugs(
    case: Case,
    setting: dict[str, Any],
    cell_lines: Any,
    make_model: Callable[[], Any],
    capture: Callable[[dict[str, Any], str], Any],
    *,
    reseed: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Train and predict one 1.6.1 model per drug, in the order of ``experiment.make_model_list``.

    Every call gets a copy of the cell line features, as in ``experiment.train_and_predict``.

    :param case: The shared case.
    :param setting: Variant settings.
    :param cell_lines: The three-view ``FeatureDataset``.
    :param make_model: Builds a fresh, built 1.6.1 model.
    :param capture: Context manager factory given the sink dictionary and the drug id; it records the fit inputs.
    :param reseed: Seed ``random``, numpy and torch with ``reference_drug_seed(ROOT_SEED, drug)`` before each drug's
        training, as v2 does. ``False`` keeps the caller's single stream, the 1.6.1 behaviour.
    :returns: ``(predictions per drug, {drug: trained model})``; the predictions are keyed ``<drug>/predictions``.
    """
    pairs, stop = variant_pairs(case, setting)
    sample_ids, drug_ids = case.arrays["sample_ids"], case.arrays["drug_ids"]
    response = case.arrays["response"]
    out: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    from drevalpy.datasets.dataset import DrugResponseDataset

    def subset(selection: np.ndarray) -> Any:
        rows = pairs[selection]
        return DrugResponseDataset(
            response=response[rows[:, 0], rows[:, 1]].astype(np.float64),
            cell_line_ids=sample_ids[rows[:, 0]],
            drug_ids=drug_ids[rows[:, 1]],
            tissues=case.arrays["tissues"][rows[:, 0]],
            dataset_name="train",
        )

    sink: dict[str, Any] = {}
    for drug_index, drug in enumerate(drug_ids):
        of_drug = pairs[:, 1] == drug_index
        fit_set = subset(of_drug & ~stop)
        stop_set = subset(of_drug & stop) if setting["early_stopping"] else None
        test_rows = case.arrays["test_pairs"][case.arrays["test_pairs"][:, 1] == drug_index]
        model = make_model()
        if reseed:
            seed_all(reference_drug_seed(ROOT_SEED, str(drug)))
        with tempfile.TemporaryDirectory() as checkpoints, capture(sink, str(drug)):
            model.train(
                fit_set,
                cell_lines.copy(),
                None,
                stop_set,
                model_checkpoint_dir=checkpoints,
            )
        predictions = model.predict(sample_ids[test_rows[:, 0]], drug_ids[test_rows[:, 1]], cell_lines.copy(), None)
        out[f"{drug}/predictions"] = np.atleast_1d(np.asarray(predictions, dtype=np.float64))
        models[str(drug)] = model
    return out, {"models": models, "sink": sink}


def reference_drug_seed(root_seed: int, drug_id: str) -> int:
    """Return the per-drug seed of v2 (``drevalpy.utils.seed.drug_seed``), without importing drevalpy.

    The 1.6.1 interpreter cannot import the v2 package, so the formula is repeated here; :func:`pinned_root_seed`
    checks the copy against the real function on the v2 side.

    :param root_seed: Root seed.
    :param drug_id: Drug identifier.
    :returns: 32-bit unsigned seed.
    """
    key = json.dumps([root_seed, drug_id], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


@contextmanager
def pinned_root_seed(root_seed: int = ROOT_SEED) -> Iterator[None]:
    """Make the v2 per-drug fit use a fixed root seed instead of one drawn from the global RNG.

    ``run_per_drug`` derives each drug's seed as ``drug_seed(random.getrandbits(32), drug_id)``. Wrapping
    ``drug_seed`` in the routing module keeps everything else of the fit unchanged and gives the 1.6.1 side a seed
    it can reproduce (:func:`reference_drug_seed`).

    :param root_seed: Root seed to use for every drug.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.components.predictors import single_drug_routing
    from drevalpy.utils.seed import drug_seed

    for probe in ("D00", "D01", "x"):
        if drug_seed(root_seed, probe) != reference_drug_seed(root_seed, probe):
            raise RuntimeError("drug_seed no longer matches reference_drug_seed; update the harness copy")

    original = single_drug_routing.drug_seed
    single_drug_routing.drug_seed = lambda _root, drug_id: original(root_seed, drug_id)
    try:
        yield
    finally:
        single_drug_routing.drug_seed = original


def shared_stream_specs(
    drugs: list[str], groups: dict[str, str], variant: str = SHARED_STREAM_VARIANT
) -> list[StepSpec]:
    """Declare the deviation steps that compare the unpatched 1.6.1 run with the v2 run of *variant*.

    :param drugs: Drug ids.
    :param groups: Network parts whose trained parameters are compared (keys of ``OMIC_GROUPS``).
    :param variant: Variant whose v2 outputs are the reference.
    :returns: Deviation steps named ``shared_stream/<drug>/<quantity>``; the declared class is the one that the
        reseeded parity steps meet.
    """
    specs = []
    for drug in drugs:
        for quantity in (*(f"params_{group}" for group in groups), "predictions"):
            specs.append(
                StepSpec(
                    f"shared_stream/{drug}/{quantity}",
                    "train",
                    v2_key=f"{variant}/{drug}/{quantity}",
                    scope=DEVIATION,
                    description=f"unpatched 1.6.1 shared RNG stream against v2 per-drug seeds, {drug}: {quantity}",
                    explanation=SHARED_STREAM_EXPLANATION,
                )
            )
    return specs
