"""Run one parity case with the 1.6.1 implementation.

Execute with the **1.6.1 interpreter**, repository root on ``PYTHONPATH``::

    PYTHONPATH=<repo> /tmp/opencode/v161/.venv/bin/python -m tools.v1_parity._v1_runner \
        --model ElasticNet --case <case_dir> --out <out_dir>

Model case modules call the helpers below from their ``run_v1`` function. ``drevalpy`` is
imported lazily so that the module can be imported (but not run) from the v2 interpreter.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, environment_info, load_model_module, save_outputs, seed_all


@dataclass
class TabularInputs:
    """1.x objects built from the standard tabular case (see ``tabular_case_arrays``).

    :param train: Training responses.
    :param test: Test responses (a sample may occur only here).
    :param cell_line_input: Gene-expression ``FeatureDataset`` over every sample.
    :param drug_input: Fingerprint ``FeatureDataset`` over every drug.
    """

    train: Any
    test: Any
    cell_line_input: Any
    drug_input: Any


def _pairs_dataset(case: Case, pairs: np.ndarray, name: str) -> Any:
    from drevalpy.datasets.dataset import DrugResponseDataset

    sample_ids, drug_ids = case.arrays["sample_ids"], case.arrays["drug_ids"]
    response = case.arrays["response"][pairs[:, 0], pairs[:, 1]].astype(np.float64)
    return DrugResponseDataset(
        response=response,
        cell_line_ids=sample_ids[pairs[:, 0]],
        drug_ids=drug_ids[pairs[:, 1]],
        tissues=case.arrays["tissues"][pairs[:, 0]],
        dataset_name=name,
    )


def tabular_inputs(
    case: Case,
    *,
    feature_dtype: str = "float64",
    expression_view: str = "gene_expression",
    drug_view: str = "fingerprints",
) -> TabularInputs:
    """Build 1.x datasets in memory from the standard tabular case, with no CSV files.

    :param case: Case holding the arrays of ``tabular_case_arrays``.
    :param feature_dtype: dtype of the feature vectors. 1.6.1 reads CSVs as ``float64``; v2
        holds features as ``float32``, so ``"float32"`` isolates logic from precision.
    :param expression_view: View name of the expression features.
    :param drug_view: View name of the drug features.
    :returns: Train/test responses and both feature datasets.
    """
    from drevalpy.datasets.dataset import FeatureDataset

    dtype = np.dtype(feature_dtype)
    expression = case.arrays["expression"].astype(dtype)
    fingerprints = case.arrays["fingerprints"].astype(dtype)
    cell_lines = FeatureDataset(
        features={sid: {expression_view: expression[i].copy()} for i, sid in enumerate(case.arrays["sample_ids"])},
        meta_info={expression_view: case.arrays["gene_names"]},
    )
    drugs = FeatureDataset(
        features={did: {drug_view: fingerprints[i].copy()} for i, did in enumerate(case.arrays["drug_ids"])}
    )
    return TabularInputs(
        train=_pairs_dataset(case, case.arrays["train_pairs"], "train"),
        test=_pairs_dataset(case, case.arrays["test_pairs"], "test"),
        cell_line_input=cell_lines,
        drug_input=drugs,
    )


def main(argv: list[str] | None = None) -> int:
    """Load the case, run the model's ``run_v1`` and write ``outputs_v1.*``.

    :param argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    module = load_model_module(args.model)
    case = Case.load(args.case)
    seed_all(case.seed)
    arrays = module.run_v1(case)
    save_outputs(args.out, "v1", arrays, environment_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
