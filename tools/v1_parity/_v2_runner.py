"""Run one parity case with the v2 implementation.

Execute with the **v2 interpreter**, repository root on ``PYTHONPATH``::

    uv run python -m tools.v1_parity._v2_runner --model ElasticNet --case <case_dir> --out <out_dir>

Model case modules call the helpers below from their ``run_v2`` function. ``drevalpy`` is
imported lazily so that the module can be imported from the 1.6.1 interpreter without
failing (it is only run in the v2 one).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tools.v1_parity._bridge import Case, environment_info, load_model_module, record_calls, save_outputs, seed_all

if TYPE_CHECKING:
    from collections.abc import Iterator

    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.split_mask import SplitMask


@dataclass
class TabularInputs:
    """v2 objects built from the standard tabular case (see ``tabular_case_arrays``).

    :param dataset: ``Dataset`` with the response matrix, gene expression and fingerprints.
    :param train: Training positions (row-major pair order, equal to the case's pair order).
    :param test: Test positions.
    """

    dataset: Dataset
    train: SplitMask
    test: SplitMask


def tabular_inputs(
    case: Case, *, expression_view: str = "gene_expression", fingerprint_view: str = "morgan_fingerprint"
) -> TabularInputs:
    """Build a v2 ``Dataset`` in memory from the standard tabular case.

    :param case: Case holding the arrays of ``tabular_case_arrays``.
    :param expression_view: Name of the expression modality.
    :param fingerprint_view: Name of the drug ``varm`` entry holding the fingerprints.
    :returns: The dataset and the train/test masks.
    """
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.split_mask import SplitMask

    sample_ids, drug_ids = case.arrays["sample_ids"], case.arrays["drug_ids"]
    response = ad.AnnData(
        X=case.arrays["response"].astype(np.float32),
        obs=pd.DataFrame({"cell_line_name": sample_ids, "tissue": case.arrays["tissues"]}, index=sample_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    response.varm[fingerprint_view] = case.arrays["fingerprints"].astype(np.float32)
    expression = ad.AnnData(
        X=case.arrays["expression"].astype(np.float32),
        obs=pd.DataFrame(index=sample_ids),
        var=pd.DataFrame(index=case.arrays["gene_names"]),
    )
    shape = (len(sample_ids), len(drug_ids))
    return TabularInputs(
        dataset=Dataset(md.MuData({"response": response, expression_view: expression}), name="parity"),
        train=SplitMask.from_pairs(case.arrays["train_pairs"], shape),
        test=SplitMask.from_pairs(case.arrays["test_pairs"], shape),
    )


def synthetic_gene_spec(preset: str) -> Any:
    """Build a zoo preset whose ``scaledGeneExpression`` keeps every gene of the view.

    The synthetic gene names are in none of the shipped lists, and the 1.6.1 in-memory inputs are already
    restricted to them, so the v2 side drops the preset's gene list.

    :param preset: Zoo preset name.
    :returns: The model spec.
    """
    from importlib.resources import files

    import yaml

    from drevalpy.models.config.io import from_dict
    from drevalpy.models.spec import ModelSpec

    config = yaml.safe_load(files("drevalpy.models.zoo").joinpath(f"{preset}.yaml").read_text())
    config["sample_featurizer"]["scaledGeneExpression"]["gene_list"] = None
    return ModelSpec(preset, from_dict(config))


@contextmanager
def capture_matrix_predictor(sink: dict[str, Any]) -> Iterator[None]:
    """Record the design matrices and the fitted estimator of any sklearn tabular predictor.

    Fills ``sink["features_train"]``, ``sink["features_test"]``, ``sink["estimator"]`` and an
    unfitted twin in ``sink["fresh_estimator"]``.

    :param sink: Dictionary receiving the captured values.
    :yields: Nothing; the patches are active inside the ``with`` block.
    """
    from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor

    def keep_estimator(predictor: Any, store: dict[str, Any]) -> None:
        store["estimator"] = predictor._estimator
        store["fresh_estimator"] = predictor._make_estimator()

    with (
        record_calls(SklearnTabularPredictor, "_fit_matrix", sink, "features_train", after=keep_estimator),
        record_calls(SklearnTabularPredictor, "_predict_matrix", sink, "features_test"),
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """Load the case, run the model's ``run_v2`` and write ``outputs_v2.*``.

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
    arrays = module.run_v2(case)
    save_outputs(args.out, "v2", arrays, environment_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
