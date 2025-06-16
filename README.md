# DrEvalPy: Python Cancer Cell Line Drug Response Prediction Suite

[![PyPI version](https://img.shields.io/pypi/v/drevalpy.svg)](https://pypi.org/project/drevalpy/)
![Python versions](https://img.shields.io/pypi/pyversions/drevalpy)
[![License](https://img.shields.io/github/license/daisybio/drevalpy)](https://opensource.org/licenses/GPL3)
[![Read the Docs](https://img.shields.io/readthedocs/drevalpy/latest.svg?label=Read%20the%20Docs)](https://drevalpy.readthedocs.io/)
[![Test status](https://github.com/daisybio/drevalpy/actions/workflows/run_tests.yml/badge.svg)](https://github.com/daisybio/drevalpy/actions?workflow=Tests)
[![Precommit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**News:** Our preprint is out on [biorxiv](https://www.biorxiv.org/content/10.1101/2025.05.26.655288v1)!

Documentation at [ReadTheDocs](https://drevalpy.readthedocs.io/en/latest/index.html#).

**Focus on Innovating Your Models — DrEval Handles the Rest!**

- DrEval is a toolkit that ensures drug response prediction evaluations are statistically sound, biologically meaningful, and reproducible.
- Focus on model innovation while using our automated standardized evaluation protocols and preprocessing workflows.
- A flexible model interface supports all model types (e.g. Machine Learning, Stats, Network-based analyses)

By contributing your model to the DrEval catalog, you can increase your work's exposure, reusability, and transferability.

![DrEval](docs/_static/img/overview.png)

Use DrEval to Build Drug Response Models That Have an Impact

    1. Maintained, up-to-date baseline catalog, no need to re-implement literature models

    2. Gold standard datasets for benchmarking

    3. Consistent application-driven evaluation

    4. Ablation studies with permutation tests

    5. Cross-study evaluation for generalization analysis

    6. Optimized nextflow pipeline for fast experiments

    7. Easy-to-use hyperparameter tuning

    8. Paper-ready visualizations to display performance

This project is a collaboration of the Technical University of Munich (TUM, Germany)
and the Freie Universität Berlin (FU, Germany).

## Installation

Using pip:

```bash
pip install drevalpy
```

Using docker:

```bash
docker pull ghcr.io/daisybio/drevalpy:main
```

From source:

```bash
git clone https://github.com/daisybio/drevalpy.git
cd drevalpy
pip install poetry
pip install poetry-plugin-export
poetry install
```

## Quickstart

To run models from the catalog, you can run:

```bash
python run_suite.py --run_id my_first_run --models NaiveTissueMeanPredictor NaiveDrugMeanPredictor --baselines NaiveMeanEffectsPredictor --dataset TOYv1 --test_mode LCO
```

This will train our baseline models which just predict the drug or tissue means or the mean drug and cell line effects.
It will evaluate in "LCO" which is the leave-cell-line-out splitting strategy using 7 fold cross validation.
The results will be stored in

```bash
results/my_first_run/TOYv1/LCO
```

You can visualize them using

```bash
python create_report.py --run_id my_first_run --dataset TOYv1
```

This will create an index.html file which you can open in your web browser.

You can also run a drug response experiment using Python:

```python
from drevalpy.experiment import drug_response_experiment
from drevalpy.models import MODEL_FACTORY
from drevalpy.datasets import AVAILABLE_DATASETS

naive_mean = MODEL_FACTORY["NaiveMeanEffectsPredictor"]
rf = MODEL_FACTORY["RandomForest"]
simple_nn = MODEL_FACTORY["SimpleNeuralNetwork"]

toyv2 = AVAILABLE_DATASETS["TOYv2"](path_data="data", measure="LN_IC50_curvecurator")

drug_response_experiment(
            models=[rf, simple_nn],
            baselines=[naive_mean],
            response_data=toyv2,
            metric="RMSE",
            n_cv_splits=7,
            test_mode="LCO",
            run_id="my_second_run",
            path_data="data",
            hyperparameter_tuning=False,
        )
```

This will run the Random Forest and Simple Neural Network models on the CTRPv2 dataset, using the Naive Mean Effects Predictor as a baseline. The results will be stored in `results/my_second_run/CTRPv2/LCO`.
To obtain evaluation metrics, you can use:

```python
from drevalpy.visualization.utils import parse_results, prep_results, write_results
import pathlib

# load data, evaluate per CV run
(
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results="results/my_second_run", dataset='TOYv2')
# reformat, calculate normalized metrics
(
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(
        evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred, pathlib.Path("data")
    )

write_results(
        path_out="results/my_second_run",
        eval_results=evaluation_results,
        eval_results_per_drug=evaluation_results_per_drug,
        eval_results_per_cl=evaluation_results_per_cell_line,
        t_vs_p=true_vs_pred,
    )
```

We recommend the use of our Nextflow pipeline for computational demanding runs and for improved reproducibility.
No knowledge of Nextflow is required to run it. The nextflow pipeline is available here: [nf-core-drugresponseeval](https://github.com/JudithBernett/nf-core-drugresponseeval).

## Example Report

[Browse our benchmark results here.](https://dilis-lab.github.io/drevalpy-report/)

## Contact

Main developers:

- [Judith Bernett](mailto:judith.bernett@tum.de), [Data Science in Systems Biology](https://www.mls.ls.tum.de/daisybio/startseite/), TUM
- [Pascal Iversen](mailto:Pascal.Iversen@hpi.de), [Data Integration in the Life Sciences](https://www.mi.fu-berlin.de/w/DILIS/WebHome), FU Berlin, Hasso-Plattner-Institut
