# DrEvalPy: Python Cancer Cell Line Drug Response Prediction Suite

Focus on Innovating Your Models — DrEval Handles the Rest!

- DrEval is a toolkit that ensures drug response prediction evaluations are statistically sound, biologically meaningful, and reproducible.
- Focus on model innovation while using our automated standardized evaluation protocols and preprocessing workflows.
- A flexible model interface supports all model types (e.g. Machine Learning, Stats, Network-based analyses)

By contributing your model to the DrEval catalog, you can increase your work's exposure, reusability, and transferability.

![DrEval](https://github.com/daisybio/drevalpy/blob/main/assets/dreval.png)

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

From Source:

```bash
conda env create -f models/SimpleNeuralNetwork/env.yml
pip install .
```

## Quickstart

To run models from the catalog, you can run:

```bash
python run_suite.py --run_id my_first_run --models ElasticNet SimpleNeuralNetwork --dataset GDSC1 --test_mode LCO
```

This will train and tune a neural network and an elastic net model on a subset of gene expression features and drug fingerprint features to predict IC50 values of the GDSC1 database. It will evaluate in "LCO" which is the leave-cell-line-out splitting strategy using 5 fold cross validation.
The results will be stored in

```bash
results/my_first_run/LCO
```

You can visualize them using

```bash
python create_report.py --run_id my_first_run
```

This will create an index.html file which you can open in your webbrowser.

You can also run a drug response experiment using Python:

```python

from drevalpy import drug_response_experiment

drug_response_experiment(
            models=["MultiOmicsNeuralNetwork"],
            baselines=["RandomForest"],
            response_data="GDSC1",
            metric="mse",
            n_cv_splits=5,
            test_mode="LPO",
            run_id="my_second_run",
        )
```

We recommend the use of our nextflow pipeline for computational demanding runs and for improved reproducibility. No knowledge of nextflow is required to run it. The nextflow pipeline is available here: [nf-core-drugresponseeval](https://github.com/JudithBernett/nf-core-drugresponseeval).

## Contact

Main developers:

- [Judith Bernett](mailto:judith.bernett@tum.de), [Data Science in Systems Biology](https://www.mls.ls.tum.de/daisybio/startseite/), TUM
- [Pascal Iversen](mailto:Pascal.Iversen@hpi.de), [Data Integration in the Life Sciences](https://www.mi.fu-berlin.de/inf/groups/ag-dilis/index.html), FU Berlin, Hasso Plattner Institute
