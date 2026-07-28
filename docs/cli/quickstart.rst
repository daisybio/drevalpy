Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Run a small LCO experiment on TOYv1 with naive predictors:

.. code-block:: bash

   drevalpy \
       --run_id my_first_run \
       --models NaiveTissueMeanPredictor NaiveDrugMeanPredictor \
       --baselines NaiveMeanEffectsPredictor \
       --dataset_name TOYv1 \
       --test_mode LCO

This trains the models to predict LN_IC50 on the TOYv1 subset of CTRPv2,
using leave-cell-line-out splits (LCO; see :doc:`/concepts/evaluation`) and
the default seven-fold CV. Results land
under:

.. code-block:: bash

   results/my_first_run/TOYv1/LCO

Build the HTML report:

.. code-block:: bash

   drevalpy report --run_id my_first_run --dataset_name TOYv1

Open ``index.html`` in that run’s results folder in your browser.

For large or highly reproducible runs, prefer the Nextflow pipeline
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_
(`GitHub <https://github.com/nf-core/drugresponseeval.git>`_). No Nextflow
knowledge is required to use it.

Next steps: :doc:`experiment` for more options, :doc:`reporting` for the
report command, and :doc:`/concepts/datasets` / :doc:`/concepts/evaluation`
for datasets and evaluation
settings.
