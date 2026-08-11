Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Run a small LCO experiment on TOYv1 with ElasticNet:

.. code-block:: bash

   drevalpy run ElasticNet --dataset TOYv1 --split-mode LCO --no-hpo

This trains ElasticNet to predict drug response on the TOYv1 subset of CTRPv2,
using leave-cell-line-out splits (LCO; see :doc:`/concepts/evaluation`).
Results are written under the output directory (default ``results/``).

Build the HTML report:

.. code-block:: bash

   drevalpy report results/ --output-dir report

Open ``report/multiqc_report.html`` in your browser.

For large or highly reproducible runs, prefer the Nextflow pipeline
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_
(`GitHub <https://github.com/nf-core/drugresponseeval.git>`_). No Nextflow
knowledge is required to use it.

Next steps: :doc:`experiments` for more options, :doc:`visualization` for the
report command, and :doc:`/concepts/datasets` / :doc:`/concepts/evaluation`
for datasets and evaluation settings.
