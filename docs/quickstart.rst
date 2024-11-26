Quickstart
----------

Make sure you have installed DrEvalPy and its dependencies (see `Installation <./installation.html>`_).

To make sure the pipeline runs, you can use the fast models NaiveDrugMeanPredictor and NaivePredictor on the Toy_Data
dataset with the LPO test mode.

.. code-block:: bash

    python run_suite.py --run_id my_first_run --models NaiveDrugMeanPredictor --baselines NaivePredictor --dataset Toy_Data --test_mode LPO

This will train the two baseline models on a subset of gene expression features and drug fingerprint features to
predict IC50 values of the GDSC1 database. It will evaluate in "LPO" which is the leave-pairs-out splitting strategy
(leave random pairs of drugs and cell lines out for testing) using 5 fold cross validation.
The results will be stored in

.. code-block:: bash

    results/my_first_run/LPO

You can visualize them using

.. code-block:: bash

    python create_report.py --run_id my_first_run


We recommend the use of our nextflow pipeline for computational demanding runs and for improved reproducibility. No
knowledge of nextflow is required to run it. The nextflow pipeline is available on the `nf-core GitHub
<https://github.com/nf-core/drugresponseeval.git>`_, the documentation can be found `here <https://nf-co.re/drugresponseeval/dev/>`_.

-  Discuss usage, development and issues on `GitHub <https://github.com/daisybio/drevalpy>`_.
-  Check the `Contributor Guide <./contributing.html>`_ if you want to participate in developing.

..
  -  Check our `tutorial notebook <https://github.com/daisybio/drevalpy/blob/development/tutorials/DrEvalPy%20Tutorial.ipynb>`_, the `usage principles <./usage.html>`_ or the `API <./API.html>`_.
  -  Consider citing `DrEvalPy <...>`_ along with original `references <./reference.html>`_.