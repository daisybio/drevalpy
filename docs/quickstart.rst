Quickstart
----------

Make sure you have installed DrEvalPy and its dependencies (see `Installation <./installation.html>`_).

To make sure the pipeline runs, you can use the fast models NaiveMeanEffectsPredictor and NaiveDrugMeanPredictor on the TOYv1 (subset of CTRPv2) or TOYv2 (subset of GDSC2)
dataset with the LCO test mode.

.. code-block:: bash

    drevalpy --run_id my_first_run --models NaiveTissueMeanPredictor NaiveDrugMeanPredictor --baselines NaiveMeanEffectsPredictor --dataset TOYv1 --test_mode LCO

This will train the three baseline models to predict LN_IC50 values of our Toy dataset which is a subset of CTRPv2.
It will evaluate in "LCO" which is the leave-cell-line-out splitting strategy
(leave random cell lines out for testing) using 7 fold cross validation.
The results will be stored in

.. code-block:: bash

    results/my_first_run/TOYv1/LCO

You can visualize them using

.. code-block:: bash

    drevalpy-report --run_id my_first_run --dataset TOYv1

This creates an index.html file which you can open in your browser to see the results of your run.

We recommend the use of our nextflow pipeline for computational demanding runs and for improved reproducibility. No
knowledge of nextflow is required to run it. The nextflow pipeline is available on the `nf-core GitHub
<https://github.com/nf-core/drugresponseeval.git>`_, the documentation can be found `here <https://nf-co.re/drugresponseeval/dev/>`_.

-  Want to test if your own model outperforms the baselines? See `Run Your Model <./runyourmodel.html>`_.
-  Discuss usage, development and issues on `GitHub <https://github.com/daisybio/drevalpy>`_.
-  Check the `Contributor Guide <./contributing.html>`_ if you want to participate in developing.
-  If you use drevalpy for your work, `please cite us <./reference.html>`_.

..
  -  Check our `tutorial notebook <https://github.com/daisybio/drevalpy/blob/development/tutorials/DrEvalPy%20Tutorial.ipynb>`_, the `usage principles <./usage.html>`_ or the `API <./API.html>`_.
