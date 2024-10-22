Jobs
====

To run models from the catalog, you can run:

.. code-block:: bash

    python run_suite.py --run_id my_first_run --models ElasticNet SimpleNeuralNetwork --dataset GDSC1 --test_mode LCO

This will train and tune a neural network and an elastic net model on a subset of gene expression features and drug fingerprint features to predict IC50 values of the GDSC1 database. It will evaluate in "LCO" which is the leave-cell-line-out splitting strategy using 5 fold cross validation. 
The results will be stored in 

.. code-block:: bash

    results/my_first_run/LCO

You can visualize them using 

.. code-block:: bash

    python create_report.py --run_id my_first_run

This will create an index.html file which you can open in your webbrowser.

You can also run a drug response experiment using Python:

.. code-block:: python

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

We recommend the use of our nextflow pipeline for computational demanding runs and for improved reproducibility. No knowledge of nextflow is required to run it. The nextflow pipeline is available here: `nf-core-drugresponseeval <https://github.com/JudithBernett/nf-core-drugresponseeval>`_.