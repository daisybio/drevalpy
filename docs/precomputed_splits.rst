Run on precomputed splits
=========================

The splits used for the leaderboard are available on `Zenodo <https://doi.org/10.5281/zenodo.12633909>`_ as ``splits.zip``.
After unzipping, you get a folder ``splits`` with one set of CSV files per cross-validation split:

.. code-block:: text

    splits/cv_split_0_train.csv
    splits/cv_split_0_validation.csv
    splits/cv_split_0_validation_es.csv
    splits/cv_split_0_early_stopping.csv
    splits/cv_split_0_test.csv
    splits/cv_split_1_train.csv
    ...

``train``, ``validation`` and ``test`` are required. ``validation_es`` and ``early_stopping`` are used by models with early stopping.
The number of CV splits is read from the files.

Standalone (``drevalpy``)
-------------------------

``drevalpy`` loads existing splits instead of creating new ones if the folder
``results/<run_id>/<dataset_name>/<test_mode>/splits`` already exists. So you only need to copy the splits there before you start your run:

.. code-block:: bash

    mkdir -p results/my_model/CTRPv2/LCO
    cp -r splits results/my_model/CTRPv2/LCO/splits

    drevalpy --run_id my_model --models YourModel --dataset_name CTRPv2 --test_mode LCO

Make sure that

* ``--run_id``, ``--dataset_name`` and ``--test_mode`` match the folder names above,
* ``--overwrite`` is **not** set (it deletes the result folder, including your splits),
* the splits belong to the test mode you choose, e.g., the LCO splits for ``--test_mode LCO``.

Nextflow pipeline
-----------------

With the `nf-core/drugresponseeval <https://github.com/nf-core/drugresponseeval>`_ pipeline, pass the script
``assets/custom_splitter_from_csvs.py`` via ``--custom_splitter_path``. The example runs from within a clone of the pipeline repository
(hence ``nextflow run .``); ``-profile`` and the cluster config ``-c`` depend on your infrastructure:

.. code-block:: bash

    nextflow run . --run_id new_leaderboard -profile gpu -c mycluster.config \
        --custom_splitter_path assets/custom_splitter_from_csvs.py \
        --baselines NaiveMeanEffectsPredictor \
        --models MyDLModel1,MyDLModel2

The path to the unzipped splits is set **inside this Python file**, so edit it before the run.
The script is part of the pipeline repository.

Your own splitting strategy
---------------------------

``--custom_splitter_path`` is also available in standalone ``drevalpy`` and is not limited to precomputed CSVs.
Point it to any Python file that defines a module-level function ``create_splits(response_data, params)``.
It must return a list with one dictionary per split. Each dictionary maps the roles ``train``, ``validation``, ``test``
(required) and ``validation_es``, ``early_stopping`` (optional) to a ``DrugResponseDataset``.
``params`` contains ``test_mode``, ``n_cv_splits``, ``validation_ratio``, ``random_state`` and ``split_early_stopping``.
The splits are validated, e.g., train/validation/test must not share cell lines in LCO mode.

.. code-block:: bash

    drevalpy --run_id my_model --models YourModel --dataset_name CTRPv2 --test_mode LCO \
        --custom_splitter_path my_splitter.py --custom_split_name my_split

``--custom_split_name`` is the optional name of the result folder (default: the test mode).

Next: :doc:`leaderboard`.
