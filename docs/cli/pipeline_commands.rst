Pipeline commands
=================

Besides the root ``drevalpy`` experiment, the CLI exposes stepwise commands
that `nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_
orchestrates. You can call them yourself for debugging or custom workflows.

For generated help and options per command, see :doc:`reference`.

Command list
------------

Typical CV evaluation path:

1. ``drevalpy viability-preprocess`` — prepare raw viability inputs
2. ``drevalpy viability-postprocess`` — postprocess fitted viability
3. ``drevalpy load-response`` — load response data for a dataset
4. ``drevalpy make-cv-pkls`` — build CV split pickles (supports
   ``--custom_splitter_path``)
5. ``drevalpy make-hpam-yamls`` — write default hyperparameters for a model
   (no search grid)
6. ``drevalpy train-cv`` — train and predict on CV folds for one hyperparameter
   file
7. ``drevalpy evaluate-hpams`` — select among YAML prediction artifacts
8. ``drevalpy test-cv`` — train on train+val and evaluate on the test fold
9. ``drevalpy make-randomization-yamls`` — prepare randomization configs
10. ``drevalpy consolidate-single-drug`` — consolidate single-drug model outputs
    (requires ``--dataset_name``; layout is
    ``outdir/run_id/dataset_name/test_mode``)
11. ``drevalpy evaluate-test`` — aggregate test metrics
12. ``drevalpy collect-results`` — collect result files for reporting
13. ``drevalpy report`` — HTML report for a local-style run layout
14. ``drevalpy make-pipeline-report`` — HTML report for nf-core result layouts

Optional final-model path after CV:

- ``drevalpy make-final-split-pkls`` — pickles for a final full-data split
- ``drevalpy tune-final-model`` — score one hyperparameter YAML on the final
  validation split (does **not** run Ray/Optuna; prefer
  ``train_final_model`` / root ``drevalpy`` for real tuning)
- ``drevalpy train-final-model`` — train the final model from a selected YAML

``make-hpam-yamls``
-------------------

``drevalpy make-hpam-yamls --model_name <Model>`` writes only
``hpam_0.yaml`` containing that model’s default hyperparameters. It does
**not** expand a search grid. Ray/Optuna tuning happens at experiment time
(root ``drevalpy`` via ``hpam_tune``), not inside this command.

Example:

.. code-block:: bash

   drevalpy make-hpam-yamls --model_name ElasticNet

When to use the root command instead
------------------------------------

For interactive local runs, prefer the root ``drevalpy`` command
(:doc:`experiment`). Use these subcommands when you need the same steps the
Nextflow pipeline runs, or when you are debugging a single stage.
