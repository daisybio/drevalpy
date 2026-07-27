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
6. ``drevalpy train-cv`` — train and predict on CV folds for one hyperparameter
   file
7. ``drevalpy evaluate-hpams`` — pick the best hyperparameters from CV
8. ``drevalpy test-cv`` — train on train+val and evaluate on the test fold
9. ``drevalpy make-randomization-yamls`` — prepare randomization configs
10. ``drevalpy consolidate-single-drug`` — consolidate single-drug model outputs
11. ``drevalpy evaluate-test`` — aggregate test metrics
12. ``drevalpy collect-results`` — collect result files for reporting
13. ``drevalpy report`` — HTML report for a local-style run layout
14. ``drevalpy make-pipeline-report`` — HTML report for nf-core result layouts

Optional final-model path after CV:

- ``drevalpy make-final-split-pkls`` — pickles for a final full-data split
- ``drevalpy tune-final-model`` — tune on the final split
- ``drevalpy train-final-model`` — train the final model

``make-hpam-yamls``
-------------------

``drevalpy make-hpam-yamls --model_name <Model>`` writes only
``hpam_0.yaml`` containing that model’s default hyperparameters. It does
**not** expand a search grid. Ray/Optuna tuning happens at experiment time
(root ``drevalpy`` or the tuning steps above), not inside this command.

Example:

.. code-block:: bash

   drevalpy make-hpam-yamls --model_name ElasticNet

When to use the root command instead
------------------------------------

For interactive local runs, prefer the root ``drevalpy`` command
(:doc:`experiment`). Use these subcommands when you need the same steps the
Nextflow pipeline runs, or when you are debugging a single stage.

Backward compatibility
----------------------

Grid YAML era
~~~~~~~~~~~~~

Before 1.6.0, ``make-hpam-yamls`` could expand hyperparameter grids into many
``hpam_*.yaml`` files. That behavior is no longer supported.
``make-hpam-yamls`` always emits a single defaults file (``hpam_0.yaml``).

``--hyperparameter_tuning`` on ``make-hpam-yamls``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``--hyperparameter_tuning`` on ``make-hpam-yamls`` suggested grid
expansion. The flag remains for nf-core compatibility, but is deprecated and
may be removed in a future release. It does not restore grid expansion;
tuning is controlled at experiment time via ``--hpo_num_samples`` /
``--no_hyperparameter_tuning`` on the root command (see
:doc:`hyperparameter_tuning`).

Legacy script names
~~~~~~~~~~~~~~~~~~~

Before 1.6.0, console scripts such as ``drevalpy-make-hpam-yamls``,
``drevalpy-train-cv``, and ``drevalpy-report`` were the usual entry points.
They remain available for backward compatibility, but are deprecated and may
be removed in a future release. Prefer ``drevalpy <subcommand>``.
