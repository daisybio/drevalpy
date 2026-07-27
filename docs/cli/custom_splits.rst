Custom splits
=============

Use an external split script when the built-in LPO/LCO/LTO/LDO CV generators
are not enough. Point ``drevalpy`` at a Python file that defines
``create_splits``, and optionally label the result directory.

Required entry point
--------------------

Your script must expose:

.. code-block:: text

   create_splits(response_data, params)

``response_data`` is the full response dataset; ``params`` carries split
settings (seed, validation ratio, fold count, ``test_mode``, and related
fields). Return a list of split dicts with ``train``, ``validation``, and
``test`` roles (and optional ``metadata``).

``--test_mode`` still matters: DrEvalPy uses it to validate overlap rules for
the protocol you claim (for example cell-line disjointness under ``LCO``). It
cannot verify that your split answers your scientific question — that is on
you.

A worked example lives at ``examples/custom_split_lco_fraction.py`` in the
repository (LCO-style train/validation/test fractions).

CLI flags
---------

- ``--custom_splitter_path`` — path to the script defining ``create_splits``
- ``--custom_split_name`` — optional label for the results subdirectory
  (defaults to ``test_mode``)

Example
-------

.. code-block:: bash

   drevalpy \
       --run_id custom_lco_fraction \
       --models NaiveDrugMeanPredictor \
       --baselines NaiveMeanEffectsPredictor \
       --dataset_name TOYv1 \
       --test_mode LCO \
       --custom_splitter_path examples/custom_split_lco_fraction.py \
       --custom_split_name LCO_fraction_0.2

Results appear under
``results/custom_lco_fraction/TOYv1/LCO_fraction_0.2/`` (or under ``LCO`` if
you omit ``--custom_split_name``).

The same ``--custom_splitter_path`` flag is available on
``drevalpy make-cv-pkls`` for nf-core-style stepwise runs — see
:doc:`pipeline_commands`.
