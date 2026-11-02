Run the curve fits of a large screen as parallel jobs
=====================================================

.. interface: both

``drevalpy curate run`` and :func:`drevalpy.curation.curate` split the curves into chunks of up
to ``--max-chunk-size`` curves (``max_chunk_size``, default 2000) and fit the chunks in
``--cores`` (``max_workers``) processes on one machine. A screen too large for one machine can be
fitted as separate jobs instead, in three steps:

``drevalpy curate plan``
   Reads the well table and writes the chunks to fit.
``drevalpy curate fit-chunk``
   Fits one chunk, on one core. Each chunk is a job of its own.
``drevalpy curate collect``
   Assembles the fitted chunks into the ``.h5ad`` file.

The result equals that of ``drevalpy curate run`` with the same options. To run the steps on a
cluster, use the Nextflow pipeline described at the end of this page. The table the steps read is
described in :doc:`bring_your_own_data`.

Run the steps by hand
---------------------

Write a small well table, and a directory for the fitted chunks:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: python

         import numpy as np
         import pandas as pd

         rng = np.random.default_rng(0)
         rows = [
             (f"S{sample}", drug, conc, 1 / (1 + conc / base) + rng.normal(0, 0.03))
             for sample in range(3)
             for drug, base in [("drugA", 0.5), ("drugB", 5.0)]
             for conc in 10.0 ** np.arange(-3, 4)
         ]
         wells = pd.DataFrame(rows, columns=["sample", "drug", "concentration", "intensity"])
         wells.to_csv("wells.csv", index=False)

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import numpy as np
         import pandas as pd

         rng = np.random.default_rng(0)
         rows = [
             (f"S{sample}", drug, conc, 1 / (1 + conc / base) + rng.normal(0, 0.03))
             for sample in range(3)
             for drug, base in [("drugA", 0.5), ("drugB", 5.0)]
             for conc in 10.0 ** np.arange(-3, 4)
         ]
         wells = pd.DataFrame(rows, columns=["sample", "drug", "concentration", "intensity"])

Then plan, fit and collect:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy curate plan wells.csv plan
         drevalpy curate fit-chunk plan/chunks/pooled-00000.parquet fitted/pooled-00000.parquet
         drevalpy curate collect plan fitted curves.h5ad

      ``plan`` takes the fit options of ``curate run`` (``--replicate-fits``, ``--min-doses``,
      ``--metadata``, ``--max-chunk-size``, ``--fit-speed``, ``--normalize``); ``fit-chunk``
      and ``collect`` read them from the plan. ``curves.h5ad`` is the file that ``drevalpy
      curate run wells.csv curves.h5ad`` writes.

   .. tab-item:: Python
      :sync: python

      The steps are commands; in Python, ``curate`` runs all three in local processes:

      .. code-block:: python

         from drevalpy.curation import curate

         curves = curate(wells, max_chunk_size=2000, max_workers=4)
         print(curves.shape)

      .. code-block:: text

         (3, 2)

      To fit the chunks as separate jobs, save the table with ``wells.to_parquet(...)`` and run
      the commands of the CLI tab.

``plan`` writes the directory ``plan/``:

``plan.json``
   The fit options and the names of the chunks, under ``"chunks"``.
``series.parquet``
   The dose series of every pair, from which ``collect`` builds the replicate-fits table.
``chunks/<name>.parquet``
   One file per chunk, such as ``pooled-00000.parquet``. With ``--replicate-fits`` there are
   ``replicate-*`` chunks as well.

The tiny table above makes a single chunk. For your own screen, run one ``fit-chunk`` job per
file in ``plan/chunks/``, each writing ``<name>.parquet`` into one directory, and run
``collect`` when all of them have finished; it stops with an error that names the chunks whose
fit is missing. The jobs can run anywhere that has the same version of DrEvalPy installed.

A chunk file holds everything its fit needs: its curves, a description of their dose range and
the fit options. Two equal chunks are byte-equal files, so a workflow manager that caches tasks
by the content of their inputs, such as Snakemake or Nextflow with ``-resume``, refits only the
chunks that changed when you add wells or rerun the plan. A new fit option or
``--max-chunk-size`` changes every chunk.

Run the steps on a cluster with Nextflow
----------------------------------------

The Nextflow pipeline `drevalpy/curate <https://github.com/drevalpy/curate>`_ runs the three steps
for you. Nextflow's executors submit the ``fit-chunk`` jobs to SLURM, AWS Batch or another
scheduler, its ``errorStrategy`` retries a job that ran out of memory with more memory, and
``-resume`` refits only the chunks whose content changed.

The pipeline fits one screen per run. ``--input`` is the screen's well table, in Parquet or CSV.
Pick the scheduler in a Nextflow config of your own, for example ``slurm.config``:

.. code-block:: groovy

   process.executor = 'slurm'
   process.queue    = 'my-partition'

Pass it with ``-c``, together with the container profile:

.. code-block:: console

   $ nextflow run drevalpy/curate -r <commit> -profile apptainer -c slurm.config \
       --input data/my_screen_wells.parquet --outdir results --replicate_fits

It writes ``results/curves.h5ad``. The fit options are the parameters ``--replicate_fits``,
``--min_doses``, ``--metadata`` (comma-separated), ``--max_chunk_size``, ``--fit_speed`` and
``--normalize``, with the meaning of the ``curate run`` options. To fit several screens, run the
pipeline once per screen. The profiles and the other settings are documented in the pipeline's
repository.
