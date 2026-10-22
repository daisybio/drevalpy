# Literature reference audit

This directory contains source independent comparison runners. They compare narrow
numerical kernels or serialized published artifacts; they do not claim full
architecture equivalence or reproduce upstream training pipelines.

The frozen provenance is pinned in [`manifest.json`](manifest.json), and the
per-model source comparison is in
[`docs/project/literature_audit/differences.md`](../../docs/project/literature_audit/differences.md).
The v2 kernel runner covers MOLIR, SuperFELTR, DIPK autoencoder modules,
GraphDRP's shared GCN stack, PharmaFormer, and SparseGO's sparse layer. SRMF
and Precily have separate authoritative runners because SRMF is MATLAB/Octave
and Precily uses an isolated TensorFlow/Keras environment.

Set `DREVALPY_ROOT` to the repository root and `UPSTREAM_REFERENCE_ROOT` to
checkouts at the commits in the manifest. Use its directory names (`GraphDRP`,
`Precily`, `MultiOmics`, `MOLI`, `SuperFELT`, `PharmaFormer`, `DIPK`, `SparseGO`),
and extract the SRMF archive into `SRMF-extracted/SRMF`. The runner writes
to `NUMERICAL_RESULTS_PATH` and does not copy upstream implementation source.

```bash
DREVALPY_ROOT="$PWD" \
UPSTREAM_REFERENCE_ROOT=/path/to/upstream \
NUMERICAL_RESULTS_PATH=/tmp/numerical-results.json \
python tools/literature_reference/numerical_compare_v2.py
```

Run `expanded_reference.py` with the same source root and
`EXPANDED_RESULTS_PATH=/tmp/expanded-reference.json`; run `fusion_reference.py`
with `FUSION_RESULTS_PATH=/tmp/fusion-reference.json`. These runners compile only
the selected source definitions, avoiding upstream training-script entry points.
They execute real source tensor operations, gradients and optimizer updates.

The normal audit environment used the Python and torch versions recorded in
`manifest.json`. The Precily
environment adds TensorFlow 2.21.0 and Keras 3.15.1 in a disposable virtualenv.
It needs access to the repository
and its PyTorch dependencies as well as the isolated TensorFlow packages. The
executed setup used Python 3.13 with the isolated environment's site-packages
first, followed by this repository's `.venv/lib/python3.13/site-packages`, on
`PYTHONPATH`. Do not add TensorFlow or Keras to the library's dependency groups.

Precily requires its isolated TensorFlow/Keras environment and a published
Model1 artifact. Set `PRECILY_MODEL1` and `PRECILY_RESULTS_PATH` before running
`compare_tf.py`. SRMF requires Octave for `check.m`; `compare.py` consumes the
resulting `reference.mat`, with paths supplied by `SRMF_REFERENCE_MAT` and
`SRMF_RESULTS_PATH`. The checked-in `srmf-reference.mat` is the small frozen audit output, and
`octave-version.txt` records the Octave version that produced it;
no upstream source is included here.

The Octave run used version 9.4.0 in a disposable Debian container. Set
`SRMF_OCTAVE_REFERENCE_ROOT` to the extracted source directory and
`SRMF_OCTAVE_OUTPUT_ROOT` to an existing output directory, then run
`octave --no-gui --quiet tools/literature_reference/check.m`. This writes both
`reference.mat` and `version.txt`; provide the latter as `SRMF_OCTAVE_VERSION`
when running `compare.py`.

`tools/literature_training_benchmark.py --output /tmp/training.json` exercises
all eight local predictors at seeds 0, 1 and 2. Add `--extended` for multi-batch
training traces. To regenerate starting-v2 evidence, execute the same harness
with the archived starting commit on `PYTHONPATH`, using that archive as the
working directory. Keep baseline and corrected outputs in separate files.

`capture_v2_baseline.py 0` replays the original low-level fixture sequence.
Set `DREVALPY_BASELINE_ROOT` to the archived starting checkout and
`BASELINE_OUTPUT_ROOT` to a fresh output directory, then repeat with seeds 1
and 2. The packaged runner was verified to reproduce all three frozen payloads.
