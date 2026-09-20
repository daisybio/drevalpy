# Architecture v2 baseline

The original before-change gates were captured from the clean starting commit
`e4ad11766bc53554eb0dd53fd29ba47fa41000c2`, archived at
`/private/tmp/drevalpy-head-e4ad117` before the architecture work began. Each
runner executed with that archive first on `PYTHONPATH`; no working-tree library
code was imported. Supplementary full-facade and isolated Precily comparisons
were subsequently executed against that unchanged archive and the final tree.
They supplement the frozen fixtures; they do not replace their expected values.

See [validation.md](validation.md) for the eight-model comparison matrix, gate
results and evidence boundaries. The [architecture guide](../v2_architecture.rst)
traces ownership and execution; the [migration guide](../v2_architecture_migration.rst)
covers the public API, extensions, checkpoints and experiments that need rerunning.

`baseline-v2-fixtures/` is the low-level eight-operation fixture sequence for
seeds 0, 1 and 2. `training/literature-training-extended.json` contains all
eight literature predictors for each seed using the four-epoch multi-batch
protocol. `facade-baseline.json` freezes `NaivePredictor` and the
scaled-gene-expression/fingerprint `ElasticNet` facade paths on the synthetic
LCO fixture for the same three seeds. It records prediction and component-state
checksums so later checks can distinguish a numerical regression from a change
to serialization.

The source-comparison runners all passed: `numerical-kernels.json` covers the
MOLIR and SuperFELTR encoders, DIPK autoencoder, GraphDRP GCN stack,
PharmaFormer and SparseGO sparse layer; `expanded-reference.json` covers the
full DIPK attention/head and SparseGO ontology networks; and
`fusion-reference.json` covers the adapted MOLIR and SuperFELTR compositions.
`srmf-reference.json` passed against the frozen result produced by Octave 9.4.0.
All float32 checks use `rtol=1e-5, atol=1e-6`; SRMF's float64 checks use
`rtol=1e-7, atol=1e-9`.

The pinned upstream source trees are available at
`/private/tmp/drevalpy-upstream` and match every manifest commit. The published
Precily Fig1/Fig1d `Model1.hdf5` artifact has SHA-256
`bef59c630f22dabe4f11ccabdd90083d88dc371229ee2f61f8bd0f63639cd7e1`.
The initial environment lacked TensorFlow and Keras. An isolated runtime was
subsequently installed and the archived starting code passed the published-weight
comparison (`precily-reference-baseline.json`), without regenerating the existing
fixtures. Its versions and command are recorded separately in the manifest.
The SRMF baseline uses frozen independent Octave 9.4.0 outputs; it does not claim
that the Python baseline command reruns Octave.

To recreate the available source gates from the archive, use the commands
recorded in `manifest.json`. The real full DIPK and SparseGO checks are
`expanded_reference.py`, not the smaller kernel runner. Their pinned upstream
sources and required files are available; the command completed with exact zero
forward, gradient and update errors.
