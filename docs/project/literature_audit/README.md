# Literature model provenance audit

This audit records upstream commits, source hashes, runtime inventories, cached
artifact hashes, and bounded numerical evidence for the eight literature models.
The immutable v2 training baseline is anchored to commit
`cf5eeb6a0e020ab7e98c9829c0d81dc00a8cdd5c`; `training-current.json` is the
final corrected snapshot.

The numerical runners compare declared scopes only. The original kernel runner
covers MOLIR, SuperFELTR, DIPK autoencoder modules, GraphDRP's shared GCN stack,
PharmaFormer, and SparseGO's sparse layer. `expanded-reference.py` adds full
DIPK attention/head and full SparseGO ontology-network forward, intermediate,
training-forward, loss, gradient, and one-update checks. `fusion-reference.py`
checks adapted MOLIR and SuperFELTR fusion compositions, including MOLIR's
`gamma=.5` loss weighting. These are source-composition regression checks and do
not claim unchanged upstream classifier or preprocessing pipelines.

Precily uses the isolated TensorFlow/Keras runner against the published Model1
artifact. SRMF uses the Octave/MATLAB runner and frozen `.mat` output. Runtime
versions and exact source/artifact hashes are in `manifest.json`; results retain
explicit scopes and do not fabricate blocked full-model parity.

Runners accept `UPSTREAM_REFERENCE_ROOT` and result-path environment variables.
See `tools/literature_reference/README.md` for commands and isolated-runtime
requirements.

`differences.md` records the per-model source comparison and classifies each
retained adaptation and corrected defect. The seeded training snapshots compare
local v2 versions; upstream runners use the declared synchronized-weight scopes.
