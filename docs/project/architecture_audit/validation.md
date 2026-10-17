# Architecture migration validation

The reference implementation is commit
`e4ad11766bc53554eb0dd53fd29ba47fa41000c2` (the corrected v2 that descends from the
correctness review's starting commit `cf5eeb6a`; see [README](README.md)). The migration preserves its corrected,
leakage-free training protocol and literature adaptations. No unexplained
numerical differences were observed in the comparisons below. This is a bounded
regression audit, not a reproduction of published benchmark scores.

## Eight-model comparison

Every row passed both the independent source comparison in its stated scope and
the starting-v2 regression checks. The latter cover seeds **0, 1 and 2** for
extended predictor training and complete public-pipeline training/prediction.

| Model        | Independent pinned reference scope                                                                                                                 | Extended training vs starting v2                            | Public pipeline vs starting v2                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------- |
| DrugGNN      | GraphDRP's three GCN layers: forward, loss, input gradient and update; pooling, cell branch and head remain documented adaptations                 | Exact predictions, metrics and validation/checkpoint traces | Exact numeric features and predictions; save/load error 0                         |
| Precily      | Published Model1 weights mapped from Keras: forward, loss, gradients and update in isolated TensorFlow float64                                     | Exact predictions, metrics and training trace               | Exact pathway/drug matrices and predictions; save/load error 0                    |
| SRMF         | Independent Octave loss, alternating updates and five-step best-factor output with matched initialization                                          | Exact predictions and metrics over eight iterations         | Exact similarities' input features and predictions; save/load error 0             |
| MOLIR        | MultiOmics encoders and original MOLI fusion/head operations composed into the documented regression adaptation; triplet/MSE, gradients and update | Exact predictions, metrics and selected-checkpoint trace    | Exact ordered omics and predictions; save/load error 0                            |
| SuperFELTR   | Super.FELT encoders, triplet gradients and frozen-encoder regression head; forward, loss and update                                                | Exact predictions, metrics and staged checkpoint traces     | Exact ordered omics and predictions; save/load error 0                            |
| PharmaFormer | Full pretraining-variant attention network: forward, loss, gradients and update                                                                    | Exact predictions, metrics and training/stopping traces     | Exact expression/BPE matrices and predictions; save/load error 0                  |
| DIPK         | Autoencoder plus full attention/head, including ragged-input adaptation, intermediates, train/eval output, gradients and update                    | Exact predictions, metrics and training/stopping traces     | Real artifact pipeline; exact numeric features and predictions; save/load error 0 |
| SparseGO     | Actual ontology helpers and complete sparse network: wiring, intermediates, train/eval output, gradients and update                                | Exact predictions, metrics and training trace               | Real ontology path; exact numeric features and predictions; save/load error 0     |

The extended benchmark uses four epochs with multiple batches, triplet sampling
where applicable, and patience one; it exercises actual stopping and best-weight
selection. SRMF uses eight iterations. The supplementary full-pipeline benchmark
uses one training epoch (two SRMF iterations) to exercise preprocessing, public
calls and complete checkpoint round-trips. These complementary runs should not
be mistaken for a single long upstream training benchmark.

All 24 full-pipeline prediction checksums match exactly. Entity identifiers,
block names, feature schemas and numeric block/matrix bytes are recorded in
`facade-reference-{baseline,postmigration}.json`; comparison results are in
`facade-reference-comparison.json`. Graph/ragged object payloads have shapes and
identities recorded, but no numeric byte checksum; their behavior is exercised
by the full predictor and artifact integration checks. The explicit projection
tuple is new metadata, so it is not expected to equal the former absent field.
Torch RNG state is unchanged by construction in all 24 cases, and seeded
training remains exact despite removing redundant component construction.

Source revisions, source/artifact hashes and the original adaptation review are
pinned in the [literature audit](../literature_audit/manifest.json) and its
[specific findings](../literature_audit/differences.md). Shared neural operations
use mapped weights and a common SGD update; this does not assert identical
upstream initialization streams or native optimizers across different recipes.
Upstream input preprocessing and classification-to-regression differences are
intentional where listed in that review. This architecture change introduces no
additional model-specific numerical correction.

Float32 tolerances were fixed at `rtol=1e-5, atol=1e-6`; float64 at
`rtol=1e-7, atol=1e-9`. Precily uses an isolated TensorFlow/Keras environment and
the published artifact hash. SRMF reuses the checksum-pinned independent Octave
9.4.0 MAT output; no live Octave executable was available for a fresh rerun.
Neither reference runtime was added to the library's normal dependencies.

## Repository and integration checks

| Gate                                                        | Result                                                            |
| ----------------------------------------------------------- | ----------------------------------------------------------------- |
| Serial full non-network suite, with coverage                | **4,561 passed, 8 skipped, 11 deselected; 95.32% coverage**       |
| Coverage gate                                               | 291 modules pass; 60% module floor, two existing exemptions       |
| Size gate                                                   | 292 modules pass; 150-statement ceiling, nine exemptions          |
| All repository hooks                                        | Pass, including fast tests, Ruff, formatting, complexity and size |
| Static typing                                               | `ty check` passes                                                 |
| Strict documentation build                                  | `sphinx-build -W` passes                                          |
| Exact public documentation snippets                         | Three executable workflow tests pass                              |
| Real DIPK raw-input train/predict/v3 save-load, seeds 0/1/2 | Three integration tests pass                                      |
| Cached real MolGNet/BIONIC artifact tests                   | Three tests pass                                                  |
| Plotting subprocess and offline wheel-build regressions     | 196 passed, one deselected                                        |

The aggregate coverage floor remains 89%. No policy floor was lowered; obsolete
exemptions were removed as their modules disappeared or became compliant.
Network tests are excluded from the full non-network command and the artifact
tests above were run separately. The eight skips are reported as skips, not
successful test execution.

The plotting/browser subprocess and offline build initially failed under the
restricted execution sandbox. Both pass unchanged with normal local subprocess
and cache access. No product workaround was introduced to hide those failures.

Commands used for the final gates:

```bash
MPLCONFIGDIR="$(mktemp -d)" uv run pytest -q -m 'not network' --cov --cov-report=json --cov-report=term-missing
uv run python tools/coverage_gate.py
uv run python tools/size_gate.py
uv run prek run --all-files
uv run ty check
MPLCONFIGDIR="$(mktemp -d)" uv run --group docs sphinx-build -W docs "$(mktemp -d)"
uv run pytest -q tests/models/test_global_models.py::TestDIPKArtifactIntegration -m network
uv run pytest -q tests/plugin/test_init.py tests/visualization/plots/test_comparison_scatter.py tests/visualization/plots/test_cross_study_table.py
```

Regression tests explicitly exercise entered-call invalidation versus Python
binding failures, one component construction per fit, custom-transform cloning,
original response units, target fitting before feature filtering, fresh retraining
after load, empty schemas and predictions, single feature computation, read-only
captured scopes, E/V separation, NaN alignment, one-pair metrics, diagnostics,
normalization, seed round-trips and independent dataset subsets. Executable user
and plugin examples use the documented public imports.

## Intended changes and migration

Direct model predictions now return original response units and persist their
fitted transform. Checkpoint format 3 deliberately rejects earlier archives.
One-pair MSE, RMSE and MAE are now defined; R² and correlations remain NaN.
These are intended API/evaluation changes, not numerical regressions to mask
with looser tolerances. Experiment/HPO results already in original units must
remain equivalent under the same valid scopes and configuration.

The [migration guide](../v2_architecture_migration.rst) identifies callers,
plugins, checkpoints and experiments to update. The
[architecture guide](../v2_architecture.rst) explains each owner and traces a fit
and prediction. Existing leakage-free results do not need retraining solely
because internals were reorganized; old checkpoints, transformed-unit consumers,
one-pair metric summaries and results from the former leaking protocol do need
the updates specified there.
