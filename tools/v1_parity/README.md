# 1.6.1 parity harness

Offline harness that runs the drevalpy **1.6.1** implementation (the unreleased head of
`origin/development`, pinned at commit `f5ff3c70`, 1.x layout `drevalpy/...`) and the **v2**
implementation on identical, seeded inputs, in separate processes and virtualenvs, and compares
the outputs under named tolerance classes. The results are committed as JSON under
`evidence/`. It is not run in CI; `tests/tools/test_v1_parity.py` unit-tests only `compare.py`,
the case format and the case-module interface.

Both packages are called `drevalpy`, so they cannot share an interpreter. Each side is a
subprocess; they exchange numpy arrays through files.

```
tools/v1_parity/
  setup_v161.sh      git worktree of f5ff3c70 + CPU-only venv with 1.6.1's pinned dependencies
  _bridge.py         Case (.npz arrays + .json metadata), outputs IO, seeding, synthetic inputs
  _v1_runner.py      runs in the 1.6.1 venv: builds 1.x objects, calls the case's run_v1
  _v2_runner.py      runs in the v2 venv: builds a v2 Dataset, calls the case's run_v2
  compare.py         tolerance classes, step comparison, verdict, evidence JSON
  run.py             orchestrator: build case -> run both sides -> compare -> write evidence
  models/<name>.py   one case module per model (see "Adding a model")
  evidence/<Name>.json   committed results
```

## Setup

```bash
tools/v1_parity/setup_v161.sh
```

- Creates the worktree `/tmp/opencode/v161` (`git worktree add --detach … f5ff3c70`; reused if it
  exists) and the venv `/tmp/opencode/v161/.venv`. The main working tree is never touched.
- Installs 1.6.1's own pins from its `requirements.txt` (a `poetry export`), minus the CUDA wheels;
  torch comes from the PyTorch CPU index at the pinned version. `drevalpy` is installed editable
  with `--no-deps`. The optional extras (rdkit, xgboost, lightgbm, gseapy, mygene, obonet) are
  installed too; set `V161_EXTRAS=""` to skip them.
- Overrides: `V161_DIR`, `V161_COMMIT`, `V161_PYTHON` (default 3.12; 1.6.1 supports >=3.11,<3.14),
  `TORCH_INDEX`.
- The script `cd`s into the worktree before calling `uv`, so the v2 `pyproject.toml`'s
  `exclude-newer` cutoff does not hide 1.6.1's pinned versions.

Reference environment recorded on 2026-10-23: Python 3.12.13, torch 2.14.1+cpu, numpy 2.5.4,
scikit-learn 1.9.1, scipy 1.18.1, `drevalpy 1.6.1`. The v2 venv (`uv sync`) had Python 3.13.13,
torch 2.13.0, numpy 2.4.6, scikit-learn 1.9.0. Each evidence file records both sets.

## Running

From the repository root, in the v2 environment:

```bash
uv run python -m tools.v1_parity.run --list            # available cases
uv run python -m tools.v1_parity.run ElasticNet        # run + write evidence/ElasticNet.json
uv run python -m tools.v1_parity.run ElasticNet --no-write --workdir /tmp/opencode/pw
```

`--workdir` keeps `case.npz`, `case.json`, `outputs_v1.*` and `outputs_v2.*` for inspection.
Options: `--seed`, `--v1-dir`. The exit code is 1 when a verdict is `mismatch`.

Each side can also be run by hand (repository root on `PYTHONPATH`):

```bash
PYTHONPATH=. /tmp/opencode/v161/.venv/bin/python -m tools.v1_parity._v1_runner --model ElasticNet --case DIR --out DIR
PYTHONPATH=. uv run python -m tools.v1_parity._v2_runner --model ElasticNet --case DIR --out DIR
```

Determinism: `random`/`numpy`/`torch` seeded with the case seed, `torch.use_deterministic_algorithms`,
`torch.set_num_threads(1)`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`.
A case module must also construct modules and iterate data in the same order on both sides.

## Tolerance classes

A class is declared per step in the case module, never loosened afterwards to make a step pass.
Numeric comparison uses `numpy.isclose(a, b, rtol, atol)` with NaNs at equal positions;
non-numeric arrays (gene names, ids) must be equal.

| Class | Rule | Use for |
|---|---|---|
| `exact` | rtol = atol = 1e-12 | gene selection and order, scalers, rank transform, mRMR selection, MolGNet pooling, forests, kNN and linear models with equal seeds, response transformations, the constant check |
| `forward` | rtol = atol = 1e-6 | neural forward passes with the 1.6.1 `state_dict` copied into the v2 network |
| `train` | rtol 1e-4, atol 1e-6 | short seeded CPU training where the RNG consumption order can be matched |
| `statistical` | Pearson >= 0.99 and abs(mean difference) <= 0.1 std | where the RNG order cannot be matched (augmentation draws, DataLoader shuffles) |

Each step has a **scope**:

- `parity`: must pass its declared class. These steps decide the verdict.
- `deviation`: a known, explained difference. Its measured size is recorded together with the
  `explanation`; it never changes the verdict. Intended deviations (plan section 4.4) go here:
  Precily GSVA per fold, globally fitted single-drug featurizers, curve-quality splits.

**Verdict** = `mismatch` if any parity step fails its class, otherwise the loosest declared class
among the parity steps (`exact` when all of them are exact). Every step also records `achieved`,
the strictest class it actually satisfies.

## Evidence schema

`evidence/<Name>.json` (sorted keys, schema version 1):

| Key | Content |
|---|---|
| `model`, `verdict`, `generated`, `schema_version` | harness name, result, ISO date, `1` |
| `case` | `seed`, `inputs_sha256` (seed + metadata + every array's dtype, shape and bytes), `meta` |
| `v1`, `v2` | `commit`, `python`, `numpy`, `torch`, `scikit-learn`, `scipy`, `pandas`, `drevalpy` versions; v2 also `dirty` (uncommitted changes under `src/`) |
| `steps` | parity steps: `name`, `declared`, `achieved`, `passed`, `max_abs_diff`, `max_rel_diff`, `pearson`, `shape_v1`, `shape_v2`, `detail`, `description` |
| `deviations` | same fields plus `explanation`; recorded, not decisive |
| `findings` | free text: differences found while building the case (fixes belong to the owning packages) |

## Adding a model

Create `models/<snake_case_name>.py`. The module is imported by **both** interpreters, so import
only the standard library, numpy and `tools.v1_parity.*` at module scope, and import `drevalpy`
inside `run_v1` / `run_v2`. It must provide:

```python
NAME = "ElasticNet"                       # harness name; evidence/<NAME>.json
FINDINGS: list[str] = [...]               # optional; copied into the evidence

def build_case(seed: int) -> Case: ...    # runs in the v2 venv; numpy only
def steps() -> list[StepSpec]: ...        # what to compare, with class and scope
def run_v1(case: Case) -> dict[str, np.ndarray]: ...   # runs in the 1.6.1 venv
def run_v2(case: Case) -> dict[str, np.ndarray]: ...   # runs in the v2 venv
```

- `build_case` returns `Case(model=NAME, seed=seed, arrays=..., meta=...)`. For the standard sample x
  drug regression inputs call `tabular_case_arrays(seed, n_samples=…, n_genes=…, n_bits=…)`
  (float32-representable values, row-major sorted train/test pairs, test-only samples); add model
  specific arrays (SMILES, graphs, weights, mask) to `arrays`, keeping strings unicode (no object
  arrays). Hyperparameters of both sides and variant lists go in `meta` (JSON only).
- `run_v1` / `run_v2` read the case and return named arrays. Use `/`-separated names such as
  `<variant>/<quantity>`. Sub-steps (features, selection, forward pass, predictions) are separate
  keys.
  - 1.6.1 helper: `_v1_runner.tabular_inputs(case)` builds `DrugResponseDataset`s and
    `FeatureDataset`s in memory (no CSVs). Skip `load_cell_line_features`: call
    `model.build_model(hp)` then `model.train(...)` / `model.predict(...)` directly.
  - v2 helper: `_v2_runner.tabular_inputs(case)` builds a `Dataset` plus train/test `SplitMask`s;
    drive it with `DRPModel(ModelSpec.from_name(NAME), hyperparameters)`, or build a
    `ModelInputBatch` and call a featurizer/predictor directly for a sub-step.
  - To see what reaches an estimator or network, wrap it with `_bridge.record_calls(owner, method,
    sink, key)`; `_v2_runner.capture_matrix_predictor(sink)` does this for every sklearn tabular
    predictor (design matrices and the fitted estimator).
  - Seeds are set from `case.seed` by the runner before `run_*` is called; re-seed inside for
    variants. Copy weights through arrays in the case (`meta`/`arrays`), e.g. the 1.6.1 `state_dict`
    written by `run_v1` and loaded in `run_v2`, if the case needs two phases.
- `steps()` returns `StepSpec(name, tolerance, v1_key=None, v2_key=None, scope="parity",
  description="", explanation="")`. Keys default to `name`; set `v1_key`/`v2_key` when the sides
  name an output differently or one 1.6.1 run is compared with several v2 runs.
- Fixing a real mismatch is not the harness' job: record it as a `finding`, declare the honest
  class, and report the bug to the package that owns the v2 file.

`tests/tools/test_v1_parity.py` automatically checks the interface of every module in `models/`.

## Examples

`models/elastic_net.py` is the worked example: 1.6.1 `ElasticNet` (`Ridge` for the first grid
point, `l1_ratio=0`) against the unmodified v2 `ElasticNet` zoo preset, compared at the estimator-input
level. Its evidence is `evidence/ElasticNet.json`.
