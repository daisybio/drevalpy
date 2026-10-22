# Follow-ups from the docs restructure

Things the docs restructure (`plans/docs-diataxis.md`) found or deliberately left out.
"Code" means the fix is a code change, not a docs change.

## Decided out of scope

- **LCO → LSO in nf-core/drugresponseeval.** The pipeline still passes `LCO`. The alias keeps it
  working and results are recorded as `LSO`. Rename it in the pipeline's parameter schema,
  and then bump the `drevalpy` version it pins.
- **Public dataset bucket.** `s3://orakl-open-source-data/drevalpy_h5mu/gen3/` returns
  `403 Forbidden` without credentials, so a new user cannot load GDSC1, GDSC2, CTRPv1, CTRPv2,
  PDX_Bruna, BeatAML2 or TOYv1. Every how-to that loads one is marked `.. snippet: network`.
- **Redirects for old documentation URLs.** The pages moved from CLI and Python guides to
  tutorials, how-to guides, reference and explanation; old addresses return 404.
- **Curation fits depend on the OpenBLAS kernel** (code, in dreval_curation): pEC50 moves by up
  to about 0.11 between machines. Recorded from the plan; not re-measured.
- **`pip install drevalpy[report]` hints.** Removed in `6e59f804`. `multiqc` is a hard
  dependency (`pyproject.toml`), and no page, source file or README mentions a `report` extra
  any more.

## Found while writing the project pages

- **README no longer carries the nf-core parameter sets of the published benchmark** (the
  `params/*.yaml` files for the main, EC50, AUC, invariant and inference runs). The restructure
  trims the README to a pitch, install, quick start and links. The blocks used `test_mode: LSO`
  and `NaiveSampleMeanPredictor`, which the pinned pipeline may not accept. Check them against
  the pipeline's current schema and move them to the nf-core docs or the paper repository.
- **Release steps in `news.rst` and `README.md`.** When v2 is published on PyPI, remove
  "It is not released yet" from the `news.rst` entry and the "Coming from DrEvalPy 1.x" note
  and the "once v2 is released" wording in the README install section, and set the entry date.
  Until then `pip install drevalpy` and `docker pull ...:latest` (the Docker image is only
  built when a release is published) give 1.6.0.
- **Default branch.** It will be `main`, but the GitHub setting is left alone for now (user
  decision). `docs/conf.py` edit links, `publish_docs.yml` and `docs/project/contributing.rst`
  already assume `main`; `git ls-remote --symref origin HEAD` still reports `development`, and
  `.github/pull_request_template.md` (line 38) still describes a development → `main` release
  PR. Update both when the branch switches.
- **`drevalpy report --dataset`** is accepted and ignored (code). It exists only for pipeline
  compatibility. Remove it when nf-core/drugresponseeval stops passing it.
- **`docs/readme.rst`** still describes DrEvalPy as a "cancer cell line" suite. The plan
  deletes the page in Wave 3.
- **The zoo loader silently drops a preset that fails validation** (code).
  `models/zoo/__init__.py::_load_builtin_entries` wraps the validation in a bare
  `except ValueError: continue`, so a zoo file whose components no longer fit (for example after
  the enforced `required_*_blocks` check) disappears from `drevalpy list zoo-presets` and from
  `ModelSpec.from_name` without a message. Log the reason or fail the load.
- **`tissue:identity:elasticNet` predicts a constant on GDSC2 + CTRPv2 LStO without tuning**
  (code). Its Pearson correlation is then undefined. The cross-study how-to pages therefore
  use the naive baselines for their examples.
- **No per-study response scaling for cross-study runs** (code). `--response-transformation` is
  fitted on the training scope, so it cannot remove the pEC50 offset between studies (GDSC2 mean
  ≈6.31, CTRPv2 ≈5.55). Consider a per-study standardization option for `concat` or LStO.
- **`dataset_registry` re-downloads a cached file without quality layers** (code).
  `_carries_curve_quality` treats such a file as stale, even when the user passes
  `CurveQuality.off()`.
