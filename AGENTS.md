# Agent notes

## Running pre-commit / committing

The hooks in `.pre-commit-config.yaml` use `language: system`, so they resolve `black`,
`flake8`, `pyupgrade`, `check-*`, etc. from `PATH` instead of installing their own copies.
Those tools live in the project venv, so prepend it before committing or running hooks:

```bash
PATH="$(pwd)/.venv/bin:$PATH" git commit -m "..."
PATH="$(pwd)/.venv/bin:$PATH" pre-commit run --all-files
```

Without this, hooks fail with "Executable `black` not found" (and the same for the other tools). Activating the venv (`source .venv/bin/activate`) works too.
