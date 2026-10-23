"""Run one parity case end to end: build the case, run both implementations, compare, record.

Usage (from the repository root, in the v2 environment)::

    uv run python -m tools.v1_parity.run ElasticNet
    uv run python -m tools.v1_parity.run --list

The 1.6.1 side needs the environment created by ``tools/v1_parity/setup_v161.sh``.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.v1_parity._bridge import (
    DEFAULT_V1_DIR,
    EVIDENCE_DIR,
    REPO_ROOT,
    V1_COMMIT,
    load_model_module,
    load_outputs,
    model_stems,
)
from tools.v1_parity.compare import Evidence, build_evidence, compare_outputs, write_evidence


def _git(directory: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed git command line
        ["git", "-C", str(directory), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip()


def _spawn(python: Path, module: str, model: str, case_dir: Path, out_dir: Path, env_extra: dict[str, str]) -> None:
    env = {**os.environ, **env_extra}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    command = [str(python), "-m", module, "--model", model, "--case", str(case_dir), "--out", str(out_dir)]
    done = subprocess.run(  # noqa: S603 - interpreter and module are harness-controlled
        command, cwd=out_dir, env=env, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        msg = f"{module} failed with exit code {done.returncode}"
        raise RuntimeError(msg)


def run_model(
    name: str,
    *,
    seed: int = 0,
    v1_dir: Path = DEFAULT_V1_DIR,
    v2_python: Path | None = None,
    workdir: Path | None = None,
    evidence_dir: Path | None = EVIDENCE_DIR,
) -> Evidence:
    """Run one model's case on both sides and compare.

    :param name: Harness model name or module stem.
    :param seed: Seed of the case.
    :param v1_dir: Directory of the 1.6.1 worktree (its ``.venv`` runs the 1.6.1 side).
    :param v2_python: Interpreter for the v2 side (defaults to the running one).
    :param workdir: Directory for the case and outputs (a temporary directory by default).
    :param evidence_dir: Where to write ``<name>.json``; ``None`` skips writing.
    :returns: The evidence record.
    :raises FileNotFoundError: If the 1.6.1 environment is missing.
    """
    v1_python = v1_dir / ".venv" / "bin" / "python"
    if not v1_python.exists():
        msg = f"{v1_python} not found; run tools/v1_parity/setup_v161.sh first"
        raise FileNotFoundError(msg)
    module = load_model_module(name)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix=f"v1_parity_{module.NAME}_"))
    work.mkdir(parents=True, exist_ok=True)
    case = module.build_case(seed)
    case.save(work)
    deterministic = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    _spawn(v1_python, "tools.v1_parity._v1_runner", module.NAME, work, work, deterministic)
    _spawn(v2_python or Path(sys.executable), "tools.v1_parity._v2_runner", module.NAME, work, work, deterministic)

    out_v1, info_v1 = load_outputs(work, "v1")
    out_v2, info_v2 = load_outputs(work, "v2")
    results = compare_outputs(module.steps(), out_v1, out_v2)
    info_v1["commit"] = _git(v1_dir, "rev-parse", "HEAD")
    info_v1["expected_commit"] = V1_COMMIT
    info_v2["commit"] = _git(REPO_ROOT, "rev-parse", "HEAD")
    info_v2["dirty"] = bool(_git(REPO_ROOT, "status", "--porcelain", "--", "src"))
    for info in (info_v1, info_v2):
        info.pop("executable", None)
    evidence = build_evidence(
        module.NAME,
        results,
        case={"seed": seed, "inputs_sha256": case.input_hash(), "meta": case.meta},
        v1=info_v1,
        v2=info_v2,
        findings=list(getattr(module, "FINDINGS", [])),
        generated=datetime.datetime.now(datetime.UTC).date().isoformat(),
    )
    if evidence_dir is not None:
        write_evidence(Path(evidence_dir) / f"{module.NAME}.json", evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: ``0`` when the verdict is not ``mismatch``, else ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="model names or module stems")
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true", help="do not write evidence JSON")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(model_stems()))
        return 0
    status = 0
    for name in args.models:
        evidence = run_model(
            name,
            seed=args.seed,
            v1_dir=args.v1_dir,
            workdir=args.workdir,
            evidence_dir=None if args.no_write else EVIDENCE_DIR,
        )
        failed = [s["name"] for s in evidence.steps if not s["passed"]]
        print(f"{evidence.model}: verdict={evidence.verdict}; {len(evidence.steps)} parity steps, {len(failed)} failed")
        for step in [*evidence.steps, *evidence.deviations]:
            print(f"  {step['scope']:9} {step['name']:28} declared={step['declared']:11} achieved={step['achieved']}")
        status |= evidence.verdict == "mismatch"
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
