"""Tolerance classes, step comparison and evidence building for the 1.6.1 parity harness.

Imports numpy only, so the unit tests need neither the 1.6.1 venv nor torch.

A model case module declares a list of :class:`StepSpec`. Each step names one output array
that both implementations produce, the tolerance class it must satisfy and its *scope*:

``parity``
    Must hold for the port to be called equivalent. These steps decide the verdict.
``deviation``
    A difference the port introduces on purpose (or that is known and explained). The
    measured size is recorded in the evidence next to the explanation, never hidden by
    loosening a parity tolerance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
#: Strictest first. ``mismatch`` is the verdict when a parity step fails its class.
CLASS_ORDER = ("exact", "forward", "train", "statistical")
MISMATCH = "mismatch"
PARITY = "parity"
DEVIATION = "deviation"


@dataclass(frozen=True)
class Tolerance:
    """Numeric acceptance rule of one tolerance class.

    :param rtol: Relative tolerance of ``numpy.isclose`` (element-wise classes).
    :param atol: Absolute tolerance of ``numpy.isclose`` (element-wise classes).
    :param statistical: When true, compare Pearson correlation and mean difference instead.
    :param min_pearson: Smallest accepted Pearson correlation (statistical class).
    :param max_mean_diff: Largest accepted ``|mean(a) - mean(b)|`` as a multiple of ``std(a)``
        (statistical class).
    """

    rtol: float = 0.0
    atol: float = 0.0
    statistical: bool = False
    min_pearson: float = 0.99
    max_mean_diff: float = 0.1


TOLERANCES: dict[str, Tolerance] = {
    "exact": Tolerance(rtol=1e-12, atol=1e-12),
    "forward": Tolerance(rtol=1e-6, atol=1e-6),
    "train": Tolerance(rtol=1e-4, atol=1e-6),
    "statistical": Tolerance(statistical=True),
}


@dataclass(frozen=True)
class StepSpec:
    """One compared output.

    :param name: Step name, unique within a case; also the evidence key.
    :param tolerance: Declared tolerance class, one of :data:`CLASS_ORDER`.
    :param v1_key: Key in the 1.6.1 outputs (defaults to *name*).
    :param v2_key: Key in the v2 outputs (defaults to *name*).
    :param scope: ``"parity"`` (decides the verdict) or ``"deviation"`` (recorded only).
    :param description: What is compared, in one line.
    :param explanation: For deviations, why the difference exists.
    """

    name: str
    tolerance: str = "exact"
    v1_key: str | None = None
    v2_key: str | None = None
    scope: str = PARITY
    description: str = ""
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate the tolerance class and scope.

        :raises ValueError: If either is unknown.
        """
        if self.tolerance not in TOLERANCES:
            msg = f"unknown tolerance class {self.tolerance!r}; expected one of {CLASS_ORDER}"
            raise ValueError(msg)
        if self.scope not in (PARITY, DEVIATION):
            msg = f"unknown step scope {self.scope!r}"
            raise ValueError(msg)


@dataclass
class StepResult:
    """Measured outcome of one :class:`StepSpec`.

    :param name: Step name.
    :param scope: Step scope.
    :param declared: Declared tolerance class.
    :param achieved: Strictest class the arrays satisfy, or ``None`` when none.
    :param passed: Whether the declared class is satisfied.
    :param max_abs_diff: Largest absolute element difference (``None`` for non-numeric data).
    :param max_rel_diff: Largest ``|a - b| / max(|a|, |b|)`` over non-zero elements.
    :param pearson: Pearson correlation of the flattened arrays (numeric data, >1 element).
    :param shape_v1: Shape of the 1.6.1 array.
    :param shape_v2: Shape of the v2 array.
    :param detail: Free-text reason, set on failures such as a missing key or shape mismatch.
    :param description: Copied from the spec.
    :param explanation: Copied from the spec.
    """

    name: str
    scope: str
    declared: str
    achieved: str | None
    passed: bool
    max_abs_diff: float | None
    max_rel_diff: float | None
    pearson: float | None
    shape_v1: list[int] | None
    shape_v2: list[int] | None
    detail: str = ""
    description: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping.

        :returns: The result as plain Python types.
        """
        return asdict(self)


def _is_numeric(array: np.ndarray) -> bool:
    return array.dtype.kind in "biuf"


def _describe_difference(a: np.ndarray, b: np.ndarray) -> str:
    """Show up to three differing elements of two equally shaped non-numeric arrays."""
    pairs = [(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist(), strict=True) if x != y][:3]
    shown = "; ".join(f"1.6.1={x!r} v2={y!r}"[:400] for x, y in pairs)
    return f"non-numeric arrays differ: {shown}"


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return 1.0 if sa == sb and np.allclose(a, b) else None
    return float(np.corrcoef(a, b)[0, 1])


def _satisfies(name: str, a: np.ndarray, b: np.ndarray, pearson: float | None) -> bool:
    tol = TOLERANCES[name]
    if tol.statistical:
        if pearson is None:
            return bool(np.allclose(a, b))
        scale = float(np.std(a)) or 1.0
        return pearson >= tol.min_pearson and abs(float(np.mean(a)) - float(np.mean(b))) <= tol.max_mean_diff * scale
    return bool(np.allclose(a, b, rtol=tol.rtol, atol=tol.atol, equal_nan=True))


def compare_arrays(spec: StepSpec, a: np.ndarray | None, b: np.ndarray | None) -> StepResult:
    """Compare the 1.6.1 array *a* with the v2 array *b* under ``spec.tolerance``.

    Non-numeric arrays (gene names, ids) must be equal. NaNs at the same positions count as
    equal. A missing array or a shape mismatch fails with an explanatory ``detail``.

    :param spec: Step to evaluate.
    :param a: Output of the 1.6.1 side (``None`` when it did not produce the key).
    :param b: Output of the v2 side (``None`` when it did not produce the key).
    :returns: The measured result.
    """
    base = {"name": spec.name, "scope": spec.scope, "declared": spec.tolerance}
    base |= {"description": spec.description, "explanation": spec.explanation}
    if a is None or b is None:
        side = "1.6.1" if a is None else "v2"
        return StepResult(
            **base,
            achieved=None,
            passed=False,
            max_abs_diff=None,
            max_rel_diff=None,
            pearson=None,
            shape_v1=None if a is None else list(a.shape),
            shape_v2=None if b is None else list(b.shape),
            detail=f"{side} produced no output for this step",
        )
    a, b = np.asarray(a), np.asarray(b)
    shapes = {"shape_v1": list(a.shape), "shape_v2": list(b.shape)}
    if a.shape != b.shape:
        return StepResult(
            **base,
            achieved=None,
            passed=False,
            max_abs_diff=None,
            max_rel_diff=None,
            pearson=None,
            detail=f"shape mismatch {a.shape} vs {b.shape}",
            **shapes,
        )
    if not (_is_numeric(a) and _is_numeric(b)):
        equal = bool(np.array_equal(a, b))
        return StepResult(
            **base,
            achieved="exact" if equal else None,
            passed=equal,
            max_abs_diff=None,
            max_rel_diff=None,
            pearson=None,
            detail="" if equal else _describe_difference(a, b),
            **shapes,
        )
    af, bf = a.astype(np.float64), b.astype(np.float64)
    nan_a, nan_b = np.isnan(af), np.isnan(bf)
    if not np.array_equal(nan_a, nan_b):
        return StepResult(
            **base,
            achieved=None,
            passed=False,
            max_abs_diff=None,
            max_rel_diff=None,
            pearson=None,
            detail="NaN positions differ",
            **shapes,
        )
    keep = ~nan_a
    af, bf = af[keep], bf[keep]
    diff = np.abs(af - bf)
    scale = np.maximum(np.abs(af), np.abs(bf))
    nonzero = scale > 0
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff[nonzero] / scale[nonzero]).max()) if nonzero.any() else 0.0
    pearson = _pearson(af, bf)
    achieved = next((c for c in CLASS_ORDER if _satisfies(c, af, bf, pearson)), None)
    return StepResult(
        **base,
        achieved=achieved,
        passed=_satisfies(spec.tolerance, af, bf, pearson),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        pearson=pearson,
        **shapes,
    )


def compare_outputs(specs: list[StepSpec], v1: dict[str, np.ndarray], v2: dict[str, np.ndarray]) -> list[StepResult]:
    """Compare every declared step of the two output sets.

    :param specs: Steps declared by the model case module.
    :param v1: Arrays produced by the 1.6.1 side.
    :param v2: Arrays produced by the v2 side.
    :returns: One result per step, in declaration order.
    :raises ValueError: If two steps share a name.
    """
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        msg = f"duplicate step names: {sorted({n for n in names if names.count(n) > 1})}"
        raise ValueError(msg)
    return [compare_arrays(s, v1.get(s.v1_key or s.name), v2.get(s.v2_key or s.name)) for s in specs]


def verdict(results: list[StepResult]) -> str:
    """Summarize parity steps into one verdict.

    The verdict is ``"mismatch"`` when any parity step fails its declared class; otherwise
    it is the loosest declared class among the parity steps (``"exact"`` when every parity
    step is exact). Deviation steps never change it.

    :param results: Step results.
    :returns: A name from :data:`CLASS_ORDER`, or ``"mismatch"``.
    :raises ValueError: If there is no parity step.
    """
    parity = [r for r in results if r.scope == PARITY]
    if not parity:
        msg = "a case needs at least one parity step"
        raise ValueError(msg)
    if not all(r.passed for r in parity):
        return MISMATCH
    return max((r.declared for r in parity), key=CLASS_ORDER.index)


@dataclass
class Evidence:
    """The committed record of one model comparison.

    :param model: Harness model name.
    :param verdict: Result of :func:`verdict`.
    :param case: Seed, input hash and case metadata.
    :param v1: Versions and commit of the 1.6.1 side.
    :param v2: Versions and commit of the v2 side.
    :param steps: Parity step results.
    :param deviations: Deviation step results (with explanations).
    :param findings: Free-text differences found while building the case.
    :param generated: ISO date of the run.
    """

    model: str
    verdict: str
    case: dict[str, Any]
    v1: dict[str, Any]
    v2: dict[str, Any]
    steps: list[dict[str, Any]]
    deviations: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    generated: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping.

        :returns: The evidence as plain Python types.
        """
        return asdict(self)


def build_evidence(
    model: str,
    results: list[StepResult],
    *,
    case: dict[str, Any],
    v1: dict[str, Any],
    v2: dict[str, Any],
    findings: list[str] | None = None,
    generated: str = "",
) -> Evidence:
    """Assemble the evidence record from step results.

    :param model: Harness model name.
    :param results: Results from :func:`compare_outputs`.
    :param case: Seed, input hash and metadata of the case.
    :param v1: Versions and commit of the 1.6.1 side.
    :param v2: Versions and commit of the v2 side.
    :param findings: Differences found while building the case.
    :param generated: ISO date of the run.
    :returns: The evidence record.
    """
    return Evidence(
        model=model,
        verdict=verdict(results),
        case=case,
        v1=v1,
        v2=v2,
        steps=[r.to_dict() for r in results if r.scope == PARITY],
        deviations=[r.to_dict() for r in results if r.scope == DEVIATION],
        findings=list(findings or []),
        generated=generated,
    )


def write_evidence(path: str | Path, evidence: Evidence) -> Path:
    """Write the evidence as sorted, indented JSON with a trailing newline.

    :param path: Target file; parent directories are created.
    :param evidence: Record to write.
    :returns: The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n")
    return target
