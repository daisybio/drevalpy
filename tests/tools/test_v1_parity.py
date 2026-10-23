"""Tests for the 1.6.1 parity harness: case IO and tolerance-class comparison.

Neither the 1.6.1 environment nor torch is needed; the runners are exercised by hand
(``tools/v1_parity/README.md``), not in CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.v1_parity._bridge import (
    Case,
    load_model_module,
    load_outputs,
    model_stems,
    record_calls,
    save_outputs,
    tabular_case_arrays,
)
from tools.v1_parity.compare import (
    CLASS_ORDER,
    DEVIATION,
    PARITY,
    StepSpec,
    build_evidence,
    compare_arrays,
    compare_outputs,
    verdict,
    write_evidence,
)


def _case(seed: int = 0) -> Case:
    return Case(
        model="Toy",
        seed=seed,
        arrays={"x": np.arange(6, dtype=np.float32).reshape(2, 3), "ids": np.array(["a", "b"])},
        meta={"alpha": 0.5, "names": ["p", "q"]},
    )


class TestCaseIO:
    def test_round_trip_preserves_arrays_and_meta(self, tmp_path):
        original = _case(3)
        original.save(tmp_path / "case")
        restored = Case.load(tmp_path / "case")
        assert restored.model == "Toy"
        assert restored.seed == 3
        assert restored.meta == original.meta
        assert restored.arrays["x"].dtype == np.float32
        np.testing.assert_array_equal(restored.arrays["x"], original.arrays["x"])
        np.testing.assert_array_equal(restored.arrays["ids"], original.arrays["ids"])

    def test_object_arrays_are_rejected(self, tmp_path):
        case = Case(model="Toy", seed=0, arrays={"bad": np.array(["a", 1], dtype=object)})
        with pytest.raises(ValueError, match="object dtype"):
            case.save(tmp_path)

    def test_input_hash_tracks_every_input(self):
        base = _case().input_hash()
        assert _case().input_hash() == base
        assert _case(seed=1).input_hash() != base
        changed = _case()
        changed.arrays["x"] = changed.arrays["x"] + 1
        assert changed.input_hash() != base
        retyped = _case()
        retyped.arrays["x"] = retyped.arrays["x"].astype(np.float64)
        assert retyped.input_hash() != base
        renamed = _case()
        renamed.meta["alpha"] = 0.6
        assert renamed.input_hash() != base

    def test_input_hash_survives_a_round_trip(self, tmp_path):
        case = _case()
        case.save(tmp_path)
        assert Case.load(tmp_path).input_hash() == case.input_hash()

    def test_outputs_round_trip(self, tmp_path):
        save_outputs(tmp_path, "v1", {"a/b": np.array([1.0, 2.0])}, {"python": "3.12"})
        arrays, info = load_outputs(tmp_path, "v1")
        np.testing.assert_array_equal(arrays["a/b"], [1.0, 2.0])
        assert info == {"python": "3.12"}


class TestTabularCaseArrays:
    def test_is_deterministic_and_seed_dependent(self):
        first, second = tabular_case_arrays(5), tabular_case_arrays(5)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        assert not np.array_equal(first["expression"], tabular_case_arrays(6)["expression"])

    def test_pairs_are_sorted_observed_and_disjoint(self):
        arrays = tabular_case_arrays(0)
        train, test = arrays["train_pairs"], arrays["test_pairs"]
        for pairs in (train, test):
            assert np.array_equal(pairs, pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))])
            assert not np.isnan(arrays["response"][pairs[:, 0], pairs[:, 1]]).any()
        train_keys = {tuple(p) for p in train.tolist()}
        assert train_keys.isdisjoint({tuple(p) for p in test.tolist()})

    def test_held_out_samples_never_occur_in_training(self):
        arrays = tabular_case_arrays(0, n_samples=20, test_only_fraction=0.25)
        held_out = set(range(15, 20))
        assert held_out.isdisjoint(set(arrays["train_pairs"][:, 0].tolist()))
        assert held_out <= set(arrays["test_pairs"][:, 0].tolist())

    def test_values_are_float32_representable(self):
        arrays = tabular_case_arrays(0)
        assert arrays["expression"].dtype == np.float32
        assert arrays["response"].dtype == np.float32
        assert set(np.unique(arrays["fingerprints"])) <= {0.0, 1.0}


class TestRecordCalls:
    class _Estimator:
        def fit(self, x):
            return "fitted"

    def test_instance_patch_copies_input_and_is_removed(self):
        estimator = self._Estimator()
        sink: dict = {}
        data = np.ones((2, 2))
        with record_calls(estimator, "fit", sink, "x"):
            assert estimator.fit(data) == "fitted"
        data[:] = 7
        assert sink["x"].max() == 1.0
        assert "fit" not in vars(estimator)

    def test_class_patch_skips_self_and_restores(self):
        sink: dict = {}
        seen: list = []
        with record_calls(self._Estimator, "fit", sink, "x", after=lambda inst, store: seen.append(inst)):
            instance = self._Estimator()
            instance.fit(np.zeros(3))
        assert sink["x"].shape == (3,)
        assert seen == [instance]
        assert self._Estimator().fit(1) == "fitted"


def _spec(tolerance: str, **kwargs) -> StepSpec:
    return StepSpec(name="s", tolerance=tolerance, **kwargs)


class TestCompareArrays:
    def test_identical_arrays_are_exact(self):
        a = np.array([1.0, 2.0, 3.0])
        result = compare_arrays(_spec("exact"), a, a.copy())
        assert result.passed
        assert result.achieved == "exact"
        assert result.max_abs_diff == 0.0

    @pytest.mark.parametrize(
        ("perturbation", "expected"),
        [(1e-14, "exact"), (5e-7, "forward"), (5e-5, "train")],
    )
    def test_achieved_class_is_the_strictest_satisfied(self, perturbation, expected):
        a = np.linspace(1.0, 2.0, 50)
        result = compare_arrays(_spec("statistical"), a, a + perturbation)
        assert result.achieved == expected
        assert result.passed

    def test_declared_class_decides_passing(self):
        a = np.linspace(1.0, 2.0, 50)
        b = a + 5e-7
        assert not compare_arrays(_spec("exact"), a, b).passed
        assert compare_arrays(_spec("forward"), a, b).passed

    def test_train_class_uses_relative_and_absolute_terms(self):
        a = np.array([1000.0, 0.0])
        within = compare_arrays(_spec("train"), a, a + np.array([0.09, 5e-7]))
        outside = compare_arrays(_spec("train"), a, a + np.array([0.2, 0.0]))
        assert within.passed
        assert not outside.passed

    def test_statistical_class_ignores_small_noise_but_not_a_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        noisy = a + rng.normal(scale=0.05, size=200)
        shifted = a + 2.0
        uncorrelated = rng.normal(size=200)
        assert compare_arrays(_spec("statistical"), a, noisy).passed
        assert not compare_arrays(_spec("statistical"), a, shifted).passed
        assert not compare_arrays(_spec("statistical"), a, uncorrelated).passed
        assert compare_arrays(_spec("statistical"), a, noisy).pearson > 0.99

    def test_constant_arrays_pass_statistical_only_when_equal(self):
        const = np.full(10, 2.0)
        assert compare_arrays(_spec("statistical"), const, const.copy()).passed
        assert not compare_arrays(_spec("statistical"), const, const + 1.0).passed

    def test_nan_positions_must_match(self):
        a = np.array([1.0, np.nan, 3.0])
        assert compare_arrays(_spec("exact"), a, a.copy()).passed
        result = compare_arrays(_spec("exact"), a, np.array([1.0, 2.0, 3.0]))
        assert not result.passed
        assert "NaN" in result.detail

    def test_shape_mismatch_fails_with_detail(self):
        result = compare_arrays(_spec("exact"), np.zeros((2, 3)), np.zeros((3, 2)))
        assert not result.passed
        assert "shape mismatch" in result.detail
        assert result.shape_v1 == [2, 3]
        assert result.shape_v2 == [3, 2]

    @pytest.mark.parametrize(("a", "b", "side"), [(None, np.zeros(2), "1.6.1"), (np.zeros(2), None, "v2")])
    def test_missing_side_fails_naming_the_side(self, a, b, side):
        result = compare_arrays(_spec("exact"), a, b)
        assert not result.passed
        assert side in result.detail

    def test_non_numeric_arrays_must_be_equal(self):
        genes = np.array(["A", "B", "C"])
        assert compare_arrays(_spec("forward"), genes, genes.copy()).achieved == "exact"
        reordered = compare_arrays(_spec("forward"), genes, genes[::-1])
        assert not reordered.passed
        assert "1.6.1='A'" in reordered.detail

    def test_integer_arrays_compare_numerically(self):
        a = np.array([1, 2, 3])
        assert compare_arrays(_spec("exact"), a, a.copy()).passed
        assert not compare_arrays(_spec("exact"), a, a + 1).passed

    def test_relative_difference_ignores_zero_pairs(self):
        result = compare_arrays(_spec("train"), np.array([0.0, 2.0]), np.array([0.0, 2.0002]))
        assert result.max_rel_diff == pytest.approx(0.0002 / 2.0002)

    def test_result_is_json_serializable(self):
        result = compare_arrays(_spec("forward"), np.ones(3), np.ones(3))
        assert json.loads(json.dumps(result.to_dict()))["name"] == "s"


class TestSteps:
    def test_unknown_tolerance_is_rejected(self):
        with pytest.raises(ValueError, match="unknown tolerance class"):
            StepSpec(name="s", tolerance="loose")

    def test_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="unknown step scope"):
            StepSpec(name="s", scope="other")

    def test_class_order_is_strictest_first(self):
        assert CLASS_ORDER == ("exact", "forward", "train", "statistical")

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate step names"):
            compare_outputs([StepSpec(name="a"), StepSpec(name="a")], {}, {})

    def test_keys_default_to_the_name_and_can_differ_per_side(self):
        specs = [StepSpec(name="same"), StepSpec(name="mapped", v1_key="old", v2_key="new")]
        v1 = {"same": np.ones(2), "old": np.arange(3.0)}
        v2 = {"same": np.ones(2), "new": np.arange(3.0)}
        assert all(r.passed for r in compare_outputs(specs, v1, v2))


class TestVerdict:
    @staticmethod
    def _results(*steps: tuple[str, str, bool]):
        specs = [StepSpec(name=f"s{i}", tolerance=tol, scope=scope) for i, (tol, scope, _) in enumerate(steps)]
        v1 = {f"s{i}": np.zeros(4) for i in range(len(steps))}
        v2 = {f"s{i}": np.zeros(4) if ok else np.full(4, 1.0) for i, (_, _, ok) in enumerate(steps)}
        return compare_outputs(specs, v1, v2)

    def test_all_exact_gives_exact(self):
        assert verdict(self._results(("exact", PARITY, True), ("exact", PARITY, True))) == "exact"

    def test_verdict_is_the_loosest_declared_class(self):
        steps = (("exact", PARITY, True), ("train", PARITY, True), ("forward", PARITY, True))
        assert verdict(self._results(*steps)) == "train"

    def test_failing_parity_step_is_a_mismatch(self):
        assert verdict(self._results(("exact", PARITY, True), ("train", PARITY, False))) == "mismatch"

    def test_deviation_steps_never_change_the_verdict(self):
        steps = (("exact", PARITY, True), ("statistical", DEVIATION, False))
        assert verdict(self._results(*steps)) == "exact"

    def test_needs_a_parity_step(self):
        with pytest.raises(ValueError, match="at least one parity step"):
            verdict(self._results(("exact", DEVIATION, True)))


class TestEvidence:
    def test_split_by_scope_and_write(self, tmp_path):
        specs = [
            StepSpec(name="p", tolerance="forward"),
            StepSpec(name="d", tolerance="statistical", scope=DEVIATION, explanation="by design"),
        ]
        results = compare_outputs(specs, {"p": np.ones(3), "d": np.arange(5.0)}, {"p": np.ones(3), "d": np.arange(5.0)})
        evidence = build_evidence(
            "Toy",
            results,
            case={"seed": 0, "inputs_sha256": "abc"},
            v1={"commit": "f5ff3c70"},
            v2={"commit": "deadbeef"},
            findings=["a finding"],
            generated="2026-10-23",
        )
        path = write_evidence(tmp_path / "evidence" / "Toy.json", evidence)
        loaded = json.loads(path.read_text())
        assert path.read_text().endswith("\n")
        assert loaded["model"] == "Toy"
        assert loaded["verdict"] == "forward"
        assert [s["name"] for s in loaded["steps"]] == ["p"]
        assert loaded["deviations"][0]["explanation"] == "by design"
        assert loaded["findings"] == ["a finding"]
        assert loaded["v1"]["commit"] == "f5ff3c70"
        assert loaded["schema_version"] == 1


class TestModelCases:
    def test_every_case_module_follows_the_interface(self):
        assert "elastic_net" in model_stems()
        for stem in model_stems():
            module = load_model_module(stem)
            assert isinstance(module.NAME, str)
            assert callable(module.build_case)
            assert callable(module.run_v1)
            assert callable(module.run_v2)
            names = [spec.name for spec in module.steps()]
            assert len(names) == len(set(names))
            assert any(spec.scope == PARITY for spec in module.steps())

    def test_lookup_accepts_name_and_stem(self):
        assert load_model_module("ElasticNet") is load_model_module("elastic_net")
        with pytest.raises(KeyError, match="no parity case"):
            load_model_module("NoSuchModel")

    def test_elastic_net_case_is_deterministic_and_step_keys_resolve(self):
        module = load_model_module("ElasticNet")
        assert module.build_case(0).input_hash() == module.build_case(0).input_hash()
        assert module.build_case(0).input_hash() != module.build_case(1).input_hash()
        variants = module.build_case(0).meta["variants"]
        for spec in module.steps():
            variant, quantity = spec.name.split("/")
            assert variant in variants
            assert quantity in {"features_train", "features_test", "params", "coef", "intercept", "predictions"}
            assert spec.v1_key.split("/")[0] in variants
