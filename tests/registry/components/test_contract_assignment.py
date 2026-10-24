"""Tests for :mod:`drevalpy.registry.components._contract_assignment`.

Both concrete registries delegate contract resolution here, so the precedence rule
(decorator argument over class-body declaration) and its errors are pinned on plain classes.
"""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.components._contract_assignment import assign_contract, declared_contract

GRAPH = FeatureContract(format=FeatureFormat.GRAPH)
MATRIX = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def _component(base: type = object, **attrs) -> type:
    return type("Component", (base,), attrs)


class TestAssignContract:
    @pytest.mark.parametrize(
        ("body", "argument", "expected"),
        [
            ({}, GRAPH, GRAPH),
            ({"contract": MATRIX}, GRAPH, GRAPH),
            ({"contract": GRAPH}, None, GRAPH),
            ({"contract": FeatureFormat.GRAPH}, None, GRAPH),
        ],
        ids=["decorator-only", "decorator-wins", "class-body-fallback", "format-shorthand"],
    )
    def test_resolution_order(self, body: dict, argument, expected: FeatureContract) -> None:
        component = _component(**body)

        assign_contract(component, "contract", argument)

        assert component.contract == expected

    def test_a_declaration_inherited_from_a_base_class_is_usable(self) -> None:
        component = _component(_component(contract=MATRIX))

        assign_contract(component, "contract", None)

        assert component.__dict__["contract"] == MATRIX

    def test_assigns_each_named_attribute_independently(self) -> None:
        component = _component()

        assign_contract(component, "sample_contract", MATRIX)
        assign_contract(component, "drug_contract", GRAPH)

        assert (component.sample_contract, component.drug_contract) == (MATRIX, GRAPH)

    def test_rejects_a_component_that_declares_nothing_naming_the_class_and_both_fixes(self) -> None:
        with pytest.raises(ValueError, match="no sample_contract declared"):
            assign_contract(_component(), "sample_contract", None)
        with pytest.raises(ValueError, match=r"Component: .*pass contract= to the .*or set it on the class body"):
            assign_contract(_component(), "contract", None)


class TestDeclaredContract:
    def test_returns_a_contract_and_promotes_a_bare_format(self) -> None:
        assert declared_contract(_component(contract=GRAPH), "contract") is GRAPH
        assert declared_contract(_component(contract=FeatureFormat.NUMERIC_MATRIX), "contract") == MATRIX

    def test_an_explicit_none_counts_as_undeclared(self) -> None:
        with pytest.raises(ValueError, match="no contract declared"):
            declared_contract(_component(contract=None), "contract")

    @pytest.mark.parametrize("value", ["numeric_matrix", 42])
    def test_an_unusable_declaration_is_invalid_not_missing_and_keeps_the_type_error_as_cause(self, value) -> None:
        with pytest.raises(ValueError, match="class-body contract is invalid") as excinfo:
            declared_contract(_component(contract=value), "contract")

        assert isinstance(excinfo.value.__cause__, TypeError)
