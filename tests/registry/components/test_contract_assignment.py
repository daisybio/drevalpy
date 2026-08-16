"""Tests for :mod:`drevalpy.registry.components._contract_assignment`.

Mirrors the private module with the underscore stripped. Both concrete registries
delegate contract resolution here, so the precedence rule - decorator argument
over class-body declaration - and its error messages are pinned once, on plain
classes, without going through a registration decorator.
"""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.components._contract_assignment import assign_contract, declared_contract

GRAPH = FeatureContract(format=FeatureFormat.GRAPH)
MATRIX = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


class TestAssignContract:
    def test_uses_the_decorator_contract_when_the_class_declares_none(self):
        class Component:
            pass

        assign_contract(Component, "contract", GRAPH)

        assert Component.contract == GRAPH

    def test_the_decorator_wins_over_a_class_body_declaration(self):
        class Component:
            contract = MATRIX

        assign_contract(Component, "contract", GRAPH)

        assert Component.contract == GRAPH

    def test_falls_back_to_the_class_body_declaration(self):
        class Component:
            contract = GRAPH

        assign_contract(Component, "contract", None)

        assert Component.contract == GRAPH

    def test_normalizes_a_class_body_format_shorthand(self):
        class Component:
            contract = FeatureFormat.GRAPH

        assign_contract(Component, "contract", None)

        assert Component.contract == GRAPH

    def test_a_declaration_inherited_from_a_base_class_is_usable(self):
        class Base:
            contract = MATRIX

        class Component(Base):
            pass

        assign_contract(Component, "contract", None)

        assert Component.__dict__["contract"] == MATRIX

    def test_assigns_each_named_attribute_independently(self):
        class Predictor:
            pass

        assign_contract(Predictor, "cell_line_contract", MATRIX)
        assign_contract(Predictor, "drug_contract", GRAPH)

        assert (Predictor.cell_line_contract, Predictor.drug_contract) == (MATRIX, GRAPH)

    def test_rejects_a_component_that_declares_nothing(self):
        class Component:
            pass

        with pytest.raises(ValueError, match="no cell_line_contract declared"):
            assign_contract(Component, "cell_line_contract", None)

    def test_the_error_names_the_class_and_both_ways_to_fix_it(self):
        class Component:
            pass

        with pytest.raises(ValueError, match=r"Component: .*pass contract= to the .*or set it on the class body"):
            assign_contract(Component, "contract", None)


class TestDeclaredContract:
    def test_returns_a_declared_contract_unchanged(self):
        class Component:
            contract = GRAPH

        assert declared_contract(Component, "contract") is GRAPH

    def test_promotes_a_bare_format_to_a_contract(self):
        class Component:
            contract = FeatureFormat.NUMERIC_MATRIX

        assert declared_contract(Component, "contract") == MATRIX

    def test_an_explicit_none_counts_as_undeclared(self):
        class Component:
            contract = None

        with pytest.raises(ValueError, match="no contract declared"):
            declared_contract(Component, "contract")

    def test_reports_an_unusable_declaration_as_invalid_rather_than_missing(self):
        class Component:
            contract = "numeric_matrix"

        with pytest.raises(ValueError, match="class-body contract is invalid"):
            declared_contract(Component, "contract")

    def test_keeps_the_underlying_type_error_as_the_cause(self):
        class Component:
            contract = 42

        with pytest.raises(ValueError, match="class-body contract is invalid") as excinfo:
            declared_contract(Component, "contract")

        assert isinstance(excinfo.value.__cause__, TypeError)
