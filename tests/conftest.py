"""Pytest configuration file for the tests directory."""

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure() -> None:
    """Change to the tests directory."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
