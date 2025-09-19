"""Pytest configuration file for the tests directory."""

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    """
    Change to the tests directory and adjust pytest settings.

    :param config: pytest config object
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Reduce flaky plugin verbosity
    config.option.flaky_report = "minimal"
