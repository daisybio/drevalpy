import os
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
