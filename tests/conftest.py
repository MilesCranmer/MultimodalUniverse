"""Pytest configuration: register `slow` marker and skip-by-default behavior."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (real hats-import pipeline calls).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that run the full hats-import pipeline (~30s each)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
