"""Shared pytest configuration: the `gpu` marker (tests that need CUDA and the checkpoint) is registered here so
`-m "not gpu"` works and unregistered-marker warnings do not appear."""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs CUDA, exllamav3 kernels or the FP8 checkpoint")
