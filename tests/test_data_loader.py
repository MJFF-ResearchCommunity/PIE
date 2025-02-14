"""
test_data_loader.py

Placeholder tests for the data loader module.
"""

import pytest
from pie.data_loader import DataLoader

def test_data_loader_stub():
    data = DataLoader.load("path/to/data", "PPMI")
    assert isinstance(data, dict), "Expected a dictionary from the loader."
    # This test currently just validates that load() returns a dict of placeholders. 