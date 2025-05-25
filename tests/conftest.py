"""pytest configuration and shared fixtures for testing.

This file is automatically discovered by pytest and used to define fixtures and hooks
that are available to all tests in this directory and its subdirectories.
"""
import os
import sys
from pathlib import Path
import pytest

# Import project path utilities
from src import PROJECT_ROOT, DATA_DIR

# Add the project root to the Python path to allow importing from src
sys.path.insert(0, str(PROJECT_ROOT))

# Example fixture - can be used in any test by adding it as a parameter
@pytest.fixture
def project_root_path() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT

# Fixture for easy access to data directory
@pytest.fixture
def data_dir() -> Path:
    """Return the path to the project's data directory."""
    return DATA_DIR

# You can add more fixtures here that will be available to all tests
# @pytest.fixture
# def sample_data():
#     """Example fixture that returns sample data."""
#     return {"key": "value"}
