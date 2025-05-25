#!/usr/bin/env python
# coding: utf-8

"""Unit tests for the dataset module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import SNPDataModule, SNPDataset, load_data

# Test configuration
TEST_CONFIG = {
    "data": {
        "input_path": "test_data.parquet",
        "seq_length": 100,
        "missing_value": 9,
        "normalize": True,
        "scale_factor": 0.5,
        "augment": False,
        "batch_size": 2,  # Smaller batch size for test data
        "num_workers": 0,  # Set to 0 for tests to avoid multiprocessing issues
        "datasplit": [4, 2, 2],  # Exact numbers that sum to 8 (number of test samples)
    }
}


# Create a fixture for test data
@pytest.fixture
def test_data():
    """Create a small test dataset."""
    # Create a small test dataset with known patterns
    data = np.array(
        [
            [0, 1, 2, 0, 1],  # Sample 1
            [1, 2, 0, 1, 2],  # Sample 2
            [2, 0, 1, 2, 0],  # Sample 3
            [0, 0, 0, 0, 0],  # Sample 4
            [1, 1, 1, 1, 1],  # Sample 5
            [2, 2, 2, 2, 2],  # Sample 6
            [9, 9, 9, 9, 9],  # Sample 7 (all missing)
            [0, 9, 1, 9, 2],  # Sample 8 (some missing)
        ],
        dtype=np.float32,
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df = pd.DataFrame(data.T)  # Transpose to match expected shape
        df.to_parquet(f.name)
        temp_path = f.name

    yield temp_path, data

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_load_data(test_data):
    """Test loading and preprocessing of data."""
    temp_path, _ = test_data
    config = TEST_CONFIG.copy()
    config["data"]["input_path"] = temp_path

    # Test loading with normalization
    data = load_data(config)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (8, 5)  # 8 samples, 5 markers

    # Check that missing values are handled
    assert not torch.any(torch.isnan(data))
    assert not torch.any(data == 9)  # No missing values should remain


def test_snp_dataset(test_data):
    """Test SNPDataset functionality."""
    temp_path, _ = test_data
    config = TEST_CONFIG.copy()
    config["data"]["input_path"] = temp_path

    dataset = SNPDataset(config)

    # Test length
    assert len(dataset) == 8

    # Test getitem
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (5,)  # 5 markers per sample


def test_snp_datamodule(test_data):
    """Test SNPDataModule functionality."""
    temp_path, _ = test_data
    config = TEST_CONFIG.copy()
    config["data"]["input_path"] = temp_path

    # Initialize datamodule
    datamodule = SNPDataModule(config)

    # Test setup
    datamodule.setup(stage="fit")

    # Check datasets
    assert hasattr(datamodule, "train_dataset")
    assert hasattr(datamodule, "val_dataset")
    assert hasattr(datamodule, "test_dataset")

    # Check dataloaders
    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    # Test batch size
    batch = next(iter(train_loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == min(
        config["data"]["batch_size"], len(datamodule.train_dataset)
    )


def test_handle_missing_values():
    """Test missing value imputation."""
    from src.dataset import handle_missing_values

    # Test data with missing values (9 represents missing)
    data = np.array(
        [
            [0, 1, 2, 9],  # First row
            [
                1,
                2,
                9,
                9,
            ],  # Second row - third column should be 0 (smallest mode when tie)
            [2, 0, 1, 9],  # Third row
            [0, 0, 0, 0],  # Fourth row
        ]
    )

    # Expected: missing values (9) should be replaced with mode of each column
    # For the third column [2,9,1,0], all values appear once, so we take the smallest (0)
    expected = np.array(
        [
            [0, 1, 2, 0],
            [1, 2, 0, 0],  # Third column is 0 (smallest mode)
            [2, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    )

    result = handle_missing_values(data, missing_value=9)
    np.testing.assert_array_equal(
        result, expected, err_msg="Missing value imputation failed"
    )


def test_normalize_data():
    """Test data normalization."""
    from src.dataset import normalize_data

    data = np.array([[0, 1, 2], [2, 1, 0]])

    expected = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])

    result = normalize_data(data)
    np.testing.assert_array_almost_equal(result, expected)


def test_scale_data():
    """Test data scaling with different factors."""
    from src.dataset import scale_data

    # Test with numpy array input
    data_np = np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]])

    # Test with default factor (0.5)
    expected_default = np.array([[0.0, 0.25, 0.5], [0.25, 0.5, 0.0]])
    result_default = scale_data(data_np)
    np.testing.assert_array_almost_equal(result_default, expected_default)

    # Test with custom factor (0.2)
    expected_custom = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.0]])
    result_custom = scale_data(data_np, factor=0.2)
    np.testing.assert_array_almost_equal(result_custom, expected_custom)

    # Test with PyTorch tensor input
    import torch

    data_tensor = torch.tensor(data_np)
    result_tensor = scale_data(data_tensor, factor=0.5)
    assert isinstance(result_tensor, torch.Tensor)
    np.testing.assert_array_almost_equal(result_tensor.numpy(), expected_default)


def test_augment_data():
    """Test data augmentation with test patterns."""
    from src.dataset import augment_data

    # Create test data with 100 markers (as expected by augment_data)
    data = np.random.rand(3, 100)  # 3 samples, 100 markers each

    # Apply augmentation
    augmented = augment_data(data)

    # Check output shape
    assert augmented.shape == data.shape

    # Check pattern application
    # First 25 markers should be 0.0
    np.testing.assert_array_equal(augmented[:, :25], np.zeros((3, 25)))

    # Next 50 markers should be 0.5
    np.testing.assert_array_equal(augmented[:, 25:75], np.full((3, 50), 0.5))

    # Next 25 markers should be 1.0
    np.testing.assert_array_equal(augmented[:, 75:100], np.ones((3, 25)))

    # Test with fewer markers than pattern expects
    data_small = np.random.rand(2, 50)  # Only 50 markers
    augmented_small = augment_data(data_small)

    # Should only apply patterns up to the available markers
    assert augmented_small.shape == (2, 50)
    np.testing.assert_array_equal(
        augmented_small[:, :25], np.zeros((2, 25))
    )  # First 25 markers = 0.0
    np.testing.assert_array_equal(
        augmented_small[:, 25:], np.full((2, 25), 0.5)
    )  # Next 25 markers = 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
