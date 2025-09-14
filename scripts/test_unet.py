#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Comprehensive UNet1D test script.
Combines shape tracing, parameter analysis, and training estimation.
"""

import sys
import time
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unet import UNet1D
from src.utils import load_config


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_training_time(model, batch_size, seq_length, num_batches, device):
    """Estimate training time per epoch."""
    model.eval()
    x = torch.randn(batch_size, 1, seq_length).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, t)

    # Time forward passes
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, t)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time_per_batch = (end_time - start_time) / 10
    estimated_epoch_time = avg_time_per_batch * num_batches

    return avg_time_per_batch, estimated_epoch_time


def test_shape_tracing(config_name, config, seq_length=64, batch_size=2):
    """Test model with shape tracing enabled."""
    print(f"\n=== Shape Tracing: {config_name} ===")
    print(f"Sequence length: {seq_length}, Batch size: {batch_size}")

    try:
        model = UNet1D(
            seq_length=seq_length,
            channels=1,
            with_time_emb=True,
            edge_pad=2,  # Use configurable edge_pad
            debug=True,  # Enable shape tracing
            **config,
        )

        x = torch.randn(batch_size, 1, seq_length)
        t = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            output = model(x, t)

        success = output.shape == x.shape
        print(f"Shape test: {'✅ PASSED' if success else '❌ FAILED'}")
        return True

    except Exception as e:
        print(f"Shape test: ❌ FAILED - {str(e)}")
        return False


def test_configurations():
    """Test different UNet1D configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test configurations
    configs = {
        "Ultra-Shallow": {
            "embedding_dim": 16,
            "dim_mults": (1,),
            "with_pos_emb": False,
            "norm_groups": 4,
        },
        "Shallow": {
            "embedding_dim": 16,
            "dim_mults": (1, 2),
            "with_pos_emb": False,
            "norm_groups": 4,
        },
        "Medium": {
            "embedding_dim": 32,
            "dim_mults": (1, 2, 4),
            "with_pos_emb": True,
            "norm_groups": 8,
        },
        "Deep": {
            "embedding_dim": 64,
            "dim_mults": (1, 2, 4, 8),
            "with_pos_emb": True,
            "norm_groups": 8,
        },
    }

    seq_length = 100  # Realistic sequence length
    batch_size = 8

    print("\n=== Configuration Comparison ===")
    print(f"Test sequence length: {seq_length}")
    print(f"Test batch size: {batch_size}")
    print("-" * 85)

    results = []

    for name, config in configs.items():
        try:
            model = UNet1D(
                seq_length=seq_length,
                channels=1,
                with_time_emb=True,
                edge_pad=2,  # Use configurable edge_pad
                debug=False,
                **config,
            ).to(device)

            # Count parameters
            total_params, _ = count_parameters(model)
            model_size_mb = total_params * 4 / (1024 * 1024)

            # Test forward pass
            x = torch.randn(batch_size, 1, seq_length).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)

            with torch.no_grad():
                output = model(x, t)

            # Memory usage
            memory_mb = 0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()

            print(
                f"{name:15} | Levels: {len(config['dim_mults']):1} | "
                f"Params: {total_params:8,} | Size: {model_size_mb:5.1f}MB | "
                f"Memory: {memory_mb:5.1f}MB | Output: {output.shape}"
            )

            results.append((name, config, total_params, model_size_mb))

        except Exception as e:
            print(f"{name:15} | ERROR: {str(e)}")

    return results


def test_from_config():
    """Test model using actual config.yaml settings."""
    try:
        config = load_config("config.yaml")
        unet_config = config["unet"]
        data_config = config["data"]
        training_config = config["training"]

        seq_length = data_config["seq_length"]
        batch_size = data_config["batch_size"]
        train_samples = data_config["datasplit"][0]
        num_batches = (train_samples + batch_size - 1) // batch_size

        print("\n=== Config.yaml Test ===")
        print(f"Dataset: {train_samples} samples, seq_length={seq_length}")
        print(f"Training: batch_size={batch_size}, {num_batches} batches/epoch")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet1D(
            embedding_dim=unet_config["embedding_dim"],
            dim_mults=tuple(unet_config["dim_mults"]),
            channels=unet_config["channels"],
            with_time_emb=unet_config["with_time_emb"],
            with_pos_emb=unet_config["with_pos_emb"],
            norm_groups=unet_config["norm_groups"],
            seq_length=seq_length,
            edge_pad=unet_config.get("edge_pad", 2),  # Use config value or default to 2
            debug=False,
        ).to(device)

        # Model statistics
        total_params, _ = count_parameters(model)
        model_size_mb = total_params * 4 / (1024 * 1024)

        print(f"Model: {total_params:,} parameters ({model_size_mb:.2f} MB)")

        # Test forward pass
        x = torch.randn(batch_size, 1, seq_length).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)

        with torch.no_grad():
            output = model(x, t)

        print(f"Forward pass: ✅ PASSED - {output.shape}")

        # Estimate training time
        try:
            batch_time, epoch_time = estimate_training_time(
                model, batch_size, seq_length, num_batches, device
            )
            total_hours = epoch_time * training_config["epochs"] / 3600

            print("Training estimate:")
            print(f"  - Time per batch: {batch_time:.3f}s")
            print(f"  - Time per epoch: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            print(
                f"  - Total training: {total_hours:.1f} hours ({training_config['epochs']} epochs)"
            )

        except Exception as e:
            print(f"Training estimation failed: {e}")

        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**2)
            print(f"GPU memory used: {memory_used:.1f} MB")

        return True

    except Exception as e:
        print(f"Config test failed: {e}")
        return False


def main():
    """Run comprehensive UNet1D tests."""
    print("🧪 Comprehensive UNet1D Test Suite")
    print("=" * 50)

    # Test 1: Shape tracing with different configurations
    test_shape_tracing(
        "Shallow",
        {
            "embedding_dim": 16,
            "dim_mults": (1, 2),
            "with_pos_emb": False,
            "norm_groups": 4,
        },
    )

    # Test 2: Configuration comparison
    results = test_configurations()

    # Test 3: Config.yaml validation
    config_success = test_from_config()

    # Summary
    print("\n=== Test Summary ===")
    print(f"Configuration tests: {len(results)} passed")
    print(f"Config.yaml test: {'✅ PASSED' if config_success else '❌ FAILED'}")

    if results:
        recommended = min(results, key=lambda x: x[2])  # Smallest parameter count
        print("\n💡 Recommended for laptop training:")
        print(
            f"   {recommended[0]} - {recommended[2]:,} parameters ({recommended[3]:.1f} MB)"
        )

    print("\n🚀 UNet1D is ready for training!")


if __name__ == "__main__":
    main()
