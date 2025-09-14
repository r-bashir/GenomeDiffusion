#!/usr/bin/env python3
"""
Test script to verify memory efficiency improvements for long sequences.
Tests the UNet1D model with gradient checkpointing and memory optimizations.
"""

import os

import psutil
import torch

from src.unet import UNet1D
from src.utils import load_config


def get_memory_usage():
    """Get current GPU and CPU memory usage."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        gpu_memory = gpu_reserved = 0

    cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**3  # GB
    return gpu_memory, gpu_reserved, cpu_memory


def test_model_memory_efficiency():
    """Test UNet1D model with different sequence lengths and optimizations."""

    print("🧪 Testing UNet1D Memory Efficiency")
    print("=" * 50)

    # Load optimized config
    config = load_config("config.yaml")

    # Test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1  # Small batch for memory testing

    # Test different sequence lengths
    test_lengths = [1000, 10000, 50000, 100000, 160858]

    for seq_len in test_lengths:
        print(f"\n📏 Testing sequence length: {seq_len:,}")

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get initial memory
            gpu_before, gpu_reserved_before, cpu_before = get_memory_usage()

            # Create model with optimized settings
            model_config = config["unet"].copy()
            model_config["seq_length"] = seq_len

            model = UNet1D(
                embedding_dim=model_config["embedding_dim"],
                dim_mults=model_config["dim_mults"],
                channels=model_config["channels"],
                with_time_emb=model_config["with_time_emb"],
                with_pos_emb=model_config["with_pos_emb"],
                norm_groups=model_config["norm_groups"],
                seq_length=seq_len,
                edge_pad=model_config["edge_pad"],
                enable_checkpointing=model_config["enable_checkpointing"],
                use_attention=model_config["use_attention"],
                attention_type=model_config["attention_type"],
                attention_heads=model_config["attention_heads"],
                attention_dim_head=model_config["attention_dim_head"],
                attention_window=model_config["attention_window"],
                num_global_tokens=model_config["num_global_tokens"],
            ).to(device)

            model.train()

            # Create test data
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            x = torch.randn(batch_size, 1, seq_len, device=device, dtype=dtype)
            t = torch.randint(0, 1000, (batch_size,), device=device)

            # Get memory after model creation
            gpu_after_model, gpu_reserved_after_model, cpu_after_model = (
                get_memory_usage()
            )

            print(
                f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}"
            )
            print(
                f"   Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.1f} MB"
            )

            # Forward pass with gradient checkpointing
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(x, t)

            # Get memory after forward pass
            gpu_after_forward, gpu_reserved_after_forward, cpu_after_forward = (
                get_memory_usage()
            )

            # Backward pass (simulate training)
            loss = output.mean()
            loss.backward()

            # Get memory after backward pass
            gpu_after_backward, gpu_reserved_after_backward, cpu_after_backward = (
                get_memory_usage()
            )

            # Print memory usage
            print("   ✅ SUCCESS - Memory usage:")
            print(f"      GPU allocated: {gpu_after_backward:.2f} GB")
            print(f"      GPU reserved:  {gpu_reserved_after_backward:.2f} GB")
            print(f"      CPU memory:    {cpu_after_backward:.2f} GB")
            print(f"      Output shape:  {output.shape}")

            # Memory efficiency metrics
            memory_per_element = (
                gpu_after_backward * 1024**3 / seq_len
            )  # bytes per sequence element
            print(f"      Memory/element: {memory_per_element:.1f} bytes")

            # Clean up
            del model, x, t, output, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ OOM ERROR at sequence length {seq_len:,}")
                print(f"      Error: {str(e)}")
                # Clear cache and continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"   ❌ ERROR: {str(e)}")
                break
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR: {str(e)}")
            break

    print("\n🎯 Memory Efficiency Test Complete!")
    print("💡 Tips for further optimization:")
    print("   - Use batch_size=1 for sequences >100K")
    print("   - Enable gradient accumulation (accumulate_grad_batches=8)")
    print("   - Use 16-bit mixed precision")
    print("   - Consider reducing embedding_dim further if needed")


def test_attention_memory_scaling():
    """Test how attention memory scales with sequence length."""

    print("\n🔍 Testing Attention Memory Scaling")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different attention configurations
    attention_configs = [
        {"type": "linear", "window_size": 512, "heads": 4},
        {"type": "linear", "window_size": 1024, "heads": 4},
        {"type": "linear", "window_size": 2048, "heads": 4},
    ]

    test_lengths = [10000, 50000, 100000]

    for config_idx, attn_config in enumerate(attention_configs):
        print(f"\n📋 Config {config_idx + 1}: {attn_config}")

        for seq_len in test_lengths:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Create minimal model for attention testing
                from src.unet import LinearAttention1D

                dim = 32
                attention = LinearAttention1D(
                    dim=dim,
                    heads=attn_config["heads"],
                    dim_head=32,
                ).to(device)

                # Test data
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                x = torch.randn(1, dim, seq_len, device=device, dtype=dtype)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    output = attention(x)

                gpu_memory, _, _ = get_memory_usage()
                print(f"   Seq {seq_len:>6,}: {gpu_memory:.3f} GB")

                del attention, x, output

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   Seq {seq_len:>6,}: OOM")
                else:
                    print(f"   Seq {seq_len:>6,}: Error - {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("🚀 Starting Memory Efficiency Tests")

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("💻 Running on CPU")

    # Run tests
    test_model_memory_efficiency()
    test_attention_memory_scaling()

    print("\n✨ All tests completed!")
