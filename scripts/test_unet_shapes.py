"""
Test script to demonstrate UNet1D shape transformations with a mock sequence.
Shows tensor shapes at each stage of the model's forward pass.
"""

import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unet import UNet1D


def test_unet_shapes():
    # Model configuration
    batch_size = 2
    seq_length = 64  # Small sequence for demonstration
    channels = 1
    embedding_dim = 32  # Memory-efficient setting
    dim_mults = (1, 2, 4)  # 3 downsampling steps
    edge_pad = 2  # Use configurable edge_pad

    print("\n=== UNet1D Shape Test ===")
    print(f"Configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Embedding dim: {embedding_dim}")
    print(f"- Dimension multipliers: {dim_mults}")
    print(f"- Edge padding: {edge_pad}")

    # Calculate expected shapes at each level
    expected_shapes = calculate_expected_shapes(
        batch_size, seq_length, embedding_dim, dim_mults, edge_pad
    )

    print(f"\n=== Expected Shape Flow ===")
    for stage, shape_info in expected_shapes.items():
        print(f"- {stage}: {shape_info}")

    # Create model with debug mode enabled
    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        seq_length=seq_length,
        edge_pad=edge_pad,
        debug=True,  # Enable shape tracing
    )

    # Create mock input
    x = torch.randn(batch_size, channels, seq_length)
    t = torch.randint(0, 1000, (batch_size,))  # Random timesteps

    print("\n=== Forward Pass with Shape Verification ===")
    print(f"Input shape: {x.shape}")

    # Use the existing debug output from the model
    print("\n--- Debug Output from UNet1D ---")
    with torch.no_grad():
        out = model(x, t)
    print("--- End Debug Output ---\n")

    print("\n=== Basic Shape Verification ===")
    # Verify basic shape consistency
    input_shape = x.shape
    output_shape = out.shape

    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {output_shape}")

    # Key verifications
    verification_passed = True

    # 1. Output shape matches input shape
    if output_shape == input_shape:
        print("✅ Output shape matches input shape")
    else:
        print(f"❌ Output shape {output_shape} doesn't match input {input_shape}")
        verification_passed = False

    # 2. Batch dimension preserved
    if output_shape[0] == input_shape[0]:
        print("✅ Batch dimension preserved")
    else:
        print(f"❌ Batch dimension changed: {input_shape[0]} -> {output_shape[0]}")
        verification_passed = False

    # 3. Channel dimension correct (should be 1 for SNP data)
    if output_shape[1] == 1:
        print("✅ Output channels correct (1 for SNP data)")
    else:
        print(f"❌ Output channels incorrect: expected 1, got {output_shape[1]}")
        verification_passed = False

    # 4. Sequence length preserved
    if output_shape[2] == input_shape[2]:
        print("✅ Sequence length preserved")
    else:
        print(f"❌ Sequence length changed: {input_shape[2]} -> {output_shape[2]}")
        verification_passed = False

    print("\n=== Final Verification ===")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Final shape assertion
    assert (
        out.shape == x.shape
    ), f"❌ Output shape {out.shape} doesn't match input shape {x.shape}"
    print("✅ Shape test passed: output matches input dimensions")

    # Verify output is not identical to input (model actually processed it)
    assert not torch.allclose(
        out, x, atol=1e-6
    ), "❌ Output is identical to input - model may not be working"
    print("✅ Output verification passed: model produced different output from input")

    # Overall verification result
    if verification_passed:
        print("✅ All intermediate shape verifications passed!")
    else:
        print("❌ Some intermediate shape verifications failed!")
        return False

    return True


def calculate_expected_shapes(
    batch_size, seq_length, embedding_dim, dim_mults, edge_pad
):
    """Calculate expected tensor shapes at each stage of UNet1D forward pass."""
    expected = {}

    # Input
    expected["Input"] = f"[{batch_size}, 1, {seq_length}]"

    # After padding
    padded_length = seq_length + 2 * edge_pad
    expected["After padding"] = f"[{batch_size}, 1, {padded_length}]"

    # After initial conv
    initial_dim = embedding_dim * dim_mults[0]
    expected["After initial conv"] = f"[{batch_size}, {initial_dim}, {padded_length}]"

    # Encoder path
    current_length = padded_length
    current_dim = initial_dim

    for i, mult in enumerate(dim_mults):
        # After encoder blocks
        expected[f"After encoder block {i}"] = (
            f"[{batch_size}, {current_dim}, {current_length}]"
        )

        # After downsampling
        if i < len(dim_mults) - 1:  # Don't downsample at the last level
            current_length = current_length // 2
            next_dim = embedding_dim * dim_mults[i + 1]
            expected[f"After downsampling {i}"] = (
                f"[{batch_size}, {next_dim}, {current_length}]"
            )
            current_dim = next_dim

    # Bottleneck
    expected["Bottleneck"] = f"[{batch_size}, {current_dim}, {current_length}]"

    # Decoder path (reverse of encoder)
    decoder_dims = [embedding_dim * mult for mult in reversed(dim_mults[:-1])]
    decoder_lengths = []

    # Calculate decoder lengths (upsampling)
    temp_length = current_length
    for i in range(len(dim_mults) - 1):
        temp_length = temp_length * 2
        decoder_lengths.append(temp_length)

    for i, (dim, length) in enumerate(zip(decoder_dims, decoder_lengths)):
        expected[f"After decoder block {i}"] = f"[{batch_size}, {dim}, {length}]"

    # Final output (after removing padding)
    expected["Final output"] = f"[{batch_size}, 1, {seq_length}]"

    return expected


if __name__ == "__main__":
    test_unet_shapes()
