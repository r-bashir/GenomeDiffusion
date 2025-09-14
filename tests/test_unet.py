import pytest
import torch

from src.unet import UNet1D


@pytest.mark.parametrize(
    "batch,channels,seq_length,dim_mults,edge_pad",
    [
        (2, 1, 32, (1, 2), 2),  # basic - sufficient length for 2 downsampling steps
        (1, 1, 16, (1, 2), 2),  # batch size 1 - sufficient length
        (4, 1, 33, (1, 2), 2),  # odd seq length - sufficient length
        (3, 1, 64, (1, 2, 4), 2),  # larger - sufficient for 3 downsampling steps
        (2, 1, 32, (1, 2, 4), 2),  # now works with edge_pad=2!
        (2, 1, 8, (1, 2, 4), 2),  # too short for 3 steps (should fail)
        (2, 1, 64, (1, 2), 4),  # test with larger edge_pad
        (2, 1, 32, (1, 2), 4),  # should fail with edge_pad=4
    ],
)
def test_unet1d_various_shapes(batch, channels, seq_length, dim_mults, edge_pad):
    embedding_dim = 8
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        with_time_emb=True,
        with_pos_emb=True,
        seq_length=seq_length,
        edge_pad=edge_pad,
        debug=False,
    )
    # Check if sequence length is sufficient for the number of downsampling steps
    min_len = seq_length
    sufficient_length = True
    for i in range(len(dim_mults)):
        min_len = (min_len + 1) // 2
        if min_len <= edge_pad:
            sufficient_length = False
            break

    if not sufficient_length:
        with pytest.raises(ValueError):
            model(x, t)
    else:
        y = model(x, t)
        assert y.shape == x.shape


@pytest.mark.parametrize(
    "with_time_emb,with_pos_emb",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_unet1d_embedding_configs(with_time_emb, with_pos_emb):
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        embedding_dim=8,
        dim_mults=(1, 2),
        channels=channels,
        with_time_emb=with_time_emb,
        with_pos_emb=with_pos_emb,
        seq_length=seq_length,
        debug=False,
    )
    y = model(x, t)
    assert y.shape == x.shape


def test_unet1d_error_on_wrong_channels():
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, 2, seq_length)  # wrong channel dim
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(seq_length=seq_length, channels=channels, edge_pad=2)
    with pytest.raises(AssertionError):
        model(x, t)


def test_unet1d_configurable_edge_pad():
    """Test that edge_pad is configurable and affects validation correctly."""
    batch, channels, seq_length = 2, 1, 32
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))

    # Test with edge_pad=2 (should work)
    model_pad2 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        debug=False,
    )
    y = model_pad2(x, t)
    assert y.shape == x.shape
    assert model_pad2.edge_pad == 2

    # Test with edge_pad=4 (should work for this length)
    model_pad4 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=4,
        debug=False,
    )
    y = model_pad4(x, t)
    assert y.shape == x.shape
    assert model_pad4.edge_pad == 4

    # Test with edge_pad=8 (should fail for seq_length=32)
    model_pad8 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=8,
        debug=False,
    )
    with pytest.raises(ValueError):
        model_pad8(x, t)


def test_unet1d_gradient_checkpointing():
    # Shape [B, C, L] - now works with shorter sequences!
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        debug=False,
    )
    model.gradient_checkpointing_enable()
    y = model(x, t)
    assert y.shape == x.shape


def test_unet1d_device_compatibility():
    # Shape [B, C, L] - now works with shorter sequences!
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        debug=False,
    )
    if torch.cuda.is_available():
        x = x.cuda()
        t = t.cuda()
        model = model.cuda()
    y = model(x, t)
    assert y.shape == x.shape
    assert y.device == x.device


def test_unet1d_reproducibility():
    # Shape [B, C, L] - now works with shorter sequences!
    batch, channels, seq_length = 2, 1, 32

    torch.manual_seed(42)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        debug=False,
    )
    y1 = model(x, t)
    torch.manual_seed(42)
    x2 = torch.randn(batch, channels, seq_length)
    t2 = torch.randint(0, 1000, (batch,))
    model2 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        debug=False,
    )
    y2 = model2(x2, t2)
    assert torch.allclose(y1, y2), "Model output is not reproducible with fixed seed"


def test_unet1d_shape_tracing():
    """Comprehensive shape tracing test with debug output."""
    # Shape [B, C, L] - now works with shorter sequences!
    batch, channels, seq_length = 2, 1, 32

    embedding_dim = 8
    dim_mults = (1, 2)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))

    print("\n=== UNet1D Shape Tracing Test ===")
    print("Configuration:")
    print(f"- Batch size: {batch}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Embedding dim: {embedding_dim}")
    print(f"- Dimension multipliers: {dim_mults}")
    print("- Edge padding: 2")

    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        with_time_emb=True,
        with_pos_emb=True,
        seq_length=seq_length,
        edge_pad=2,
        debug=True,  # Enable detailed shape tracing
    )

    print("\n=== Forward Pass with Shape Tracing ===")
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        y = model(x, t)

    print("\n=== Output Verification ===")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, f"Output shape {y.shape} does not match input {x.shape}"
    print("✅ Shape test passed: output matches input dimensions")


def test_unet1d_detailed_shape_validation():
    """Test shape validation with different configurations."""
    configurations = [
        {"name": "Ultra-shallow", "seq_len": 16, "dim_mults": (1,), "edge_pad": 2},
        {"name": "Shallow", "seq_len": 32, "dim_mults": (1, 2), "edge_pad": 2},
        {"name": "Medium", "seq_len": 64, "dim_mults": (1, 2, 4), "edge_pad": 2},
        {"name": "Edge_pad=4", "seq_len": 64, "dim_mults": (1, 2), "edge_pad": 4},
    ]

    for config in configurations:
        batch_size = 2
        x = torch.randn(batch_size, 1, config["seq_len"])
        t = torch.randint(0, 1000, (batch_size,))

        model = UNet1D(
            embedding_dim=8,
            dim_mults=config["dim_mults"],
            channels=1,
            seq_length=config["seq_len"],
            edge_pad=config["edge_pad"],
            debug=False,
        )

        with torch.no_grad():
            y = model(x, t)

        assert (
            y.shape == x.shape
        ), f"{config['name']}: Output shape {y.shape} != input {x.shape}"
        print(f"✅ {config['name']}: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_unet1d_shape_tracing()
    print("UNet1D shape tracing test passed.")
