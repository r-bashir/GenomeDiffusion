import pytest
import torch

from src.unet import UNet1D


@pytest.mark.parametrize(
    "batch,channels,seq_length,dim_mults,edge_pad",
    [
        (2, 1, 32, (1, 2), 2),  # basic - sufficient for 2 downsampling steps
        (1, 1, 16, (1, 2), 2),  # batch size 1
        (4, 1, 33, (1, 2), 2),  # odd seq length
        (3, 1, 64, (1, 2, 4), 2),  # deeper
        (2, 1, 8, (1, 2, 4), 2),  # too short for 3 steps (should fail)
        (2, 1, 64, (1, 2), 4),  # larger edge_pad works
        (2, 1, 32, (1, 2), 4),  # should fail with edge_pad=4
    ],
)
def test_unet1d_various_shapes(batch, channels, seq_length, dim_mults, edge_pad):
    embedding_dim = 8
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    # Use strict resize for even lengths; allow non-strict (zero pad/crop) for odd lengths
    strict = seq_length % 2 == 0
    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        with_time_emb=True,
        with_pos_emb=True,
        seq_length=seq_length,
        edge_pad=edge_pad,
        strict_resize=strict,
        use_attention=False,
    )
    # Check if sequence length is sufficient for the number of downsampling steps
    min_len = seq_length
    sufficient_length = True
    for _ in range(len(dim_mults)):
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
        strict_resize=True,
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
        strict_resize=True,
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
        strict_resize=True,
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
        strict_resize=True,
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
        strict_resize=True,
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
        strict_resize=True,
    )
    if torch.cuda.is_available():
        x = x.cuda()
        t = t.cuda()
        model = model.cuda()
    y = model(x, t)
    assert y.shape == x.shape
    assert y.device == x.device


def test_unet1d_reproducibility():
    """Given a fixed seed, identical models and inputs produce identical outputs."""
    batch, channels, seq_length = 2, 1, 32

    torch.manual_seed(42)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))

    torch.manual_seed(123)
    model1 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        strict_resize=True,
        use_attention=False,
    )
    torch.manual_seed(123)
    model2 = UNet1D(
        seq_length=seq_length,
        channels=channels,
        dim_mults=(1, 2),
        edge_pad=2,
        strict_resize=True,
        use_attention=False,
    )
    y1 = model1(x, t)
    y2 = model2(x, t)
    assert torch.allclose(y1, y2), "Model output is not reproducible with fixed seed"


def test_unet1d_strict_resize_and_padding_behavior():
    """Directly test the internal _resize_to_length behavior for strict and non-strict modes."""
    x = torch.randn(2, 4, 10)
    model_strict = UNet1D(seq_length=32, dim_mults=(1, 2), strict_resize=True)
    with pytest.raises(RuntimeError):
        _ = model_strict._resize_to_length(x, 12)

    model_nonstrict = UNet1D(
        seq_length=32, dim_mults=(1, 2), strict_resize=False, pad_value=0.0
    )
    y = model_nonstrict._resize_to_length(x, 12)
    assert y.shape[-1] == 12
    assert torch.allclose(
        y[..., -2:], torch.zeros_like(y[..., -2:])
    ), "Padding should be zeros at the right"

    z = model_nonstrict._resize_to_length(x, 8)
    assert z.shape[-1] == 8


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
            strict_resize=True,
        )

        with torch.no_grad():
            y = model(x, t)

        assert (
            y.shape == x.shape
        ), f"{config['name']}: Output shape {y.shape} != input {x.shape}"


def test_unet1d_contains_expected_nonlinearities():
    """Ensure the model contains SiLU activations in key modules when attention is off."""
    model = UNet1D(seq_length=32, dim_mults=(1, 2), use_attention=False)
    activations = [m for m in model.modules() if isinstance(m, torch.nn.SiLU)]
    assert len(activations) > 0, "Expected SiLU activations present in UNet1D"
