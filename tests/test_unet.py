import pytest
import torch

from src.unet import UNet1D


@pytest.mark.parametrize(
    "batch,channels,seq_length",
    [
        (2, 1, 32),  # basic
        (1, 1, 16),  # batch size 1 (should fail)
        (4, 1, 33),  # odd seq length
        (3, 1, 64),  # larger
        (2, 1, 8),  # very short (should fail)
    ],
)
def test_unet1d_various_shapes(batch, channels, seq_length):
    embedding_dim = 8
    dim_mults = (1, 2, 4)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        with_time_emb=True,
        with_pos_emb=True,
        seq_length=seq_length,
        debug=False,
    )
    edge_pad = 4
    min_len = seq_length + 2 * edge_pad
    for _ in range(len(dim_mults)):
        min_len = (min_len + 1) // 2
    if min_len <= edge_pad:
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
    model = UNet1D(seq_length=seq_length, channels=channels)
    with pytest.raises(AssertionError):
        model(x, t)


def test_unet1d_gradient_checkpointing():
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length, channels=channels, dim_mults=(1, 2, 4), debug=False
    )
    model.gradient_checkpointing_enable()
    y = model(x, t)
    assert y.shape == x.shape


def test_unet1d_device_compatibility():
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length, channels=channels, dim_mults=(1, 2, 4), debug=False
    )
    if torch.cuda.is_available():
        x = x.cuda()
        t = t.cuda()
        model = model.cuda()
    y = model(x, t)
    assert y.shape == x.shape
    assert y.device == x.device


def test_unet1d_reproducibility():
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    torch.manual_seed(42)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        seq_length=seq_length, channels=channels, dim_mults=(1, 2, 4), debug=False
    )
    y1 = model(x, t)
    torch.manual_seed(42)
    x2 = torch.randn(batch, channels, seq_length)
    t2 = torch.randint(0, 1000, (batch,))
    model2 = UNet1D(
        seq_length=seq_length, channels=channels, dim_mults=(1, 2, 4), debug=False
    )
    y2 = model2(x2, t2)
    assert torch.allclose(y1, y2), "Model output is not reproducible with fixed seed"


def test_unet1d_shape_tracing():
    # Shape [B, C, L]
    batch, channels, seq_length = 2, 1, 32

    embedding_dim = 8
    dim_mults = (1, 2, 4)
    x = torch.randn(batch, channels, seq_length)
    t = torch.randint(0, 1000, (batch,))
    model = UNet1D(
        embedding_dim=embedding_dim,
        dim_mults=dim_mults,
        channels=channels,
        with_time_emb=True,
        with_pos_emb=True,
        seq_length=seq_length,
        debug=True,
    )
    y = model(x, t)
    assert y.shape == x.shape, f"Output shape {y.shape} does not match input {x.shape}"


if __name__ == "__main__":
    test_unet1d_shape_tracing()
    print("UNet1D shape tracing test passed.")
