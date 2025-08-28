import pytest
import torch

from src.unet import Attention1D, LinearAttention1D, SparseAttention1D


@pytest.mark.parametrize(
    "attn_cls,kwargs",
    [
        (Attention1D, {}),
        (LinearAttention1D, {}),
        (SparseAttention1D, {"window_size": 64}),
    ],
)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("C", [8, 16])
@pytest.mark.parametrize("L", [16, 57, 128, 257, 512, 1000])
@torch.no_grad()
def test_attention_preserves_shape(attn_cls, kwargs, B, C, L):
    torch.manual_seed(0)
    x = torch.randn(B, C, L)

    # heads * dim_head must match C
    # choose heads that divide C and dim_head accordingly
    # prefer 4 heads when possible, otherwise fall back to 2 or 1
    if C % 4 == 0:
        heads = 4
    elif C % 2 == 0:
        heads = 2
    else:
        heads = 1
    dim_head = C // heads

    attn = attn_cls(dim=C, heads=heads, dim_head=dim_head, **kwargs)
    y = attn(x)

    assert (
        y.shape == x.shape
    ), f"{attn_cls.__name__} changed shape: in {tuple(x.shape)} -> out {tuple(y.shape)}"


@torch.no_grad()
def test_sparse_attention_handles_non_multiple_window():
    # L not a multiple of window_size
    B, C, L = 2, 12, 250
    window_size = 64

    # make heads divide C
    heads = 4 if C % 4 == 0 else 3 if C % 3 == 0 else 1
    dim_head = C // heads

    x = torch.randn(B, C, L)
    attn = SparseAttention1D(
        dim=C, heads=heads, dim_head=dim_head, window_size=window_size
    )
    y = attn(x)

    assert y.shape == x.shape
