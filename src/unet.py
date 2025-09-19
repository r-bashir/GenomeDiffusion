#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""Based on `https://huggingface.co/blog/annotated-diffusion` that explains
using the original DDPM by Ho et al. 2022 on images i.e. 2D dataset. We
adapted the code for 1-dimensional SNP genomic dataset. It uses dual skip
connections, which is slightly different from traditional single-skip U-Net.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.sinusoidal_embedding import (
    SinusoidalPositionEmbeddings,
    SinusoidalTimeEmbeddings,
)


class Residual(nn.Module):
    """Residual connection wrapper: output = fn(x) + x."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class DownsampleConv(nn.Module):
    """1D downsampling with robust odd-length sequence handling."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.conv = nn.Conv1d(dim, dim_out, 4, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        # Handle odd sequence lengths with ZERO padding (genomic ends should not mirror)
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (1, 0), mode="constant", value=0.0)
        x = self.conv(x)
        return self.act(x)


class UpsampleConv(nn.Module):
    """1D upsampling using transposed convolution for exact reconstruction."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.conv = nn.ConvTranspose1d(dim, dim_out, 4, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        # Keep symmetry with downsampling by adding a lightweight activation
        x = self.conv(x)
        return self.act(x)


# ========== ResnetBlock Modules ==========
# The canonical form that uses FiLM Conditioning (in the first ConvBlock) and Dropout
# (in both ConvBlocks). The approach is used by Kenneweg et al. (2025) for UNet/DDPM.


class ConvBlock(nn.Module):
    """
    Basic 1D convolutional block: Conv1D + GroupNorm + (FiLM) + SiLU + (Dropout).
    Flow: x → Conv1d → GroupNorm → (FiLM conditioning) → SiLU → (Dropout) → y
    """

    def __init__(self, dim, dim_out, groups=8, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # New: FiLM conditioning if scale & shift are provided
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    """
    1D ResNet block with FiLM-style time embedding conditioning (for DDPM).
    Flow: output = block2(block1(x, scale_shift)) + res_conv(x)

    Where:
        - x: [B, C_in, L] input sequence
        - time_emb (optional): [B, time_dim] embedding
            -> projected by Linear -> SiLU              # New
            -> reshaped and split into (scale, shift)   # New
            -> injected into GroupNorm inside block1    # New
        - res_conv: 1x1 Conv1d if C_in != C_out, else identity
        - block1, block2: Conv1d → GroupNorm → (FiLM if available) → SiLU
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        time_dim=None,
        groups=8,
        dropout=0.0,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.use_scale_shift_norm = use_scale_shift_norm
        if time_dim is not None:
            if self.use_scale_shift_norm:
                # FiLM: produce (scale, shift) — Linear -> SiLU
                self.mlp = nn.Sequential(nn.Linear(time_dim, dim_out * 2), nn.SiLU())
            else:
                # Additive: unified order with FiLM — Linear -> SiLU
                self.mlp = nn.Sequential(nn.Linear(time_dim, dim_out), nn.SiLU())
        else:
            self.mlp = None

        # New: ConvBlocks now support optional scale_shift
        self.block1 = ConvBlock(dim_in, dim_out, groups=groups, dropout=dropout)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups, dropout=dropout)
        self.res_conv = (
            nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        # FiLM (scale-shift) mode: inject inside block1's GroupNorm
        if self.use_scale_shift_norm:
            scale_shift = None
            if self.mlp is not None and time_emb is not None:
                t_proj = self.mlp(time_emb)  # [B, 2*C]
                t_proj = t_proj.unsqueeze(-1)  # [B, 2*C, 1]
                scale_shift = t_proj.chunk(2, dim=1)  # (scale, shift)
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h)
        else:
            # Additive mode: apply after block1, before block2
            h = self.block1(x, scale_shift=None)
            if self.mlp is not None and time_emb is not None:
                t_proj = self.mlp(time_emb)  # [B, C]
                t_proj = t_proj.unsqueeze(-1)  # [B, C, 1]
                h = h + t_proj
            h = self.block2(h)

        return h + self.res_conv(x)


# ========== Attention Modules ==========
class PreNorm(nn.Module):
    """Pre-normalization wrapper for improved training stability."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention1D(nn.Module):
    """Full self-attention with O(n²) complexity but highest expressiveness.

    Captures all pairwise interactions but impractical for very long sequences.
    Best for short sequences (<5k SNPs) where memory allows.

    Args:
        dim (int): Input dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension per head
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        assert n > 0, "Sequence length must be positive"
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.view(b, -1, n)
        return self.to_out(out)


class LinearAttention1D(nn.Module):
    """Linear attention with O(n) complexity but reduced expressiveness.

    Memory-efficient alternative that approximates attention via factorization.
    Best for extremely long sequences where memory is critical.

    Args:
        dim (int): Input dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension per head
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, n = x.shape
        assert n > 0, "Sequence length must be positive"
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)

        # Apply scaling before softmax
        q = q * self.scale
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.view(b, -1, n)
        return self.to_out(out)


class SparseAttention1D(nn.Module):
    """Windowed attention with O(n×w) complexity and near-full expressiveness.

    Processes sequence in local windows (size w) with optional global tokens.
    Best balance for long sequences (10k-200k SNPs).

    Args:
        dim (int): Input dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension per head
        window_size (int): Local attention window size
        num_global_tokens (int): Global tokens for cross-window attention
    """

    def __init__(self, dim, heads=4, dim_head=32, window_size=512, num_global_tokens=0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

        if num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, dim))

    def forward(self, x):
        b, c, n = x.shape
        assert n > 0, "Sequence length must be positive"
        assert self.window_size > 0, "Window size must be positive"
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)
        q = q * self.scale

        # Process in local windows with fixed dimension handling
        out = torch.zeros_like(v)
        for i in range(0, n, self.window_size):
            j = min(i + self.window_size, n)

            # Use consistent window boundaries
            q_local = q[..., i:j]
            k_local = k[..., i:j]  # Same window as q
            v_local = v[..., i:j]  # Same window as q

            sim = torch.einsum("b h d i, b h d j -> b h i j", q_local, k_local)
            attn = sim.softmax(dim=-1)
            out[..., i:j] = torch.einsum("b h i j, b h d j -> b h d i", attn, v_local)

        return self.to_out(out.view(b, -1, n))


# ========== UNet1D Architecture for Genomic Diffusion Models ==========
class UNet1D(nn.Module):
    """
    1D U-Net for genomic SNP sequence modeling in diffusion models.

    A time-conditional U-Net architecture designed for denoising genomic
    sequences in diffusion-based generative models. Processes SNP data
    as 1D sequences with shape [B, C, L] where C is always equal to 1.
    """

    def __init__(
        self,
        emb_dim=512,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        with_time_emb=True,
        time_dim=128,
        with_pos_emb=True,
        pos_dim=128,
        norm_groups=8,
        seq_length=160858,
        edge_pad=2,
        enable_checkpointing=True,
        strict_resize=False,
        pad_value=0.0,
        dropout=0.0,
        use_scale_shift_norm=False,
        use_attention=False,
        attention_heads=4,
        attention_dim_head=32,
        **kwargs,
    ):
        super().__init__()

        # Base parameters
        self.emb_dim = emb_dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.time_dim = time_dim
        self.with_pos_emb = with_pos_emb
        self.pos_dim = pos_dim
        self.norm_groups = norm_groups
        self.seq_length = seq_length
        self.edge_pad = edge_pad
        self.use_gradient_checkpointing = enable_checkpointing
        self.strict_resize = strict_resize
        self.pad_value = pad_value
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.attention_dim_head = attention_dim_head

        # INITIAL CONVOLUTION
        init_dim = 16
        kernel_size = 7
        padding = (kernel_size - 1) // 2
        self.init_conv = nn.Conv1d(
            channels, init_dim, kernel_size=kernel_size, padding=padding
        )  # input: [B, 1, L] → output: [B, init_dim, L]

        # STANDARD FEATURE DIMENSIONS
        # dims shape: [16, 16, 32, 64, 128]
        # dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        # in_out shape: [(16,16), (16,32), (32,64), (64,128)]
        # in_out = list(zip(dims[:-1], dims[1:]))

        # PROGRESSIVE FEATURE DIMENSIONS, dim_mults > 8 are clamped to 256 and 512
        # dims shape: [16, 16, 32, 64, 128]
        dims = [init_dim]
        for i, mult in enumerate(dim_mults):
            max_dim = 256 if i < 2 else 512
            dims.append(min(init_dim * mult, max_dim))
        # in_out shape: [(16,16), (16,32), (32,64), (64,128)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # TIME EMBEDDINGS
        if self.with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.emb_dim),
                nn.Linear(self.emb_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
        else:
            self.time_dim = None
            self.time_mlp = None

        # POSITION EMBEDDINGS
        if self.with_pos_emb:
            self.pos_emb = SinusoidalPositionEmbeddings(self.pos_dim)
            self.pos_proj = nn.Conv1d(self.pos_dim, init_dim, 1)
            self.pos_alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.pos_emb = None
            self.pos_proj = None
            self.pos_alpha = None

        # ===== UNet1D ARCHITECTURE =====
        num_resolutions = len(in_out)
        resnet_block = partial(
            ResnetBlock,
            groups=self.norm_groups,
            dropout=self.dropout,
            use_scale_shift_norm=self.use_scale_shift_norm,
        )

        # ENCODER / DOWNSAMPLING
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # LinearAttention1D for encoder (memory-efficient for long sequences)
            attn_block = (
                Residual(
                    PreNorm(
                        dim_in,
                        LinearAttention1D(
                            dim_in,
                            heads=self.attention_heads,
                            dim_head=self.attention_dim_head,
                        ),
                    )
                )
                if self.use_attention
                else nn.Identity()
            )

            # Downsampling Path
            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in, time_dim=self.time_dim),
                        resnet_block(dim_in, dim_in, time_dim=self.time_dim),
                        attn_block,
                        (
                            DownsampleConv(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        # BOTTLENECK
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_dim=self.time_dim)

        # Attention1D for bottleneck (full attention for maximum expressiveness)
        self.mid_attn = (
            Residual(
                PreNorm(
                    mid_dim,
                    Attention1D(
                        mid_dim,
                        heads=self.attention_heads,
                        dim_head=self.attention_dim_head,
                    ),
                )
            )
            if self.use_attention
            else nn.Identity()
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_dim=self.time_dim)

        # DECODER / UPSAMPLING
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # LinearAttention1D for decoder (memory-efficient for long sequences)
            attn_block = (
                Residual(
                    PreNorm(
                        dim_out,
                        LinearAttention1D(
                            dim_out,
                            heads=self.attention_heads,
                            dim_head=self.attention_dim_head,
                        ),
                    )
                )
                if self.use_attention
                else nn.Identity()
            )

            # Upsampling Path
            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out, time_dim=self.time_dim),
                        resnet_block(dim_out + dim_in, dim_out, time_dim=self.time_dim),
                        attn_block,
                        (
                            UpsampleConv(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        # OUTPUT
        self.out_dim = channels
        self.final_res_block = resnet_block(
            dims[0] * 2, dims[0], time_dim=self.time_dim
        )
        self.final_conv = nn.Conv1d(dims[0], self.out_dim, 1)  # output: [B, C, L]

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage during training."""
        self.use_gradient_checkpointing = True

    def _resize_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Adjust 1D feature map length to match target_len.

        Strict by default: any mismatch raises an error to surface indexing bugs.
        With strict_resize=False, only zero padding (or right-cropping) is used.

        Args:
            x: Tensor of shape [B, C, L]
            target_len: desired length L_out

        Returns:
            Tensor with shape [B, C, target_len]
        """
        cur_len = x.size(-1)
        if cur_len == target_len:
            return x
        if self.strict_resize:
            raise RuntimeError(
                f"Length mismatch during U-Net skip alignment: got {cur_len}, expected {target_len}. "
                f"Set strict_resize=False to allow zero padding/cropping, but investigate indexing first."
            )
        # Non-strict path: prefer zero padding or right-cropping; never interpolate
        if cur_len < target_len:
            pad_right = target_len - cur_len
            return F.pad(x, (0, pad_right), mode="constant", value=self.pad_value)
        # cur_len > target_len: crop on the right
        return x[..., :target_len]

    def forward(self, x, time):
        """
        Forward pass for noise prediction in diffusion models.

        Args:
            x (torch.Tensor): Noisy SNP sequences [B, 1, L]
            time (torch.Tensor): Diffusion timesteps [B]

        Returns:
            torch.Tensor: Predicted noise [B, 1, L]

        Raises:
            ValueError: If sequence too short for downsampling levels
        """
        # INPUT
        B, C, L = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Original x
        original_x = x

        # POSITIONAL EMBEDDING (post-stem): Positional embedding injection
        # happens after stem conv to avoid 1-channel bottleneck (Moved).

        # EDGE PADDING CHECK
        edge_pad = self.edge_pad
        min_len = L
        for i in range(len(self.dim_mults)):
            min_len = (min_len + 1) // 2  # Downsampling by 2 at each step
            if min_len <= edge_pad:
                raise ValueError(
                    f"Input sequence length {L} too short for {len(self.dim_mults)} downsampling steps and edge_pad={edge_pad}."
                    f"At downsampling step {i}, length after downsampling would be {min_len} which is not enough for edge_pad={edge_pad}."
                    f"Increase seq_length or reduce dim_mults/edge_pad."
                )

        # INITIAL CONVOLUTION
        # [B, 1, L] → [B, init_dim, L]
        x = self.init_conv(x)

        # POSITIONAL EMBEDDING (post-stem): inject after the stem so we can project
        # the sinusoidal position to init_dim channels and add it. Doing this on the
        # raw 1-channel input would bottleneck positional information into a single
        # channel, which we explicitly avoid here.
        if self.with_pos_emb and self.pos_emb is not None:
            positions = torch.arange(L, device=x.device).expand(B, -1)  # [B, L]
            pos = self.pos_emb(positions)  # [B, L, pos_dim]
            pos = pos.permute(0, 2, 1)  # [B, pos_dim, L]
            pos = self.pos_proj(pos)  # [B, init_dim, L]
            # Add with optional gate
            x = x + (self.pos_alpha * pos if hasattr(self, "pos_alpha") else pos)

        # TIME EMBEDDING
        t = self.time_mlp(time) if self.time_mlp else None  # Project time to time_dim

        # RESIDUAL CONNECTION
        r = x

        # ENCODER / DOWNSAMPLING
        h = []
        for block1, block2, attn, downsample in self.downs:
            if self.use_gradient_checkpointing:
                x = checkpoint(block1, x, t, use_reentrant=False)
            else:
                x = block1(x, t)

            h.append(x)  # Save after block1

            if self.use_gradient_checkpointing:
                x = checkpoint(block2, x, t, use_reentrant=False)
                x = checkpoint(attn, x, use_reentrant=False)
            else:
                x = block2(x, t)
                x = attn(x)

            h.append(x)  # Save after block2 + attn

            x = downsample(x)

        # BOTTLENECK
        x = (  # mid_block1
            checkpoint(self.mid_block1, x, t, use_reentrant=False)
            if self.use_gradient_checkpointing
            else self.mid_block1(x, t)
        )
        x = (  # mid_attn
            checkpoint(self.mid_attn, x, use_reentrant=False)
            if self.use_gradient_checkpointing
            else self.mid_attn(x)
        )
        x = (  # mid_block2
            checkpoint(self.mid_block2, x, t, use_reentrant=False)
            if self.use_gradient_checkpointing
            else self.mid_block2(x, t)
        )

        # DECODER / UPSAMPLING
        for block1, block2, attn, upsample in self.ups:
            skip2 = h.pop()

            # Align lengths before concatenation
            x = self._resize_to_length(x, skip2.size(-1))
            x = torch.cat((x, skip2), dim=1)  # Use skip 1

            if self.use_gradient_checkpointing:
                x = checkpoint(block1, x, t, use_reentrant=False)
            else:
                x = block1(x, t)

            skip1 = h.pop()

            # Align lengths before concatenation
            x = self._resize_to_length(x, skip1.size(-1))
            x = torch.cat((x, skip1), dim=1)  # Use skip 2

            if self.use_gradient_checkpointing:
                x = checkpoint(block2, x, t, use_reentrant=False)
                x = checkpoint(attn, x, use_reentrant=False)
            else:
                x = block2(x, t)
                x = attn(x)

            x = upsample(x)

        # OUTPUT
        x = self._resize_to_length(x, r.size(-1))
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return original_x - self.final_conv(x)
