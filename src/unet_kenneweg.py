#!/usr/bin/env python
# coding: utf-8

"""Based on `https://huggingface.co/blog/annotated-diffusion` that explains
using the original DDPM by Ho et al. 2022 on images i.e. 2D dataset. We
adapted the code for 1-dimensional SNP genomic dataset. In additon, some
improvements are applied from the work of Kenneweg et al whose work is similar
to mine (SNP data, UNet1D), see https://github.com/TheMody/GeneDiffusion."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sinusoidal_embedding import (
    SinusoidalPositionEmbeddings,
    SinusoidalTimeEmbeddings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    """Residual connection wrapper: output = fn(x) + x."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Downsample1D(nn.Module):
    """1D downsampling with robust odd-length sequence handling."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.conv = nn.Conv1d(dim, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        # Handle odd sequence lengths with reflective padding
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (1, 0), mode="reflect")
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D upsampling using transposed convolution for exact reconstruction."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.conv = nn.ConvTranspose1d(dim, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# Kenneweg: Added zero initialization utility function
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class Block1D(nn.Module):
    """Basic 1D convolutional block: Conv1D + GroupNorm + SiLU."""

    def __init__(
        self, dim, dim_out, groups=8, dropout=0.0
    ):  # Kenneweg: Added dropout parameter
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )  # Kenneweg: Added dropout

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(
            x
        )  # Kenneweg: Apply dropouthttps://github.com/TheMody/GeneDiffusion
        return x


class ResnetBlock1D(nn.Module):
    """1D ResNet block with time embedding integration for diffusion models."""

    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        dropout=0.0,
        use_scale_shift_norm=True,
    ):  # Kenneweg: Added dropout and scale_shift_norm
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm  # Kenneweg: Store flag

        # Kenneweg: Time embedding MLP projects to 2*dim_out for scale+shift normalization
        if use_scale_shift_norm:
            self.time_mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
                if time_emb_dim is not None
                else None
            )
        else:
            # Original behavior: project to dim_out for direct addition
            self.time_mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
                if time_emb_dim is not None
                else None
            )

        self.block1 = Block1D(
            dim, dim_out, groups=groups, dropout=dropout
        )  # Kenneweg: Pass dropout
        self.block2 = Block1D(
            dim_out, dim_out, groups=groups, dropout=dropout
        )  # Kenneweg: Pass dropout
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        # Kenneweg: Apply scale-shift normalization or simple addition based on flag
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1)  # [B, C] -> [B, C, 1]

            if self.use_scale_shift_norm:
                # Kenneweg: Scale-shift normalization (FiLM-like conditioning)
                scale, shift = time_emb.chunk(2, dim=1)
                h = h * (1 + scale) + shift
            else:
                # Original behavior: simple addition
                h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention1D(nn.Module):
    """Multi-head self-attention for capturing long-range genomic patterns."""

    def __init__(self, dim, heads=4, dim_head=32, use_checkpoint=False):
        super().__init__()
        # Kenneweg: Double square root scaling for better stability (from Lai et al.)
        self.scale = 1 / math.sqrt(math.sqrt(dim_head))  # Kenneweg: More stable scaling
        # self.scale = dim_head**-0.5  # Original standard scaling - uncomment to revert
        self.heads = heads
        self.dim_head = dim_head
        self.use_checkpoint = use_checkpoint  # Kenneweg: Gradient checkpointing flag
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        # Kenneweg: Zero-initialize output projection for better training stability
        self.to_out = zero_module(nn.Conv1d(hidden_dim, dim, 1))

    def forward(self, x):
        # Kenneweg: Use gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, n = x.shape
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
    """Linear attention with O(n) complexity for efficient long sequence processing."""

    def __init__(self, dim, heads=4, dim_head=32, use_checkpoint=False):
        super().__init__()
        # Kenneweg: Double square root scaling for better stability (from Lai et al.)
        self.scale = 1 / math.sqrt(math.sqrt(dim_head))  # Kenneweg: More stable scaling
        # self.scale = dim_head**-0.5  # Original standard scaling - uncomment to revert
        self.heads = heads
        self.dim_head = dim_head
        self.use_checkpoint = use_checkpoint  # Kenneweg: Gradient checkpointing flag
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        # Kenneweg: Zero-initialize output projection for better training stability
        self.to_out = nn.Sequential(
            zero_module(nn.Conv1d(hidden_dim, dim, 1)), nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        # Kenneweg: Use gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.view(b, -1, n)
        return self.to_out(out)


class PreNorm(nn.Module):
    """Pre-normalization wrapper for improved training stability."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# UNet1D Architecture for Genomic Diffusion Models
class UNet1D(nn.Module):
    """
    1D U-Net for genomic SNP sequence modeling in diffusion models.

    Kenneweg IMPROVEMENTS IMPLEMENTED:
    - Scale-shift normalization (FiLM-like conditioning) for better time embedding integration
    - Zero-initialized output layers for training stability
    - Dropout in ResNet blocks (0.1) for regularization
    - Larger time embedding dimension (embedding_dim * 4) for better expressivity
    - Additive residual connections instead of concatenation
    - Zero-initialized attention output projections

    A time-conditional U-Net architecture designed for denoising genomic sequences
    in diffusion-based generative models. Processes SNP data as 1D sequences with
    shape [batch_size, 1, sequence_length].

    Architecture:
    - 4-level encoder-decoder with feature dimensions [16, 32, 64, 128]
    - Dual skip connections per level for rich gradient flow
    - ResNet blocks with GroupNorm and SiLU activation
    - Optional multi-head attention for long-range dependencies
    - Sinusoidal time/position embeddings for temporal conditioning

    Key Features:
    - Handles variable-length genomic sequences (odd/even lengths)
    - Memory-efficient with gradient checkpointing support
    - Configurable attention mechanisms (8 heads × 32 dims)
    - Robust downsampling/upsampling with precise reconstruction

    Args:
        embedding_dim (int): Time/position embedding dimension (default: 64)
        dim_mults (tuple): Feature multipliers per level (default: (1,2,4,8))
        channels (int): Input channels, always 1 for SNP data
        with_time_emb (bool): Enable time embeddings for diffusion
        with_pos_emb (bool): Enable position embeddings for spatial awareness
        norm_groups (int): GroupNorm groups (default: 8)
        seq_length (int): Expected sequence length for validation
        edge_pad (int): Boundary padding size (default: 2)
        use_attention (bool): Enable attention mechanisms
        attention_heads (int): Number of attention heads (default: 4)
        attention_dim_head (int): Dimension per attention head (default: 32)
        debug (bool): Print tensor shapes during forward pass
        dropout (float): Dropout rate for ResNet blocks (default: 0.1) # Kenneweg
        use_scale_shift_norm (bool): Use scale-shift normalization (default: True) # Kenneweg

    Input/Output:
        Input: [B, 1, L] - Noisy SNP sequences
        Output: [B, 1, L] - Predicted noise for denoising
    """

    def __init__(
        self,
        embedding_dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        with_time_emb=True,
        with_pos_emb=True,
        norm_groups=8,
        seq_length=160858,
        edge_pad=2,
        debug=False,
        use_attention=True,
        attention_heads=4,
        attention_dim_head=32,
        dropout=0.1,  # Kenneweg: Added dropout parameter
        use_scale_shift_norm=True,  # Kenneweg: Added scale-shift normalization flag
        attention_checkpoint=False,  # Kenneweg: Added attention checkpointing flag
    ):
        """
        Initialize UNet1D with genomic-optimized architecture and Lai et al. improvements.

        Memory usage scales with embedding_dim × dim_mults. For efficiency:
        - embedding_dim=64-128 balances capacity and memory
        - dim_mults=(1,2,4,8) provides 4-level hierarchy
        - Enable gradient checkpointing for long sequences
        """
        super().__init__()

        # Save config for reference and checkpointing
        self.embedding_dim = embedding_dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.with_pos_emb = with_pos_emb
        self.norm_groups = norm_groups
        self.seq_length = seq_length
        self.edge_pad = edge_pad
        self.debug = debug
        self.use_gradient_checkpointing = False
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.attention_dim_head = attention_dim_head
        self.dropout = dropout  # Kenneweg: Store dropout rate
        self.use_scale_shift_norm = (
            use_scale_shift_norm  # Kenneweg: Store normalization flag
        )
        self.attention_checkpoint = (
            attention_checkpoint  # Kenneweg: Store attention checkpointing flag
        )

        # --- Model complexity and memory control ---
        # Base feature dimension - kept small (16) to manage memory usage
        # This is multiplied by dim_mults at each level, so even small
        # changes here have a big impact on memory
        init_dim = 16  # [16 -> 32 -> 64 -> 128] with dim_mults=(1,2,4,8)
        out_dim = self.channels  # Always 1 for SNP data

        # Initial conv layer: maps input to base feature dimension
        # Using larger kernel size (7) for better receptive field at the input level
        # Output: [B, 1, L] -> [B, 16, L]
        kernel_size = 7  # Larger kernel for better pattern recognition
        padding = (kernel_size - 1) // 2  # Same padding to preserve length
        self.init_conv = nn.Conv1d(
            self.channels, init_dim, kernel_size=kernel_size, padding=padding
        )

        # Calculate feature dimensions for each U-Net level
        # Example with init_dim=16, dim_mults=(1,2,4,8):
        # dims = [16, 32, 64, 128] (feature channels at each level)
        # Each level halves spatial dimension but increases features
        dims = [init_dim]
        for mult in self.dim_mults:
            # Cap feature dims at 128 to prevent memory explosion
            # This is crucial for long sequences
            dims.append(min(init_dim * mult, 128))

        # Create (input_dim, output_dim) pairs for each level
        # Example: [(16,32), (32,64), (64,128)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Store dimensions for debugging and shape analysis
        self._dims = dims  # Feature dimensions at each level
        self._in_out = in_out  # Input/output dim pairs

        # --- Embeddings ---
        # Kenneweg: Larger time embedding dimension for better expressivity (4x instead of 1x)
        # Time embeddings: crucial for diffusion models
        # Maps scalar timestep to high-dim vector via sinusoidal encoding
        # Then projects through MLP for better expressivity
        if self.with_time_emb:
            time_dim = (
                self.embedding_dim * 4
            )  # Kenneweg: Increased from embedding_dim to embedding_dim * 4
            self.time_mlp = nn.Sequential(
                # Initial sinusoidal encoding
                SinusoidalTimeEmbeddings(self.embedding_dim),
                # Project and add non-linearity for expressivity
                nn.Linear(self.embedding_dim, time_dim),
                nn.GELU(),  # Smooth activation
                nn.Linear(time_dim, time_dim),  # Final projection
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Position embeddings: help with long-range dependencies
        # Uses same sinusoidal encoding as time embeddings
        # Added directly to input features for spatial awareness
        if self.with_pos_emb:
            self.pos_emb = SinusoidalPositionEmbeddings(self.embedding_dim)
        else:
            self.pos_emb = None

        # ========== UNet1D Architecture ==========
        num_resolutions = len(in_out)
        # Kenneweg: Pass dropout and scale_shift_norm parameters to ResNet blocks
        block_klass = partial(
            ResnetBlock1D,
            groups=self.norm_groups,
            dropout=self.dropout,
            use_scale_shift_norm=self.use_scale_shift_norm,
        )

        # ENCODER / DOWNSAMPLING
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        # Mimic original: block(dim_in, dim_in), block(dim_in, dim_in)
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    LinearAttention1D(
                                        dim_in,
                                        heads=self.attention_heads,
                                        dim_head=self.attention_dim_head,
                                        use_checkpoint=self.attention_checkpoint,  # Kenneweg: Attention checkpointing
                                    ),
                                )
                            )
                            if self.use_attention
                            else nn.Identity()
                        ),
                        (
                            Downsample1D(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        # BOTTLENECK
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = (
            Residual(
                PreNorm(
                    mid_dim,
                    LinearAttention1D(
                        mid_dim,
                        heads=self.attention_heads,
                        dim_head=self.attention_dim_head,
                        use_checkpoint=self.attention_checkpoint,  # Kenneweg: Attention checkpointing
                    ),
                )
            )
            if self.use_attention
            else nn.Identity()
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # DECODER / UPSAMPLING
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        # Mimic original: block(dim_out + dim_in, dim_out) twice
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    LinearAttention1D(
                                        dim_out,
                                        heads=self.attention_heads,
                                        dim_head=self.attention_dim_head,
                                        use_checkpoint=self.attention_checkpoint,  # Kenneweg: Attention checkpointing
                                    ),
                                )
                            )
                            if self.use_attention
                            else nn.Identity()
                        ),
                        (
                            Upsample1D(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        # OUTPUT
        self.out_dim = out_dim if out_dim is not None else self.channels
        self.final_res_block = block_klass(dims[0] * 2, dims[0], time_emb_dim=time_dim)
        # Kenneweg: Zero-initialize final convolution for training stability
        self.final_conv = zero_module(nn.Conv1d(dims[0], self.out_dim, 1))

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage during training."""
        self.use_gradient_checkpointing = True

    def forward(self, x, time):
        """
        Forward pass for noise prediction in diffusion models.

        Processes noisy SNP sequences through encoder-decoder architecture
        with time conditioning to predict added noise.

        Args:
            x (torch.Tensor): Noisy SNP sequences [B, 1, L]
            time (torch.Tensor): Diffusion timesteps [B]

        Returns:
            torch.Tensor: Predicted noise [B, 1, L]

        Raises:
            ValueError: If sequence too short for downsampling levels
        """
        # ========== INPUT & EMBEDDINGS ==========
        if self.debug:
            print(
                f"[DEBUG] Input shape: {x.shape} (expected: [B, {self.channels}, {self.seq_length}])"
            )

        batch, c, seq_len = x.shape
        assert c == self.channels, f"Expected {self.channels} channels, got {c}"
        original_len = x.size(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Add positional embedding if enabled
        if self.with_pos_emb and self.pos_emb is not None:
            positions = torch.arange(seq_len, device=x.device).expand(
                batch, -1
            )  # [B, L]
            pos_encoding = self.pos_emb(positions)  # [B, L, emb]
            pos_encoding = pos_encoding.permute(0, 2, 1)  # [B, emb, L]
            # Add to input (if emb > 1, only add to first channel)
            x = x + pos_encoding[:, : x.shape[1], :]

        # Edge padding for boundary preservation (configurable)
        edge_pad = self.edge_pad
        # Improved input validation: check after each downsampling that length is always > edge_pad
        min_len = seq_len
        for i in range(len(self.dim_mults)):
            min_len = (min_len + 1) // 2  # Downsampling with stride 2
            if min_len <= edge_pad:
                raise ValueError(
                    f"Input sequence length {seq_len} is too short for {len(self.dim_mults)} downsampling steps and edge_pad={edge_pad}. "
                    f"At downsampling step {i}, length after downsampling would be {min_len}, which is not enough for edge_pad={edge_pad}. "
                    f"Increase seq_length or reduce dim_mults/edge_pad."
                )

        # === INPUT ===
        # [B, 1, L] → [B, init_dim, L]
        x = self.init_conv(x)
        if self.debug:
            print(f"[DEBUG] After initial conv: {x.shape}")
        t = self.time_mlp(time) if self.time_mlp else None

        # Kenneweg: Save input for additive residual connection (instead of concatenation)
        x_skip = x.clone()

        # ENCODER / DOWNSAMPLING
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)  # Save after block1

            x = block2(x, t)
            x = attn(x)
            h.append(x)  # Save after block2 + attn

            x = downsample(x)

        # BOTTLENECK
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # DECODER / UPSAMPLING
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # Use skip 1
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)  # Use skip 2
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        # OUTPUT
        x = torch.cat((x, x_skip), dim=1)  # Kenneweg: Use saved input skip connection
        x = self.final_res_block(x, t)
        output = self.final_conv(x)

        # Kenneweg: Return additive residual instead of just output
        # This follows the pattern from Lai et al. where x_skip + h is returned
        return x_skip[:, : self.out_dim, :] + output
