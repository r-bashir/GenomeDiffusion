#!/usr/bin/env python
# coding: utf-8

"""Based on `https://huggingface.co/blog/annotated-diffusion` that explains
using the original DDPM by Ho et al. 2022 on images i.e. 2D dataset. We
adapted the code for 1-dimensional SNP genomic dataset."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    """Enhanced 1D downsampling with anti-aliasing and feature preservation."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        # Anti-aliasing downsampling to preserve high-frequency genomic patterns
        self.pre_conv = nn.Conv1d(
            dim, dim, 3, padding=1, groups=dim
        )  # Depthwise smoothing
        self.downsample = nn.Conv1d(dim, dim_out, 4, stride=2, padding=1)

        # Additional feature processing
        self.post_conv = nn.Conv1d(dim_out, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, dim_out), dim_out)  # Ensure divisibility
        self.act = nn.SiLU()

    def forward(self, x):
        # Handle odd sequence lengths with reflective padding
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (1, 0), mode="reflect")

        # Anti-aliasing pre-processing
        x = self.pre_conv(x)
        x = self.downsample(x)

        # Feature enhancement after downsampling
        x = self.post_conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Upsample1D(nn.Module):
    """Enhanced 1D upsampling with feature refinement and detail preservation."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        # Learnable upsampling with feature refinement
        self.upsample = nn.ConvTranspose1d(dim, dim_out, 4, stride=2, padding=1)

        # Post-upsampling refinement to reduce checkerboard artifacts
        self.refine = nn.Sequential(
            nn.Conv1d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(min(8, dim_out), dim_out),  # Ensure divisibility
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding=1),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.refine(x)
        return x


class Block1D(nn.Module):
    """Enhanced 1D convolutional block with multi-scale receptive fields."""

    def __init__(self, dim, dim_out, groups=8, use_multiscale=True):
        super().__init__()
        self.use_multiscale = use_multiscale

        if use_multiscale:
            # Multi-scale dilated convolutions for genomic patterns
            self.conv_1 = nn.Conv1d(
                dim, dim_out // 4, 3, padding=1, dilation=1
            )  # Local patterns
            self.conv_2 = nn.Conv1d(
                dim, dim_out // 4, 3, padding=2, dilation=2
            )  # Medium patterns
            self.conv_4 = nn.Conv1d(
                dim, dim_out // 4, 3, padding=4, dilation=4
            )  # Long patterns
            self.conv_8 = nn.Conv1d(
                dim, dim_out // 4, 3, padding=8, dilation=8
            )  # Very long patterns
            self.fusion = nn.Conv1d(dim_out, dim_out, 1)  # Combine multi-scale features
        else:
            self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)

        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.use_multiscale:
            # Multi-scale feature extraction
            out1 = self.conv_1(x)
            out2 = self.conv_2(x)
            out4 = self.conv_4(x)
            out8 = self.conv_8(x)
            x = torch.cat([out1, out2, out4, out8], dim=1)
            x = self.fusion(x)
        else:
            x = self.proj(x)

        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock1D(nn.Module):
    """Enhanced 1D ResNet block with squeeze-excitation and improved time integration."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, use_se=True):
        super().__init__()
        self.use_se = use_se

        # Enhanced time embedding with gating mechanism
        self.time_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),  # For both scale and shift
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block1D(dim, dim_out, groups=groups, use_multiscale=True)
        self.block2 = Block1D(dim_out, dim_out, groups=groups, use_multiscale=True)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # Squeeze-and-Excitation for channel attention
        if use_se:
            se_dim = max(1, dim_out // 8)  # Ensure at least 1 channel
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(dim_out, se_dim, 1),
                nn.SiLU(),
                nn.Conv1d(se_dim, dim_out, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        # Enhanced time embedding with scale and shift
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim=1)
            scale = scale.unsqueeze(-1)  # [B, C] -> [B, C, 1]
            shift = shift.unsqueeze(-1)  # [B, C] -> [B, C, 1]
            h = h * (1 + scale) + shift  # Affine transformation

        h = self.block2(h)

        # Apply squeeze-excitation attention
        if self.use_se:
            se_weights = self.se(h)
            h = h * se_weights

        return h + self.res_conv(x)


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


# UNet1D Architecture for Genomic Diffusion Models
class UNet1D(nn.Module):
    """
    1D U-Net for genomic SNP sequence modeling in diffusion models.

    A time-conditional U-Net architecture designed for denoising genomic sequences
    in diffusion-based generative models. Processes SNP data as 1D sequences with
        shape [B, C=1, L].

    Architecture:
    - 4-level encoder-decoder with progressive capacity scaling [16, 32, 64, 128+]
    - Dual skip connections per level for rich gradient flow
    - Enhanced ResNet blocks with multi-scale convolutions and squeeze-excitation
    - Specialized attention: LinearAttention1D (encoder/decoder), Attention1D (bottleneck)
    - Sinusoidal time/position embeddings for temporal conditioning

    Key Features:
    - Handles variable-length genomic sequences (odd/even lengths)
    - Memory-efficient with gradient checkpointing support
    - Enhanced multi-scale pattern recognition with dilated convolutions
    - Anti-aliasing downsampling and artifact-reducing upsampling
    - Progressive capacity scaling: up to 512 channels at deepest levels
    - Specialized attention per path: Linear (memory-efficient) + Full (high expressiveness)

    Args:
        embedding_dim (int): Time/position embedding dimension (default: 64)
        dim_mults (tuple): Feature multipliers per level (default: (1,2,4,8))
        channels (int): Input channels, always 1 for SNP data
        with_time_emb (bool): Enable time embeddings for diffusion
        with_pos_emb (bool): Enable position embeddings for spatial awareness
        norm_groups (int): GroupNorm groups (default: 8)
        seq_length (int): Expected sequence length for validation
        edge_pad (int): Boundary padding size (default: 2)
        enable_checkpointing (bool): Enable gradient checkpointing for memory efficiency
        use_attention (bool): Enable attention mechanisms
        attention_heads (int): Number of attention heads (default: 4)
        attention_dim_head (int): Dimension per attention head (default: 32)

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
        enable_checkpointing=True,
        use_attention=True,  # Enable attention mechanisms
        attention_heads=4,  # Number of attention heads
        attention_dim_head=32,  # Dimension per attention head
        **kwargs,  # Accept additional arguments
    ):
        """
        Initialize UNet1D with genomic-optimized architecture.

        Memory usage scales with embedding_dim × dim_mults. For efficiency:
        - embedding_dim=64-128 balances capacity and memory
        - dim_mults=(1,2,4,8) provides 4-level hierarchy
        - Enable gradient checkpointing for long sequences
        """
        super().__init__()

        # Base Parameters
        self.embedding_dim = embedding_dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.with_pos_emb = with_pos_emb
        self.norm_groups = norm_groups
        self.seq_length = seq_length
        self.edge_pad = edge_pad
        self.use_gradient_checkpointing = enable_checkpointing

        # Attention Parameters
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.attention_dim_head = attention_dim_head

        # --- Model complexity and memory control ---
        # Base feature dimension - kept small (16) to manage memory usage
        # This is multiplied by dim_mults at each level, so even small
        # changes here have a big impact on memory
        init_dim = 16  # [16 -> 32 -> 64 -> 128] with dim_mults=(1,2,4,8)
        out_dim = self.channels  # Always 1 for SNP data

        # Enhanced initial conv layer with multi-scale pattern recognition
        # Using multiple kernel sizes to capture genomic patterns at different scales
        # Output: [B, 1, L] → [B, 16, L]
        self.init_conv = nn.Sequential(
            # Multi-scale initial feature extraction
            nn.Conv1d(
                self.channels, init_dim // 4, kernel_size=3, padding=1
            ),  # Local SNP patterns
            nn.Conv1d(
                init_dim // 4, init_dim // 2, kernel_size=7, padding=3
            ),  # Gene-level patterns
            nn.Conv1d(
                init_dim // 2, init_dim, kernel_size=15, padding=7
            ),  # Regulatory patterns
            nn.GroupNorm(8, init_dim),
            nn.SiLU(),
        )

        # Calculate feature dimensions for each U-Net level
        # Increased capacity for better genomic pattern modeling
        # Example with init_dim=16, dim_mults=(1,2,4,8):
        # dims = [16, 32, 64, 128, 256] (higher capacity at deeper levels)
        dims = [init_dim]
        for i, mult in enumerate(self.dim_mults):
            # Progressive capacity increase - more capacity where spatial resolution is lower
            if i < 2:
                max_dim = 128  # Conservative for early levels
            elif i < 3:
                max_dim = 256  # More capacity for mid levels
            else:
                max_dim = 512  # High capacity for deepest levels
            dims.append(min(init_dim * mult, max_dim))

        # Create (input_dim, output_dim) pairs for each level
        # Example: [(16,32), (32,64), (64,128)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Store dimensions for debugging and shape analysis
        self._dims = dims  # Feature dimensions at each level
        self._in_out = in_out  # Input/output dim pairs

        # --- Embeddings ---
        # Time embeddings: crucial for diffusion models
        # Maps scalar timestep to high-dim vector via sinusoidal encoding
        # Then projects through MLP for better expressivity
        # Output dimension matches embedding_dim for consistent scale
        if self.with_time_emb:
            time_dim = self.embedding_dim  # Consistent dimension for stability
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
        block_klass = partial(ResnetBlock1D, groups=self.norm_groups)

        # ENCODER / DOWNSAMPLING
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # LinearAttention1D for encoder path (memory-efficient)
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
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        attn_block,
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
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # DECODER / UPSAMPLING
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # LinearAttention1D for decoder path (memory-efficient)
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
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        attn_block,
                        (
                            Upsample1D(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        # Enhanced OUTPUT with multi-scale final processing
        self.out_dim = out_dim if out_dim is not None else self.channels
        self.final_res_block = block_klass(dims[0] * 2, dims[0], time_emb_dim=time_dim)

        # Multi-scale final convolution for better reconstruction
        self.final_conv = nn.Sequential(
            # Progressive refinement with different kernel sizes
            nn.Conv1d(
                dims[0], dims[0] // 2, kernel_size=7, padding=3
            ),  # Capture larger patterns
            nn.GroupNorm(min(8, dims[0] // 2), dims[0] // 2),  # Ensure divisibility
            nn.SiLU(),
            nn.Conv1d(
                dims[0] // 2, max(8, dims[0] // 4), kernel_size=5, padding=2
            ),  # Medium patterns, ensure >=8 channels
            nn.GroupNorm(
                min(8, max(8, dims[0] // 4)), max(8, dims[0] // 4)
            ),  # Ensure divisibility
            nn.SiLU(),
            nn.Conv1d(
                max(8, dims[0] // 4), self.out_dim, kernel_size=3, padding=1
            ),  # Fine details
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage during training."""
        self.use_gradient_checkpointing = True

    def _resize_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad or crop a 1D feature map along the sequence dimension to match target_len.

        Uses reflective padding when increasing length to preserve boundary information.
        This helps resolve off-by-one mismatches introduced by odd-length down/upsampling.

        Args:
            x: Tensor of shape [B, C, L]
            target_len: desired length L_out

        Returns:
            Tensor with shape [B, C, target_len]
        """
        cur_len = x.size(-1)
        if cur_len == target_len:
            return x
        if cur_len < target_len:
            pad_right = target_len - cur_len
            # (left, right) padding
            return F.pad(x, (0, pad_right), mode="reflect")
        # cur_len > target_len: crop on the right
        return x[..., :target_len]

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
        batch, c, seq_len = x.shape
        assert c == self.channels, f"Expected {self.channels} channels, got {c}"
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

        # INPUT
        # [B, 1, L] → [B, init_dim, L]
        x = self.init_conv(x)
        t = self.time_mlp(time) if self.time_mlp else None

        # Save residual connection from after initial conv (for final output)
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
        if self.use_gradient_checkpointing:
            x = checkpoint(self.mid_block1, x, t, use_reentrant=False)
            x = checkpoint(self.mid_attn, x, use_reentrant=False)
            x = checkpoint(self.mid_block2, x, t, use_reentrant=False)
        else:
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

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
        # Align with initial residual
        x = self._resize_to_length(x, r.size(-1))
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
