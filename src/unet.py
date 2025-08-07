#!/usr/bin/env python
# coding: utf-8

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
    """Generic residual connection wrapper.

    Applies a function and adds the result to the input: f(x) + x
    Useful for creating residual connections around any function.

    Args:
        fn: Function to apply before adding residual connection
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class DownsampleConv(nn.Module):
    """1D Downsampling using strided convolution.

    Reduces sequence length by half using stride=2 convolution.
    Handles odd sequence lengths with reflective padding to preserve edge patterns.

    Args:
        dim (int): Number of input/output channels
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        # Handle odd sequence lengths with reflective padding
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (1, 0), mode="reflect")

        return self.conv(x)


class UpsampleConv(nn.Module):
    """1D Upsampling using transposed convolution.

    Doubles sequence length using stride=2 transposed convolution.
    Uses kernel_size=4 with padding=1 for clean upsampling.

    Args:
        dim (int): Number of input/output channels
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    """1D Convolutional Block with GroupNorm and SiLU activation.

    Standard building block: Conv1d -> GroupNorm -> SiLU
    Uses 'same' padding to preserve sequence length.

    Args:
        dim_in (int): Input channels
        dim_out (int): Output channels
        groups (int): Number of groups for GroupNorm (default: 8)
    """

    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Residual block with two convolutions and optional time embedding.

    Architecture: x -> ConvBlock -> [+time_emb] -> ConvBlock -> output
                  |                                            ^
                  +---> [projection if needed] ---------------+

    Args:
        dim (int): Input channels
        dim_out (int): Output channels
        time_emb_dim (int, optional): Time embedding dimension. If None, no time embedding.
        groups (int): Number of groups for GroupNorm (default: 8)
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # Time embedding MLP (if enabled)
        self.mlp_time = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        # Two convolutional blocks
        self.block1 = ConvBlock(dim, dim_out, groups=groups)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups)

        # Residual projection (if input/output dims differ)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        identity = x
        h = self.block1(x)

        # Add time embedding if provided
        if self.mlp_time is not None and time_emb is not None:
            time_emb = self.mlp_time(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1)
            if h.size(-1) > 1:
                time_emb = time_emb.expand(-1, -1, h.size(-1))
            h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(identity)


# Noise Predictor
class UNet1D(nn.Module):
    """
    1D U-Net for SNP Sequence Modeling.

    This model is designed for 1D SNP sequence data with strict input shape [B, 1, L],
    where B is batch size, 1 is the fixed channel dimension, and L is sequence length.

    Key Features:
    - Adjustable depth and width via `dim_mults` and `embedding_dim`
    - Sinusoidal time and (optional) position embeddings
    - Residual blocks with GroupNorm and SiLU activation
    - Downsampling/Upsampling via strided convolutions
    - Memory-efficient handling of long sequences
    - Gradient checkpointing support
    - Detailed shape tracking (when debug=True)

    Memory Efficiency:
    - Feature dimensions are capped at 128 channels
    - Gradient checkpointing can be enabled for training
    - Smaller dim_mults (e.g., (1,2,4)) reduce memory usage
    - Reduced embedding_dim (e.g., 32) for lighter embeddings

    Sequence Length Requirements:
    - Input length must be sufficient for all downsampling steps
    - Each downsampling halves the sequence length
    - Edge padding (4 on each side) preserves sequence boundaries
    - Minimum length depends on dim_mults and edge_pad size

    Args:
        embedding_dim (int): Dimensionality for time/pos embeddings. Lower values (32-64)
            reduce memory usage but may affect model capacity.
        dim_mults (tuple): Feature dimension multipliers at each level. Each value
            multiplies the base dimension (16). More values = deeper network but
            requires longer sequences. Example: (1,2,4) for memory efficiency.
        channels (int): Input channels, fixed at 1 for SNP data.
        with_time_emb (bool): Whether to use sinusoidal time embeddings.
        with_pos_emb (bool): Whether to use sinusoidal position embeddings.
        norm_groups (int): Number of groups in ResNet GroupNorm layers.
        seq_length (int): Expected input sequence length. Used for validation and
            debugging. For SNP data, typically 160858.
        debug (bool): If True, prints tensor shapes at each step of forward pass.

    Shape Flow (with default settings):
    1. Input: [B, 1, L]
    2. Edge padding: [B, 1, L+8]
    3. Initial conv: [B, 16, L+8]
    4. Encoder: Progressive downsampling, doubling channels
    5. Bottleneck: Deepest features
    6. Decoder: Progressive upsampling with skip connections
    7. Output: [B, 1, L] (matches input)
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
    ):
        """
        Initialize the UNet1D model with specified architecture parameters.

        The model's memory usage and computational requirements are primarily determined
        by embedding_dim and dim_mults. For memory-constrained environments:
        - Use embedding_dim=32 (reduced from 64)
        - Use dim_mults=(1,2,4) (reduced from (1,2,4,8))
        - Enable gradient_checkpointing during training

        Args:
            embedding_dim (int, optional): Dimension of time and position embeddings.
                Defaults to 64. Lower values (32) reduce memory usage.
            dim_mults (tuple, optional): Feature multipliers at each U-Net level.
                Defaults to (1,2,4,8). Each value multiplies the base dimension (16).
                More values = deeper network but requires longer input sequences.
            channels (int, optional): Number of input channels. Defaults to 1.
                Should remain 1 for SNP data.
            with_time_emb (bool, optional): Whether to use time embeddings.
                Defaults to True. Required for diffusion models.
            with_pos_emb (bool, optional): Whether to use position embeddings.
                Defaults to True. Helps with long-range dependencies.
            norm_groups (int, optional): Number of groups in GroupNorm.
                Defaults to 8. Should be a factor of the feature dimensions.
            seq_length (int, optional): Expected input sequence length.
                Defaults to 160858 (typical SNP sequence length).
            debug (bool, optional): Whether to print tensor shapes during forward pass.
                Defaults to False. Useful for debugging shape mismatches.
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

        # --- Model complexity and memory control ---
        # Base feature dimension - kept small (16) to manage memory usage
        # This is multiplied by dim_mults at each level, so even small
        # changes here have a big impact on memory
        init_dim = 16  # [16 -> 32 -> 64 -> 128] with dim_mults=(1,2,4,8)
        out_dim = self.channels  # Always 1 for SNP data

        # Initial conv layer: maps input to base feature dimension
        # kernel_size=3 balances receptive field and memory
        # Output: [B, 1, L] -> [B, 16, L]
        kernel_size = 3  # Small kernel for efficiency
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

        # --- UNet Architecture ---
        # Helper for consistent ResNet block creation
        # Groups=8 balances between independence and stability
        resnet_block = partial(ResnetBlock, groups=self.norm_groups)
        num_resolutions = len(in_out)  # Number of up/down sampling levels

        # --- Encoder (Downsampling) Path ---
        # Progressive feature expansion with spatial reduction
        # Each level: 2x ResNet + optional downsample
        # Spatial dim halves while features double (up to cap)
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        # Two ResNet blocks for feature processing
                        resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                        resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                        nn.Identity(),  # Future: could add attention here
                        # Downsample except at final level
                        DownsampleConv(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # --- Bottleneck ---
        # Process features at lowest resolution
        # Two ResNet blocks without changing dimensions
        mid_dim = dims[-1]  # Deepest feature dimension
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- Decoder (Upsampling) Path ---
        # Mirror of encoder path with skip connections
        # Each level: Upsample -> Concat skip -> 2x ResNet
        # Spatial dim doubles while features halve
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        # First ResNet: process concatenated features
                        # Input has 2x channels due to skip connection
                        resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        # Second ResNet: process at target dimension
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        nn.Identity(),  # Future: could add attention here
                        # Upsample except at final level
                        UpsampleConv(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # --- Output Projection ---
        # Final processing at base resolution
        # Projects back to 1 channel (SNP data)
        self.final_conv = nn.Sequential(
            resnet_block(dims[0], dims[0]),  # Final feature processing
            nn.Conv1d(dims[0], out_dim, kernel_size=1, padding="same"),  # To SNP space
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = True

    def forward(self, x, time):
        """
        Forward pass through the UNet1D model.

        The data flow follows the canonical U-Net architecture:
        1. Input Processing:
           - Add position embeddings (if enabled)
           - Apply edge padding (4 on each side)
           - Initial convolution to base dimension

        2. Encoder Path (Downsampling):
           - Multiple levels based on dim_mults
           - Each level: 2x ResNet blocks + downsampling
           - Features double while spatial dim halves
           - Skip connections saved for decoder

        3. Bottleneck:
           - Two ResNet blocks at deepest level
           - Processes features at lowest resolution

        4. Decoder Path (Upsampling):
           - Mirrors encoder path in reverse
           - Each level: Upsample + concat skip + 2x ResNet
           - Features halve while spatial dim doubles
           - Careful handling of skip connection alignment

        5. Output Projection:
           - Final ResNet + conv to output channels
           - Removes padding to match input length

        Args:
            x (torch.Tensor): Input SNP data tensor.
                Shape: [batch_size, channels=1, seq_length]
                Note: seq_length must be sufficient for downsampling steps
            time (torch.Tensor): Diffusion timestep embeddings.
                Shape: [batch_size]
                Range: Typically 0 to num_timesteps-1

        Returns:
            torch.Tensor: Denoised/processed output.
                Shape: [batch_size, channels=1, seq_length]
                Note: Output length matches input exactly

        Raises:
            ValueError: If input sequence length is too short for the
                number of downsampling steps and edge padding (4).
                Minimum length depends on dim_mults length.
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

        # Initial conv: [B, 1, L] → [B, init_dim, L]
        x = self.init_conv(x)
        if self.debug:
            print(f"[DEBUG] After initial conv: {x.shape}")
        t = self.time_mlp(time) if self.time_mlp else None

        # ========== ENCODER / DOWNSAMPLING ==========
        h = []  # skip connections
        for i, (block1, block2, _, downsample) in enumerate(self.downs):
            # Residual blocks (optionally with time embedding)
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block1, x, t, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(block2, x, t, use_reentrant=False)
            else:
                x = block1(x, t)
                x = block2(x, t)
            if self.debug:
                print(f"[DEBUG] After encoder block {i}: {x.shape}")
            # Save features before downsampling for skip connection
            h.append(x)
            # Downsample for next level (halves sequence length)
            x = downsample(x)
            if self.debug:
                print(f"[DEBUG] After downsampling {i}: {x.shape}")
            # (Shape after downsampling: [B, C, L//2])

        # ========== BOTTLENECK ==========
        x = self.mid_block1(x, t)
        if self.debug:
            print(f"[DEBUG] After bottleneck block 1: {x.shape}")
        x = self.mid_block2(x, t)
        if self.debug:
            print(f"[DEBUG] After bottleneck block 2: {x.shape}")

        # ========== DECODER / UPSAMPLING ==========
        for i, (block1, block2, _, upsample) in enumerate(self.ups):
            # Get matching skip connection from encoder
            skip_x = h.pop()

            # Upsample features
            x = upsample(x)

            if self.debug:
                print(f"[DEBUG] After upsampling {i}: {x.shape} (skip: {skip_x.shape})")

            # Skip connection should match upsampled size
            if skip_x.size(-1) != x.size(-1):
                # Always pad skip connection to match upsampled size
                diff = x.size(-1) - skip_x.size(-1)
                left_pad = diff // 2
                right_pad = diff - left_pad
                skip_x = F.pad(skip_x, (left_pad, right_pad), mode="reflect")

            # Concatenate features
            x = torch.cat((x, skip_x), dim=1)
            if self.debug:
                print(f"[DEBUG] After concat {i}: {x.shape}")
            # Residual blocks (optionally with time embedding)
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block1, x, t, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(block2, x, t, use_reentrant=False)
            else:
                x = block1(x, t)
                x = block2(x, t)
            if self.debug:
                print(f"[DEBUG] After decoder block {i}: {x.shape}")
            # (Shape after upsampling: [B, C, L*2])

        # ========== OUTPUT PROJECTION ==========
        x = self.final_conv(x)
        if self.debug:
            print(f"[DEBUG] After final conv: {x.shape}")

        # Center crop or pad to match input length
        if x.size(-1) != original_len:
            if x.size(-1) > original_len:
                # Center crop if too long
                start = (x.size(-1) - original_len) // 2
                x = x[:, :, start : start + original_len]
            else:
                # Reflect pad if too short
                diff = original_len - x.size(-1)
                left_pad = diff // 2
                right_pad = diff - left_pad
                x = F.pad(x, (left_pad, right_pad), mode="reflect")

        if self.debug:
            print(
                f"[DEBUG] Output shape: {x.shape} (should match input: {original_len})"
            )
        return x
