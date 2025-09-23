#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from .sinusoidal_embedding import SinusoidalTimeEmbeddings
from .utils import setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = setup_logging(name="MLP")


# ZeroModel
def zero_out_model_parameters(model):
    """
    Sets all weights and biases of nn.Linear layers in the model
    to zero. This makes the model predict zeros for any input.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


# IdentityModel
class IdentityModel(nn.Module):
    """
    Identity noise predictor for debugging: always returns input x as predicted noise.
    Drop-in replacement for MLP/UNet1D in DiffusionModel.
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
        **kwargs,
    ):
        super().__init__()
        # Base parameters
        self.emb_dim = emb_dim
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.time_dim = time_dim
        self.seq_length = seq_length

    def forward(self, x, time):
        return x


# LinearMLP
class LinearMLP(nn.Module):
    """
    Linear MLP for noise prediction in diffusion models. It is designed as drop-in
    replacement for UNet1D. Global n→n linear baseline (flatten + linear + unflatten).
    """

    def __init__(
        self,
        emb_dim=512,  # Sinusoidal embedding dimension
        dim_mults=(1, 2, 4, 8),  # Unused, but kept for interface compatibility
        channels=1,  # Input channels (SNP data channels is always 1)
        with_time_emb=True,  # To include time embeddings
        time_dim=128,  # Final projected time embedding dimension
        with_pos_emb=True,  # Unused, but kept for interface compatibility
        pos_dim=128,  # Unused, but kept for interface compatibility
        norm_groups=8,  # Unused, but kept for interface compatibility
        seq_length=160858,  # Expected sequence length (number of SNP markers)
        **kwargs,
    ):
        super().__init__()
        # Base parameters
        self.emb_dim = emb_dim
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.time_dim = time_dim
        self.seq_length = seq_length

        # Input dimension (flattened sequence)
        self.input_dim = channels * seq_length

        # Time embedding dimension
        self.time_dim = time_dim if with_time_emb else 0

        # Flatten layer
        self.flatten = nn.Flatten()

        # Time embedding network (emb_dim -> time_dim)
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.emb_dim),
                nn.Linear(self.emb_dim, self.time_dim),
            )
        else:
            self.time_mlp = None

        # Linear layer (flattened input + time embedding)
        self.linear = nn.Linear(self.input_dim + self.time_dim, self.input_dim)

    def forward(self, x, time):
        """Forward pass for LinearMLP.

        Args:
            x (torch.Tensor): Input SNP data of shape [B, C, L].
            time (torch.Tensor): Diffusion timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise with same shape as input.
        """
        B, C, L = x.shape

        # Flatten the input
        x_flat = self.flatten(x)

        # Concatenate time embedding if enabled
        if self.with_time_emb and self.time_mlp is not None:
            t_emb = self.time_mlp(time)  # [B, time_dim]
            x_flat = torch.cat([x_flat, t_emb], dim=1)  # [B, C*L + time_dim]

        out = self.linear(x_flat)
        return out.view(B, C, L)


# ComplexMLP
class ComplexMLP(nn.Module):
    """Deep MLP for noise prediction in diffusion models. It is designed as drop-in replacement for UNet1D."""

    def __init__(
        self,
        emb_dim=512,  # Sinusoidal embedding dimension
        dim_mults=(1, 2, 4, 8),  # Unused, but kept for interface compatibility
        channels=1,  # Input channels (SNP data channels is always 1)
        with_time_emb=True,  # To include time embeddings
        time_dim=128,  # Final projected time embedding dimension
        with_pos_emb=True,  # Unused, but kept for interface compatibility
        pos_dim=128,  # Unused, but kept for interface compatibility
        norm_groups=8,  # Unused, but kept for interface compatibility
        seq_length=160858,  # Expected sequence length (number of SNP markers)
        **kwargs,
    ):
        super().__init__()

        # Base parameters
        self.emb_dim = emb_dim
        self.channels = channels
        self.seq_length = seq_length
        self.with_time_emb = with_time_emb

        # Input dimension (flattened sequence)
        self.input_dim = channels * seq_length

        # Time embedding dimension
        self.time_dim = time_dim if with_time_emb else 0

        # Flatten layer
        self.flatten = nn.Flatten()

        # Hidden dimensions
        self.hidden_dims = [1024, 512, 256, 1024]

        # Time embedding network (emb_dim -> time_dim -> GELU -> time_dim)
        if self.with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.emb_dim),
                nn.Linear(self.emb_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
        else:
            self.time_mlp = None

        # Linear input layer (flattened input + time embedding)
        self.input_layer = nn.Linear(
            self.input_dim + self.time_dim, self.hidden_dims[0]
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                self._make_residual_block(self.hidden_dims[i], self.hidden_dims[i + 1])
            )

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.input_dim)

    def _make_residual_block(self, in_dim, out_dim):
        """Create a residual block with layer normalization."""
        resnet_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        return resnet_block

    def forward(self, x, time):
        """Forward pass for ComplexMLP.

        Args:
            x (torch.Tensor): Input SNP data of shape [B, C, L].
            time (torch.Tensor): Diffusion timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise with same shape as input.
        """
        B, C, L = x.shape

        # Flatten the input
        x_flat = self.flatten(x)  # [B, C*L]

        # Process time embedding if enabled
        if self.with_time_emb and self.time_mlp is not None:
            t_emb = self.time_mlp(time)  # [B, time_dim]
            # Concatenate flattened input with time embedding
            x_t = torch.cat([x_flat, t_emb], dim=1)  # [B, C*L + time_dim]
        else:
            x_t = x_flat

        # Input layer
        h = self.input_layer(x_t)

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            h_new = layer(h)
            # Add residual connection if dimensions match
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

        # Output layer
        output = self.output_layer(h)

        # Reshape back to original dimensions
        return output.view(B, C, L)
