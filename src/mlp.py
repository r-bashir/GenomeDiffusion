#!/usr/bin/env python
# coding: utf-8

import math

import torch
import torch.nn as nn

from .all_models import SinusoidalTimeEmbeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Time Embeddings
class SinusoidalTimeEmbeddings(nn.Module):
    """Sinusoidal positional embedding (used for time steps in diffusion models)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# MLP
class MLP(nn.Module):
    """Deep MLP for noise prediction in diffusion models.

    A replacement for UNet1D that uses a deep MLP architecture instead of convolutional layers.
    Maintains the same interface as UNet1D for easy swapping in the diffusion model.

    Note: This is designed to work with SNP data where each example has a large number of markers.
    For memory efficiency, we use a dynamic architecture that scales with sequence length.

    Recommended deeper architecture:
        hidden_dims = [512, 1024, 512, 256, 128]
    This can improve model expressiveness and stability, especially at extreme timesteps.

    Example usage:
        # In your MLP __init__, replace the hidden_dims line with:
        hidden_dims = [512, 1024, 512, 256, 128]

        # Or, to experiment:
        # hidden_dims = [2048, 1024, 512, 256, 128]

    """

    def __init__(
        self,
        embedding_dim=64,  # Embedding dimension for time embeddings
        dim_mults=(1, 2, 4, 8),  # Used for compatibility with UNet1D interface
        channels=1,  # Input channels (SNP data has a single channel)
        with_time_emb=True,  # Whether to include time embeddings
        with_pos_emb=True,  # Not used in MLP but kept for interface compatibility
        resnet_block_groups=8,  # Not used in MLP but kept for interface compatibility
        seq_length=1000,  # Expected sequence length (number of SNP markers)
    ):
        super().__init__()

        self.channels = channels
        self.seq_length = seq_length
        self.with_time_emb = with_time_emb

        print(f"Initializing MLP with sequence length: {seq_length}")

        # Calculate input dimension (flattened sequence + time embedding)
        self.input_dim = channels * seq_length

        # Time embedding network
        if with_time_emb:
            self.time_dim = embedding_dim
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(embedding_dim),
                nn.Linear(embedding_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
        else:
            self.time_dim = 0
            self.time_mlp = None

        # Hidden dimensions
        hidden_dims = [2048, 1024, 512, 256, 128]

        print(f"Using hidden dimensions: {hidden_dims}")

        # MLP layers
        self.flatten = nn.Flatten()

        # Input layer
        self.input_layer = nn.Linear(self.input_dim + self.time_dim, hidden_dims[0])

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                self._make_residual_block(hidden_dims[i], hidden_dims[i + 1])
            )

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], self.input_dim)

    def _make_residual_block(self, in_dim, out_dim):
        """Create a residual block with layer normalization."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, time):
        """Forward pass for DeepMLP.

        Args:
            x (torch.Tensor): Input SNP data of shape [batch, channels, seq_len].
            time (torch.Tensor): Diffusion timesteps of shape [batch].

        Returns:
            torch.Tensor: Predicted noise with same shape as input.
        """
        batch_size, channels, seq_len = x.shape

        # Flatten the input
        x_flat = self.flatten(x)  # [batch, channels*seq_len]

        # Process time embedding if enabled
        if self.with_time_emb and self.time_mlp is not None:
            t_emb = self.time_mlp(time)  # [batch, time_dim]
            # Concatenate flattened input with time embedding
            x_t = torch.cat(
                [x_flat, t_emb], dim=1
            )  # [batch, channels*seq_len + time_dim]
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
        return output.view(batch_size, channels, seq_len)

    def gradient_checkpointing_enable(self):
        """Dummy method for compatibility with UNet1D interface."""
        pass


# Simple Linear Model
class SimpleLinearModel(nn.Module):
    """Simple linear model for noise prediction in diffusion models.

    A minimal implementation that uses a single linear layer without activation functions.
    Maintains the same interface as UNet1D and MLP for easy swapping in the diffusion model.

    This model follows the reviewer's suggestion to test with "the most simple single dense layer
    for n SNP inputs to n SNP outputs, no activation function" to evaluate baseline performance.
    """

    def __init__(
        self,
        embedding_dim=64,  # Unused but kept for interface compatibility
        dim_mults=(1, 2, 4, 8),  # Unused but kept for interface compatibility
        channels=1,  # Input channels (SNP data has a single channel)
        with_time_emb=False,  # Whether to include time embeddings
        with_pos_emb=False,  # Unused but kept for interface compatibility
        resnet_block_groups=8,  # Unused but kept for interface compatibility
        seq_length=1000,  # Expected sequence length (number of SNP markers)
    ):
        super().__init__()

        self.channels = channels
        self.seq_length = seq_length
        self.with_time_emb = with_time_emb

        print(f"Initializing SimpleLinearModel with sequence length: {seq_length}")

        # Calculate input dimension
        self.input_dim = channels * seq_length

        # Time embedding network (optional)
        if with_time_emb:
            self.time_dim = embedding_dim
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.time_dim = 0
            self.time_mlp = None

        # Flatten layer
        self.flatten = nn.Flatten()

        # Simple linear transformation
        # If time embedding is used, we add its dimensions to the input
        self.linear = nn.Linear(self.input_dim + self.time_dim, self.input_dim)

    def forward(self, x, time):
        """Forward pass for SimpleLinearModel.

        Args:
            x (torch.Tensor): Input SNP data of shape [batch, channels, seq_len].
            time (torch.Tensor): Diffusion timesteps of shape [batch].

        Returns:
            torch.Tensor: Predicted noise with same shape as input.
        """
        batch_size, channels, seq_len = x.shape

        # Flatten the input
        x_flat = self.flatten(x)  # [batch, channels*seq_len]

        # Process time embedding if enabled
        if self.with_time_emb and self.time_mlp is not None:
            t_emb = self.time_mlp(time)  # [batch, time_dim]
            # Concatenate flattened input with time embedding
            x_t = torch.cat(
                [x_flat, t_emb], dim=1
            )  # [batch, channels*seq_len + time_dim]
        else:
            x_t = x_flat

        # Apply linear transformation
        output = self.linear(x_t)

        # Reshape back to original dimensions
        return output.view(batch_size, channels, seq_len)

    def gradient_checkpointing_enable(self):
        """Dummy method for compatibility with UNet1D interface."""
        pass


def zero_out_model_parameters(model):
    """
    Sets all weights and biases of nn.Linear layers in the model to zero.
    This makes the model predict zeros for any input.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
