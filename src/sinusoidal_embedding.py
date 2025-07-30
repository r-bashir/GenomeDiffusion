#!/usr/bin/env python
# coding: utf-8

import math

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Time Embeddings
class SinusoidalTimeEmbeddings(nn.Module):
    """
    Generates sinusoidal embeddings for scalar time steps, as used in diffusion models.

    Args:
        dim (int): The embedding dimension (must be even).

    Input shape:
        - time: Tensor of shape (batch_size,) or (batch_size, 1), containing time steps (float or int).

    Output shape:
        - Tensor of shape (batch_size, dim), containing sinusoidal embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Compute sinusoidal time embeddings.

        Args:
            time (torch.Tensor): Tensor of shape (batch_size,) or (batch_size, 1) with time steps.
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, dim)
        """
        # Ensure time is (batch_size, 1)
        if time.dim() == 1:
            time = time.unsqueeze(-1)

        device = time.device
        half_dim = self.dim // 2

        # Compute frequency bands
        freq_const = math.log(10000) / (half_dim - 1)
        freq_bands = torch.exp(torch.arange(half_dim, device=device) * -freq_const)

        # Apply frequencies to time
        args = time * freq_bands  # (batch_size, half_dim)

        # Concatenate sine and cosine
        embeddings = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch_size, dim)
        return embeddings


# Position Embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal embeddings for 1D sequence positions (as in Transformers).

    Args:
        dim (int): The embedding dimension (must be even).

    Input shape:
        - positions: Tensor of shape (batch_size, seq_len) with integer position indices (or float positions).

    Output shape:
        - Tensor of shape (batch_size, seq_len, dim), containing sinusoidal embeddings for each position.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, positions):
        """
        Compute sinusoidal position embeddings for a batch of sequences.

        Args:
            positions (torch.Tensor): Tensor of shape (batch_size, seq_len) with position indices.
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, seq_len, dim)
        """
        device = positions.device
        half_dim = self.dim // 2

        # Compute frequency bands
        freq_const = math.log(10000) / (half_dim - 1)
        freq_bands = torch.exp(torch.arange(half_dim, device=device) * -freq_const)

        # Apply frequencies to positions (broadcast)
        args = positions.unsqueeze(-1) * freq_bands  # (batch_size, seq_len, half_dim)

        # Concatenate sine and cosine
        embeddings = torch.cat(
            [args.sin(), args.cos()], dim=-1
        )  # (batch_size, seq_len, dim)
        return embeddings
