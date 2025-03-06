#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from functools import partial

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim=None, groups=8):
        super().__init__()
        
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dim))
        else:
            self.mlp = None
            
        self.block1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
        )
        
        self.res_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.mlp and time_emb is not None:
            h += self.mlp(time_emb)[:, :, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class ForwardDiffusion:
    def __init__(
            self,
            num_diffusion_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
        ):
        self._num_diffusion_timesteps = num_diffusion_timesteps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._betas = np.linspace(
            self._beta_start, self._beta_end, self._num_diffusion_timesteps
        )
        alphas_bar = self._get_alphas_bar()
        self._alphas = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)
        self._sigmas = torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32)
    
    def _get_alphas_bar(self):
        alphas = 1 - self._betas
        alphas_bar = np.cumprod(alphas)
        return alphas_bar
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._alphas[t]
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._sigmas[t]
    
    def sample(
            self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
        ) -> torch.Tensor:
        """Forward diffusion: q(xt | x0)."""
        return self.alpha(t)[:, None] * x0 + self.sigma(t)[:, None] * eps

class UNet1D(nn.Module):
    def __init__(
            self,
            embedding_dim=32,  # Memory-optimized from 64
            dim_mults=(1, 2, 4),  # Memory-optimized: removed 8
            channels=1,  # For SNP data
            with_time_emb=True,
            resnet_block_groups=8,
        ):
        super().__init__()
        
        self.channels = channels
        # Initialize with same channel count
        init_dim = embedding_dim
        out_dim = channels
        
        # Initial convolution with explicit padding
        self.init_conv = nn.Conv1d(channels, init_dim, kernel_size=3, padding=1)
        
        # For handling odd-length sequences at each level
        self.pad_if_odd = lambda x: torch.nn.functional.pad(x, (0, x.shape[-1] % 2), mode='replicate')
        
        # Compute dimensions (memory-optimized)
        self.dims = [init_dim] + [min(embedding_dim * m, 64) for m in dim_mults]  # Cap maximum channels at 64
        in_out = list(zip(self.dims[:-1], self.dims[1:]))
        
        # Time embeddings
        if with_time_emb:
            time_dim = embedding_dim
            self.time_mlp = nn.Sequential(
                SinusoidalPositionalEmbeddings(embedding_dim),
                nn.Linear(embedding_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        # UNet structure
        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)
        
        # Downsampling
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                nn.Conv1d(dim_out, dim_out, kernel_size=2, stride=2, padding=0) if not is_last else nn.Identity()
            ]))
        
        # Middle blocks
        mid_dim = self.dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        # Upsampling (memory-optimized)
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            # For the final upsampling block, output channels should match init_dim
            out_dim = init_dim if is_last else dim_in
            self.ups.append(nn.ModuleList([
                resnet_block(dim_out * 2, out_dim, time_emb_dim=time_dim),
                resnet_block(out_dim, out_dim, time_emb_dim=time_dim),
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=2, stride=2, padding=0, output_padding=0) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            resnet_block(self.dims[0], self.dims[0]),
            nn.Conv1d(self.dims[0], out_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Ensure input is [B, C, L]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Store original length
        orig_len = x.shape[-1]
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Time embeddings
        t = self.time_mlp(time) if self.time_mlp else None
        
        # Downsampling
        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            # Pad if odd length before downsampling
            x = self.pad_if_odd(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Upsampling with correct channel handling
        h = h[::-1]  # Reverse skip connections
        for i, (block1, block2, upsample) in enumerate(self.ups):
                # Get target dim for this level
            target_dim = self.dims[-(i+2)]  # Index from end since we're going up
            
            # Pad if odd length before upsampling
            x = self.pad_if_odd(x)
            x = upsample(x)
            
            # Concatenate and ensure proper channel count
            skip = h.pop(0)
            x = torch.cat((x, skip), dim=1)
            
            # Process through blocks
            x = block1(x, t)
            x = block2(x, t)
        
        x = self.final_conv(x)
        
        # Ensure output matches input length
        if x.shape[-1] > orig_len:
            x = x[..., :orig_len]
        elif x.shape[-1] < orig_len:
            x = torch.nn.functional.pad(x, (0, orig_len - x.shape[-1]), mode='replicate')
            
        return x
