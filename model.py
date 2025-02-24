import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🔹 Sinusoidal Positional Encoding for Time Steps
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: Tensor of shape [batch_size] containing the time steps.

        Returns:
            Tensor of shape [batch_size, embedding_dim] (Sinusoidal embeddings)
        """
        half_dim = self.embedding_dim // 2
        e = math.log(10000.0) / (half_dim - 1)
        inv_freq = torch.exp(-e * torch.arange(half_dim).float()).to(timesteps.device)
        emb = timesteps[:, None] * inv_freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.unsqueeze(2)  # Reshape for 1D conv compatibility

# 🔹 Residual Block (Same as before)
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        x = F.silu(self.conv1(x))
        x = self.norm(self.conv2(x))
        return F.silu(x + residual)

# 🔹 Downsampling Block
class DownBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = ResBlock1D(in_channels, out_channels)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.resblock(x)
        return self.downsample(x), x  # Downsampled + skip connection

# 🔹 Upsampling Block
class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock = ResBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.resblock(x)

# 🔹 U-Net with Time Embeddings
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, time_emb_dim=32):
        super().__init__()
        self.time_embed = nn.Linear(time_emb_dim, base_channels)
        self.down1 = DownBlock1D(in_channels, base_channels)
        self.down2 = DownBlock1D(base_channels, base_channels * 2)
        self.down3 = DownBlock1D(base_channels * 2, base_channels * 4)

        self.mid = ResBlock1D(base_channels * 4, base_channels * 4)

        self.up3 = UpBlock1D(base_channels * 4, base_channels * 2)
        self.up2 = UpBlock1D(base_channels * 2, base_channels)
        self.up1 = UpBlock1D(base_channels, out_channels)

    def forward(self, x, t_emb):
        """
        x: Noisy SNP data of shape [batch_size, 1, num_snps]
        t_emb: Time embeddings of shape [batch_size, time_emb_dim]
        """
        t_emb = self.time_embed(t_emb).unsqueeze(2)  # Expand for broadcasting in 1D convs

        x, skip1 = self.down1(x + t_emb)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        x = self.mid(x)

        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        return x

# 🔹 Diffusion Model with DDPM
class DDPM(nn.Module):
    def __init__(self, snp_length, unet_channels=32, time_emb_dim=32, num_timesteps=1000):
        super().__init__()
        self.unet = UNet1D(in_channels=1, out_channels=1, base_channels=unet_channels, time_emb_dim=time_emb_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Precompute variance schedule
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(0.0001, 0.02, num_timesteps)  # Linear schedule
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward(self, x_noisy, t):
        """
        x_noisy: Noisy SNP data [batch_size, 1, num_snps]
        t: Time step tensor [batch_size]
        """
        t_emb = self.time_embedding(t)  # Get sinusoidal embeddings
        return self.unet(x_noisy, t_emb)

    def sample(self, batch_size, snp_length, device="cuda"):
        """
        Generate SNP samples using DDPM.
        """
        x = torch.randn(batch_size, 1, snp_length, device=device)  # Start from pure noise

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x, t_tensor)

            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            # Reverse diffusion step
            mean = (x - beta_t * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                noise = torch.randn_like(x)  # Add noise except at t=0
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean  # Last step, no noise

        return x  # Return denoised SNP sequences
