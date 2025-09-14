import numpy as np
import torch
import torch.nn as nn


# ResNet Block for U-Net
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        # Skip connection handling if channels change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x + shortcut


# Down Block for U-Net
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(in_channels if i == 0 else out_channels, out_channels)
                for i in range(block_depth)
            ]
        )
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        skips = []
        for res_block in self.res_blocks:
            x = res_block(x)
            skips.append(x)
        return self.downsample(x), skips


# Up Block for U-Net
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(in_channels if i == 0 else out_channels, out_channels)
                for i in range(block_depth)
            ]
        )

    def forward(self, x, skips):
        x = self.upsample(x)
        for res_block in self.res_blocks:
            skip = skips.pop()
            x = x + skip  # Add skip connection
            x = res_block(x)
        return x


# Full U-Net Model (unchanged from your implementation)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, widths, block_depth):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        prev_channels = in_channels
        for width in widths:
            self.down_blocks.append(DownBlock(prev_channels, width, block_depth))
            prev_channels = width

        self.mid_blocks = nn.ModuleList(
            [
                ResNetBlock(widths[-1], 2 * widths[-1]),
                ResNetBlock(2 * widths[-1], widths[-1]),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for width in reversed(widths):
            self.up_blocks.append(
                UpBlock(
                    width, out_channels if width == widths[0] else width, block_depth
                )
            )

    def forward(self, x):
        skips_list = []
        for down in self.down_blocks:
            x, skips = down(x)
            skips_list.append(skips)

        for mid in self.mid_blocks:
            x = mid(x)

        skips_list.reverse()  # Reverse skip connections for upsampling
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips_list[i])  # Use skip connections for upsampling

        return x


# SNP-specific Diffusion Process
class SNPDiffusionProcess:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
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

        # SNP-specific valid values (excluding missing value 9.0)
        self.valid_values = torch.tensor([0.0, 0.5, 1.0])

    @property
    def tmin(self):
        return 1

    @property
    def tmax(self):
        return self._num_diffusion_timesteps

    def _get_alphas_bar(self) -> np.ndarray:
        alphas = 1.0 - self._betas
        alphas_bar = np.cumprod(alphas)
        # Add 1 in front to simplify indexing (t=0 means no noise)
        alphas_bar = np.concatenate(([1.0], alphas_bar))
        return alphas_bar

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._alphas.to(t.device)[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._sigmas.to(t.device)[t.long()]

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Draws samples from the forward diffusion process q(xt|x0), handling missing data."""
        # Create a mask for valid (non-missing) values
        valid_mask = x0 != 9.0

        # Apply diffusion only to valid values
        alpha_t = self.alpha(t).view(
            -1, 1, 1, 1
        )  # Shape for broadcasting with 4D tensor
        sigma_t = self.sigma(t).view(-1, 1, 1, 1)

        # For valid values, apply diffusion; for missing values, keep them as is
        xt = torch.where(
            valid_mask,
            alpha_t * x0 + sigma_t * eps,
            torch.tensor(9.0, device=x0.device),
        )

        return xt

    def predict_denoise(
        self, model_output: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Converts model output to predicted x0 and handles discretization."""
        # Mask for missing values
        missing_mask = xt == 9.0

        # Predict x0 from model output for non-missing values
        alpha_t = self.alpha(t).view(-1, 1, 1, 1)
        sigma_t = self.sigma(t).view(-1, 1, 1, 1)
        predicted_x0 = (xt - sigma_t * model_output) / alpha_t

        # Discretize predictions to valid SNP values
        # Find closest valid value for each prediction
        flat_preds = predicted_x0[~missing_mask].view(-1, 1)
        distances = torch.abs(flat_preds - self.valid_values.to(xt.device))
        closest_indices = torch.argmin(distances, dim=1)
        discretized_values = self.valid_values[closest_indices].to(xt.device)

        # Create the final result tensor
        result = torch.full_like(xt, 9.0)  # Start with all missing
        result[~missing_mask] = discretized_values

        return result


# Complete SNP Diffusion Model with U-Net
class SNPDiffusionModel(nn.Module):
    def __init__(
        self,
        snp_shape,  # Expected shape of SNP data after reshaping to 2D
        embedding_dim=32,
        widths=(32, 64, 96, 128),
        block_depth=2,
    ):
        super().__init__()
        self.snp_shape = snp_shape

        # Initialize U-Net
        self.unet = UNet(
            in_channels=embedding_dim * 2,  # Combined embedding of data and noise level
            out_channels=embedding_dim,
            widths=widths,
            block_depth=block_depth,
        )

        # Embedding for noise levels
        self.embedding = SNPSinusoidalEmbedding(embedding_dim=embedding_dim)

        # Input projection to create embeddings
        self.input_proj = nn.Conv2d(
            in_channels=1, out_channels=embedding_dim, kernel_size=1
        )

        # Output projection
        self.output_proj = nn.Conv2d(
            in_channels=embedding_dim, out_channels=1, kernel_size=1
        )
        nn.init.zeros_(self.output_proj.weight)  # Initialize output layer to zeros

        # Diffusion process
        self.diffusion = SNPDiffusionProcess()

    def _reshape_to_2d(self, x):
        """Reshape SNP data to 2D format suitable for convolution"""
        batch_size = x.shape[0]
        # Reshape to [batch_size, 1, height, width]
        return x.view(batch_size, 1, self.snp_shape[0], self.snp_shape[1])

    def _reshape_to_original(self, x, original_shape):
        """Reshape back to original format"""
        batch_size = x.shape[0]
        return x.view(batch_size, *original_shape)

    def forward(self, noisy_snps, t):
        """Forward pass through the model

        Args:
            noisy_snps: Tensor of noisy SNP data, shape [batch_size, ...]
            t: Diffusion timesteps, shape [batch_size]

        Returns:
            Predicted noise
        """
        original_shape = noisy_snps.shape[1:]

        # Reshape to 2D for convolution
        x = self._reshape_to_2d(noisy_snps)

        # Get noise level embeddings
        noise_embeddings = self.embedding(t)
        # Expand dimensions to match spatial dimensions
        noise_embeddings = noise_embeddings.unsqueeze(-1).unsqueeze(-1)
        noise_embeddings = noise_embeddings.expand(-1, -1, x.shape[2], x.shape[3])

        # Project input to embedding dimension
        x_emb = self.input_proj(x)

        # Concatenate embeddings
        combined = torch.cat([x_emb, noise_embeddings], dim=1)

        # Forward through U-Net
        unet_output = self.unet(combined)

        # Project back to original dimension
        output = self.output_proj(unet_output)

        # Reshape to original format
        output = self._reshape_to_original(output, original_shape)

        return output

    def loss_fn(self, x0, t=None):
        """Calculate the loss for training

        Args:
            x0: Original SNP data, shape [batch_size, ...]
            t: Optional timesteps. If None, will be sampled randomly.

        Returns:
            Loss value
        """
        batch_size = x0.shape[0]

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(
                self.diffusion.tmin,
                self.diffusion.tmax + 1,
                (batch_size,),
                device=x0.device,
            )

        # Sample noise
        noise = torch.randn_like(x0)

        # Get noisy samples
        noisy_snps = self.diffusion.sample(x0, t, noise)

        # Get model prediction (predicted noise)
        pred_noise = self(noisy_snps, t)

        # Calculate loss only on non-missing values
        valid_mask = x0 != 9.0

        # Mean squared error on predicted noise
        loss = torch.mean((pred_noise[valid_mask] - noise[valid_mask]) ** 2)

        return loss

    @torch.no_grad()
    def sample(self, shape, device, starting_noise=None):
        """Generate samples using the diffusion model

        Args:
            shape: Shape of samples to generate
            device: Device to use
            starting_noise: Optional starting noise. If None, will be sampled randomly.

        Returns:
            Generated samples
        """
        # Start from random noise if not provided
        if starting_noise is None:
            x = torch.randn(*shape, device=device)
        else:
            x = starting_noise

        # Iterative denoising
        for t in range(self.diffusion.tmax, 0, -1):
            # Create timestep batch
            t_batch = torch.full((shape[0],), t, device=device)

            # Get model prediction (predicted noise)
            pred_noise = self(x, t_batch)

            # Get alpha and sigma for this step
            alpha_t = self.diffusion.alpha(t_batch).view(-1, 1, 1, 1)
            sigma_t = self.diffusion.sigma(t_batch).view(-1, 1, 1, 1)

            # No noise at the last step
            if t > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            # Denoise step
            x = (x - sigma_t * pred_noise) / alpha_t

            # Add noise according to posterior distribution
            posterior_std = sigma_t * z
            x = x + posterior_std

        # Final discretization
        return self.diffusion.predict_denoise(
            pred_noise, x, torch.ones(shape[0], device=device)
        )


# Example usage:
def train_snp_diffusion_model(
    snp_data, num_epochs=100, batch_size=32, learning_rate=1e-4
):
    """Train the SNP diffusion model

    Args:
        snp_data: Tensor of SNP data with shape [num_samples, ...]
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Trained model
    """
    # Determine appropriate 2D shape for convolution
    # For data of shape [160858, 2067], we might reshape to something like [401, 401]
    # Exact shape depends on your specific requirements
    data_size = (
        snp_data.shape[1] * snp_data.shape[2]
        if len(snp_data.shape) > 2
        else snp_data.shape[1]
    )
    side_length = int(np.ceil(np.sqrt(data_size)))
    snp_shape = (side_length, side_length)

    # Initialize model
    model = SNPDiffusionModel(snp_shape=snp_shape)

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    snp_data = snp_data.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle and batch data
        indices = torch.randperm(len(snp_data))
        total_loss = 0.0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            x_batch = snp_data[batch_indices]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass and loss calculation
            loss = model.loss_fn(x_batch)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        avg_loss = total_loss / (len(snp_data) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return model
