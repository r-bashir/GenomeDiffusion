class DiffusionModel(nn.Module):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse diffusion (denoising)
    processes for SNP data. The forward process gradually adds noise to the data following
    a predefined schedule, while the reverse process learns to denoise the data using a
    UNet1D architecture.
    """

    def __init__(self, hparams: Dict):
        """
        Initialize the diffusion model with hyperparameters.

        Args:
            hparams: Dictionary containing model hyperparameters with the following structure:
                    {
                        'input_path': str,  # Path to input data
                        'diffusion': {'num_diffusion_timesteps': int, 'beta_start': float, 'beta_end': float},
                        'time_sampler': {'tmin': int, 'tmax': int},
                        'unet': {'embedding_dim': int, 'dim_mults': List[int], ...},
                        'data': {'seq_length': int, 'batch_size': int, 'num_workers': int}
                    }
        """
        super().__init__()

        # Set data shape
        self._data_shape = (hparams["unet"]["channels"], hparams["data"]["seq_length"])

        # Initialize components from hyperparameters
        self._forward_diffusion = DDPM(
            num_diffusion_timesteps=hparams["diffusion"]["num_diffusion_timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
        )

        self._time_sampler = UniformDiscreteTimeSampler(
            tmin=hparams["time_sampler"]["tmin"], tmax=hparams["time_sampler"]["tmax"]
        )

        self.unet = UNet1D(
            embedding_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            resnet_block_groups=hparams["unet"]["resnet_block_groups"],
        )

    def predict_added_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added during forward diffusion.
        This is a key part of the DDPM's reverse process - the UNet learns to
        predict what noise was added, which helps us denoise the data.

        Args:
            x (torch.Tensor): Noisy input tensor at timestep t
            t (torch.Tensor): Current timestep tensor

        Returns:
            torch.Tensor: Predicted noise that was added during forward diffusion
        """
        return self.unet(x, t)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: Predicted noise
        """
        return self.forward_step(batch)

    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: Predicted noise
        """
        # Sample time and noise
        t = self._time_sampler.sample(shape=(batch.shape[0],))
        eps = torch.randn_like(batch)

        # Forward diffusion process
        xt = self._forward_diffusion.sample(batch, t, eps)

        # Debugging print statements
        print(f"Shape before, ensuring 1 channel: {xt.shape}")

        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(xt.shape) == 2:  # If shape is (batch_size, seq_len)
            xt = xt.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        elif xt.shape[1] != 1:  # If incorrect number of channels
            print(f"Unexpected number of channels: {xt.shape[1]}, reshaping...")
            xt = xt[:, :1, :]  # Force to 1 channel

        print(f"Final shape before UNet: {xt.shape}")

        # Predict noise added during forward diffusion
        return self.predict_added_noise(xt, t)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.
        xt = alpha(t) * x0 + sigma(t)**2 * eps

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: MSE loss
        """
        # Sample noise
        eps = torch.randn_like(batch)

        # Get model predictions
        pred_eps = self.forward_step(batch)

        # Compute MSE loss
        return torch.mean((pred_eps - eps) ** 2)

    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes loss at specific timesteps.

        Args:
            x0 (torch.Tensor): Original clean input data (batch_size, channels, seq_len).
            eps (torch.Tensor): Sampled noise of same shape as x0.
            timesteps (torch.Tensor): Timesteps to evaluate.

        Returns:
            torch.Tensor: Loss values for each timestep.
        """
        losses = []
        for t in timesteps:
            t = int(t.item()) * torch.ones((x0.shape[0],), dtype=torch.int32)
            # Forward diffusion at timestep t
            xt = self._forward_diffusion.sample(x0, t, eps)

            # Predict noise that was added
            predicted_noise = self.predict_added_noise(xt, t)

            # Compute loss at this timestep
            loss = torch.mean((predicted_noise - eps) ** 2)
            losses.append(loss)

        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion step to estimate x_{t-1} given x_t.

        Args:
            xt (torch.Tensor): Noisy input at time t of shape (batch_size, channels, seq_len).
            t (int): Current timestep.

        Returns:
            torch.Tensor: Estimated previous timestep data.
        """
        # Move input to device and create timestep tensor
        xt = xt.to(device)
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=device)

        # Predict noise that was added during forward diffusion
        eps_pred = self.predict_added_noise(xt, t)

        # Compute reverse process parameters
        if t > 1:
            sqrt_a_t = self._forward_diffusion.alpha(t) / self._forward_diffusion.alpha(
                t - 1
            )
        else:
            sqrt_a_t = self._forward_diffusion.alpha(t)

        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t**2
        inv_sigma_t = 1.0 / self._forward_diffusion.sigma(t)

        # Compute mean and standard deviation
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        std = torch.sqrt(beta_t)

        # Add noise scaled by standard deviation
        z = torch.randn_like(xt, device=device)
        return mean + std * z

    def sample(self, sample_size: int) -> torch.Tensor:
        """
        Samples from the learned reverse diffusion process without conditioning.
        Implements the full reverse diffusion chain, starting from pure noise and
        gradually denoising to generate SNP data.

        Args:
            sample_size (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (sample_size, channels, seq_len).
        """
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn((sample_size,) + self._data_shape)

            # Gradually denoise
            for t in range(self._forward_diffusion.tmax, 0, -1):
                x = self._reverse_process_step(x, t)

            # Ensure output is in correct range (typically [0,1] for SNP data)
            x = torch.clamp(x, 0, 1)
        return x
