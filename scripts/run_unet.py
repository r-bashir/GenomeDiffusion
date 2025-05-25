#!/usr/bin/env python
# coding: utf-8

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from .utils import load_config


def main():
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load config
    config = load_config("config.yaml")
    input_path = config.get("input_path")

    print(f"Using device: {device}")

    # Initialize dataset
    print("\nInitializing dataset...")
    dataset = SNPDataset(input_path)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Get batch
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"Batch shape [B, seq_len]: {batch.shape}")

    # Prepare input (ensure [B, C, L] format for UNet)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    print(f"Batch shape [B, C, seq_len]: {batch.shape}")

    # Initialize DDPM
    print("\nInitializing DDPM...")
    forward_diffusion = DDPM(
        num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02
    )

    # Move forward diffusion tensors to device
    forward_diffusion._alphas = forward_diffusion._alphas.to(device)
    forward_diffusion._sigmas = forward_diffusion._sigmas.to(device)

    # Initialize UNet
    print("\nInitializing UNet1D...")
    unet_config = config.get("unet", {})
    unet = UNet1D(
        embedding_dim=unet_config.get("embedding_dim", 32),
        dim_mults=unet_config.get("dim_mults", [1, 2, 4]),
        channels=unet_config.get("channels", 1),
        with_time_emb=unet_config.get("with_time_emb", True),
        resnet_block_groups=unet_config.get("resnet_block_groups", 8),
    ).to(device)

    # Test UNet with different timesteps
    timesteps = [0, 250, 500, 750, 999]

    # For visualization, use a single sample
    x0 = batch[0:1]  # [1, 1, seq_len]
    print(f"Single sample shape: {x0.shape}")

    # Create figure for visualization
    fig, axs = plt.subplots(len(timesteps), 3, figsize=(15, 15))

    for i, t in enumerate(timesteps):
        # Sample noise
        noise = torch.randn_like(x0)
        t_tensor = torch.tensor([t], device=device)

        # Apply forward diffusion to get noisy data
        noisy_data = forward_diffusion.sample(x0, t_tensor, noise)

        # UNet should predict the noise that was added
        noise_pred = unet(noisy_data, t_tensor)

        # Ensure dimensions match
        if noise_pred.shape != noise.shape:
            print(
                f"WARNING: Shape mismatch at t={t}: noise_pred {noise_pred.shape}, noise {noise.shape}"
            )
            if noise_pred.shape[2] != noise.shape[2]:
                noise_pred = F.interpolate(
                    noise_pred, size=noise.shape[2], mode="linear"
                )

        # Calculate MSE loss
        mse_loss = F.mse_loss(noise_pred, noise).item()
        print(f"t={t}, MSE Loss: {mse_loss:.6f}")

        # Plot original data, noisy data, and noise prediction
        axs[i, 0].plot(x0[0, 0].cpu().detach().numpy(), linewidth=1)
        axs[i, 0].set_title(f"t={t}: Original")

        axs[i, 1].plot(noisy_data[0, 0].cpu().detach().numpy(), linewidth=1)
        axs[i, 1].set_title(f"t={t}: Noisy")

        # Plot actual vs predicted noise
        axs[i, 2].plot(noise[0, 0].cpu().detach().numpy(), linewidth=1, label="Actual")
        axs[i, 2].plot(
            noise_pred[0, 0].cpu().detach().numpy(),
            linewidth=1,
            label="Predicted",
            alpha=0.7,
        )
        axs[i, 2].set_title(f"t={t}: Noise (MSE: {mse_loss:.4f})")
        axs[i, 2].legend()

    plt.tight_layout()
    plt.savefig("unet_noise_prediction.png")
    plt.close()

    # Test with full batch
    print("\nTesting with full batch...")
    t_batch = torch.tensor([500] * batch.shape[0], device=device)
    noise_batch = torch.randn_like(batch)

    # Apply forward diffusion
    noisy_batch = forward_diffusion.sample(batch, t_batch, noise_batch)

    # Predict noise with UNet
    pred_noise_batch = unet(noisy_batch, t_batch)

    # Calculate batch MSE loss
    batch_mse = F.mse_loss(pred_noise_batch, noise_batch).item()
    print(f"Batch MSE Loss: {batch_mse:.6f}")

    # Test gradient flow
    print("\nTesting gradient flow...")

    # Zero gradients
    if hasattr(unet, "zero_grad"):
        unet.zero_grad()

    # Forward pass
    pred_noise = unet(noisy_batch, t_batch)

    # Calculate loss
    loss = F.mse_loss(pred_noise, noise_batch)

    # Backward pass
    loss.backward()

    # Check if gradients exist
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in unet.parameters()
    )
    print(f"UNet has gradients: {has_gradients}")

    print("\nUNet test complete!")


if __name__ == "__main__":
    main()
