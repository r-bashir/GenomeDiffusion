#!/usr/bin/env python
# coding: utf-8


# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .utils import load_config


def main() -> None:
    """Run tests for the DDPM implementation.
    This function performs several tests:
    1. Tests the forward diffusion process with different timesteps
    2. Tests batch processing with varied timesteps
    3. Validates noise levels through signal-to-noise ratio analysis

    Results are saved as plots and metrics are printed to stdout.
    """
    try:
        print(f"Using device: {DEVICE}")

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Load config
        config = load_config("../config.yaml")
        print("\nInitializing dataset...")

        # Initialize dataset
        dataset = SNPDataset(
            input_path=config.get("input_path"),
            seq_length=config.get("data").get("seq_length"),
        )

        # Initialize dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=config.get("batch_size"),
            shuffle=True,
            num_workers=config.get("num_workers"),
            pin_memory=True,
        )

        # Get batch
        batch = next(iter(train_loader))  # Shape: [B, seq_len]
        batch = batch.unsqueeze(1).to(DEVICE)  # [B, 1, seq_len]
        # ------------------ Test DDPM
        print("\nTesting DDPM...")
        forward_diffusion = DDPM(
            diffusion_steps=config.get("diffusion").get("diffusion_steps"),
            beta_start=config.get("diffusion").get("beta_start"),
            beta_end=config.get("diffusion").get("beta_end"),
        )

        # Move forward diffusion tensors to device
        forward_diffusion.to(DEVICE)

        # Test different timesteps
        timesteps = [0, 250, 500, 750, 999]

        # 1. Test with single sample for visualization
        x0_single = batch[0:1]  # Take first sample [1, seq_len]

        # Create figure for visualization
        plt.figure(figsize=(15, 5))

        for i, t in enumerate(timesteps):
            # Sample noise
            eps = torch.randn_like(x0_single).to(DEVICE)
            t_tensor = torch.tensor([t], device=DEVICE)
            # Apply forward diffusion
            xt = forward_diffusion.sample(x0_single, t_tensor, eps)

            print(f"\nt_tensor.shape: {t_tensor.shape}")
            print(f"eps.shape: {eps.shape}")
            print(f"xt.shape: {xt.shape}")
            # Plot results - move to CPU for plotting if needed
            xt_cpu = xt.cpu() if DEVICE.type == "cuda" else xt
            plt.subplot(1, len(timesteps), i + 1)
            plt.plot(
                xt_cpu[0, 0].detach().numpy(), linewidth=1, color="blue", alpha=0.8
            )
            plt.title(f"t={t}")

        plt.tight_layout()
        plt.savefig("ddpm_timesteps.png")
        plt.close()

        # 2. Test with full batch and different timesteps
        print("\n2. Testing with full batch...")

        # Assign different timesteps to each batch element
        batch_size = batch.shape[0]
        # TODO: use a time sampler instead of linspace
        varied_timesteps = torch.linspace(0, 999, batch_size).long().to(DEVICE)
        print(f"Varied timesteps shape: {varied_timesteps.shape}")

        # Sample noise
        eps = torch.randn_like(batch).to(DEVICE)
        # Apply forward diffusion
        xt_batch = forward_diffusion.sample(batch, varied_timesteps, eps)
        print(f"Output batch shape: {xt_batch.shape}")

        # 3. Validate noise levels
        print("\n3. Validating noise levels:")
        t_start = torch.tensor([0], device=DEVICE)
        t_mid = torch.tensor([500], device=DEVICE)
        t_end = torch.tensor([999], device=DEVICE)

        # Use same noise for fair comparison
        same_noise = torch.randn_like(x0_single).to(DEVICE)
        # Get samples at different timesteps
        x_start = forward_diffusion.sample(x0_single, t_start, same_noise)
        x_mid = forward_diffusion.sample(x0_single, t_mid, same_noise)
        x_end = forward_diffusion.sample(x0_single, t_end, same_noise)

        # Calculate signal-to-noise ratio
        # For 3D tensors, calculate variance along the sequence dimension
        if len(x_start.shape) == 3:
            # Calculate variance along the sequence dimension (dim=2)
            snr_start = (
                x_start.var(dim=2).mean() / (x_start - x0_single).var(dim=2).mean()
            ).item()
            snr_mid = (
                x_mid.var(dim=2).mean() / (x_mid - x0_single).var(dim=2).mean()
            ).item()
            snr_end = (
                x_end.var(dim=2).mean() / (x_end - x0_single).var(dim=2).mean()
            ).item()
        else:
            # Original calculation for 2D tensors
            snr_start = (x_start.var() / (x_start - x0_single).var()).item()
            snr_mid = (x_mid.var() / (x_mid - x0_single).var()).item()
            snr_end = (x_end.var() / (x_end - x0_single).var()).item()

        print(f"SNR at t={forward_diffusion.tmin}: {snr_start:.4f}")
        print(f"SNR at t=500: {snr_mid:.4f}")
        print(f"SNR at t=999: {snr_end:.4f}")

    except Exception as e:
        print(f"Error in main(): {e}")


if __name__ == "__main__":
    main()
