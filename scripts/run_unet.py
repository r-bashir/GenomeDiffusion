#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SNPDataset
from src.forward_diffusion import ForwardDiffusion
from src.unet import UNet1D
from src.utils import load_config, set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="UNet1D Sanity Checks: shapes, MSE vs true noise, gradient flow."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unet_sanity"))
    parser.add_argument(
        "--plot", action="store_true", help="Save a simple PNG visualization."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output",
    )
    return parser.parse_args()


def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="unet")
    logger.info("Starting 'run_unet.py' script.")

    # Set global seed
    set_seed(seed=42)

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = SNPDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    batch = next(iter(dataloader))  # Shape: [B, L]
    logger.info(f"Batch shape [B, L]: {batch.shape}, and dim: {batch.dim()}")

    # Prepare input (ensure [B, C, L] format for UNet)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, L]
    logger.info(f"Batch shape [B, C, L]: {batch.shape}")

    # Prepare output directory
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize ForwardDiffusion
    logger.info("Initializing ForwardDiffusion...")
    forward_diff = ForwardDiffusion(
        time_steps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        beta_schedule=config["diffusion"]["beta_schedule"],
    ).to(device)

    # Initialize UNet
    logger.info("Initializing UNet1D...")
    unet_config = config.get("unet", {})
    unet = UNet1D(
        embedding_dim=unet_config.get("embedding_dim", 32),
        dim_mults=unet_config.get("dim_mults", [1, 2, 4]),
        channels=unet_config.get("channels", 1),
        with_time_emb=unet_config.get("with_time_emb", True),
        with_pos_emb=unet_config.get("with_pos_emb", False),
        norm_groups=unet_config.get("norm_groups", 8),
        seq_length=config["data"]["seq_length"],
        edge_pad=config["unet"]["edge_pad"],
        enable_checkpointing=config["unet"]["enable_checkpointing"],
        strict_resize=config["unet"].get("strict_resize", True),
        pad_value=config["unet"].get("pad_value", 0.0),
        use_attention=config["unet"].get("use_attention", False),
        attention_heads=config["unet"].get("attention_heads", 4),
        attention_dim_head=config["unet"].get("attention_dim_head", 32),
    ).to(device)

    # Print model information
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32

    # Print model configuration and statistics
    print("\nModel Configuration:")
    print(f"  - Embedding dim: {unet_config.get('embedding_dim', 32)}")
    print(f"  - Dimension multipliers: {unet_config.get('dim_mults', [1, 2, 4])}")
    print(f"  - Norm groups: {unet_config.get('norm_groups', 8)}")
    print(f"  - Sequence length: {config['data']['seq_length']}")
    print("Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {model_size_mb:.2f} MB\n")

    # Optional: print expected down/up length flow to aid debugging
    if args.debug:

        def down_lengths(L, steps):
            lens = [L]
            cur = L
            for _ in range(steps):
                # mirror DownsampleConv stride-2 conv with optional 1 pad when odd
                if cur % 2 != 0:
                    cur = cur + 1
                cur = cur // 2
                lens.append(cur)
            return lens

        dims = unet_config.get("dim_mults", [1, 2, 4])
        steps = len(dims)
        L0 = batch.size(-1)
        dflow = down_lengths(L0, steps)
        print("Expected length flow (down path):", dflow)
        # The up path mirrors the down path in reverse with ConvTranspose1d
        upflow = list(reversed(dflow))
        print("Expected length flow (up path):  ", upflow)

    # Test UNet with different timesteps
    timesteps = [0, 250, 500, 750, 999]

    # For visualization, use a single sample
    x0 = batch[0:1]  # [1, 1, seq_len]
    print(f"Single sample shape: {x0.shape}\n")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(len(timesteps), 3, figsize=(12, 3 * len(timesteps)))

    for i, t in enumerate(timesteps):
        # Sample noise
        noise = torch.randn_like(x0)
        t_tensor = torch.tensor([t], device=device)

        # Apply forward diffusion to get noisy data
        noisy_data = forward_diff.sample(x0, t_tensor, noise)

        # UNet should predict the noise that was added
        noise_pred = unet(noisy_data, t_tensor)

        # Ensure dimensions match strictly (model should handle this or raise early)
        assert noise_pred.shape == noise.shape, (
            f"Shape mismatch at t={t}: noise_pred {noise_pred.shape}, noise {noise.shape}. "
            f"Check UNet strict_resize/padding configuration."
        )

        # Calculate MSE loss
        mse_loss = F.mse_loss(noise_pred, noise).item()
        print(f"t={t}, MSE Loss: {mse_loss:.6f}")

        if args.plot:
            # Plot original data, noisy data, and noise prediction
            axs[i, 0].plot(x0[0, 0].cpu().detach().numpy(), linewidth=1)
            axs[i, 0].set_title(f"t={t}: Original")

            axs[i, 1].plot(noisy_data[0, 0].cpu().detach().numpy(), linewidth=1)
            axs[i, 1].set_title(f"t={t}: Noisy")

            # Plot actual vs predicted noise
            axs[i, 2].plot(
                noise[0, 0].cpu().detach().numpy(), linewidth=1, label="Actual"
            )
            axs[i, 2].plot(
                noise_pred[0, 0].cpu().detach().numpy(),
                linewidth=1,
                label="Predicted",
                alpha=0.7,
            )
            axs[i, 2].set_title(f"t={t}: Noise (MSE: {mse_loss:.4f})")
            axs[i, 2].legend()

    if args.plot:
        fig.tight_layout()
        fig.savefig(output_dir / "unet_prediction.png")
        plt.close()

    # Test with full batch
    print("\nTesting with full batch...")
    t_batch = torch.tensor([500] * batch.shape[0], device=device)
    noise_batch = torch.randn_like(batch)

    # Apply forward diffusion
    noisy_batch = forward_diff.sample(batch, t_batch, noise_batch)

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

    print(f"UNet has gradients: {has_gradients}\n")

    # End of UNet1D Testing
    logger.info("UNet1D complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
