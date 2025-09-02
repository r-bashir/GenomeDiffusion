import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_model_from_checkpoint, set_seed, setup_logging
from utils.ddpm_utils import (
    compute_locality_metrics,
    format_metrics_report,
    plot_locality_analysis,
)

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse Arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Locality Test")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    return parser.parse_args()


# Main Function
def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="locality")
    logger.info("Starting run_locality_test script.")

    # Set global seed
    set_seed(seed=42)

    try:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully from checkpoint on %s", device)
        logger.info("Model config loaded from checkpoint:")
        print(f"\n{config}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    output_dir = Path(args.checkpoint).parent.parent / "locality_test"
    output_dir.mkdir(exist_ok=True)

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    test_batch = next(iter(test_loader)).to(device)
    logger.info(f"Batch shape: {test_batch.shape}, and dim: {test_batch.dim()}")

    # Select a single sample and ensure shape [1, 1, seq_len]
    logger.info(f"Adding channel dim, and selecting single sample")
    sample_idx = 0
    x0 = test_batch[sample_idx : sample_idx + 1].unsqueeze(1)
    logger.info(f"x0 shape: {x0.shape} and dim: {x0.dim()}")
    logger.info(f"x0 unique values: {torch.unique(x0)}")

    # === BEGIN: Locality Experiment ===
    logger.info("Running SNP locality experiment (varying SNP 60)...")
    batch_size = 1
    snp_index = 59
    true_noise_values = np.round(np.arange(0, 0.51, 0.01), 2)
    eps_thetas = []

    # Run experiment
    for val in true_noise_values:
        x0_test = x0.clone()
        x0_test[..., snp_index] = val
        with torch.no_grad():
            eps_theta = model.noise_predictor(
                x0_test, torch.zeros(batch_size, device=device)
            )

            mean_pred_noise = eps_theta.mean().item()
            print(f"Mean of predicted noise (ε_θ): {mean_pred_noise:.6f}")
        eps_thetas.append(eps_theta.cpu().numpy().squeeze())

    eps_thetas = np.stack(eps_thetas)  # shape: (len(true_noise_values), seq_len)

    # Compute metrics and generate plots
    metrics = compute_locality_metrics(true_noise_values, eps_thetas, snp_index)
    plot_locality_analysis(true_noise_values, eps_thetas, snp_index, output_dir)

    # Save metrics report
    report = format_metrics_report(metrics)
    with open(output_dir / f"snp{snp_index+1}_locality_metrics.txt", "w") as f:
        f.write(report)
    logger.info("Metrics Report:")
    print(f"\n{report}\n")

    # Predicted noise histogram at SNP 60
    plt.figure(figsize=(7, 5))
    plt.hist(eps_thetas.flatten(), bins=100, density=True)
    plt.xlabel(f"Predicted Noise (ε_θ) at SNP {snp_index+1}")
    plt.ylabel("Density")
    plt.title("Noise Histogram")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"snp{snp_index+1}_locality_hist.png", dpi=150)
    plt.close()

    # === END: Locality Experiment ===

    logger.info("Locality test complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
