import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DiffusionModel
from src.utils import load_config, set_seed, setup_logging
from utils.ddpm_utils import plot_locality_analysis

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Model
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Loads a DiffusionModel from a checkpoint and moves it to the specified device.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        model: The loaded DiffusionModel (on the correct device, in eval mode)
        config: The config/hparams dictionary from the checkpoint
    """
    from src import DiffusionModel

    model = DiffusionModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    config = model.hparams
    model = model.to(device)
    model.eval()
    return model, config


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
    perturbed_values = np.round(np.arange(0, 0.51, 0.01), 2)
    outputs = []

    # Run experiment
    for val in perturbed_values:
        x0_test = x0.clone()
        x0_test[..., snp_index] = val
        with torch.no_grad():
            noise = model.noise_predictor(
                x0_test, torch.zeros(batch_size, device=device)
            )
        outputs.append(noise.cpu().numpy().squeeze())

    outputs = np.stack(outputs)  # shape: (len(perturbed_values), seq_len)

    # Plot perturb_value vs model output at SNP 59
    plt.figure(figsize=(7, 5))
    plt.plot(perturbed_values, outputs[:, snp_index], marker="o")
    plt.xlabel(f"Perturbed Value at SNP {snp_index+1}")
    plt.ylabel(f"Model Output at SNP {snp_index+1}")
    plt.title(f"Locality Curve: SNP {snp_index+1}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"snp{snp_index+1}_locality_curve.png", dpi=150)
    plt.close()

    # Plot locality analysis
    plot_locality_analysis(perturbed_values, outputs, snp_index, output_dir)

    # === END: Locality Experiment ===

    logger.info("Locality test complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
