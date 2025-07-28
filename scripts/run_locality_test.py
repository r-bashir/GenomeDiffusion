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


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    model = DiffusionModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    config = model.hparams
    model = model.to(device)
    model.eval()
    return model, config


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal locality test for noise predictor"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(name="locality_test")
    set_seed(42)

    # Load model and config
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    logger.info("Model loaded successfully.")

    seq_len = config.get("seq_length", config.get("data", {}).get("seq_length", 100))
    snp_index = 59

    # Output directory
    output_dir = Path(args.checkpoint).parent.parent / "locality_test"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    test_batch = next(iter(test_loader)).to(device)
    logger.info(
        f"Test batch has shape: {test_batch.shape}, and dim: {test_batch.dim()}"
    )

    # Select a single sample and ensure shape [1, 1, seq_len]
    sample_idx = 0
    x0 = test_batch[sample_idx : sample_idx + 1].unsqueeze(1)
    logger.info(f"x0 shape: {x0.shape} and dim: {x0.dim()}")
    logger.info(f"x0 unique values: {torch.unique(x0)}")
    logger.info(f"First 10 values: {x0[0, 0, :10]}")

    # --- Construct test inputs ---
    batch_size = 1
    baseline_value = 0.0
    perturb_delta = 0.15

    # Perfect x0
    # x0_perfect = x0.clone()
    x0_perfect = torch.full((batch_size, 1, seq_len), baseline_value, device=device)

    # Perturbed x0 (down and up)
    x0_down = x0_perfect.clone()
    x0_down[0, 0, snp_index] = baseline_value - perturb_delta

    x0_up = x0_perfect.clone()
    x0_up[0, 0, snp_index] = baseline_value + perturb_delta

    # --- Run noise predictor ---
    with torch.no_grad():
        noise_perfect = model.noise_predictor(
            x0_perfect, torch.zeros(batch_size, device=device)
        )
        noise_down = model.noise_predictor(
            x0_down, torch.zeros(batch_size, device=device)
        )
        noise_up = model.noise_predictor(x0_up, torch.zeros(batch_size, device=device))

    noise_perfect = noise_perfect.cpu().numpy().squeeze()
    noise_down = noise_down.cpu().numpy().squeeze()
    noise_up = noise_up.cpu().numpy().squeeze()

    perfect = x0_perfect.cpu().numpy().squeeze()
    down = x0_down.cpu().numpy().squeeze()
    up = x0_up.cpu().numpy().squeeze()

    # --- Analysis ---
    def report_noise(label, noise):
        print(f"\n{label} noise output:")
        print(noise)
        print(f"On-target (SNP {snp_index+1}): {noise[snp_index]:.4f}")
        print(f"Off-target max: {np.max(np.abs(np.delete(noise, snp_index))):.4e}")
        print(f"Off-target mean: {np.mean(np.abs(np.delete(noise, snp_index))):.4e}")

    report_noise("Perfect x0", noise_perfect)
    report_noise(
        f"Perturbed x0 (SNP {snp_index+1} = {baseline_value-perturb_delta:.2f})",
        noise_down,
    )
    report_noise(
        f"Perturbed x0 (SNP {snp_index+1} = {baseline_value+perturb_delta:.2f})",
        noise_up,
    )

    # --- Subplots: input, output for each case ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cases = [
        (perfect, noise_perfect, np.zeros_like(noise_perfect), "Perfect x0"),
        (
            down,
            noise_down,
            np.eye(1, seq_len, snp_index).squeeze() * (-perturb_delta),
            f"Perturbed Down (SNP {snp_index+1} = {baseline_value-perturb_delta:.2f})",
        ),
        (
            up,
            noise_up,
            np.eye(1, seq_len, snp_index).squeeze() * (perturb_delta),
            f"Perturbed Up (SNP {snp_index+1} = {baseline_value+perturb_delta:.2f})",
        ),
    ]
    for ax, (x_in, x_out, x_exp, title) in zip(axes, cases):
        ax.plot(x_in, label="Input to Model", color="tab:blue", alpha=0.7)
        ax.plot(x_out, label="Model Output", color="tab:orange")
        # Optionally plot expectation if needed:
        # ax.plot(x_exp, '--', color="black", label="Expected Output", linewidth=2)
        ax.axvline(snp_index, color="red", linestyle="--", label=f"SNP {snp_index+1}")
        ax.set_title(title)
        ax.set_xlabel("SNP Position")
        ax.set_ylim(-0.5, 0.8)
        ax.legend(loc="best")
    axes[0].set_ylabel("Value / Predicted Noise")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        output_dir / f"locality_test_{baseline_value}_{perturb_delta}.png", dpi=150
    )
    plt.close()

    # Zoomed-in plot around SNP 60 (all cases together)
    zoom_radius = 10
    start = max(0, snp_index - zoom_radius)
    end = min(seq_len, snp_index + zoom_radius + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(
        range(start, end), perfect[start:end], "--", color="orange", label="Perfect x0"
    )
    plt.plot(
        range(start, end),
        noise_perfect[start:end],
        color="orange",
        label="Model: Perfect x0",
    )
    plt.plot(range(start, end), up[start:end], "--", color="green", label="Up")
    plt.plot(
        range(start, end),
        noise_up[start:end],
        color="green",
        label=f"Model: Up ({baseline_value+perturb_delta:.2f})",
    )
    plt.plot(range(start, end), down[start:end], "--", color="blue", label="Down")
    plt.plot(
        range(start, end),
        noise_down[start:end],
        color="blue",
        label=f"Model: Down ({baseline_value-perturb_delta:.2f})",
    )
    plt.axvline(snp_index, color="red", linestyle="--", label=f"SNP {snp_index+1}")
    plt.ylim(-0.5, 0.8)
    plt.xlabel("SNP Position")
    plt.ylabel("Predicted Noise")
    plt.title(f"Noise Output (Zoomed): SNP {snp_index+1} ±{zoom_radius}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"locality_test_zoom_{baseline_value}_{perturb_delta}.png", dpi=150
    )
    plt.close()

    # Save report
    with open(
        output_dir / f"locality_test_report_{baseline_value}_{perturb_delta}.txt", "w"
    ) as f:
        f.write("Minimal Locality Test for Noise Predictor\n")
        f.write(f"SNP index tested: {snp_index+1}\n\n")
        f.write("Perfect x0 noise output:\n")
        f.write(np.array2string(noise_perfect, precision=4, separator=", ") + "\n")
        f.write(f"On-target: {noise_perfect[snp_index]:.4f}\n")
        f.write(
            f"Off-target max: {np.max(np.abs(np.delete(noise_perfect, snp_index))):.4e}\n"
        )
        f.write(
            f"Off-target mean: {np.mean(np.abs(np.delete(noise_perfect, snp_index))):.4e}\n\n"
        )

        f.write(f"Perturbed x0 (down) noise output:\n")
        f.write(np.array2string(noise_down, precision=4, separator=", ") + "\n")
        f.write(f"On-target: {noise_down[snp_index]:.4f}\n")
        f.write(
            f"Off-target max: {np.max(np.abs(np.delete(noise_down, snp_index))):.4e}\n"
        )
        f.write(
            f"Off-target mean: {np.mean(np.abs(np.delete(noise_down, snp_index))):.4e}\n\n"
        )

        f.write(f"Perturbed x0 (up) noise output:\n")
        f.write(np.array2string(noise_up, precision=4, separator=", ") + "\n")
        f.write(f"On-target: {noise_up[snp_index]:.4f}\n")
        f.write(
            f"Off-target max: {np.max(np.abs(np.delete(noise_up, snp_index))):.4e}\n"
        )
        f.write(
            f"Off-target mean: {np.mean(np.abs(np.delete(noise_up, snp_index))):.4e}\n"
        )

    logger.info("Minimal locality test complete! Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
