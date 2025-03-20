#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from diffusion import SNPDataset
from test_models import DDPM, UNet1D, DiffusionModel

# Set global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_models():
    """Test ForwardDiffusion and UNet1D with real SNP data from config."""
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(config_path='config.yaml')
    input_path = config.get('input_path')
    print("\nInitializing dataset...")
    
    # Initialize dataset
    dataset = SNPDataset(input_path)
    
    # Initialize dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get batch
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"Batch shape [B, seq_len]: {batch.shape}")
    
    # Prepare input (ensure [B, C, L] format for both models)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    print(f"Batch shape [B, C, seq_len]: {batch.shape}")
    
    # ------------------ Test DiffusionModel
    print("\nTesting DiffusionModel...")
    model = DiffusionModel(hparams=config)
  
    # Prepare input for the model
    x_input = batch  # [B, 1, seq_len]
    print(f"Input shape before model: {x_input.shape}")
    t_input = torch.tensor([500] * x_input.shape[0], dtype=torch.float32, device=device)  # Use middle timestep
    print(f"Shape of input to first conv layer: {x_input.shape}")

    # Test the model
    try:
        print("\nTesting full DiffusionModel...")
        model_output = model(x_input)  # Call the model directly with input
        print(f"Model output shape: {model_output.shape}")

        print("\nComputing loss...")
        loss_value = model.compute_loss(x_input)
        print(f"Computed loss: {loss_value.item():.6f}")

        # Visualize model output if needed
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOUT OF MEMORY ERROR!")
            print("Try reducing batch_size or model dimensions")
        else:
            print(f"\nError during DiffusionModel test: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error during DiffusionModel test: {str(e)}")
        
    
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_models()
