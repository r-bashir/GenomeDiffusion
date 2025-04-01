"""Callback for model inference and evaluation metrics."""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, auc, confusion_matrix


class InferenceCallback(Callback):
    """Callback for model inference and evaluation metrics.
    
    This callback collects model outputs during testing and computes various evaluation
    metrics including ROC curves, confusion matrices, and classification metrics.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the callback.
        
        Args:
            output_dir: Directory to save plots and metrics. If None, will use logger's directory.
        """
        super().__init__()
        self.output_dir = output_dir
        self._test_outputs: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

    def on_test_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect outputs and targets after each test batch.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The current PyTorch Lightning module
            outputs: Outputs from the test step
            batch: The input batch
            batch_idx: The index of the current batch
            dataloader_idx: The index of the current dataloader
        """
        if isinstance(batch, (tuple, list)):
            batch = batch[0]  # Get just the data, not labels if present
            
        # Store model outputs and targets
        if "model_output" in outputs:
            self._test_outputs.append(outputs["model_output"].detach().cpu())
            self._test_targets.append(batch.detach().cpu())

    def on_test_epoch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute evaluation metrics at the end of the test epoch.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The current PyTorch Lightning module
        """
        if not self._test_outputs or not self._test_targets:
            return

        # Get output directory
        output_dir = self.output_dir
        if output_dir is None and trainer.logger is not None:
            if hasattr(trainer.logger, "log_dir"):
                output_dir = trainer.logger.log_dir

        if output_dir is None:
            print("Warning: No output directory specified. Metrics will only be logged.")
            
        # Concatenate all batches
        outputs = torch.cat(self._test_outputs, dim=0)
        targets = torch.cat(self._test_targets, dim=0)

        # Compute and log reconstruction metrics
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        pl_module.log("test_mse", mse)
        pl_module.log("test_mae", mae)

        # Convert to binary for classification metrics
        predictions_binary = (outputs > 0.5).float()
        targets_binary = (targets > 0.5).float()

        try:
            # Compute ROC curve and area
            fpr, tpr, _ = roc_curve(targets_binary.flatten().numpy(), 
                                  outputs.flatten().numpy())
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()

            # Compute confusion matrix
            cm = confusion_matrix(targets_binary.flatten().numpy(), 
                                predictions_binary.flatten().numpy())
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                "roc_auc": roc_auc
            }

            # Log all metrics
            for name, value in metrics.items():
                pl_module.log(f"test_{name}", value)

            # Save metrics to file
            if output_dir is not None:
                metrics_file = os.path.join(output_dir, "test_metrics.txt")
                with open(metrics_file, "w") as f:
                    for name, value in metrics.items():
                        f.write(f"{name}: {value:.4f}\n")
                    
                # Save confusion matrix
                np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)

        except ImportError:
            print("Warning: scikit-learn not found. Skipping ROC and confusion matrix metrics.")
        except Exception as e:
            print(f"Warning: Error computing classification metrics: {str(e)}")
        finally:
            # Clear stored predictions
            self._test_outputs.clear()
            self._test_targets.clear()
