"""Callback for model evaluation metrics."""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, auc, confusion_matrix


class EvaluationCallback(Callback):
    """Callback for model evaluation metrics.

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
        if isinstance(outputs, dict) and "model_output" in outputs:
            self._test_outputs.append(outputs["predicted"].detach().cpu())
            self._test_targets.append(outputs["model_output"].detach().cpu())

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

        # Determine output directory
        if self.output_dir is not None:
            # If output_dir was explicitly provided (e.g., in test.py), use it
            output_dir = self.output_dir
        else:
            # During training, store in version-specific directory
            if trainer.logger:
                if (
                    hasattr(trainer.logger, "version")
                    and trainer.logger.version is not None
                ):
                    # Get the version-specific directory
                    log_dir = trainer.logger.log_dir
                    output_dir = os.path.join(log_dir, "evaluation")
                else:
                    # Fallback if no version available
                    base_dir = trainer.logger.save_dir or "output"
                    output_dir = os.path.join(base_dir, "evaluation")
            else:
                # Ultimate fallback
                output_dir = os.path.join("output", "evaluation")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Concatenate all batches
        outputs = torch.cat(self._test_outputs, dim=0)
        targets = torch.cat(self._test_targets, dim=0)

        # Compute and log reconstruction metrics
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        pl_module.log("test_mse", mse)
        pl_module.log("test_mae", mae)

        # Convert tensors to float32 before computing metrics
        outputs_float = outputs.float()
        targets_float = targets.float()

        # Convert to binary for classification metrics
        predictions_binary = (outputs_float > 0.5).float()
        targets_binary = (targets_float > 0.5).float()

        try:
            # Compute ROC curve and area
            fpr, tpr, _ = roc_curve(
                targets_binary.flatten().numpy(), outputs_float.flatten().numpy()
            )
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")

            # Save ROC curve
            roc_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved to {roc_path}")

            # Compute confusion matrix
            cm = confusion_matrix(
                targets_binary.flatten().numpy(), predictions_binary.flatten().numpy()
            )

            # Plot confusion matrix
            plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

            # Save confusion matrix
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")

        except Exception as e:
            print(f"Warning: Could not compute classification metrics: {str(e)}")

        # Clear stored outputs
        self._test_outputs.clear()
        self._test_targets.clear()
