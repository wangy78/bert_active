"""Training loop for DistilBERT fine-tuning.

This module provides a Trainer class that handles the training and evaluation
of text classification models using manual training loops with mixed precision
training, gradient clipping, and learning rate scheduling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    import wandb
    from bert_active.config.experiment import TrainerConfig
    from bert_active.data.dataset import TextClassificationDataset
    from bert_active.models.classifier import ModelWrapper


class Trainer:
    """Trainer for fine-tuning text classification models.

    Handles the training loop with AdamW optimizer, linear warmup scheduling,
    mixed precision training, and gradient clipping. Provides evaluation with
    standard classification metrics.

    Attributes:
        model: Wrapped model containing the PreTrainedModel and device.
        config: Training configuration with hyperparameters.
    """

    def __init__(
        self,
        model: ModelWrapper,
        config: TrainerConfig,
        wandb_run: wandb.Run | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.wandb_run = wandb_run
        self._global_step = 0

    def train(self, train_dataset: TextClassificationDataset) -> dict[str, float]:
        """Run training loop for the configured number of epochs.

        Returns:
            Dictionary with at minimum ``train_loss`` (average over all epochs).
        """
        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate total training steps and warmup steps
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.model.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                optimizer.zero_grad()

                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.config.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                total_loss += loss.item()
                total_batches += 1

                progress_bar.set_postfix({"loss": loss.item()})

                self._global_step += 1
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"train/loss_step": loss.item(), "train/lr": scheduler.get_last_lr()[0]},
                        step=self._global_step,
                    )

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train/loss_epoch": avg_epoch_loss, "train/epoch": epoch + 1},
                    step=self._global_step,
                )

        avg_train_loss = total_loss / total_batches

        return {"train_loss": avg_train_loss}

    def evaluate(self, eval_dataset: TextClassificationDataset) -> dict[str, float]:
        """Run evaluation on the provided dataset.

        Returns:
            Dictionary with ``eval_loss``, ``eval_accuracy``, and ``eval_f1_macro``.
        """
        # Create DataLoader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

        # Evaluation mode
        self.model.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(eval_loader, desc="Evaluating")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits

                # Track loss
                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(eval_loader)
        accuracy = sum(pred == label for pred, label in zip(all_predictions, all_labels, strict=True)) / len(all_labels)
        f1_macro = f1_score(all_labels, all_predictions, average="macro")

        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_f1_macro": f1_macro,
        }
