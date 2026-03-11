"""Metrics tracking and learning curves for active learning experiments.

This module provides utilities for tracking training and evaluation metrics
across active learning rounds and computing learning curve statistics.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class MetricsTracker:
    """Track metrics across active learning rounds and compute learning curves.

    Attributes:
        experiment_name: Name of the experiment for file naming
        history: List of metric dictionaries, one per round
        sample_selection_history: List of sample selection dictionaries, tracking
            which samples (indices and labels) are selected each round
    """

    def __init__(self, experiment_name: str) -> None:
        """Initialize metrics tracker.

        Args:
            experiment_name: Name of the experiment for saving outputs
        """
        self.experiment_name = experiment_name
        self.history: list[dict[str, Any]] = []
        self.sample_selection_history: list[dict[str, Any]] = []

    def log_round(
        self,
        round_num: int,
        n_labeled: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ) -> None:
        """Log metrics for a single active learning round.

        Args:
            round_num: Round number (0-indexed)
            n_labeled: Total number of labeled samples at this round
            train_metrics: Dictionary of training metrics (e.g., {"train_loss": 0.5})
            eval_metrics: Dictionary of evaluation metrics (e.g., {"eval_loss": 0.3,
                         "eval_accuracy": 0.92, "eval_f1_macro": 0.91})
        """
        entry = {
            "round": round_num,
            "n_labeled": n_labeled,
            **train_metrics,
            **eval_metrics,
        }
        self.history.append(entry)

    def log_sample_selection(
        self,
        round_num: int,
        indices: NDArray[np.intp],
        labels: NDArray[np.intp],
    ) -> None:
        """Record sample selection details and statistics.

        Args:
            round_num: Round number (1-indexed, matching the round these samples will be used in)
            indices: Array of selected sample indices
            labels: Array of corresponding labels
        """
        label_counts = dict(Counter(labels.tolist()))

        entry = {
            "round": round_num,
            "indices": indices.tolist(),
            "labels": labels.tolist(),
            "label_counts": label_counts,
        }
        self.sample_selection_history.append(entry)

    def save(self, output_dir: str) -> None:
        """Save metrics history to JSON file.

        Args:
            output_dir: Directory to save metrics JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_file = output_path / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_learning_curves(self, output_dir: str) -> None:
        """Plot learning curves (accuracy and F1 vs n_labeled).

        Creates a figure with two subplots:
        - Left: Evaluation accuracy vs number of labeled samples
        - Right: Evaluation F1 (macro) vs number of labeled samples

        Args:
            output_dir: Directory to save the plot
        """
        if not self.history:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract data
        n_labeled_list = [entry["n_labeled"] for entry in self.history]
        accuracy_list = [entry.get("eval_accuracy", 0.0) for entry in self.history]
        f1_list = [entry.get("eval_f1_macro", 0.0) for entry in self.history]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(n_labeled_list, accuracy_list, marker="o", linewidth=2)
        ax1.set_xlabel("Number of Labeled Samples")
        ax1.set_ylabel("Evaluation Accuracy")
        ax1.set_title("Learning Curve: Accuracy")
        ax1.grid(True, alpha=0.3)

        # Plot F1
        ax2.plot(n_labeled_list, f1_list, marker="o", linewidth=2, color="orange")
        ax2.set_xlabel("Number of Labeled Samples")
        ax2.set_ylabel("Evaluation F1 (Macro)")
        ax2.set_title("Learning Curve: F1")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = output_path / f"{self.experiment_name}_learning_curves.png"
        plt.savefig(plot_file, dpi=100, bbox_inches="tight")
        plt.close()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the experiment.

        Returns:
            Dictionary with experiment_name, n_rounds, final_accuracy, final_f1,
            best_accuracy, and best_f1
        """
        if not self.history:
            return {
                "experiment_name": self.experiment_name,
                "n_rounds": 0,
                "final_accuracy": None,
                "final_f1": None,
                "best_accuracy": None,
                "best_f1": None,
            }

        accuracy_list = [entry.get("eval_accuracy", 0.0) for entry in self.history]
        f1_list = [entry.get("eval_f1_macro", 0.0) for entry in self.history]

        return {
            "experiment_name": self.experiment_name,
            "n_rounds": len(self.history),
            "final_accuracy": accuracy_list[-1],
            "final_f1": f1_list[-1],
            "best_accuracy": max(accuracy_list) if accuracy_list else None,
            "best_f1": max(f1_list) if f1_list else None,
        }

    def compute_auc(self, metric_key: str = "eval_accuracy") -> float:
        """Compute area under the learning curve using trapezoidal rule.

        The AUC is computed over (n_labeled, metric_value) pairs and normalized
        by the total annotation budget (max n_labeled).

        Args:
            metric_key: Key of the metric to compute AUC for (default: "eval_accuracy")

        Returns:
            Normalized AUC value (area / max_n_labeled)
        """
        if not self.history or len(self.history) < 2:
            return 0.0

        n_labeled_list = np.array([entry["n_labeled"] for entry in self.history])
        metric_list = np.array([entry.get(metric_key, 0.0) for entry in self.history])

        # Compute area under curve using trapezoidal rule
        auc = float(np.trapezoid(metric_list, n_labeled_list))

        # Normalize by total annotation budget
        max_n_labeled = n_labeled_list[-1]
        if max_n_labeled > 0:
            auc = auc / max_n_labeled

        return auc
