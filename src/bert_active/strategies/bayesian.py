"""Bayesian active learning strategies using MC Dropout.

This module implements BALD (Bayesian Active Learning by Disagreement),
which uses Monte Carlo Dropout to estimate model uncertainty via mutual information.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase

from bert_active.data.dataset import DataPool
from bert_active.data.tokenization import build_dataset
from bert_active.models.classifier import ModelWrapper
from bert_active.strategies.base import Strategy


class BALDStrategy(Strategy):
    """Bayesian Active Learning by Disagreement (BALD) using MC Dropout.

    Selects instances with highest mutual information between model predictions
    obtained via MC Dropout, measuring disagreement in the predictive distribution.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        n_drop: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize BALD strategy.

        Args:
            pool: Data pool with labeled and unlabeled instances
            model: ModelWrapper with predict_proba_dropout method
            tokenizer: Tokenizer for text encoding
            max_length: Max sequence length (default 128)
            n_drop: Number of MC dropout passes (default 10)
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_drop = n_drop

    def query(self, n: int) -> NDArray[np.intp]:
        """Query n most uncertain instances using BALD (mutual information).

        Args:
            n: Number of instances to select

        Returns:
            Array of pool-level indices for selected instances
        """
        # Get unlabeled indices and texts
        unlabeled_indices = self.pool.unlabeled_indices
        unlabeled_texts = self.pool.get_unlabeled_texts()

        # Build dataset for unlabeled data
        dataset = build_dataset(
            tokenizer=self.tokenizer,
            texts=unlabeled_texts,
            labels=None,
            max_length=self.max_length,
        )

        # Get mean probabilities via MC dropout
        # Also populates self.model._mc_probs with shape (n_drop, n_samples, n_classes)
        mean_probs = self.model.predict_proba_dropout(dataset, n_drop=self.n_drop)  # shape: (n_unlabeled, n_classes)

        # Access individual MC predictions
        mc_probs = self.model._mc_probs  # shape: (n_drop, n_unlabeled, n_classes)

        # Compute BALD score (mutual information)
        # H(mean) = -sum_c mean_probs[c] * log(mean_probs[c])
        h_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)  # shape: (n_unlabeled,)

        # H(p) = -sum_c p[c] * log(p[c]) computed per MC pass
        # mean_H = mean over MC passes of H(p_i) for each sample
        entropy_per_pass = -np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=2)  # shape: (n_drop, n_unlabeled)
        mean_h = np.mean(entropy_per_pass, axis=0)  # shape: (n_unlabeled,)

        # BALD score = mutual information = H(mean) - mean_H
        # Higher score = higher disagreement = more informative
        bald_scores = h_mean - mean_h  # shape: (n_unlabeled,)

        # Select top n by score (descending order)
        top_n_relative_indices = np.argsort(-bald_scores)[:n]

        # Map back to pool-level indices
        selected_pool_indices = unlabeled_indices[top_n_relative_indices]

        return selected_pool_indices
