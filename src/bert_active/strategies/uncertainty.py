"""Uncertainty-based active learning strategies.

This module implements three uncertainty sampling strategies:
- LeastConfidenceStrategy: selects instances with lowest max probability
- MarginStrategy: selects instances with smallest margin between top two predictions
- EntropyStrategy: selects instances with highest prediction entropy
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


class LeastConfidenceStrategy(Strategy):
    """Select instances with lowest max prediction probability (highest uncertainty)."""

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize strategy.

        Args:
            pool: Data pool with labeled and unlabeled instances
            model: ModelWrapper with predict_proba method
            tokenizer: Tokenizer for text encoding
            max_length: Max sequence length (default 128)
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def query(self, n: int) -> NDArray[np.intp]:
        """Query n most uncertain instances using least confidence.

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

        # Get prediction probabilities
        probs = self.model.predict_proba(dataset)  # shape: (n_unlabeled, n_classes)

        # Compute least confidence scores: 1.0 - max_prob (higher = more uncertain)
        scores = 1.0 - probs.max(axis=1)

        # Select top n by score (descending order)
        top_n_relative_indices = np.argsort(-scores)[:n]

        # Map back to pool-level indices
        selected_pool_indices = unlabeled_indices[top_n_relative_indices]

        return selected_pool_indices


class MarginStrategy(Strategy):
    """Select instances with smallest margin between top two predictions (highest uncertainty)."""

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize strategy.

        Args:
            pool: Data pool with labeled and unlabeled instances
            model: ModelWrapper with predict_proba method
            tokenizer: Tokenizer for text encoding
            max_length: Max sequence length (default 128)
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def query(self, n: int) -> NDArray[np.intp]:
        """Query n most uncertain instances using margin sampling.

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

        # Get prediction probabilities
        probs = self.model.predict_proba(dataset)  # shape: (n_unlabeled, n_classes)

        # Sort probabilities to get top two
        probs_sorted = np.sort(probs, axis=1)
        margin = probs_sorted[:, -1] - probs_sorted[:, -2]

        # Compute margin scores: negative margin (higher = smaller margin = more uncertain)
        scores = -margin

        # Select top n by score (descending order)
        top_n_relative_indices = np.argsort(-scores)[:n]

        # Map back to pool-level indices
        selected_pool_indices = unlabeled_indices[top_n_relative_indices]

        return selected_pool_indices


class EntropyStrategy(Strategy):
    """Select instances with highest prediction entropy (highest uncertainty)."""

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize strategy.

        Args:
            pool: Data pool with labeled and unlabeled instances
            model: ModelWrapper with predict_proba method
            tokenizer: Tokenizer for text encoding
            max_length: Max sequence length (default 128)
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def query(self, n: int) -> NDArray[np.intp]:
        """Query n most uncertain instances using entropy sampling.

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

        # Get prediction probabilities
        probs = self.model.predict_proba(dataset)  # shape: (n_unlabeled, n_classes)

        # Compute entropy: -sum(p * log(p)) with numerical stability
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Select top n by score (descending order)
        top_n_relative_indices = np.argsort(-entropy)[:n]

        # Map back to pool-level indices
        selected_pool_indices = unlabeled_indices[top_n_relative_indices]

        return selected_pool_indices
