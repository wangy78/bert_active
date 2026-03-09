"""Greedy k-Center CoreSet strategy for active learning.

Reference: "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
(Sener & Savarese, ICLR 2018)

The strategy selects samples that are furthest from the current labeled set in
embedding space, maximizing coverage and diversity of the unlabeled pool.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from transformers import PreTrainedTokenizerBase

from bert_active.data.dataset import DataPool
from bert_active.data.tokenization import build_dataset
from bert_active.models.classifier import ModelWrapper
from bert_active.strategies.base import Strategy


class CoreSetStrategy(Strategy):
    """Greedy k-Center CoreSet strategy for active learning.

    Selects samples that maximize the minimum distance to any labeled sample
    in the embedding space. This provides representation coverage of the pool.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the CoreSet strategy.

        Args:
            pool: The data pool containing labeled and unlabeled samples.
            model: The model wrapper for extracting embeddings.
            tokenizer: Tokenizer for encoding texts.
            max_length: Maximum sequence length for tokenization. Defaults to 128.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def query(self, n: int) -> NDArray[np.intp]:
        """Select n samples from the unlabeled pool using greedy k-Center.

        Args:
            n: Number of samples to select.

        Returns:
            Array of indices (pool-level) of selected samples.
        """
        # Get all pool texts and embeddings
        all_texts = self.pool.texts
        dataset = build_dataset(
            self.tokenizer,
            all_texts,
            labels=None,
            max_length=self.max_length,
        )
        embeddings = self.model.get_embeddings(dataset)

        # Get labeled and unlabeled indices
        labeled_idx = self.pool.labeled_indices
        unlabeled_idx = self.pool.unlabeled_indices

        # Run greedy k-center selection
        num_to_select = min(n, len(unlabeled_idx))
        selected_indices = self._greedy_k_center(
            embeddings,
            labeled_idx,
            unlabeled_idx,
            num_to_select,
        )

        return selected_indices

    def _greedy_k_center(
        self,
        embeddings: NDArray[np.floating[Any]],
        labeled_idx: NDArray[np.intp],
        unlabeled_idx: NDArray[np.intp],
        n: int,
    ) -> NDArray[np.intp]:
        """Greedy k-center algorithm.

        Iteratively selects the unlabeled point with maximum minimum distance
        to any labeled point.

        Args:
            embeddings: Full pool embeddings (n_pool, emb_dim).
            labeled_idx: Indices of labeled samples.
            unlabeled_idx: Indices of unlabeled samples.
            n: Number of samples to select.

        Returns:
            Array of selected pool-level indices.
        """
        selected: list[int] = []
        current_labeled = set(labeled_idx.tolist())

        for _ in range(n):
            if len(unlabeled_idx) == 0:
                break

            # Compute distances from unlabeled to all current labeled points
            unlabeled_emb = embeddings[unlabeled_idx]
            labeled_pool_idx = np.array(sorted(current_labeled), dtype=np.intp)
            labeled_emb = embeddings[labeled_pool_idx]

            # Shape: (n_unlabeled, n_labeled)
            distances = cdist(unlabeled_emb, labeled_emb, metric="euclidean")

            # Minimum distance to nearest labeled point for each unlabeled sample
            min_distances = distances.min(axis=1)

            # Select unlabeled sample with maximum minimum distance
            max_idx = np.argmax(min_distances)
            selected_pool_idx = unlabeled_idx[max_idx]

            selected.append(int(selected_pool_idx))
            current_labeled.add(int(selected_pool_idx))

            # Remove selected sample from unlabeled pool
            unlabeled_idx = np.delete(unlabeled_idx, max_idx)

        return np.array(selected, dtype=np.intp)
