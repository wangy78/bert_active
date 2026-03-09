"""BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy for active learning.

BADGE uses gradient embeddings to select a diverse batch of samples via k-means++ initialization.
The gradient embedding for sample x with embedding e and predicted class ŷ is computed as:
    g_c(x) = e * (1[c == ŷ] - p_c(x))
where p_c(x) is the predicted probability for class c.

Reference: Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (ICLR 2020)
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


def _kmeans_plus_plus(
    embeddings: NDArray[np.floating[Any]],
    n: int,
) -> NDArray[np.intp]:
    """K-means++ initialization: select n diverse centers from embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        n: Number of centers to select.

    Returns:
        Array of indices of selected centers.
    """
    n_samples = embeddings.shape[0]
    if n >= n_samples:
        return np.arange(n_samples, dtype=np.intp)

    rng = np.random.default_rng()

    # First center: select randomly
    selected_indices = [rng.integers(0, n_samples)]

    # Iteratively select remaining centers
    for _ in range(n - 1):
        # Compute squared distances from each sample to nearest selected center
        selected_embeddings = embeddings[selected_indices]
        distances = cdist(
            embeddings,
            selected_embeddings,
            metric="euclidean",
        )
        min_distances = distances.min(axis=1)
        min_distances_sq = min_distances**2

        # Probability proportional to distance
        probabilities = min_distances_sq / min_distances_sq.sum()

        # Select next center
        next_center = rng.choice(n_samples, p=probabilities)
        selected_indices.append(next_center)

    return np.array(selected_indices, dtype=np.intp)


class BADGEStrategy(Strategy):
    """BADGE: Batch Active learning by Diverse Gradient Embeddings.

    Uses gradient embeddings computed from the model's predictions to select
    diverse batches of unlabeled samples via k-means++ initialization.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the BADGE strategy.

        Args:
            pool: The data pool containing labeled and unlabeled samples.
            model: The model wrapper with gradient embedding computation.
            tokenizer: Tokenizer for encoding texts.
            max_length: Maximum sequence length for tokenization (default 128).
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def query(self, n: int) -> NDArray[np.intp]:
        """Select n samples from the unlabeled pool using BADGE.

        Args:
            n: Number of samples to select.

        Returns:
            Array of pool-level indices of selected samples.
        """
        unlabeled_indices = self.pool.unlabeled_indices
        num_to_select = min(n, len(unlabeled_indices))

        # Handle edge case: if requesting more than available
        if num_to_select >= len(unlabeled_indices):
            return unlabeled_indices

        # Get unlabeled texts
        unlabeled_texts = self.pool.get_unlabeled_texts()

        # Build dataset for unlabeled samples (no labels needed)
        dataset = build_dataset(
            self.tokenizer,
            unlabeled_texts,
            labels=None,
            max_length=self.max_length,
        )

        # Compute gradient embeddings for unlabeled samples
        grad_embeddings = self.model.get_grad_embeddings(dataset)

        # Run k-means++ on gradient embeddings to select diverse samples
        selected_relative_indices = _kmeans_plus_plus(grad_embeddings, num_to_select)

        # Map relative indices back to pool-level indices
        selected_pool_indices = unlabeled_indices[selected_relative_indices]

        return selected_pool_indices
