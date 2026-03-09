"""Random sampling baseline strategy for active learning."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bert_active.data.dataset import DataPool
from bert_active.models.classifier import ModelWrapper
from bert_active.strategies.base import Strategy


class RandomStrategy(Strategy):
    """Random sampling strategy that selects unlabeled samples at random.

    This is the simplest baseline strategy for active learning, selecting
    n samples uniformly at random from the unlabeled pool.
    """

    def __init__(self, pool: DataPool, model: ModelWrapper, **kwargs: Any) -> None:
        """Initialize the random strategy.

        Args:
            pool: The data pool containing labeled and unlabeled samples.
            model: The model wrapper (unused in random strategy).
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(pool, model, **kwargs)

    def query(self, n: int) -> NDArray[np.intp]:
        """Select n samples at random from the unlabeled pool.

        Args:
            n: Number of samples to select.

        Returns:
            Array of indices (pool-level) of selected samples.
        """
        unlabeled = self.pool.unlabeled_indices
        num_to_select = min(n, len(unlabeled))

        rng = np.random.default_rng()
        selected_indices = rng.choice(unlabeled, size=num_to_select, replace=False)

        return selected_indices
