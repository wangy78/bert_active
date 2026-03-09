"""Abstract base class for active learning query strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bert_active.data.dataset import DataPool
from bert_active.models.classifier import ModelWrapper


class Strategy(ABC):
    """Base class for pool-based active learning strategies.

    Each strategy implements `query()` to select indices from the unlabeled pool.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        **kwargs: Any,
    ) -> None:
        self.pool = pool
        self.model = model

    @abstractmethod
    def query(self, n: int) -> NDArray[np.intp]:
        """Select n indices from the unlabeled pool to label next.

        Args:
            n: Number of samples to query.

        Returns:
            Array of indices into the full pool (not relative to unlabeled subset).
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
