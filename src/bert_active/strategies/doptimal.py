"""D-Optimal experimental design strategy for active learning.

Greedily selects samples that maximise |X^T X| in embedding space.
Each step picks the unlabelled point with the highest leverage score
  s(x) = x^T (X^T X)^{-1} x
which equals the multiplicative gain in det(X^T X) from adding x.

Reference:
  Federov (1972) "Theory of Optimal Experiments"
  Yu et al. (2006) "Active Learning via Transductive Experimental Design"
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


class DOptimalStrategy(Strategy):
    """Greedy D-Optimal (maximum leverage score) query strategy.

    At each query step, maintains (X^T X + ridge * I)^{-1} and selects the
    unlabelled point with the highest leverage score.  Uses a rank-1
    Sherman-Morrison update to keep the inverse cheap across steps.

    Args:
        ridge: Regularisation added to X^T X before inversion (default 1e-4).
               Prevents singularity when the labeled set is small.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        ridge: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ridge = ridge

    def query(self, n: int) -> NDArray[np.intp]:
        """Select n samples with greedy D-optimal design.

        Args:
            n: Number of samples to select.

        Returns:
            Array of pool-level indices.
        """
        all_texts = self.pool.texts
        dataset = build_dataset(
            self.tokenizer,
            all_texts,
            labels=None,
            max_length=self.max_length,
        )
        embeddings = self.model.get_embeddings(dataset).astype(np.float64)

        labeled_idx = self.pool.labeled_indices
        unlabeled_idx = self.pool.unlabeled_indices.copy()

        num_to_select = min(n, len(unlabeled_idx))

        # Initialise (X^T X + ridge*I)^{-1} from the current labeled set
        d = embeddings.shape[1]
        XtX = self.ridge * np.eye(d)
        if len(labeled_idx) > 0:
            X_lab = embeddings[labeled_idx]
            XtX += X_lab.T @ X_lab
        XtX_inv = np.linalg.inv(XtX)

        selected: list[int] = []

        for _ in range(num_to_select):
            if len(unlabeled_idx) == 0:
                break

            # Leverage scores: s_i = x_i^T (X^T X)^{-1} x_i
            U = embeddings[unlabeled_idx]          # (m, d)
            scores = np.einsum("md,dd,md->m", U, XtX_inv, U)

            best_local = int(np.argmax(scores))
            best_pool_idx = int(unlabeled_idx[best_local])

            selected.append(best_pool_idx)

            # Rank-1 Sherman-Morrison update:
            # (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
            x = embeddings[best_pool_idx]          # (d,)
            Ainv_x = XtX_inv @ x                   # (d,)
            denom = 1.0 + float(x @ Ainv_x)
            XtX_inv -= np.outer(Ainv_x, Ainv_x) / denom

            unlabeled_idx = np.delete(unlabeled_idx, best_local)

        return np.array(selected, dtype=np.intp)
