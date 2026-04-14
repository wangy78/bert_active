"""TypiClust strategy for active learning cold-start.

Reference: "Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets"
(Hacohen et al., ICML 2022)

At small budgets, selecting *typical* (high-density) points outperforms diversity-based
methods like CoreSet. The algorithm clusters the unlabeled pool into K clusters, then
selects the most typical point from each cluster, where typicality is measured as the
inverse average distance to the k nearest neighbours.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from transformers import PreTrainedTokenizerBase

from bert_active.data.dataset import DataPool
from bert_active.data.tokenization import build_dataset
from bert_active.models.classifier import ModelWrapper
from bert_active.strategies.base import Strategy


class TypiClustStrategy(Strategy):
    """TypiClust: cluster then pick the most typical point per cluster.

    Steps per query:
    1. Embed all unlabeled points with the current model.
    2. Cluster into n clusters (one per query budget).
    3. Within each cluster, rank by typicality = 1 / mean_dist_to_knn.
    4. Return the most typical unlabeled point per cluster.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        knn: int = 20,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            pool: Data pool with labeled/unlabeled split.
            model: Model wrapper for computing embeddings.
            tokenizer: Tokenizer for encoding texts.
            max_length: Max sequence length for tokenisation.
            knn: Number of neighbours used to estimate typicality density.
        """
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.knn = knn

    def query(self, n: int) -> NDArray[np.intp]:
        """Select n samples by clustering + typicality ranking.

        Args:
            n: Number of samples to select.

        Returns:
            Array of pool-level indices of selected samples.
        """
        unlabeled_idx = self.pool.unlabeled_indices
        n = min(n, len(unlabeled_idx))

        all_texts = self.pool.texts
        dataset = build_dataset(
            self.tokenizer,
            all_texts,
            labels=None,
            max_length=self.max_length,
        )
        embeddings = self.model.get_embeddings(dataset)
        unlabeled_emb = embeddings[unlabeled_idx]

        # Typicality: 1 / mean distance to knn within the unlabeled pool
        k = min(self.knn, len(unlabeled_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn.fit(unlabeled_emb)
        distances, _ = nn.kneighbors(unlabeled_emb)
        # distances[:, 0] is self (distance=0), skip it
        mean_dist = distances[:, 1:].mean(axis=1)
        typicality = 1.0 / (mean_dist + 1e-10)

        # Cluster unlabeled pool into n clusters
        kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, n_init=3)
        cluster_ids = kmeans.fit_predict(unlabeled_emb)

        # From each cluster pick the most typical point
        selected_local: list[int] = []
        for c in range(n):
            mask = cluster_ids == c
            if not mask.any():
                continue
            local_indices = np.where(mask)[0]
            best_local = local_indices[np.argmax(typicality[local_indices])]
            selected_local.append(int(best_local))

        return unlabeled_idx[np.array(selected_local, dtype=np.intp)]
