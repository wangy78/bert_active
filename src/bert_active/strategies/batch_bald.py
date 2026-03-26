"""BatchBALD: BALD uncertainty filtering + KMeans diversity in embedding space.

Selects a diverse batch by:
1. Computing BALD (mutual information) scores via MC Dropout
2. Pre-filtering top uncertain candidates (batch_size * candidate_ratio)
3. Extracting CLS embeddings for candidates
4. Running KMeans clustering to ensure spatial diversity
5. Picking the sample closest to each cluster center
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from transformers import PreTrainedTokenizerBase

from bert_active.data.dataset import DataPool
from bert_active.data.tokenization import build_dataset
from bert_active.models.classifier import ModelWrapper
from bert_active.strategies.base import Strategy


class BatchBALDStrategy(Strategy):
    """BALD + KMeans diversity batch selection.

    Uses BALD scores to pre-filter uncertain candidates, then clusters
    their embeddings with KMeans to select a diverse batch.
    """

    def __init__(
        self,
        pool: DataPool,
        model: ModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        n_drop: int = 10,
        candidate_ratio: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(pool, model, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_drop = n_drop
        self.candidate_ratio = candidate_ratio

    def query(self, n: int) -> NDArray[np.intp]:
        unlabeled_indices = self.pool.unlabeled_indices
        unlabeled_texts = self.pool.get_unlabeled_texts()

        dataset = build_dataset(
            tokenizer=self.tokenizer,
            texts=unlabeled_texts,
            labels=None,
            max_length=self.max_length,
        )

        # --- BALD scores ---
        mean_probs = self.model.predict_proba_dropout(dataset, n_drop=self.n_drop)
        mc_probs = self.model._mc_probs  # (T, N, C)

        h_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        ent_per_pass = -np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=2)
        mean_h = np.mean(ent_per_pass, axis=0)
        bald_scores = h_mean - mean_h  # (N,)

        # --- Pre-filter top candidates by BALD ---
        n_candidates = min(n * self.candidate_ratio, len(unlabeled_indices))
        top_candidate_idx = np.argsort(-bald_scores)[:n_candidates]

        # Edge case: fewer candidates than requested
        if n_candidates <= n:
            return unlabeled_indices[top_candidate_idx[:n]]

        # --- Extract embeddings for candidates ---
        candidate_texts = [unlabeled_texts[i] for i in top_candidate_idx]
        candidate_dataset = build_dataset(
            tokenizer=self.tokenizer,
            texts=candidate_texts,
            labels=None,
            max_length=self.max_length,
        )
        embeddings = self.model.get_embeddings(candidate_dataset)  # (n_candidates, dim)

        # --- KMeans clustering for diversity ---
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        # Pick closest to each cluster center
        selected: list[int] = []
        for cluster_id in range(n):
            cluster_mask = kmeans.labels_ == cluster_id
            cluster_local = np.where(cluster_mask)[0]
            cluster_embs = embeddings[cluster_mask]

            distances = np.linalg.norm(
                cluster_embs - kmeans.cluster_centers_[cluster_id], axis=1
            )
            closest = cluster_local[np.argmin(distances)]
            selected.append(closest)

        selected_arr = np.array(selected, dtype=np.intp)
        # Map: selected -> top_candidate_idx -> unlabeled_indices
        return unlabeled_indices[top_candidate_idx[selected_arr]]
