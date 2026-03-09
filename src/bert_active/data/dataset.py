"""AG News dataset loading and DataPool for active learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from datasets import DatasetDict, load_dataset
from numpy.typing import NDArray
from torch.utils.data import Dataset


@dataclass
class DataPool:
    """Pool-based active learning data manager.

    Tracks which samples are labeled via a boolean mask.
    Provides views into labeled / unlabeled subsets without copying data.
    """

    texts: list[str]
    labels: NDArray[np.intp]
    labeled_mask: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.texts)
        self.labeled_mask = np.zeros(n, dtype=bool)

    @property
    def n_pool(self) -> int:
        return len(self.texts)

    @property
    def n_labeled(self) -> int:
        return int(self.labeled_mask.sum())

    @property
    def n_unlabeled(self) -> int:
        return self.n_pool - self.n_labeled

    @property
    def labeled_indices(self) -> NDArray[np.intp]:
        return np.where(self.labeled_mask)[0]

    @property
    def unlabeled_indices(self) -> NDArray[np.intp]:
        return np.where(~self.labeled_mask)[0]

    def label(self, indices: NDArray[np.intp]) -> None:
        """Mark samples at given indices as labeled."""
        self.labeled_mask[indices] = True

    def initialize(self, n_init: int, seed: int = 42, balanced: bool = True) -> NDArray[np.intp]:
        """Select initial labeled set, optionally balanced across classes.

        Returns the selected indices.
        """
        rng = np.random.default_rng(seed)
        if balanced:
            unique_labels = np.unique(self.labels)
            per_class = n_init // len(unique_labels)
            remainder = n_init % len(unique_labels)
            selected: list[NDArray[np.intp]] = []
            for i, label in enumerate(unique_labels):
                class_indices = np.where(self.labels == label)[0]
                n_select = per_class + (1 if i < remainder else 0)
                chosen = rng.choice(class_indices, size=min(n_select, len(class_indices)), replace=False)
                selected.append(chosen)
            init_indices = np.concatenate(selected)
        else:
            init_indices = rng.choice(self.n_pool, size=n_init, replace=False)
        self.label(init_indices)
        return init_indices

    def get_labeled_texts(self) -> list[str]:
        idx = self.labeled_indices
        return [self.texts[i] for i in idx]

    def get_labeled_labels(self) -> NDArray[np.intp]:
        return self.labels[self.labeled_indices]

    def get_unlabeled_texts(self) -> list[str]:
        idx = self.unlabeled_indices
        return [self.texts[i] for i in idx]


class TextClassificationDataset(Dataset[dict[str, object]]):
    """Simple torch Dataset wrapping tokenized text + labels."""

    def __init__(
        self,
        input_ids: NDArray[np.intp],
        attention_mask: NDArray[np.intp],
        labels: NDArray[np.intp] | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        import torch

        item: dict[str, object] = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_ag_news(seed: int = 42, test_size: int | None = None) -> tuple[DataPool, TextClassificationDataset]:
    """Load AG News and return (train_pool, test_dataset).

    The test dataset is pre-tokenized later; here we just load raw data.
    Returns DataPool for train, and a placeholder that will be tokenized by the caller.
    """
    ds: DatasetDict = load_dataset("ag_news")  # type: ignore[assignment]
    train_split = ds["train"]
    test_split = ds["test"]

    train_texts: list[str] = list(train_split["text"])
    train_labels = np.array(train_split["label"], dtype=np.intp)

    pool = DataPool(texts=train_texts, labels=train_labels)

    # Test data — raw for now, tokenization happens externally
    test_texts: list[str] = list(test_split["text"])
    test_labels_raw = np.array(test_split["label"], dtype=np.intp)

    if test_size is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(test_texts), size=min(test_size, len(test_texts)), replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels_raw = test_labels_raw[indices]

    # Return raw test data as a "dataset" with empty arrays — caller tokenizes
    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),  # placeholder
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),  # placeholder
        labels=test_labels_raw,
    )
    # Store raw texts for tokenization
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset
