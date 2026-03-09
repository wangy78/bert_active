"""DistilBERT tokenization utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bert_active.data.dataset import TextClassificationDataset


def create_tokenizer(model_name: str = "distilbert-base-uncased") -> PreTrainedTokenizerBase:
    """Create a DistilBERT tokenizer."""
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 128,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Tokenize a list of texts, returning (input_ids, attention_mask) as numpy arrays."""
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_ids: NDArray[np.intp] = np.array(encoding["input_ids"], dtype=np.intp)
    attention_mask: NDArray[np.intp] = np.array(encoding["attention_mask"], dtype=np.intp)
    return input_ids, attention_mask


def build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    labels: NDArray[np.intp] | None = None,
    max_length: int = 128,
) -> TextClassificationDataset:
    """Tokenize texts and build a TextClassificationDataset."""
    input_ids, attention_mask = tokenize_texts(tokenizer, texts, max_length)
    return TextClassificationDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def tokenize_test_dataset(
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: TextClassificationDataset,
    max_length: int = 128,
) -> TextClassificationDataset:
    """Tokenize the raw test dataset (replaces placeholder arrays)."""
    raw_texts: list[str] = test_dataset._raw_texts  # type: ignore[attr-defined]
    input_ids, attention_mask = tokenize_texts(tokenizer, raw_texts, max_length)
    return TextClassificationDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=test_dataset.labels,
    )
