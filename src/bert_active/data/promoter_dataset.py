"""Promoter dataset loading for cross-species transfer learning.

Loads species-specific promoter data from local text files (TATA + nonTATA merged),
generates negative samples, and returns DataPool / test sets compatible with
the active learning pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bert_active.data.dataset import DataPool, TextClassificationDataset
from bert_active.data.dna_dataset import _split_to_pool
from bert_active.data.negative_generator import generate_negative_set


def _load_sequences(filepath: str | Path) -> list[str]:
    """Load sequences from a text file, one sequence per line."""
    path = Path(filepath)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_promoter_species(
    species: str,
    data_dir: str,
    seed: int = 42,
    test_split: float = 0.2,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load all promoter data for one species (TATA + nonTATA merged).

    Reads {species}_pos_TATA.txt and {species}_pos_nonTATA.txt,
    generates 1:1 negative samples, and splits into train pool + test set.

    Args:
        species: Species prefix, e.g. "hs" or "mm".
        data_dir: Directory containing the promoter text files.
        seed: Random seed for negative generation and splitting.
        test_split: Fraction reserved for testing.

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    data_path = Path(data_dir)
    tata_seqs = _load_sequences(data_path / f"{species}_pos_TATA.txt")
    nontata_seqs = _load_sequences(data_path / f"{species}_pos_nonTATA.txt")

    # Merge TATA + nonTATA as positive samples
    pos_sequences = tata_seqs + nontata_seqs
    pos_labels = np.ones(len(pos_sequences), dtype=np.intp)

    # Generate negatives
    neg_sequences, neg_labels = generate_negative_set(pos_sequences, seed=seed)

    # Combine
    all_sequences = pos_sequences + neg_sequences
    all_labels = np.concatenate([pos_labels, neg_labels])

    return _split_to_pool(all_sequences, all_labels, seed, test_split)


def load_transfer_data(
    source_species: str,
    target_species: str,
    data_dir: str,
    seed: int = 42,
    test_split: float = 0.2,
) -> tuple[list[str], NDArray[np.intp], DataPool, TextClassificationDataset, list[str]]:
    """Load source domain (full) and target domain (pool + test) data.

    Args:
        source_species: Source species prefix (e.g. "hs").
        target_species: Target species prefix (e.g. "mm").
        data_dir: Directory containing the promoter text files.
        seed: Random seed.
        test_split: Test fraction for target species.

    Returns:
        (source_texts, source_labels,
         target_pool, target_test_dataset, target_test_texts)
    """
    data_path = Path(data_dir)

    # Source domain: load all data (no train/test split needed)
    src_tata = _load_sequences(data_path / f"{source_species}_pos_TATA.txt")
    src_nontata = _load_sequences(data_path / f"{source_species}_pos_nonTATA.txt")
    src_pos = src_tata + src_nontata
    src_pos_labels = np.ones(len(src_pos), dtype=np.intp)
    src_neg, src_neg_labels = generate_negative_set(src_pos, seed=seed)
    source_texts = src_pos + src_neg
    source_labels = np.concatenate([src_pos_labels, src_neg_labels])

    # Target domain: split into pool + test
    target_pool, target_test_dataset, target_test_texts = load_promoter_species(
        species=target_species,
        data_dir=data_dir,
        seed=seed,
        test_split=test_split,
    )

    return source_texts, source_labels, target_pool, target_test_dataset, target_test_texts
