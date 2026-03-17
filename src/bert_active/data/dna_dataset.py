"""DNA sequence datasets for active learning with DNA-BERT.

Supports:
- UCI Promoter Gene Sequences Dataset (small, benchmark)
- NCBI Eukaryotic Promoter Database (EPD) (large, real-world)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bert_active.data.dataset import DataPool, TextClassificationDataset


def seq_to_kmer(seq: str, k: int = 6) -> str:
    """Convert DNA sequence to k-mer format for DNA-BERT.

    DNA-BERT uses overlapping k-mer tokenization. For example:
        ATCGATCG with k=3 -> "ATC TCG CGA GAT ATC TCG"

    Args:
        seq: Raw DNA sequence (ATCG format)
        k: k-mer size (default 6, as used in DNABERT paper)

    Returns:
        Space-separated k-mer string
    """
    # Remove whitespace and convert to uppercase
    seq = seq.replace(" ", "").replace("\n", "").upper()

    # Validate DNA sequence
    valid_bases = set("ATCGN")
    if not all(base in valid_bases for base in seq):
        raise ValueError(f"Invalid DNA sequence: {seq}")

    # Generate overlapping k-mers
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    return " ".join(kmers)


def load_uci_promoter(
    seed: int = 42,
    test_split: float = 0.2,
    k: int | None = None,  # None = no k-mer (for DNABERT-2), 6 = k-mer (for DNABERT-1)
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load UCI Promoter Gene Sequences Dataset.

    Dataset: https://archive.ics.uci.edu/dataset/67
    - Binary classification: promoter (+) vs non-promoter (-)
    - 106 samples (53 promoters, 53 non-promoters)
    - Sequence length: 57 bp

    Args:
        seed: Random seed for train/test split
        test_split: Fraction of data for testing (default 0.2)
        k: k-mer size for DNA-BERT tokenization (default 6)

    Returns:
        - DataPool for training (pool-based active learning)
        - TextClassificationDataset for testing (placeholder, needs tokenization)
        - test_texts_raw: Raw k-mer sequences for test set
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        msg = "ucimlrepo is required. Install with: pip install ucimlrepo"
        raise ImportError(msg) from e

    # Fetch dataset
    promoter_data = fetch_ucirepo(id=67)

    # Extract features and labels
    X = promoter_data.data.features  # DataFrame with sequence columns
    y = promoter_data.data.targets  # Series with +/- labels

    # Combine sequence columns into single DNA string
    sequences = []
    for _, row in X.iterrows():
        # Each row has columns p-50, p-49, ..., p0, p1, ..., p+7
        # Concatenate them in order
        seq_parts = [str(row[col]) for col in sorted(X.columns)]
        seq = "".join(seq_parts)
        sequences.append(seq)

    # Convert labels to numeric: + (promoter) -> 1, - (non-promoter) -> 0
    labels = np.array([1 if label == "+" else 0 for label in y.values], dtype=np.intp)

    # Convert to k-mer format (if k is specified)
    if k is not None:
        sequences_kmer = [seq_to_kmer(seq, k=k) for seq in sequences]
    else:
        # DNABERT-2: use raw sequences
        sequences_kmer = sequences

    # Train/test split
    rng = np.random.default_rng(seed)
    n_samples = len(sequences_kmer)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_split)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Create DataPool for training
    train_texts = [sequences_kmer[i] for i in train_indices]
    train_labels = labels[train_indices]
    pool = DataPool(texts=train_texts, labels=train_labels)

    # Test data (raw, to be tokenized by caller)
    test_texts = [sequences_kmer[i] for i in test_indices]
    test_labels = labels[test_indices]

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),  # placeholder
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),  # placeholder
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def load_epd_promoter(
    data_dir: str | Path,
    seed: int = 42,
    test_split: float = 0.2,
    k: int | None = None,  # None = no k-mer (for DNABERT-2)
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load NCBI Eukaryotic Promoter Database (EPD).

    Expected file format (CSV):
        sequence,label
        ATCGATCG...,1
        GGCCAATT...,0

    Where:
        - sequence: raw DNA sequence (ATCG format)
        - label: 1 (promoter) or 0 (non-promoter)

    Args:
        data_dir: Directory containing 'epd_promoter.csv'
        seed: Random seed for train/test split
        test_split: Fraction of data for testing (default 0.2)
        k: k-mer size for DNA-BERT tokenization (default 6)
        max_samples: Maximum number of samples to load (for memory efficiency)

    Returns:
        - DataPool for training
        - TextClassificationDataset for testing (placeholder)
        - test_texts_raw: Raw k-mer sequences for test set
    """
    data_path = Path(data_dir) / "epd_promoter.csv"

    if not data_path.exists():
        msg = f"EPD dataset not found at {data_path}. Please download and prepare the dataset."
        raise FileNotFoundError(msg)

    # Load CSV
    df = pd.read_csv(data_path)

    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed)

    # Extract sequences and labels
    sequences_raw = df["sequence"].tolist()
    labels = np.array(df["label"].values, dtype=np.intp)

    # Convert to k-mer format (if k is specified)
    if k is not None:
        sequences_kmer = [seq_to_kmer(seq, k=k) for seq in sequences_raw]
    else:
        # DNABERT-2: use raw sequences
        sequences_kmer = sequences_raw

    # Train/test split
    rng = np.random.default_rng(seed)
    n_samples = len(sequences_kmer)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_split)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Create DataPool
    train_texts = [sequences_kmer[i] for i in train_indices]
    train_labels = labels[train_indices]
    pool = DataPool(texts=train_texts, labels=train_labels)

    # Test data
    test_texts = [sequences_kmer[i] for i in test_indices]
    test_labels = labels[test_indices]

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def prepare_epd_from_fasta(
    fasta_positive: str | Path,
    fasta_negative: str | Path,
    output_csv: str | Path,
    seq_length: int = 200,
) -> None:
    """Prepare EPD dataset from FASTA files.

    This helper function converts FASTA files (positive and negative samples)
    into the CSV format expected by load_epd_promoter().

    Args:
        fasta_positive: Path to FASTA file with promoter sequences
        fasta_negative: Path to FASTA file with non-promoter sequences
        output_csv: Output CSV path
        seq_length: Standardize all sequences to this length (truncate/pad)
    """
    try:
        from Bio import SeqIO
    except ImportError as e:
        msg = "biopython is required. Install with: pip install biopython"
        raise ImportError(msg) from e

    sequences = []
    labels = []

    # Load positive samples
    for record in SeqIO.parse(fasta_positive, "fasta"):
        seq = str(record.seq).upper()[:seq_length]
        # Pad if too short
        if len(seq) < seq_length:
            seq += "N" * (seq_length - len(seq))
        sequences.append(seq)
        labels.append(1)

    # Load negative samples
    for record in SeqIO.parse(fasta_negative, "fasta"):
        seq = str(record.seq).upper()[:seq_length]
        if len(seq) < seq_length:
            seq += "N" * (seq_length - len(seq))
        sequences.append(seq)
        labels.append(0)

    # Create DataFrame and save
    df = pd.DataFrame({"sequence": sequences, "label": labels})
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} sequences to {output_csv}")
    print(f"  - Promoters: {sum(labels)}")
    print(f"  - Non-promoters: {len(labels) - sum(labels)}")
