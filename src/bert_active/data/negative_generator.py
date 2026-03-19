"""Negative sample generation for promoter sequences.

Paper method: split 300bp into 20 blocks of 15bp, randomly keep 8 blocks,
replace the other 12 with random nucleotides. ~40% retention preserves
some conserved motifs (e.g. TATA box) in a fraction of negatives.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NUCLEOTIDES = list("ACGT")


def generate_negative_sample(
    seq: str,
    num_blocks: int = 20,
    keep_blocks: int = 8,
    rng: np.random.Generator | None = None,
) -> str:
    """Generate one negative sample from a positive promoter sequence.

    Args:
        seq: Positive promoter sequence (300bp expected).
        num_blocks: Number of sub-blocks to divide the sequence into.
        keep_blocks: Number of blocks to keep from the original sequence.
        rng: Numpy random generator for reproducibility.

    Returns:
        A shuffled negative sequence of the same length.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_len = len(seq)
    block_size = seq_len // num_blocks
    remainder = seq_len % num_blocks

    # Build block boundaries
    blocks: list[tuple[int, int]] = []
    start = 0
    for i in range(num_blocks):
        end = start + block_size + (1 if i < remainder else 0)
        blocks.append((start, end))
        start = end

    # Choose which blocks to keep
    keep_idx = set(rng.choice(num_blocks, size=keep_blocks, replace=False).tolist())

    # Build negative sequence
    result = list(seq)
    for i, (s, e) in enumerate(blocks):
        if i not in keep_idx:
            for j in range(s, e):
                result[j] = rng.choice(NUCLEOTIDES)

    return "".join(result)


def generate_negative_set(
    sequences: list[str],
    seed: int = 42,
) -> tuple[list[str], NDArray[np.intp]]:
    """Generate 1:1 negative samples for a list of positive sequences.

    Args:
        sequences: List of positive promoter sequences.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (negative_sequences, labels) where labels are all 0.
    """
    rng = np.random.default_rng(seed)
    negatives = [generate_negative_sample(seq, rng=rng) for seq in sequences]
    labels = np.zeros(len(negatives), dtype=np.intp)
    return negatives, labels
