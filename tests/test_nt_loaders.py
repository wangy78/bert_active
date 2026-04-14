"""Smoke tests for NT downstream task loaders."""
import numpy as np
import pytest

from bert_active.data.dna_dataset import (
    load_nt_h3k4me3,
    load_nt_enhancers,
    load_nt_enhancers_types,
)


def test_load_nt_h3k4me3_shapes():
    pool, test_ds, test_texts = load_nt_h3k4me3(seed=42, max_samples=200)
    assert len(pool.texts) > 0
    assert len(pool.texts) == len(pool.labels)
    assert pool.labels.dtype == np.intp
    assert set(np.unique(pool.labels)).issubset({0, 1})
    assert len(test_texts) == len(test_ds.labels)


def test_load_nt_enhancers_shapes():
    pool, test_ds, test_texts = load_nt_enhancers(seed=42, max_samples=200)
    assert len(pool.texts) > 0
    assert set(np.unique(pool.labels)).issubset({0, 1})
    assert len(test_texts) == len(test_ds.labels)


def test_load_nt_enhancers_types_shapes():
    pool, test_ds, test_texts = load_nt_enhancers_types(seed=42, max_samples=200)
    assert len(pool.texts) > 0
    assert set(np.unique(pool.labels)).issubset({0, 1, 2})
    assert len(test_texts) == len(test_ds.labels)
