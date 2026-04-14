"""Smoke tests for NT downstream task loaders."""
import numpy as np
import pytest

from bert_active.data.dna_dataset import (
    load_nt_h3k4me3,
    load_nt_enhancers,
    load_nt_enhancers_types,
)

from bert_active.config.experiment import PretrainConfig, ExperimentConfig
import tempfile, os, yaml


def test_pretrain_config_source_dataset_name():
    cfg = PretrainConfig(source_dataset_name="nt_h3k4me3")
    assert cfg.source_dataset_name == "nt_h3k4me3"


def test_pretrain_config_default_none():
    cfg = PretrainConfig()
    assert cfg.source_dataset_name is None


def test_pretrain_config_from_yaml():
    data = {
        "experiment_name": "test",
        "pretrain": {
            "enabled": True,
            "source_dataset_name": "nt_h3k4me3",
            "num_epochs": 5,
            "learning_rate": 2e-5,
            "batch_size": 16,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.pretrain.source_dataset_name == "nt_h3k4me3"
    finally:
        os.unlink(path)


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
