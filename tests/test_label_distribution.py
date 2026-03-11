"""Tests for label distribution tracking functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from bert_active.evaluation.metrics import MetricsTracker


def test_metrics_tracker_initializes_with_empty_selection_history():
    """Test that MetricsTracker initializes sample_selection_history."""
    tracker = MetricsTracker(experiment_name="test")

    assert hasattr(tracker, "sample_selection_history")
    assert tracker.sample_selection_history == []
    assert isinstance(tracker.sample_selection_history, list)


def test_log_sample_selection_records_entry():
    """Test that log_sample_selection records indices, labels, and counts."""
    tracker = MetricsTracker(experiment_name="test")

    indices = np.array([42, 157, 891, 234], dtype=np.intp)
    labels = np.array([2, 0, 1, 3], dtype=np.intp)

    tracker.log_sample_selection(
        round_num=1,
        indices=indices,
        labels=labels,
    )

    assert len(tracker.sample_selection_history) == 1
    entry = tracker.sample_selection_history[0]

    assert entry["round"] == 1
    assert entry["indices"] == [42, 157, 891, 234]
    assert entry["labels"] == [2, 0, 1, 3]
    assert entry["label_counts"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_log_sample_selection_computes_label_counts():
    """Test that label_counts correctly aggregates repeated labels."""
    tracker = MetricsTracker(experiment_name="test")

    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.intp)
    labels = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.intp)

    tracker.log_sample_selection(round_num=1, indices=indices, labels=labels)

    entry = tracker.sample_selection_history[0]
    assert entry["label_counts"] == {0: 2, 1: 3, 2: 1, 3: 2}


def test_log_sample_selection_multiple_rounds():
    """Test logging sample selection for multiple rounds."""
    tracker = MetricsTracker(experiment_name="test")

    # Round 1
    tracker.log_sample_selection(
        round_num=1,
        indices=np.array([1, 2, 3, 4], dtype=np.intp),
        labels=np.array([0, 1, 2, 3], dtype=np.intp),
    )

    # Round 2
    tracker.log_sample_selection(
        round_num=2,
        indices=np.array([5, 6, 7, 8], dtype=np.intp),
        labels=np.array([1, 1, 2, 3], dtype=np.intp),
    )

    assert len(tracker.sample_selection_history) == 2
    assert tracker.sample_selection_history[0]["round"] == 1
    assert tracker.sample_selection_history[1]["round"] == 2
    assert tracker.sample_selection_history[1]["label_counts"] == {1: 2, 2: 1, 3: 1}


def test_log_sample_selection_validates_array_lengths():
    """Test that mismatched array lengths raise ValueError."""
    tracker = MetricsTracker(experiment_name="test")

    indices = np.array([1, 2, 3], dtype=np.intp)
    labels = np.array([0, 1], dtype=np.intp)

    with pytest.raises(ValueError, match="Length mismatch"):
        tracker.log_sample_selection(round_num=1, indices=indices, labels=labels)


def test_save_includes_sample_selection_history():
    """Test that save() writes both metrics and sample_selection to JSON."""
    tracker = MetricsTracker(experiment_name="test_exp")

    # Log some metrics
    tracker.log_round(
        round_num=0,
        n_labeled=40,
        train_metrics={"train_loss": 0.5},
        eval_metrics={"eval_accuracy": 0.85},
    )

    # Log sample selection
    tracker.log_sample_selection(
        round_num=1,
        indices=np.array([1, 2, 3, 4], dtype=np.intp),
        labels=np.array([0, 1, 2, 3], dtype=np.intp),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker.save(tmpdir)

        json_path = Path(tmpdir) / "test_exp_metrics.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "sample_selection" in data
        assert len(data["metrics"]) == 1
        assert len(data["sample_selection"]) == 1
        assert data["metrics"][0]["round"] == 0
        assert data["sample_selection"][0]["round"] == 1


def test_save_backward_compatible_with_empty_selection():
    """Test that save() works when sample_selection_history is empty."""
    tracker = MetricsTracker(experiment_name="test_exp")

    tracker.log_round(
        round_num=0,
        n_labeled=40,
        train_metrics={"train_loss": 0.5},
        eval_metrics={"eval_accuracy": 0.85},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker.save(tmpdir)

        json_path = Path(tmpdir) / "test_exp_metrics.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "sample_selection" in data
        assert len(data["metrics"]) == 1
        assert data["sample_selection"] == []
