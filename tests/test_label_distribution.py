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
