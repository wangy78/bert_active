# Label Distribution Tracking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track and visualize sample label distribution across active learning rounds

**Architecture:** Extend MetricsTracker to record sample selection history (indices, labels, counts) and generate stacked bar + line plots showing class distribution trends. Integrate into ActiveLearningLoop after each query.

**Tech Stack:** NumPy (data), matplotlib (visualization), pytest (testing), collections.Counter (label counting)

---

## Chunk 1: Core Infrastructure

### Task 1: Initialize sample_selection_history field

**Files:**
- Modify: `src/bert_active/evaluation/metrics.py:25-32`
- Test: `tests/test_label_distribution.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_label_distribution.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_label_distribution.py::test_metrics_tracker_initializes_with_empty_selection_history -v`

Expected: FAIL with "AttributeError: 'MetricsTracker' object has no attribute 'sample_selection_history'"

- [ ] **Step 3: Write minimal implementation**

In `src/bert_active/evaluation/metrics.py`, modify `__init__`:

```python
def __init__(self, experiment_name: str) -> None:
    """Initialize metrics tracker.

    Args:
        experiment_name: Name of the experiment for saving outputs
    """
    self.experiment_name = experiment_name
    self.history: list[dict[str, Any]] = []
    self.sample_selection_history: list[dict[str, Any]] = []  # NEW
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_label_distribution.py::test_metrics_tracker_initializes_with_empty_selection_history -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bert_active/evaluation/metrics.py tests/test_label_distribution.py
git commit -m "feat: add sample_selection_history to MetricsTracker

Initialize empty list to track sample selection across rounds.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Implement log_sample_selection()

**Files:**
- Modify: `src/bert_active/evaluation/metrics.py` (add method after `log_round`)
- Test: `tests/test_label_distribution.py`

- [ ] **Step 1: Add Counter import**

In `src/bert_active/evaluation/metrics.py`, add to imports at top:

```python
from collections import Counter
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_label_distribution.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_label_distribution.py::test_log_sample_selection_records_entry -v`

Expected: FAIL with "AttributeError: 'MetricsTracker' object has no attribute 'log_sample_selection'"

- [ ] **Step 4: Write minimal implementation**

In `src/bert_active/evaluation/metrics.py`, add method after `log_round()`:

```python
def log_sample_selection(
    self,
    round_num: int,
    indices: NDArray[np.intp],
    labels: NDArray[np.intp],
) -> None:
    """Record sample selection details and statistics.

    Args:
        round_num: Round number (1-indexed, matching the round these samples will be used in)
        indices: Array of selected sample indices
        labels: Array of corresponding labels
    """
    label_counts = dict(Counter(labels.tolist()))

    entry = {
        "round": round_num,
        "indices": indices.tolist(),
        "labels": labels.tolist(),
        "label_counts": label_counts,
    }
    self.sample_selection_history.append(entry)
```

- [ ] **Step 5: Add numpy typing import**

At the top of `src/bert_active/evaluation/metrics.py`, ensure this import exists:

```python
from numpy.typing import NDArray
```

If it doesn't exist, add it after the numpy import.

- [ ] **Step 6: Run all tests to verify they pass**

Run: `pytest tests/test_label_distribution.py -v`

Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/bert_active/evaluation/metrics.py tests/test_label_distribution.py
git commit -m "feat: implement log_sample_selection method

Add method to record sample indices, labels, and label counts for each
active learning round. Uses Counter to aggregate label statistics.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Modify save() to include sample_selection

**Files:**
- Modify: `src/bert_active/evaluation/metrics.py:58-69`
- Test: `tests/test_label_distribution.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_label_distribution.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_label_distribution.py::test_save_includes_sample_selection_history -v`

Expected: FAIL - JSON structure doesn't match (missing "metrics" and "sample_selection" keys)

- [ ] **Step 3: Modify save() implementation**

In `src/bert_active/evaluation/metrics.py`, replace the `save()` method:

```python
def save(self, output_dir: str) -> None:
    """Save metrics history and sample selection to JSON file.

    Args:
        output_dir: Directory to save metrics JSON
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine both histories
    data = {
        "metrics": self.history,
        "sample_selection": self.sample_selection_history,
    }

    metrics_file = output_path / f"{self.experiment_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_label_distribution.py::test_save_includes_sample_selection_history tests/test_label_distribution.py::test_save_backward_compatible_with_empty_selection -v`

Expected: Both tests PASS

- [ ] **Step 5: Run all existing tests to ensure backward compatibility**

Run: `pytest tests/test_config.py tests/test_tokenization.py -v`

Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/bert_active/evaluation/metrics.py tests/test_label_distribution.py
git commit -m "feat: modify save() to include sample selection data

Update JSON output format to include both metrics and sample_selection
sections. Maintains backward compatibility when selection history is empty.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Chunk 2: Visualization

### Task 4: Implement plot_label_distribution()

**Files:**
- Modify: `src/bert_active/evaluation/metrics.py` (add method after `plot_learning_curves`)
- Test: `tests/test_label_distribution.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_label_distribution.py`:

```python
def test_plot_label_distribution_creates_file():
    """Test that plot_label_distribution creates PNG file."""
    tracker = MetricsTracker(experiment_name="test_exp")

    # Add sample selection for multiple rounds
    tracker.log_sample_selection(
        round_num=1,
        indices=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.intp),
        labels=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.intp),
    )
    tracker.log_sample_selection(
        round_num=2,
        indices=np.array([9, 10, 11, 12], dtype=np.intp),
        labels=np.array([0, 1, 2, 3], dtype=np.intp),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker.plot_label_distribution(tmpdir)

        plot_path = Path(tmpdir) / "test_exp_label_distribution.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


def test_plot_label_distribution_with_empty_history():
    """Test that plot_label_distribution handles empty history gracefully."""
    tracker = MetricsTracker(experiment_name="test_exp")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise error
        tracker.plot_label_distribution(tmpdir)

        # Should not create file
        plot_path = Path(tmpdir) / "test_exp_label_distribution.png"
        assert not plot_path.exists()


def test_plot_label_distribution_with_single_round():
    """Test that plot_label_distribution works with single round."""
    tracker = MetricsTracker(experiment_name="test_exp")

    tracker.log_sample_selection(
        round_num=1,
        indices=np.array([1, 2, 3, 4], dtype=np.intp),
        labels=np.array([0, 1, 2, 3], dtype=np.intp),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker.plot_label_distribution(tmpdir)

        plot_path = Path(tmpdir) / "test_exp_label_distribution.png"
        assert plot_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_label_distribution.py::test_plot_label_distribution_creates_file -v`

Expected: FAIL with "AttributeError: 'MetricsTracker' object has no attribute 'plot_label_distribution'"

- [ ] **Step 3: Write minimal implementation**

In `src/bert_active/evaluation/metrics.py`, add method after `plot_learning_curves()`:

```python
def plot_label_distribution(self, output_dir: str, num_classes: int = 4) -> None:
    """Generate label distribution visualizations.

    Creates two subplots:
    1. Stacked bar chart - shows per-round class distribution
    2. Line plot - shows per-class trend across rounds

    Args:
        output_dir: Directory to save the plot
        num_classes: Number of classes in the dataset (default: 4 for AG News)
    """
    if not self.sample_selection_history:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    rounds = [entry["round"] for entry in self.sample_selection_history]

    # Build count matrix: rows=rounds, cols=classes
    count_matrix = np.zeros((len(rounds), num_classes), dtype=int)
    for i, entry in enumerate(self.sample_selection_history):
        for label, count in entry["label_counts"].items():
            count_matrix[i, label] = count

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Stacked bar chart
    bottom = np.zeros(len(rounds))
    colors = plt.cm.tab10(np.linspace(0, 0.4, num_classes))

    for class_idx in range(num_classes):
        ax1.bar(
            rounds,
            count_matrix[:, class_idx],
            bottom=bottom,
            label=f"Class {class_idx}",
            color=colors[class_idx],
        )
        bottom += count_matrix[:, class_idx]

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Sample Distribution per Round (Stacked)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Line plot
    for class_idx in range(num_classes):
        ax2.plot(
            rounds,
            count_matrix[:, class_idx],
            marker="o",
            label=f"Class {class_idx}",
            color=colors[class_idx],
            linewidth=2,
        )

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Sample Distribution Trend by Class")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_path / f"{self.experiment_name}_label_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_label_distribution.py::test_plot_label_distribution_creates_file tests/test_label_distribution.py::test_plot_label_distribution_with_empty_history tests/test_label_distribution.py::test_plot_label_distribution_with_single_round -v`

Expected: All 3 tests PASS

- [ ] **Step 5: Run all label distribution tests**

Run: `pytest tests/test_label_distribution.py -v`

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/bert_active/evaluation/metrics.py tests/test_label_distribution.py
git commit -m "feat: implement plot_label_distribution visualization

Add method to generate stacked bar chart and line plot showing class
distribution across active learning rounds. Handles empty history and
single-round cases gracefully.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Chunk 3: Integration

### Task 5: Integrate with ActiveLearningLoop

**Files:**
- Modify: `src/bert_active/engine/active_loop.py:157-161`

- [ ] **Step 1: Modify active_loop.py to track sample selection**

In `src/bert_active/engine/active_loop.py`, replace lines 157-161:

**Before:**
```python
if round_num < self.config.active_learning.n_rounds - 1:
    query_indices = self.strategy.query(
        n=self.config.active_learning.n_query,
    )
    self.pool.label(query_indices)
```

**After:**
```python
if round_num < self.config.active_learning.n_rounds - 1:
    query_indices = self.strategy.query(
        n=self.config.active_learning.n_query,
    )
    query_labels = self.pool.labels[query_indices]
    self.pool.label(query_indices)

    # Track sample selection for next round
    self.metrics_tracker.log_sample_selection(
        round_num=round_num + 1,
        indices=query_indices,
        labels=query_labels,
    )
```

- [ ] **Step 2: Verify integration by running existing tests**

Run: `pytest tests/ -v`

Expected: All tests PASS (including existing tests)

- [ ] **Step 3: Commit**

```bash
git add src/bert_active/engine/active_loop.py
git commit -m "feat: integrate sample selection tracking in ActiveLearningLoop

Track selected sample indices and labels after each query operation.
Passes data to MetricsTracker for recording and visualization.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Add visualization call to run.py

**Files:**
- Modify: `src/bert_active/run.py:161-162`

- [ ] **Step 1: Add plot_label_distribution call**

In `src/bert_active/run.py`, after line 162, add:

```python
logger.info("Plotting learning curves...")
metrics.plot_learning_curves(str(output_path))

logger.info("Plotting label distribution...")
metrics.plot_label_distribution(str(output_path))  # NEW
```

The section should now look like:

```python
logger.info(f"Saving metrics to {output_path}")
metrics.save(str(output_path))

logger.info("Plotting learning curves...")
metrics.plot_learning_curves(str(output_path))

logger.info("Plotting label distribution...")
metrics.plot_label_distribution(str(output_path))

summary = metrics.get_summary()
```

- [ ] **Step 2: Commit**

```bash
git add src/bert_active/run.py
git commit -m "feat: add label distribution plotting to main workflow

Call plot_label_distribution after plot_learning_curves in the
experiment completion workflow.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Chunk 4: Final Validation

### Task 7: Integration test with real experiment

**Files:**
- Test: Manual integration test
- Reference: `configs/default.yaml`

- [ ] **Step 1: Create minimal test config**

Create `configs/test_label_dist.yaml`:

```yaml
experiment_name: test_label_dist
output_dir: outputs/test_label_dist

model:
  name: google/bert_uncased_L-2_H-128_A-2
  num_labels: 4
  max_length: 128
  attn_implementation: eager
  torch_dtype: bfloat16

data:
  seed: 42
  test_size: 1000

trainer:
  device: mps
  learning_rate: 2.0e-5
  weight_decay: 0.01
  num_epochs: 2
  batch_size: 32
  warmup_ratio: 0.1

active_learning:
  strategy: random
  n_rounds: 2
  n_query: 20
  n_init: 40
  strategy_params: null

wandb:
  enabled: false
  project: bert-active-learning
  entity: null
  tags: []
  notes: null
```

- [ ] **Step 2: Run integration test**

Run: `python -m bert_active.run --config configs/test_label_dist.yaml`

Expected:
- Training completes successfully
- Files created in `outputs/test_label_dist/`:
  - `test_label_dist_metrics.json` (with "metrics" and "sample_selection" sections)
  - `test_label_dist_learning_curves.png`
  - `test_label_dist_label_distribution.png` (NEW)

- [ ] **Step 3: Verify JSON structure**

Run: `python -c "import json; data = json.load(open('outputs/test_label_dist/test_label_dist_metrics.json')); print('metrics:', len(data['metrics'])); print('sample_selection:', len(data['sample_selection']))"`

Expected output:
```
metrics: 2
sample_selection: 1
```

(2 rounds of metrics, 1 sample selection logged for round 1→2)

- [ ] **Step 4: Verify PNG files exist**

Run: `ls -lh outputs/test_label_dist/*.png`

Expected:
- `test_label_dist_learning_curves.png` (existing)
- `test_label_dist_label_distribution.png` (new, ~50-100KB)

- [ ] **Step 5: Manually inspect visualization**

Open `outputs/test_label_dist/test_label_dist_label_distribution.png`

Verify:
- Left panel: Stacked bar chart with 1 bar (round 1)
- Right panel: 4 lines (one per class) with 1 data point each
- Clear labels, legend, grid

- [ ] **Step 6: Clean up test outputs**

Run: `rm -rf outputs/test_label_dist configs/test_label_dist.yaml`

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 8: Final commit**

```bash
git add configs/test_label_dist.yaml
git commit -m "test: add integration test config for label distribution

Minimal config for testing label distribution tracking end-to-end.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Files Modified:**
- `src/bert_active/evaluation/metrics.py` (~110 lines added)
- `src/bert_active/engine/active_loop.py` (~6 lines added)
- `src/bert_active/run.py` (~2 lines added)

**Files Created:**
- `tests/test_label_distribution.py` (~150 lines)
- `configs/test_label_dist.yaml` (test config)

**Total Changes:** ~268 lines added

**Verification:**
- All unit tests pass ✓
- Integration test produces expected outputs ✓
- Backward compatibility maintained ✓
- Visualization quality verified ✓

## Next Steps

After implementation:
1. Run full experiment with multiple strategies to compare label distributions
2. Consider adding wandb integration for online visualization
3. Add cumulative distribution plots as future enhancement
