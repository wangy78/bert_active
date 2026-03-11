# Active Learning Label Distribution Tracking Design

**Date:** 2026-03-11
**Author:** Claude Sonnet 4.5
**Status:** Proposed

## Overview

Add tracking and visualization of sample label distribution across active learning rounds to understand sampling strategy behavior and class balance over time.

## Goals

1. **Track** which samples (by index and label) are selected each round
2. **Visualize** label distribution trends across rounds with stacked bar charts and line plots
3. **Store** both detailed selection data and statistical summaries
4. **Integrate** seamlessly with existing `MetricsTracker` infrastructure

## Non-Goals

- Real-time terminal output of distributions (save to files only)
- Wandb integration (may add later)
- Analysis of sampling strategy effectiveness (just tracking/visualization)

## Architecture

### Data Flow

```
ActiveLearningLoop.run()
  └─> For each round:
      1. strategy.query(n) → query_indices
      2. Get labels: query_labels = pool.labels[query_indices]
      3. pool.label(query_indices)
      4. metrics_tracker.log_sample_selection(
           round_num + 1,
           query_indices,
           query_labels
         )

MetricsTracker:
  - Maintains: self.sample_selection_history: list[dict]
  - Each entry: {round, indices, labels, label_counts}
  - New methods:
    * log_sample_selection() - record selection data
    * plot_label_distribution() - generate visualizations
    * save() - save metrics + selections to JSON
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Track in `ActiveLearningLoop` | Data available immediately after query |
| Store in `MetricsTracker` | Reuse existing metrics infrastructure |
| Save detailed + summary data | Enables both quick review and deep analysis |
| Stacked bar + line plots | Shows per-round distribution and trends |
| No terminal output | Avoid cluttering training logs |

## Data Structures

### MetricsTracker Extension

```python
class MetricsTracker:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.history: list[dict[str, Any]] = []  # existing
        self.sample_selection_history: list[dict[str, Any]] = []  # NEW
```

### Sample Selection Entry

```python
{
    "round": 1,                                # Round number (1-indexed)
    "indices": [42, 157, 891, 234],           # Selected sample indices
    "labels": [2, 0, 1, 3],                   # Corresponding labels
    "label_counts": {                          # Pre-computed statistics
        0: 8,   # Class 0: 8 samples
        1: 12,  # Class 1: 12 samples
        2: 6,   # Class 2: 6 samples
        3: 14   # Class 3: 14 samples
    }
}
```

### Saved JSON Format

```json
{
  "metrics": [
    {
      "round": 0,
      "n_labeled": 40,
      "train_loss": 0.523,
      "eval_accuracy": 0.8234,
      "eval_f1_macro": 0.8123
    },
    ...
  ],
  "sample_selection": [
    {
      "round": 1,
      "indices": [42, 157, ...],
      "labels": [2, 0, ...],
      "label_counts": {0: 8, 1: 12, 2: 6, 3: 14}
    },
    ...
  ]
}
```

## Implementation Details

### File 1: `src/bert_active/engine/active_loop.py`

**Location:** `ActiveLearningLoop.run()` method, lines 157-161

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
    query_labels = self.pool.labels[query_indices]  # Get labels
    self.pool.label(query_indices)

    # Track sample selection for next round
    self.metrics_tracker.log_sample_selection(
        round_num=round_num + 1,
        indices=query_indices,
        labels=query_labels,
    )
```

### File 2: `src/bert_active/evaluation/metrics.py`

#### New Method: `log_sample_selection()`

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
    from collections import Counter

    label_counts = dict(Counter(labels.tolist()))

    entry = {
        "round": round_num,
        "indices": indices.tolist(),
        "labels": labels.tolist(),
        "label_counts": label_counts,
    }
    self.sample_selection_history.append(entry)
```

#### New Method: `plot_label_distribution()`

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

#### Modified Method: `save()`

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

## Visualization Design

### Stacked Bar Chart (Left Panel)

- **X-axis:** Round number
- **Y-axis:** Number of samples
- **Bars:** Each round is one bar, stacked by class
- **Purpose:** See per-round class distribution and balance

### Line Plot (Right Panel)

- **X-axis:** Round number
- **Y-axis:** Number of samples
- **Lines:** One line per class
- **Purpose:** See how each class's selection changes over time

### Example Output

```
{experiment_name}_label_distribution.png
├─ Left: Stacked bars showing class mix each round
└─ Right: Lines showing per-class trends
```

## Integration Points

### Calling `plot_label_distribution()`

Add to `run.py` after the active learning loop completes:

```python
metrics = loop.run()
metrics.save(output_dir="outputs")
metrics.plot_learning_curves(output_dir="outputs")
metrics.plot_label_distribution(output_dir="outputs")  # NEW
```

### Backward Compatibility

- All changes are additive (no breaking changes)
- Existing code continues to work without modifications
- `log_sample_selection()` is a new method, not modifying existing interfaces

## Testing Strategy

### Unit Tests

1. **Test `log_sample_selection()`**
   - Verify entry structure
   - Check label_counts computation
   - Test with different array sizes

2. **Test `plot_label_distribution()`**
   - Verify file creation
   - Test with empty history (should no-op)
   - Test with single round
   - Test with multiple rounds

3. **Test `save()` modifications**
   - Verify JSON structure includes both metrics and sample_selection
   - Test backward compatibility with empty sample_selection_history

### Integration Tests

1. Run a small active learning experiment (2-3 rounds)
2. Verify JSON output contains both sections
3. Verify PNG file is created
4. Manually inspect visualization quality

## File Changes Summary

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `active_loop.py` | Modification | ~8 lines added |
| `metrics.py` | Addition | ~100 lines added |
| `run.py` | Modification | ~1 line added |

**Total:** ~109 lines added, 0 lines removed

## Example Output

### JSON Output (`experiment_metrics.json`)

```json
{
  "metrics": [
    {"round": 0, "n_labeled": 40, "eval_accuracy": 0.8234},
    {"round": 1, "n_labeled": 80, "eval_accuracy": 0.8567},
    {"round": 2, "n_labeled": 120, "eval_accuracy": 0.8789}
  ],
  "sample_selection": [
    {
      "round": 1,
      "indices": [42, 157, 891, ...],
      "labels": [2, 0, 1, ...],
      "label_counts": {0: 8, 1: 12, 2: 6, 3: 14}
    },
    {
      "round": 2,
      "indices": [123, 456, ...],
      "labels": [1, 3, ...],
      "label_counts": {0: 10, 1: 15, 2: 8, 3: 7}
    }
  ]
}
```

### Visualization Output

File: `experiment_label_distribution.png`

Two-panel figure showing:
- Left: Stacked bars (total 40 samples per round, colored by class)
- Right: Four lines tracking each class's selection count

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Memory overhead from storing indices | Low | Low | Indices are small integers; negligible |
| Visualization quality issues | Medium | Low | Use matplotlib best practices, test with real data |
| Breaking existing code | Low | High | All changes are additive; existing tests verify |
| Incorrect label tracking | Low | Medium | Unit tests verify label extraction and counting |

## Future Enhancements

- Add cumulative class distribution plot
- Support for wandb integration
- Configurable terminal output option
- Per-strategy comparison visualization
- Class imbalance metrics (e.g., Gini coefficient)

## References

- Existing `MetricsTracker` implementation: `src/bert_active/evaluation/metrics.py`
- Active learning loop: `src/bert_active/engine/active_loop.py`
- AG News dataset (4 classes): `src/bert_active/data/dataset.py`
