# NT Cross-Task Transfer Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add H3K4me3 → enhancers (binary) and H3K4me3 → enhancers_types (3-class) cross-task transfer learning experiments to the bert-active pipeline.

**Architecture:** Add NT downstream dataset loaders, extend `PretrainConfig` with `source_dataset_name`, generalize checkpoint naming in `active_loop.py`, then create 10 config YAMLs and a queue script.

**Tech Stack:** HuggingFace `datasets`, existing `bert_active` pipeline, YAML configs, `run_queue_lib`

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `src/bert_active/data/dna_dataset.py` | Modify | Add 3 new loader functions |
| `src/bert_active/config/experiment.py` | Modify | Add `source_dataset_name` field to `PretrainConfig` |
| `src/bert_active/engine/active_loop.py` | Modify | Add NT dataset branches + generalize checkpoint key |
| `configs/nt_enhancer_baseline_random.yaml` | Create | A1 no-pretrain baseline |
| `configs/nt_enhancer_transfer_random.yaml` | Create | A1 full fine-tune + random |
| `configs/nt_enhancer_transfer_badge.yaml` | Create | A1 full fine-tune + BADGE |
| `configs/nt_enhancer_frozen_badge.yaml` | Create | A1 frozen backbone + BADGE |
| `configs/nt_enhancer_frozen_doptimal.yaml` | Create | A1 frozen backbone + D-Optimal |
| `configs/nt_enhancer_types_baseline_random.yaml` | Create | A2 no-pretrain baseline |
| `configs/nt_enhancer_types_transfer_random.yaml` | Create | A2 full fine-tune + random |
| `configs/nt_enhancer_types_transfer_badge.yaml` | Create | A2 full fine-tune + BADGE |
| `configs/nt_enhancer_types_frozen_badge.yaml` | Create | A2 frozen backbone + BADGE |
| `configs/nt_enhancer_types_frozen_doptimal.yaml` | Create | A2 frozen backbone + D-Optimal |
| `run_queue_nt_transfer.py` | Create | 10 configs × 3 seeds = 30 jobs |
| `tests/test_nt_loaders.py` | Create | Smoke tests for new loaders + config parsing |

---

## Task 1: Add NT dataset loaders

**Files:**
- Modify: `src/bert_active/data/dna_dataset.py`
- Create: `tests/test_nt_loaders.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_nt_loaders.py
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/saltfish/Files/Coding/bert-active
python -m pytest tests/test_nt_loaders.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'load_nt_h3k4me3'`

- [ ] **Step 3: Add loaders to dna_dataset.py**

Append these three functions at the end of `src/bert_active/data/dna_dataset.py`, before `_split_to_pool`:

```python
def load_nt_h3k4me3(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load H3K4me3 histone mark dataset from NT downstream tasks.

    Binary classification: H3K4me3 mark present (1) vs absent (0).
    ~30,000 human sequences, 200bp. Used as source for cross-task transfer.
    Has train/test splits.
    """
    ds: DatasetDict = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks", "H3K4me3"
    )  # type: ignore[assignment]

    train_texts: list[str] = list(ds["train"]["sequence"])
    train_labels = np.array(ds["train"]["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=min(max_samples, len(train_texts)), replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    pool = DataPool(texts=train_texts, labels=train_labels)

    test_texts: list[str] = list(ds["test"]["sequence"])
    test_labels = np.array(ds["test"]["label"], dtype=np.intp)

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def load_nt_enhancers(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load enhancers dataset from NT downstream tasks.

    Binary classification: enhancer (1) vs background (0).
    ~14,000 human sequences, 200bp. Target for A1 cross-task transfer.
    Has train/test splits.
    """
    ds: DatasetDict = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks", "enhancers"
    )  # type: ignore[assignment]

    train_texts: list[str] = list(ds["train"]["sequence"])
    train_labels = np.array(ds["train"]["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=min(max_samples, len(train_texts)), replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    pool = DataPool(texts=train_texts, labels=train_labels)

    test_texts: list[str] = list(ds["test"]["sequence"])
    test_labels = np.array(ds["test"]["label"], dtype=np.intp)

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def load_nt_enhancers_types(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load enhancers_types dataset from NT downstream tasks.

    3-class classification: strong enhancer (0) / weak enhancer (1) / background (2).
    ~14,000 human sequences, 200bp. Target for A2 cross-task transfer.
    Has train/test splits.
    """
    ds: DatasetDict = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks", "enhancers_types"
    )  # type: ignore[assignment]

    train_texts: list[str] = list(ds["train"]["sequence"])
    train_labels = np.array(ds["train"]["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=min(max_samples, len(train_texts)), replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    pool = DataPool(texts=train_texts, labels=train_labels)

    test_texts: list[str] = list(ds["test"]["sequence"])
    test_labels = np.array(ds["test"]["label"], dtype=np.intp)

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_nt_loaders.py -v
```

Expected: all 3 tests PASS (downloads data on first run, ~1-2 min)

- [ ] **Step 5: Commit**

```bash
git add src/bert_active/data/dna_dataset.py tests/test_nt_loaders.py
git commit -m "feat: add NT downstream task loaders for H3K4me3, enhancers, enhancers_types"
```

---

## Task 2: Extend PretrainConfig and generalize checkpoint naming

**Files:**
- Modify: `src/bert_active/config/experiment.py` (lines 64-69)
- Modify: `src/bert_active/engine/active_loop.py` (lines 185-189)

- [ ] **Step 1: Write failing config test**

Add to `tests/test_nt_loaders.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_nt_loaders.py::test_pretrain_config_source_dataset_name -v
```

Expected: `TypeError: PretrainConfig.__init__() got an unexpected keyword argument 'source_dataset_name'`

- [ ] **Step 3: Add field to PretrainConfig**

In `src/bert_active/config/experiment.py`, change `PretrainConfig` from:

```python
@dataclass
class PretrainConfig:
    enabled: bool = False
    source_species: str = "hs"
    num_epochs: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 16
```

to:

```python
@dataclass
class PretrainConfig:
    enabled: bool = False
    source_species: str = "hs"
    source_dataset_name: str | None = None  # NT cross-task transfer source, e.g. "nt_h3k4me3"
    num_epochs: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 16
```

- [ ] **Step 4: Generalize checkpoint key in active_loop.py run()**

In `src/bert_active/engine/active_loop.py`, change the checkpoint directory computation (lines 186-189) from:

```python
        if self.config.pretrain.enabled and self.source_texts is not None:
            ckpt_dir = Path("checkpoints") / (
                f"pretrained_{self.config.pretrain.source_species}"
                f"_seed{self.config.data.seed}"
            )
```

to:

```python
        if self.config.pretrain.enabled and self.source_texts is not None:
            source_key = (
                self.config.pretrain.source_dataset_name
                or self.config.pretrain.source_species
            )
            ckpt_dir = Path("checkpoints") / (
                f"pretrained_{source_key}"
                f"_seed{self.config.data.seed}"
            )
```

- [ ] **Step 5: Run all config tests**

```bash
python -m pytest tests/test_nt_loaders.py -v -k "config"
```

Expected: 3 config tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/bert_active/config/experiment.py src/bert_active/engine/active_loop.py tests/test_nt_loaders.py
git commit -m "feat: add source_dataset_name to PretrainConfig, generalize checkpoint key"
```

---

## Task 3: Add NT dataset branches to active_loop.py

**Files:**
- Modify: `src/bert_active/engine/active_loop.py` (lines 10-17, 119-129)

- [ ] **Step 1: Add imports at top of active_loop.py**

In `src/bert_active/engine/active_loop.py`, change the import block (lines 12-17):

```python
from bert_active.data.dna_dataset import (
    load_dna_core_promoter,
    load_gue_fungi_species,
    load_gue_virus_covid,
    load_gue_virus_species,
    load_human_nontata_promoters,
)
```

to:

```python
from bert_active.data.dna_dataset import (
    load_dna_core_promoter,
    load_gue_fungi_species,
    load_gue_virus_covid,
    load_gue_virus_species,
    load_human_nontata_promoters,
    load_nt_enhancers,
    load_nt_enhancers_types,
    load_nt_h3k4me3,
)
```

- [ ] **Step 2: Add NT dataset loading branches**

In `src/bert_active/engine/active_loop.py`, replace the `else` clause at line 124:

```python
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: promoter, dna_core_promoter, human_nontata_promoters, "
                f"gue_fungi_species, gue_virus_species, gue_virus_covid"
            )
```

with:

```python
        elif dataset_name == "nt_enhancers":
            self.pool, test_dataset, test_texts = load_nt_enhancers(
                seed=config.data.seed,
                max_samples=max_samples,
            )
            if config.pretrain.enabled:
                source_pool, _, _ = load_nt_h3k4me3(seed=config.data.seed)
                self.source_texts = source_pool.texts
                self.source_labels = source_pool.labels
        elif dataset_name == "nt_enhancer_types":
            self.pool, test_dataset, test_texts = load_nt_enhancers_types(
                seed=config.data.seed,
                max_samples=max_samples,
            )
            if config.pretrain.enabled:
                source_pool, _, _ = load_nt_h3k4me3(seed=config.data.seed)
                self.source_texts = source_pool.texts
                self.source_labels = source_pool.labels
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: promoter, dna_core_promoter, human_nontata_promoters, "
                f"gue_fungi_species, gue_virus_species, gue_virus_covid, "
                f"nt_enhancers, nt_enhancer_types"
            )
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from bert_active.engine.active_loop import ActiveLearningLoop; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/bert_active/engine/active_loop.py
git commit -m "feat: add nt_enhancers and nt_enhancer_types dataset branches to active loop"
```

---

## Task 4: Create A1 config files (enhancers, binary)

**Files:** Create 5 YAML files in `configs/`

- [ ] **Step 1: Create nt_enhancer_baseline_random.yaml**

```yaml
experiment_name: "nt_enhancer_baseline_random"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 2
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancers"
  seed: 42

pretrain:
  enabled: false

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "random"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer", "baseline"]
  notes: "Enhancer detection from scratch, random AL (no pretrain baseline)"

output_dir: "outputs"
```

- [ ] **Step 2: Create nt_enhancer_transfer_random.yaml**

```yaml
experiment_name: "nt_enhancer_transfer_random"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 2
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancers"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "random"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer", "full-finetune"]
  notes: "H3K4me3 pretrain -> enhancer target, full fine-tune + random AL"

output_dir: "outputs"
```

- [ ] **Step 3: Create nt_enhancer_transfer_badge.yaml**

```yaml
experiment_name: "nt_enhancer_transfer_badge"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 2
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancers"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "badge"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer", "full-finetune", "badge"]
  notes: "H3K4me3 pretrain -> enhancer target, full fine-tune + BADGE AL"

output_dir: "outputs"
```

- [ ] **Step 4: Create nt_enhancer_frozen_badge.yaml**

```yaml
experiment_name: "nt_enhancer_frozen_badge"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 2
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancers"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 1e-3
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "badge"
  freeze_backbone: true

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer", "frozen-backbone", "badge"]
  notes: "H3K4me3 pretrain -> enhancer target, frozen backbone (linear probe) + BADGE AL"

output_dir: "outputs"
```

- [ ] **Step 5: Create nt_enhancer_frozen_doptimal.yaml**

```yaml
experiment_name: "nt_enhancer_frozen_doptimal"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 2
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancers"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 1e-3
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "doptimal"
  freeze_backbone: true

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer", "frozen-backbone", "doptimal"]
  notes: "H3K4me3 pretrain -> enhancer target, frozen backbone (linear probe) + D-Optimal AL"

output_dir: "outputs"
```

- [ ] **Step 6: Verify all A1 configs parse without error**

```bash
python -c "
from bert_active.config.experiment import ExperimentConfig
for f in [
    'configs/nt_enhancer_baseline_random.yaml',
    'configs/nt_enhancer_transfer_random.yaml',
    'configs/nt_enhancer_transfer_badge.yaml',
    'configs/nt_enhancer_frozen_badge.yaml',
    'configs/nt_enhancer_frozen_doptimal.yaml',
]:
    cfg = ExperimentConfig.from_yaml(f)
    print(f'{f}: {cfg.experiment_name} num_labels={cfg.model.num_labels} pretrain={cfg.pretrain.enabled}')
"
```

Expected: 5 lines printed, no errors.

- [ ] **Step 7: Commit**

```bash
git add configs/nt_enhancer_*.yaml
git commit -m "feat: add A1 enhancer transfer learning config files"
```

---

## Task 5: Create A2 config files (enhancers_types, 3-class)

**Files:** Create 5 YAML files in `configs/`

- [ ] **Step 1: Create all 5 A2 configs**

`configs/nt_enhancer_types_baseline_random.yaml`:
```yaml
experiment_name: "nt_enhancer_types_baseline_random"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 3
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancer_types"
  seed: 42

pretrain:
  enabled: false

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "random"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer_types", "baseline"]
  notes: "Enhancer types (3-class) from scratch, random AL (no pretrain baseline)"

output_dir: "outputs"
```

`configs/nt_enhancer_types_transfer_random.yaml`:
```yaml
experiment_name: "nt_enhancer_types_transfer_random"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 3
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancer_types"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "random"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer_types", "full-finetune"]
  notes: "H3K4me3 pretrain -> enhancer types (3-class), full fine-tune + random AL"

output_dir: "outputs"
```

`configs/nt_enhancer_types_transfer_badge.yaml`:
```yaml
experiment_name: "nt_enhancer_types_transfer_badge"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 3
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancer_types"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 2e-5
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "badge"
  freeze_backbone: false

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer_types", "full-finetune", "badge"]
  notes: "H3K4me3 pretrain -> enhancer types (3-class), full fine-tune + BADGE AL"

output_dir: "outputs"
```

`configs/nt_enhancer_types_frozen_badge.yaml`:
```yaml
experiment_name: "nt_enhancer_types_frozen_badge"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 3
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancer_types"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 1e-3
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "badge"
  freeze_backbone: true

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer_types", "frozen-backbone", "badge"]
  notes: "H3K4me3 pretrain -> enhancer types (3-class), frozen backbone (linear probe) + BADGE AL"

output_dir: "outputs"
```

`configs/nt_enhancer_types_frozen_doptimal.yaml`:
```yaml
experiment_name: "nt_enhancer_types_frozen_doptimal"

model:
  name: "zhihan1996/DNABERT-2-117M"
  num_labels: 3
  attn_implementation: "flash_attention_2"
  torch_dtype: "float32"
  max_length: 200

data:
  dataset_name: "nt_enhancer_types"
  seed: 42

pretrain:
  enabled: true
  source_dataset_name: "nt_h3k4me3"
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 16

trainer:
  learning_rate: 1e-3
  weight_decay: 0.01
  batch_size: 16
  eval_batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  device: "cuda"

active_learning:
  n_init: 100
  n_query: 100
  n_rounds: 20
  strategy: "doptimal"
  freeze_backbone: true

wandb:
  enabled: true
  project: "bert-active-dna"
  entity: "codycjy"
  tags: ["nt_downstream", "transfer", "h3k4me3_to_enhancer_types", "frozen-backbone", "doptimal"]
  notes: "H3K4me3 pretrain -> enhancer types (3-class), frozen backbone (linear probe) + D-Optimal AL"

output_dir: "outputs"
```

- [ ] **Step 2: Verify all A2 configs parse**

```bash
python -c "
from bert_active.config.experiment import ExperimentConfig
for f in [
    'configs/nt_enhancer_types_baseline_random.yaml',
    'configs/nt_enhancer_types_transfer_random.yaml',
    'configs/nt_enhancer_types_transfer_badge.yaml',
    'configs/nt_enhancer_types_frozen_badge.yaml',
    'configs/nt_enhancer_types_frozen_doptimal.yaml',
]:
    cfg = ExperimentConfig.from_yaml(f)
    print(f'{f}: {cfg.experiment_name} num_labels={cfg.model.num_labels} pretrain={cfg.pretrain.enabled}')
"
```

Expected: 5 lines, `num_labels=3` for all.

- [ ] **Step 3: Commit**

```bash
git add configs/nt_enhancer_types_*.yaml
git commit -m "feat: add A2 enhancer_types transfer learning config files"
```

---

## Task 6: Create queue script and dry-run verification

**Files:**
- Create: `run_queue_nt_transfer.py`

- [ ] **Step 1: Create run_queue_nt_transfer.py**

```python
"""Job queue for NT cross-task transfer experiments.

Conditions (per target):
  1. baseline_random        — From scratch, random AL (no transfer)
  2. transfer_random        — H3K4me3 pretrain, full fine-tune, random AL
  3. transfer_badge         — H3K4me3 pretrain, full fine-tune, BADGE AL
  4. frozen_badge           — H3K4me3 pretrain, frozen backbone, BADGE AL
  5. frozen_doptimal        — H3K4me3 pretrain, frozen backbone, D-Optimal AL

Two targets:
  A1: nt_enhancers (binary)
  A2: nt_enhancer_types (3-class)

Total: 10 configs × 3 seeds = 30 jobs

Usage:
  python run_queue_nt_transfer.py --gpus 0 1 2 3
  python run_queue_nt_transfer.py --gpus 0 --dry-run
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

jobs = [
    # A1: enhancers (binary)
    Job("configs/nt_enhancer_baseline_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_transfer_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_transfer_badge.yaml",      seeds=SEEDS),
    Job("configs/nt_enhancer_frozen_badge.yaml",        seeds=SEEDS),
    Job("configs/nt_enhancer_frozen_doptimal.yaml",     seeds=SEEDS),
    # A2: enhancers_types (3-class)
    Job("configs/nt_enhancer_types_baseline_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_types_transfer_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_types_transfer_badge.yaml",      seeds=SEEDS),
    Job("configs/nt_enhancer_types_frozen_badge.yaml",        seeds=SEEDS),
    Job("configs/nt_enhancer_types_frozen_doptimal.yaml",     seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(jobs, default_log_dir="logs/nt_transfer")
```

- [ ] **Step 2: Dry-run to verify all 30 jobs listed**

```bash
python run_queue_nt_transfer.py --gpus 0 1 2 3 --dry-run
```

Expected: 30 lines printed, each showing a config + seed combination.

- [ ] **Step 3: Commit**

```bash
git add run_queue_nt_transfer.py
git commit -m "feat: add NT transfer learning queue script (30 jobs: 10 configs x 3 seeds)"
```

---

## Self-Review

**Spec coverage check:**
- [x] NT loaders (H3K4me3, enhancers, enhancers_types) — Task 1
- [x] PretrainConfig.source_dataset_name field — Task 2
- [x] Generalized checkpoint key in active_loop.run() — Task 2
- [x] Active loop branches for nt_enhancers, nt_enhancer_types — Task 3
- [x] A1 configs (5 files, num_labels=2) — Task 4
- [x] A2 configs (5 files, num_labels=3) — Task 5
- [x] run_queue_nt_transfer.py (30 jobs) — Task 6
- [x] frozen configs use lr=1e-3 (matches mm2hs_frozen pattern) — Tasks 4, 5

**Placeholder scan:** None found.

**Type consistency:**
- `load_nt_h3k4me3` returns `tuple[DataPool, TextClassificationDataset, list[str]]` — same signature as all other loaders
- `source_dataset_name: str | None = None` — used as `config.pretrain.source_dataset_name` in active_loop.py
- In NT branches, `source_pool.texts` and `source_pool.labels` assigned to `self.source_texts` / `self.source_labels` — matches how `run()` reads them
