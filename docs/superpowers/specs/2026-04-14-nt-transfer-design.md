# NT Downstream Tasks Cross-Task Transfer Learning Design

**Date:** 2026-04-14
**Status:** Approved

## Problem

Current mm→hs transfer learning experiment shows poor transfer signal because mouse and human promoter sequences are highly conserved. The model learns the target task well from scratch, making the transfer advantage unmeasurable.

## Goal

Demonstrate meaningful cross-task transfer learning using datasets from `InstaDeepAI/nucleotide_transformer_downstream_tasks`, where source and target share chromatin regulatory features but are distinct enough that limited labeled target data benefits from pretraining.

## Dataset Pair

- **Source (pretraining):** `H3K4me3` — histone mark at active transcription start sites, binary classification, ~30k human sequences, 200bp
- **Target A1:** `enhancers` — enhancer vs. background, binary, ~14k sequences
- **Target A2:** `enhancers_types` — strong enhancer / weak enhancer / background, 3-class, ~14k sequences

Biological rationale: H3K4me3 marks active chromatin near promoters; enhancers share open chromatin features but are mechanistically distinct. Transfer is non-trivial but biologically grounded.

## Implementation

### 1. New Dataset Loaders (`src/bert_active/data/dna_dataset.py`)

Add three functions:
- `load_nt_h3k4me3(seed, max_samples)` — source pretraining data
- `load_nt_enhancers(seed, max_samples)` — A1 target
- `load_nt_enhancers_types(seed, max_samples)` — A2 target

All load from `InstaDeepAI/nucleotide_transformer_downstream_tasks` with the respective config name, using the dataset's existing train/test split.

### 2. Pretrain Config Extension (`src/bert_active/config/experiment.py`)

Add `source_dataset_name: str | None = None` to `PretrainConfig`. Backward compatible — existing species-based logic unchanged.

### 3. Active Loop Extension (`src/bert_active/engine/active_loop.py`)

Add elif branches for `nt_enhancers` and `nt_enhancer_types` dataset names. When `pretrain.enabled`, load H3K4me3 as source and run the existing pretrain→AL pipeline. Register new dataset names in the error message.

### 4. Config Files (10 total)

**A1 — enhancers (num_labels=2):**
| File | Pretrain | Freeze | Strategy |
|------|----------|--------|----------|
| `nt_enhancer_baseline_random.yaml` | No | No | random |
| `nt_enhancer_transfer_random.yaml` | H3K4me3 | No | random |
| `nt_enhancer_transfer_badge.yaml` | H3K4me3 | No | badge |
| `nt_enhancer_frozen_badge.yaml` | H3K4me3 | Yes | badge |
| `nt_enhancer_frozen_doptimal.yaml` | H3K4me3 | Yes | doptimal |

**A2 — enhancers_types (num_labels=3):**
Same 5 conditions with `num_labels=3`.

### 5. Queue Script (`run_queue_nt_transfer.py`)

10 configs × 3 seeds (42, 43, 44) = **30 jobs**

### Active Learning Parameters

Consistent with mm2hs experiments:
- `n_init: 100`, `n_query: 100`, `n_rounds: 20` (total budget: 2100 labeled samples)
- `pretrain.num_epochs: 10`, `learning_rate: 2e-5`
- `trainer.num_epochs: 5` per AL round

## Scientific Questions

1. Does pretraining on H3K4me3 improve AL efficiency on enhancer detection vs. from-scratch baseline?
2. Does freezing the backbone (linear probing) preserve or destroy the transfer benefit?
3. Does D-Optimal acquisition outperform BADGE when the embedding space is fixed (frozen backbone)?
4. Is the transfer benefit larger for the harder 3-class target (A2) than binary (A1)?

## WandB Tags

- A1 runs: `["nt_downstream", "transfer", "h3k4me3_to_enhancer"]`
- A2 runs: `["nt_downstream", "transfer", "h3k4me3_to_enhancer_types"]`
