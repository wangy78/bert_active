"""DNA sequence datasets for active learning with DNA-BERT.

Supports:
- dnagpt/dna_core_promoter (59k samples, 70bp sequences)
- katarinagresova/Genomic_Benchmarks_human_nontata_promoters (36k samples, 251bp sequences)
- leannmlindsey/GUE fungi_species_20 (10k samples, 20-class fungal species classification)
- leannmlindsey/GUE virus_species_40 (5k samples, 40-class viral species classification)
"""

from __future__ import annotations

import numpy as np
from datasets import DatasetDict, load_dataset
from numpy.typing import NDArray

from bert_active.data.dataset import DataPool, TextClassificationDataset


def load_dna_core_promoter(
    seed: int = 42,
    test_split: float = 0.2,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load dnagpt/dna_core_promoter dataset.

    Binary classification: core promoter (1) vs non-core promoter (0).
    59,196 samples, 70bp sequences. Only has train split, so we manually split.

    Args:
        seed: Random seed for train/test split.
        test_split: Fraction for testing (default 0.2).
        max_samples: Limit total samples (optional).

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    ds = load_dataset("dnagpt/dna_core_promoter", split="train")

    sequences: list[str] = list(ds["sequence"])
    labels = np.array(ds["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sequences), size=min(max_samples, len(sequences)), replace=False)
        sequences = [sequences[i] for i in idx]
        labels = labels[idx]

    return _split_to_pool(sequences, labels, seed, test_split)


def load_human_nontata_promoters(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load katarinagresova/Genomic_Benchmarks_human_nontata_promoters dataset.

    Binary classification: promoter (1) vs non-promoter (0).
    27,100 train + 9,031 test samples, 251bp sequences.
    Already has train/test splits.

    Args:
        seed: Random seed (used only if max_samples is set).
        max_samples: Limit training samples (optional).

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    ds: DatasetDict = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")  # type: ignore[assignment]

    # Train split -> DataPool
    train_texts: list[str] = list(ds["train"]["seq"])
    train_labels = np.array(ds["train"]["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=min(max_samples, len(train_texts)), replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    pool = DataPool(texts=train_texts, labels=train_labels)

    # Test split
    test_texts: list[str] = list(ds["test"]["seq"])
    test_labels = np.array(ds["test"]["label"], dtype=np.intp)

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def load_gue_fungi_species(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load GUE fungi_species_20 dataset.

    20-class fungal species classification. ~10,000 samples, ~500bp sequences.
    Has train/test splits.

    Args:
        seed: Random seed (used if max_samples is set).
        max_samples: Limit training samples (optional).

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    ds: DatasetDict = load_dataset("leannmlindsey/GUE", "fungi_species_20")  # type: ignore[assignment]

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


def load_gue_virus_species(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load GUE virus_species_40 dataset.

    40-class viral species classification. ~5,000 samples.
    Has train/test splits.

    Args:
        seed: Random seed (used if max_samples is set).
        max_samples: Limit training samples (optional).

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    ds: DatasetDict = load_dataset("leannmlindsey/GUE", "virus_species_40")  # type: ignore[assignment]

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


def load_gue_virus_covid(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load GUE virus_covid dataset.

    9-class viral variant classification. ~73,000 train samples.
    Has train/test splits.

    Args:
        seed: Random seed (used if max_samples is set).
        max_samples: Limit training samples (optional).

    Returns:
        (DataPool, test_dataset_placeholder, test_texts)
    """
    ds: DatasetDict = load_dataset("leannmlindsey/GUE", "virus_covid")  # type: ignore[assignment]

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



def _load_nt_task(
    task_name: str,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Helper: load a single task from NT downstream tasks dataset by filtering on 'task' column.

    NOTE: The named-config API (load_dataset(..., task_name)) does NOT work for this dataset.
    The dataset only exposes a single 'default' config, so all 18 tasks are packed into one
    split and differentiated by a 'task' column.  Attempting to pass the task name as a
    config name raises:
        ValueError: BuilderConfig '<task>' not found. Available: ['default']
    Therefore the .filter() approach below is intentional and necessary.
    """
    ds: DatasetDict = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks"
    )  # type: ignore[assignment]

    train_split = ds["train"].filter(lambda x: x["task"] == task_name)
    test_split = ds["test"].filter(lambda x: x["task"] == task_name)

    train_texts: list[str] = list(train_split["sequence"])
    train_labels = np.array(train_split["label"], dtype=np.intp)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=min(max_samples, len(train_texts)), replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    pool = DataPool(texts=train_texts, labels=train_labels)

    test_texts: list[str] = list(test_split["sequence"])
    test_labels = np.array(test_split["label"], dtype=np.intp)

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts


def load_nt_h3k4me3(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load H3K4me3 histone mark dataset from NT downstream tasks.

    Binary classification: H3K4me3 mark present (1) vs absent (0).
    ~30,000 human sequences, 200bp. Used as source for cross-task transfer.
    Has train/test splits.
    """
    return _load_nt_task("H3K4me3", seed=seed, max_samples=max_samples)


def load_nt_enhancers(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load enhancers dataset from NT downstream tasks.

    Binary classification: enhancer (1) vs background (0).
    ~14,000 human sequences, 200bp. Target for A1 cross-task transfer.
    Has train/test splits.
    """
    return _load_nt_task("enhancers", seed=seed, max_samples=max_samples)


def load_nt_enhancers_types(
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Load enhancers_types dataset from NT downstream tasks.

    3-class classification: strong enhancer (0) / weak enhancer (1) / background (2).
    ~14,000 human sequences, 200bp. Target for A2 cross-task transfer.
    Has train/test splits.
    """
    return _load_nt_task("enhancers_types", seed=seed, max_samples=max_samples)


def _split_to_pool(
    sequences: list[str],
    labels: NDArray[np.intp],
    seed: int,
    test_split: float,
) -> tuple[DataPool, TextClassificationDataset, list[str]]:
    """Split sequences into DataPool (train) and test set."""
    rng = np.random.default_rng(seed)
    n = len(sequences)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_split)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_texts = [sequences[i] for i in train_idx]
    train_labels = labels[train_idx]
    pool = DataPool(texts=train_texts, labels=train_labels)

    test_texts = [sequences[i] for i in test_idx]
    test_labels = labels[test_idx]

    test_dataset = TextClassificationDataset(
        input_ids=np.zeros((len(test_texts), 1), dtype=np.intp),
        attention_mask=np.zeros((len(test_texts), 1), dtype=np.intp),
        labels=test_labels,
    )
    test_dataset._raw_texts = test_texts  # type: ignore[attr-defined]

    return pool, test_dataset, test_texts
