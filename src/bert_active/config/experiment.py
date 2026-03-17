"""YAML experiment configuration schema and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 4
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    max_length: int = 128


@dataclass
class DataConfig:
    dataset_name: str = "ag_news"
    seed: int = 42
    test_size: int | None = None  # None = use full test set (AG News only)
    data_dir: str | None = None  # Directory for EPD dataset
    max_samples: int | None = None  # Max samples to load (EPD only)


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "cuda"


@dataclass
class ActiveLearningConfig:
    n_init: int = 100
    n_query: int = 100
    n_rounds: int = 20
    strategy: str = "entropy"
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "bert-active"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "outputs"
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from YAML file, merging with defaults."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        config = cls()
        if "model" in raw:
            config.model = _merge_dataclass(ModelConfig, raw["model"])
        if "data" in raw:
            config.data = _merge_dataclass(DataConfig, raw["data"])
        if "trainer" in raw:
            config.trainer = _merge_dataclass(TrainerConfig, raw["trainer"])
        if "active_learning" in raw:
            config.active_learning = _merge_dataclass(ActiveLearningConfig, raw["active_learning"])
        if "wandb" in raw:
            config.wandb = _merge_dataclass(WandbConfig, raw["wandb"])
        if "output_dir" in raw:
            config.output_dir = raw["output_dir"]
        if "experiment_name" in raw:
            config.experiment_name = raw["experiment_name"]
        return config


def _merge_dataclass(cls: type[Any], overrides: dict[str, Any]) -> Any:
    """Create dataclass instance from dict, ignoring unknown keys and converting types."""
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    converted = {}

    for key, value in overrides.items():
        if key not in fields:
            continue  # Ignore unknown keys

        field_type = fields[key].type
        # Handle string type annotations (from __future__ import annotations)
        type_name = field_type if isinstance(field_type, str) else field_type.__name__

        # Convert string values to the correct type if needed
        if isinstance(value, str) and type_name != "str":
            try:
                if type_name == "float":
                    converted[key] = float(value)
                elif type_name == "int":
                    converted[key] = int(value)
                else:
                    converted[key] = value
            except (ValueError, TypeError):
                converted[key] = value
        else:
            converted[key] = value

    return cls(**converted)
