"""Active learning main loop orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bert_active.config.experiment import ExperimentConfig, TrainerConfig
from bert_active.data.dna_dataset import (
    load_dna_core_promoter,
    load_gue_fungi_species,
    load_gue_human_ensembl_regulatory,
    load_gue_virus_species,
    load_human_nontata_promoters,
)
from bert_active.data.promoter_dataset import load_promoter_species, load_transfer_data
from bert_active.data.tokenization import (
    build_dataset,
    create_tokenizer,
)
from bert_active.engine.trainer import Trainer
from bert_active.evaluation.metrics import MetricsTracker
from bert_active.models.classifier import ModelWrapper, create_model
from bert_active.strategies.badge import BADGEStrategy
from bert_active.strategies.base import Strategy
from bert_active.strategies.batch_bald import BatchBALDStrategy
from bert_active.strategies.bayesian import BALDStrategy
from bert_active.strategies.coreset import CoreSetStrategy
from bert_active.strategies.doptimal import DOptimalStrategy
from bert_active.strategies.random import RandomStrategy
from bert_active.strategies.uncertainty import (
    EntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
)

if TYPE_CHECKING:
    import wandb

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "random": RandomStrategy,
    "least_confidence": LeastConfidenceStrategy,
    "margin": MarginStrategy,
    "entropy": EntropyStrategy,
    "bald": BALDStrategy,
    "badge": BADGEStrategy,
    "batch_bald": BatchBALDStrategy,
    "coreset": CoreSetStrategy,
    "doptimal": DOptimalStrategy,
}


class ActiveLearningLoop:
    """Orchestrates the active learning training loop."""

    def __init__(
        self,
        config: ExperimentConfig,
        wandb_run: wandb.Run | None = None,
    ) -> None:
        self.config = config
        self.wandb_run = wandb_run

        # Create tokenizer
        self.tokenizer = create_tokenizer(config.model.name)

        # Load dataset based on config
        dataset_name = config.data.dataset_name.lower()
        max_samples = getattr(config.data, "max_samples", None)

        # For transfer learning: store source data (None if not applicable)
        self.source_texts: list[str] | None = None
        self.source_labels = None

        if dataset_name == "promoter":
            target_species = config.data.target_species or "mm"
            data_dir = config.data.data_dir or "."
            if config.pretrain.enabled:
                (
                    self.source_texts,
                    self.source_labels,
                    self.pool,
                    test_dataset,
                    test_texts,
                ) = load_transfer_data(
                    source_species=config.pretrain.source_species,
                    target_species=target_species,
                    data_dir=data_dir,
                    seed=config.data.seed,
                )
            else:
                self.pool, test_dataset, test_texts = load_promoter_species(
                    species=target_species,
                    data_dir=data_dir,
                    seed=config.data.seed,
                )
        elif dataset_name == "dna_core_promoter":
            self.pool, test_dataset, test_texts = load_dna_core_promoter(
                seed=config.data.seed,
                max_samples=max_samples,
            )
        elif dataset_name == "human_nontata_promoters":
            self.pool, test_dataset, test_texts = load_human_nontata_promoters(
                seed=config.data.seed,
                max_samples=max_samples,
            )
        elif dataset_name == "gue_fungi_species":
            self.pool, test_dataset, test_texts = load_gue_fungi_species(
                seed=config.data.seed,
                max_samples=max_samples,
            )
        elif dataset_name == "gue_virus_species":
            self.pool, test_dataset, test_texts = load_gue_virus_species(
                seed=config.data.seed,
                max_samples=max_samples,
            )
        elif dataset_name == "gue_human_ensembl_regulatory":
            self.pool, test_dataset, test_texts = load_gue_human_ensembl_regulatory(
                seed=config.data.seed,
                max_samples=max_samples,
            )
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: promoter, dna_core_promoter, human_nontata_promoters, "
                f"gue_fungi_species, gue_virus_species, gue_human_ensembl_regulatory"
            )

        # Tokenize test set
        self.test_dataset = build_dataset(
            tokenizer=self.tokenizer,
            texts=test_texts,
            labels=test_dataset.labels,
            max_length=config.model.max_length,
        )

        # Create model
        model = create_model(
            model_name=config.model.name,
            num_labels=config.model.num_labels,
            attn_implementation=config.model.attn_implementation,
            torch_dtype=config.model.torch_dtype,
        )

        # Wrap model
        self.model_wrapper = ModelWrapper(
            model=model,
            device=config.trainer.device,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model_wrapper,
            config=config.trainer,
            wandb_run=wandb_run,
        )

        # Look up and instantiate strategy
        strategy_class = STRATEGY_REGISTRY[config.active_learning.strategy]
        strategy_kwargs = {
            "tokenizer": self.tokenizer,
            "max_length": config.model.max_length,
            **(config.active_learning.strategy_params or {}),
        }
        self.strategy = strategy_class(
            pool=self.pool,
            model=self.model_wrapper,
            **strategy_kwargs,
        )

        # Create metrics tracker
        self.metrics_tracker = MetricsTracker(
            experiment_name=config.experiment_name,
        )

    def run(self) -> MetricsTracker:
        """Run the active learning loop.

        Returns:
            MetricsTracker containing all metrics from the experiment.
        """
        # Source domain pretraining (transfer learning)
        if self.config.pretrain.enabled and self.source_texts is not None:
            print("=== Source domain pretraining ===")
            source_dataset = build_dataset(
                tokenizer=self.tokenizer,
                texts=self.source_texts,
                labels=self.source_labels,
                max_length=self.config.model.max_length,
            )
            pretrain_config = TrainerConfig(
                learning_rate=self.config.pretrain.learning_rate,
                batch_size=self.config.pretrain.batch_size,
                num_epochs=self.config.pretrain.num_epochs,
                weight_decay=self.config.trainer.weight_decay,
                warmup_ratio=self.config.trainer.warmup_ratio,
                max_grad_norm=self.config.trainer.max_grad_norm,
                eval_batch_size=self.config.trainer.eval_batch_size,
                device=self.config.trainer.device,
            )
            pretrain_trainer = Trainer(
                model=self.model_wrapper,
                config=pretrain_config,
                wandb_run=self.wandb_run,
            )
            pretrain_trainer.train(source_dataset)
            print("=== Source domain pretraining complete ===")

        if self.config.active_learning.init_strategy == "doptimal":
            self._initialize_doptimal(self.config.active_learning.n_init)
        else:
            self.pool.initialize(
                n_init=self.config.active_learning.n_init,
                seed=self.config.data.seed,
            )

        for round_num in range(self.config.active_learning.n_rounds):
            train_dataset = build_dataset(
                tokenizer=self.tokenizer,
                texts=self.pool.get_labeled_texts(),
                labels=self.pool.get_labeled_labels(),
                max_length=self.config.model.max_length,
            )

            train_metrics = self.trainer.train(train_dataset)
            eval_metrics = self.trainer.evaluate(self.test_dataset)

            n_labeled = self.pool.n_labeled
            self.metrics_tracker.log_round(
                round_num=round_num,
                n_labeled=n_labeled,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

            query_indices = None
            query_label_counts = {}
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

                from collections import Counter
                query_label_counts = Counter(query_labels.tolist())

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "al/round": round_num,
                        "al/n_labeled": n_labeled,
                        "al/train_loss": train_metrics["train_loss"],
                        "al/eval_loss": eval_metrics["eval_loss"],
                        "al/eval_accuracy": eval_metrics["eval_accuracy"],
                        "al/eval_f1_macro": eval_metrics["eval_f1_macro"],
                        **{f"al/query_label_{k}": v for k, v in query_label_counts.items()},
                    },
                )

            print(
                f"Round {round_num}: "
                f"n_labeled={n_labeled}, "
                f"eval_accuracy={eval_metrics['eval_accuracy']:.4f}, "
                f"eval_f1={eval_metrics['eval_f1_macro']:.4f}"
            )

        return self.metrics_tracker

    def _initialize_doptimal(self, n_init: int) -> None:
        """Select initial labeled set using D-Optimal design on pretrained embeddings.

        Uses the pretrained (untrained) model embeddings to greedily maximise
        det(X^T X) for the initial pool, giving a more informative cold start
        than random balanced sampling.
        """
        from bert_active.data.tokenization import build_dataset
        from bert_active.strategies.doptimal import DOptimalStrategy

        print(f"=== D-Optimal cold-start initialisation (n_init={n_init}) ===")

        all_texts = self.pool.texts
        dataset = build_dataset(
            self.tokenizer,
            all_texts,
            labels=None,
            max_length=self.config.model.max_length,
        )
        embeddings = self.model_wrapper.get_embeddings(dataset).astype(float)

        d = embeddings.shape[1]
        ridge = 1e-4
        XtX_inv = (1.0 / ridge) * np.eye(d)

        all_indices = np.arange(len(all_texts), dtype=np.intp)
        remaining = all_indices.copy()
        selected: list[int] = []

        for _ in range(min(n_init, len(all_texts))):
            U = embeddings[remaining]
            scores = np.einsum("md,dd,md->m", U, XtX_inv, U)
            best_local = int(np.argmax(scores))
            best_idx = int(remaining[best_local])
            selected.append(best_idx)

            x = embeddings[best_idx]
            Ainv_x = XtX_inv @ x
            denom = 1.0 + float(x @ Ainv_x)
            XtX_inv -= np.outer(Ainv_x, Ainv_x) / denom

            remaining = np.delete(remaining, best_local)

        init_indices = np.array(selected, dtype=np.intp)
        self.pool.label(init_indices)
        print(f"=== D-Optimal init complete: {len(init_indices)} samples labeled ===")
