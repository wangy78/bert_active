"""Active learning main loop orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bert_active.config.experiment import ExperimentConfig
from bert_active.data.dataset import load_ag_news
from bert_active.data.tokenization import (
    build_dataset,
    create_tokenizer,
    tokenize_test_dataset,
)
from bert_active.engine.trainer import Trainer
from bert_active.evaluation.metrics import MetricsTracker
from bert_active.models.classifier import ModelWrapper, create_model
from bert_active.strategies.badge import BADGEStrategy
from bert_active.strategies.base import Strategy
from bert_active.strategies.bayesian import BALDStrategy
from bert_active.strategies.coreset import CoreSetStrategy
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
    "coreset": CoreSetStrategy,
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

        # Load AG News dataset
        self.pool, test_dataset = load_ag_news(
            seed=config.data.seed,
            test_size=config.data.test_size,
        )

        # Tokenize test dataset
        self.test_dataset = tokenize_test_dataset(
            tokenizer=self.tokenizer,
            test_dataset=test_dataset,
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

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "al/round": round_num,
                        "al/n_labeled": n_labeled,
                        "al/train_loss": train_metrics["train_loss"],
                        "al/eval_loss": eval_metrics["eval_loss"],
                        "al/eval_accuracy": eval_metrics["eval_accuracy"],
                        "al/eval_f1_macro": eval_metrics["eval_f1_macro"],
                    },
                )

            print(
                f"Round {round_num}: "
                f"n_labeled={n_labeled}, "
                f"eval_accuracy={eval_metrics['eval_accuracy']:.4f}, "
                f"eval_f1={eval_metrics['eval_f1_macro']:.4f}"
            )

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

        return self.metrics_tracker
