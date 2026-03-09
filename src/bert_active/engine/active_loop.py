"""Active learning main loop orchestration."""

from __future__ import annotations

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

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the active learning loop.

        Args:
            config: Experiment configuration containing all settings.
        """
        self.config = config

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
        # Initialize pool with initial labeled samples
        self.pool.initialize(
            n_init=self.config.active_learning.n_init,
            seed=self.config.data.seed,
        )

        # Active learning loop
        for round_num in range(self.config.active_learning.n_rounds):
            # Build training dataset from labeled data
            train_dataset = build_dataset(
                tokenizer=self.tokenizer,
                texts=self.pool.get_labeled_texts(),
                labels=self.pool.get_labeled_labels(),
                max_length=self.config.model.max_length,
            )

            # Train model
            train_metrics = self.trainer.train(train_dataset)

            # Evaluate on test set
            eval_metrics = self.trainer.evaluate(self.test_dataset)

            # Log metrics
            n_labeled = self.pool.n_labeled
            self.metrics_tracker.log_round(
                round_num=round_num,
                n_labeled=n_labeled,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

            # Print round summary
            print(
                f"Round {round_num}: "
                f"n_labeled={n_labeled}, "
                f"eval_accuracy={eval_metrics['eval_accuracy']:.4f}, "
                f"eval_f1={eval_metrics['eval_f1_macro']:.4f}"
            )

            # Query strategy for next batch (skip on last round)
            if round_num < self.config.active_learning.n_rounds - 1:
                query_indices = self.strategy.query(
                    n=self.config.active_learning.n_query,
                )
                self.pool.label(query_indices)

        return self.metrics_tracker
