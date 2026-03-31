"""
CLI entry point for BERT Active Learning experiments.

This module provides the main command-line interface for running active learning
experiments with configurable parameters. It handles argument parsing, config loading,
experiment execution, and results visualization.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import wandb
from bert_active.config.experiment import ExperimentConfig
from bert_active.engine.active_loop import ActiveLearningLoop

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main CLI entry point for BERT active learning experiments.

    Parses command-line arguments, loads configuration, applies overrides,
    runs the active learning loop, and saves/visualizes results.
    """
    parser = argparse.ArgumentParser(
        description="Run BERT-based active learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override active learning strategy from config",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Override number of active learning rounds",
    )
    parser.add_argument(
        "--n-query",
        type=int,
        default=None,
        help="Override number of samples to query per round",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Override size of initial labeled set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run experiment with multiple seeds, e.g. --seeds 42 43 44",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Enable wandb logging (overrides config)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=None,
        help="Disable wandb logging (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration from YAML
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Determine seeds to run
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [None]  # use seed from config

    logger.info(f"Running with seeds: {seeds}")

    for seed in seeds:
        logger.info(f"Loading configuration from {config_path}")
        config = ExperimentConfig.from_yaml(str(config_path))

        # Apply CLI overrides
        if args.strategy is not None:
            config.active_learning.strategy = args.strategy
            logger.info(f"Overriding strategy: {args.strategy}")

        if args.n_rounds is not None:
            config.active_learning.n_rounds = args.n_rounds
            logger.info(f"Overriding n_rounds: {args.n_rounds}")

        if args.n_query is not None:
            config.active_learning.n_query = args.n_query
            logger.info(f"Overriding n_query: {args.n_query}")

        if args.n_init is not None:
            config.active_learning.n_init = args.n_init
            logger.info(f"Overriding n_init: {args.n_init}")

        if args.output_dir is not None:
            config.output_dir = args.output_dir
            logger.info(f"Overriding output_dir: {args.output_dir}")

        if seed is not None:
            config.data.seed = seed
            logger.info(f"Overriding seed: {seed}")

        # Update experiment name to reflect CLI overrides
        if args.strategy is not None or args.n_rounds is not None or args.n_query is not None:
            al = config.active_learning
            config.experiment_name = f"{config.experiment_name}_q{al.n_query}_r{al.n_rounds}"
            logger.info(f"Updated experiment_name: {config.experiment_name}")

        # Append seed to experiment name and tags when seed is explicitly set via CLI
        if args.seeds is not None or args.seed is not None:
            config.experiment_name = f"{config.experiment_name}_seed{config.data.seed}"
            if config.wandb.tags is not None and f"seed{config.data.seed}" not in config.wandb.tags:
                config.wandb.tags = list(config.wandb.tags) + [f"seed{config.data.seed}"]

        if args.wandb:
            config.wandb.enabled = True
        elif args.no_wandb:
            config.wandb.enabled = False

        _print_config_summary(config)
        _run_single(config)


def _run_single(config: ExperimentConfig) -> None:
    wandb_run: wandb.Run | None = None
    if config.wandb.enabled:
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.experiment_name,
            config=_config_to_dict(config),
            tags=config.wandb.tags,
            notes=config.wandb.notes,
        )

    try:
        logger.info("Starting active learning experiment...")
        loop = ActiveLearningLoop(config, wandb_run=wandb_run)
        metrics = loop.run()

        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving metrics to {output_path}")
        metrics.save(str(output_path))

        logger.info("Plotting learning curves...")
        metrics.plot_learning_curves(str(output_path))

        logger.info("Plotting label distribution...")
        metrics.plot_label_distribution(str(output_path), num_classes=config.model.num_labels)

        summary = metrics.get_summary()
        logger.info("Active learning experiment completed successfully!")
        logger.info(f"Results summary:\n{summary}")

        if wandb_run is not None:
            wandb_run.summary.update(summary)

        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"Output directory: {output_path}")
        print(f"Results summary:\n{summary}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _print_config_summary(config: ExperimentConfig) -> None:
    print(f"\n{'=' * 60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Strategy: {config.active_learning.strategy}")
    print(f"Number of Rounds: {config.active_learning.n_rounds}")
    print(f"Samples per Query: {config.active_learning.n_query}")
    print(f"Initial Labeled Set: {config.active_learning.n_init}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Random Seed: {config.data.seed}")
    print(f"Wandb: {'enabled' if config.wandb.enabled else 'disabled'}")
    print(f"{'=' * 60}\n")


def _config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    return {
        "model": dataclasses.asdict(config.model),
        "data": dataclasses.asdict(config.data),
        "trainer": dataclasses.asdict(config.trainer),
        "active_learning": dataclasses.asdict(config.active_learning),
        "output_dir": config.output_dir,
        "experiment_name": config.experiment_name,
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
