"""Active learning query strategies."""

from bert_active.strategies.badge import BADGEStrategy
from bert_active.strategies.base import Strategy
from bert_active.strategies.batch_bald import BatchBALDStrategy
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
    "batch_bald": BatchBALDStrategy,
    "coreset": CoreSetStrategy,
}

__all__ = [
    "BADGEStrategy",
    "BALDStrategy",
    "BatchBALDStrategy",
    "CoreSetStrategy",
    "EntropyStrategy",
    "LeastConfidenceStrategy",
    "MarginStrategy",
    "RandomStrategy",
    "STRATEGY_REGISTRY",
    "Strategy",
]
