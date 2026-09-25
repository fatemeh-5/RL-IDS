"""Advanced RL algorithms beyond B0–B10 (PPO, Rainbow, C51, …)."""

from __future__ import annotations

from agent.training.advanced.registry import (
    ADVANCED_ORDER,
    ADVANCED_REGISTRY,
    AdvancedSpec,
    get_advanced,
    list_advanced,
)
from agent.training.advanced.dispatch import train_advanced

__all__ = [
    "ADVANCED_ORDER",
    "ADVANCED_REGISTRY",
    "AdvancedSpec",
    "get_advanced",
    "list_advanced",
    "train_advanced",
]
