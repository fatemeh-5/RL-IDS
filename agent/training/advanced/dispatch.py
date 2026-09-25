"""Dispatch advanced trainers."""

from __future__ import annotations

from typing import Any, Callable

from agent.training.advanced.registry import get_advanced


def _load_trainers() -> dict[str, Callable[..., dict[str, Any]]]:
    from agent.training.advanced.ppo import train_ppo
    from agent.training.advanced.policy_grad import train_a2c, train_reinforce
    from agent.training.advanced.value_based import (
        train_apex,
        train_c51,
        train_m_dqn,
        train_qr_dqn,
        train_r2d2,
        train_rainbow,
    )
    from agent.training.advanced.actor_critic_extra import (
        train_acer,
        train_acktr,
        train_impala,
        train_trpo,
    )
    from agent.training.advanced.continuous_adapt import train_ddpg, train_sac, train_td3

    return {
        "ppo": train_ppo,
        "c51": train_c51,
        "qr_dqn": train_qr_dqn,
        "rainbow": train_rainbow,
        "m_dqn": train_m_dqn,
        "a2c": train_a2c,
        "reinforce": train_reinforce,
        "trpo": train_trpo,
        "r2d2": train_r2d2,
        "apex": train_apex,
        "impala": train_impala,
        "acer": train_acer,
        "acktr": train_acktr,
        "sac": train_sac,
        "td3": train_td3,
        "ddpg": train_ddpg,
    }


TRAINERS: dict[str, Callable[..., dict[str, Any]]] = {}


def _ensure_trainers() -> dict[str, Callable[..., dict[str, Any]]]:
    global TRAINERS
    if not TRAINERS:
        TRAINERS = _load_trainers()
    return TRAINERS


def train_advanced(trainer: str, env: Any, **kwargs: Any) -> dict[str, Any]:
    key = str(trainer).strip().lower()
    trainers = _ensure_trainers()
    if key not in trainers:
        known = ", ".join(sorted(trainers))
        raise KeyError(f"Unknown advanced trainer '{trainer}'. Known: {known}")
    return trainers[key](env, **kwargs)


def is_advanced_experiment(experiment_id: str) -> bool:
    try:
        get_advanced(experiment_id)
        return True
    except KeyError:
        return False


# Eager populate for `trainer in TRAINERS` checks in runners
TRAINERS.update(_load_trainers())
