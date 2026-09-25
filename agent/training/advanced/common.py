"""Shared helpers for advanced trainers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.utils.checkpointing import load_training_checkpoint, save_training_checkpoint


def cfg_get(cfg: dict[str, Any] | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE advantages + returns. shapes (T,)."""
    t_steps = len(rewards)
    advantages = np.zeros(t_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_steps)):
        mask = 1.0 - float(dones[t])
        next_v = next_value if t == t_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def soft_update(online, target, tau: float = 0.005) -> None:
    o_w = online.get_weights()
    t_w = target.get_weights()
    target.set_weights([tau * o + (1.0 - tau) * t for o, t in zip(o_w, t_w)])


def hard_update(online, target) -> None:
    target.set_weights(online.get_weights())


def maybe_resume(
    checkpoint_dir: Path | None,
    mode: str,
    verbose: bool,
) -> tuple[int, Any, Any, list[dict[str, Any]], dict[str, Any]]:
    start_episode = 0
    online = None
    target = None
    history_rows: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}
    if checkpoint_dir is None:
        return start_episode, online, target, history_rows, extra
    ckpt = load_training_checkpoint(checkpoint_dir)
    if ckpt is not None:
        online = ckpt["online_model"]
        target = ckpt.get("target_model")
        history_rows = list(ckpt["history_rows"])
        start_episode = int(ckpt["episode"]) + 1
        extra = dict(ckpt.get("state") or {})
        if verbose:
            print(f"Resuming from episode {start_episode}")
    elif mode == "resume":
        raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")
    return start_episode, online, target, history_rows, extra


def checkpoint(
    checkpoint_dir: Path | None,
    episode: int,
    online_model,
    history_rows: list[dict[str, Any]],
    save_every_episodes: int,
    episodes: int,
    target_model=None,
    extra: dict[str, Any] | None = None,
) -> None:
    if checkpoint_dir is None or save_every_episodes <= 0:
        return
    if (episode + 1) % save_every_episodes == 0 or episode == episodes - 1:
        save_training_checkpoint(
            checkpoint_dir,
            episode=episode,
            online_model=online_model,
            target_model=target_model,
            history_rows=history_rows,
            extra=extra,
        )


def finish(
    online_model,
    history_rows: list[dict[str, Any]],
    overall_start: float,
    target_model=None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "online_model": online_model,
        "target_model": target_model,
        "history_df": pd.DataFrame(history_rows),
        "elapsed_seconds": time.time() - overall_start,
        **extra,
    }


def epsilon_action(rng: np.random.Generator, q_row: np.ndarray, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_row)))
    return int(np.argmax(q_row))


def sample_policy_action(rng: np.random.Generator, logits_or_probs: np.ndarray) -> int:
    x = np.asarray(logits_or_probs, dtype=np.float64).reshape(-1)
    if x.min() < 0 or not np.isclose(x.sum(), 1.0, atol=1e-3):
        # logits -> softmax
        x = x - np.max(x)
        e = np.exp(x)
        x = e / e.sum()
    x = np.clip(x, 1e-8, 1.0)
    x = x / x.sum()
    return int(rng.choice(len(x), p=x))
