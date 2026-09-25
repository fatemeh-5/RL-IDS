"""B0 baseline trainer (Untitled paper-style softmax DQN)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.buffers import ReplayBuffer
from agent.utils.checkpointing import load_training_checkpoint, save_training_checkpoint
from agent.envs import BaselineIDSEnvironment
from agent.models import build_baseline_dqn


def baseline_replay_update(model, replay_buffer, batch_size, gamma=0.97) -> dict[str, float]:
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    current_outputs = model.predict(states, verbose=0)
    next_outputs = model.predict(next_states, verbose=0)
    max_next_outputs = np.max(next_outputs, axis=1)
    bellman_targets = rewards + (1.0 - dones) * gamma * max_next_outputs
    target_outputs = current_outputs.copy()
    target_outputs[np.arange(batch_size), actions] = bellman_targets
    history = model.fit(
        states, target_outputs, epochs=1, batch_size=batch_size, verbose=0
    )
    return {
        "loss": float(history.history["loss"][0]),
        "mean_reward": float(np.mean(rewards)),
        "mean_target": float(np.mean(bellman_targets)),
        "batch_size": int(batch_size),
    }


def train_baseline(
    env: BaselineIDSEnvironment,
    *,
    episodes: int = 31,
    episode_length: int = 256,
    batch_size: int = 16,
    gamma: float = 0.97,
    replay_capacity: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.996,
    learning_rate: float = 0.001,
    seed: int = 42,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    B0 training loop.

    Epsilon decays after every replay update (paper-style), not once per episode.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    start_episode = 0
    history_rows: list[dict[str, Any]] = []
    model = None

    if checkpoint_dir is not None:
        ckpt = load_training_checkpoint(checkpoint_dir)
        if ckpt is not None:
            model = ckpt["online_model"]
            history_rows = list(ckpt["history_rows"])
            start_episode = int(ckpt["episode"]) + 1
            if verbose:
                print(f"Resuming B0 from episode {start_episode}")
        elif mode == "resume":
            raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")

    if model is None:
        model = build_baseline_dqn(
            n_features=env.n_features, learning_rate=learning_rate
        )

    buffer = ReplayBuffer(capacity=replay_capacity, random_state=seed)
    epsilon = epsilon_start
    # Approximate resume epsilon: decay once per prior episode * episode_length / 2
    if start_episode > 0:
        # Prefer stored epsilon if present
        if history_rows:
            epsilon = float(history_rows[-1].get("Epsilon", epsilon_start))

    overall_start = time.time()
    for episode in range(start_episode, episodes):
        episode_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        exploration = 0
        exploitation = 0
        losses: list[float] = []

        for _ in range(episode_length):
            if rng.random() <= epsilon:
                action = int(rng.integers(0, 2))
                exploration += 1
            else:
                q = model.predict(
                    np.asarray(state, dtype=np.float32).reshape(1, -1), verbose=0
                )[0]
                action = int(np.argmax(q))
                exploitation += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                update = baseline_replay_update(
                    model, buffer, batch_size=batch_size, gamma=gamma
                )
                losses.append(update["loss"])
                if epsilon > epsilon_min:
                    epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if done:
                break

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / episode_length,
            "Episode_Accuracy": correct / episode_length,
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Epsilon": epsilon,
            "Exploration_Actions": exploration,
            "Exploitation_Actions": exploitation,
            "Episode_Time_Seconds": time.time() - episode_start,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | "
                f"Acc: {row['Episode_Accuracy']:.3f} | "
                f"Eps: {epsilon:.4f} | "
                f"t: {row['Episode_Time_Seconds']:.1f}s"
            )

        if (
            checkpoint_dir is not None
            and save_every_episodes > 0
            and ((episode + 1) % save_every_episodes == 0 or episode == episodes - 1)
        ):
            save_training_checkpoint(
                checkpoint_dir,
                episode=episode,
                online_model=model,
                history_rows=history_rows,
                extra={"epsilon": epsilon},
            )

    return {
        "online_model": model,
        "target_model": None,
        "history_df": pd.DataFrame(history_rows),
        "epsilon": epsilon,
        "elapsed_seconds": time.time() - overall_start,
    }
