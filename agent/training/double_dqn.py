"""Shared Double-DQN training loop with mid-run checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.buffers import ReplayBuffer
from agent.utils.checkpointing import load_training_checkpoint, save_training_checkpoint
from agent.models import clone_compiled_model, synchronize_target_network


def double_dqn_replay_update(
    online_model,
    target_model,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float = 0.97,
) -> dict[str, float]:
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    current_q_values = online_model.predict(states, verbose=0)
    next_online_q_values = online_model.predict(next_states, verbose=0)
    next_target_q_values = target_model.predict(next_states, verbose=0)

    best_next_actions = np.argmax(next_online_q_values, axis=1)
    selected_next_q_values = next_target_q_values[
        np.arange(batch_size),
        best_next_actions,
    ]
    bellman_targets = rewards + (1.0 - dones) * gamma * selected_next_q_values

    target_q_values = current_q_values.copy()
    target_q_values[np.arange(batch_size), actions] = bellman_targets

    history = online_model.fit(
        states,
        target_q_values,
        epochs=1,
        batch_size=batch_size,
        verbose=0,
    )
    loss = float(history.history["loss"][0])
    td_errors = bellman_targets - current_q_values[np.arange(batch_size), actions]

    return {
        "loss": loss,
        "mean_reward": float(np.mean(rewards)),
        "mean_bellman_target": float(np.mean(bellman_targets)),
        "mean_absolute_td_error": float(np.mean(np.abs(td_errors))),
        "max_absolute_td_error": float(np.max(np.abs(td_errors))),
        "terminal_samples": int(dones.sum()),
        "batch_size": int(batch_size),
    }


def train_loop(
    env,
    *,
    episodes: int = 31,
    episode_length: int | None = None,
    batch_size: int = 16,
    gamma: float = 0.97,
    replay_capacity: int = 1000,
    target_update_interval: int = 250,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.90,
    epsilon_schedule: str = "episode",
    learning_rate: float = 0.001,
    seed: int = 42,
    online_model=None,
    target_model=None,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train Double DQN on any IDS env that implements reset/step.

    mode:
      - train: start fresh (or continue if checkpoint exists and mode=resume)
      - resume: require checkpoint and continue
    """

    episode_length = episode_length or getattr(env, "episode_length", 256)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    random = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    start_episode = 0
    history_rows: list[dict[str, Any]] = []

    if mode in {"train", "resume"} and checkpoint_dir is not None:
        ckpt = load_training_checkpoint(checkpoint_dir)
        if ckpt is not None:
            online_model = ckpt["online_model"]
            target_model = ckpt["target_model"]
            history_rows = list(ckpt["history_rows"])
            start_episode = int(ckpt["episode"]) + 1
            if verbose:
                print(f"Resuming from episode {start_episode} ({checkpoint_dir})")
        elif mode == "resume":
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")

    if online_model is None:
        from agent.models import build_double_dqn

        online_model = build_double_dqn(
            n_features=env.n_features,
            learning_rate=learning_rate,
        )
    if target_model is None:
        target_model = clone_compiled_model(online_model, learning_rate=learning_rate)

    buffer = ReplayBuffer(capacity=replay_capacity, random_state=seed)
    epsilon = epsilon_start
    # If resuming, decay epsilon for completed episodes
    if start_episode > 0 and epsilon_schedule == "episode":
        for _ in range(start_episode):
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    gradient_update_count = 0
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        episode_start = time.time()
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0.0
        correct = 0
        losses: list[float] = []

        for _step in range(episode_length):
            if random.random() <= epsilon:
                action = int(random.integers(0, 2))
            else:
                q_values = online_model(
                    np.asarray(state, dtype=np.float32).reshape(1, -1),
                    training=False,
                ).numpy()[0]
                action = int(np.argmax(q_values))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)

            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                update = double_dqn_replay_update(
                    online_model=online_model,
                    target_model=target_model,
                    replay_buffer=buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                )
                losses.append(update["loss"])
                gradient_update_count += 1
                if gradient_update_count % target_update_interval == 0:
                    synchronize_target_network(online_model, target_model)
                # B1-style: decay epsilon after every replay update
                if epsilon_schedule in {"step", "replay", "per_update"}:
                    if epsilon > epsilon_min:
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if done:
                break

        # B2/B3-style: decay once per episode
        if epsilon_schedule == "episode":
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        row = {
            "episode": episode,
            "total_reward": total_reward,
            "accuracy": correct / episode_length,
            "epsilon": epsilon,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "seconds": time.time() - episode_start,
            "gradient_updates": gradient_update_count,
        }
        history_rows.append(row)

        if verbose:
            print(
                f"Episode {episode}/{episodes - 1} | "
                f"reward={total_reward:.1f} | "
                f"acc={row['accuracy']:.3f} | "
                f"eps={epsilon:.3f} | "
                f"t={row['seconds']:.1f}s"
            )

        if (
            checkpoint_dir is not None
            and save_every_episodes > 0
            and ((episode + 1) % save_every_episodes == 0 or episode == episodes - 1)
        ):
            save_training_checkpoint(
                checkpoint_dir,
                episode=episode,
                online_model=online_model,
                target_model=target_model,
                history_rows=history_rows,
                extra={"epsilon": epsilon, "gradient_updates": gradient_update_count},
            )
            if verbose:
                print(f"  checkpoint saved -> {checkpoint_dir}")

    return {
        "online_model": online_model,
        "target_model": target_model,
        "history_df": pd.DataFrame(history_rows),
        "epsilon": epsilon,
        "gradient_updates": gradient_update_count,
        "elapsed_seconds": time.time() - overall_start,
    }
