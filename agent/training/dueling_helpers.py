"""B6 dueling Double DQN + PER trainer (Untitled Tasks 75–78)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.utils.checkpointing import load_training_checkpoint, save_training_checkpoint
from agent.envs import BaselineIDSEnvironment
from agent.models import build_dueling_dqn, synchronize_target_network
from agent.buffers.prioritized import PrioritizedReplayBuffer, calculate_per_beta


def prioritized_double_dqn_update(
    online_model,
    target_model,
    replay_buffer: PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float = 0.97,
    beta: float = 0.4,
) -> dict[str, float]:
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        indices,
        importance_weights,
    ) = replay_buffer.sample(batch_size=batch_size, beta=beta)

    current_q_values = online_model.predict(states, verbose=0)
    next_online_q_values = online_model.predict(next_states, verbose=0)
    next_target_q_values = target_model.predict(next_states, verbose=0)

    best_next_actions = np.argmax(next_online_q_values, axis=1)
    selected_next_q_values = next_target_q_values[
        np.arange(batch_size), best_next_actions
    ]
    bellman_targets = rewards + (1.0 - dones) * gamma * selected_next_q_values
    selected_current_q_values = current_q_values[np.arange(batch_size), actions]
    td_errors = bellman_targets - selected_current_q_values

    target_q_values = current_q_values.copy()
    target_q_values[np.arange(batch_size), actions] = bellman_targets

    history = online_model.fit(
        states,
        target_q_values,
        sample_weight=importance_weights,
        epochs=1,
        batch_size=batch_size,
        verbose=0,
    )
    replay_buffer.update_priorities(indices=indices, td_errors=td_errors)

    return {
        "loss": float(history.history["loss"][0]),
        "mean_reward": float(np.mean(rewards)),
        "mean_bellman_target": float(np.mean(bellman_targets)),
        "mean_absolute_td_error": float(np.mean(np.abs(td_errors))),
        "mean_importance_weight": float(np.mean(importance_weights)),
        "batch_size": int(batch_size),
    }


def train_b6(
    env: BaselineIDSEnvironment,
    *,
    episodes: int = 31,
    episode_length: int = 256,
    batch_size: int = 16,
    gamma: float = 0.97,
    replay_capacity: int = 1000,
    per_alpha: float = 0.60,
    beta_start: float = 0.40,
    beta_end: float = 1.00,
    priority_epsilon: float = 1e-6,
    target_update_interval: int = 250,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.90,
    learning_rate: float = 0.001,
    seed: int = 42,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    start_episode = 0
    history_rows: list[dict[str, Any]] = []
    online_model = None
    target_model = None

    if checkpoint_dir is not None:
        ckpt = load_training_checkpoint(checkpoint_dir)
        if ckpt is not None:
            online_model = ckpt["online_model"]
            target_model = ckpt["target_model"]
            history_rows = list(ckpt["history_rows"])
            start_episode = int(ckpt["episode"]) + 1
            if verbose:
                print(f"Resuming B6 from episode {start_episode}")
        elif mode == "resume":
            raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")

    if online_model is None:
        online_model = build_dueling_dqn(
            input_features=env.n_features,
            learning_rate=learning_rate,
            model_name="B6_Dueling_Online_Network",
        )
        target_model = build_dueling_dqn(
            input_features=env.n_features,
            learning_rate=learning_rate,
            model_name="B6_Dueling_Target_Network",
        )
        synchronize_target_network(online_model, target_model)

    buffer = PrioritizedReplayBuffer(
        capacity=replay_capacity,
        alpha=per_alpha,
        priority_epsilon=priority_epsilon,
        random_state=seed,
    )
    epsilon = epsilon_start
    if history_rows:
        epsilon = float(history_rows[-1].get("Epsilon_End", epsilon_start))

    gradient_update_count = 0
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        episode_start = time.time()
        beta = calculate_per_beta(episode, episodes, beta_start, beta_end)
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        exploration = 0
        exploitation = 0
        losses: list[float] = []
        td_errors: list[float] = []
        epsilon_at_start = epsilon

        for _ in range(episode_length):
            if rng.random() <= epsilon:
                action = int(rng.integers(0, 2))
                exploration += 1
            else:
                q = online_model(
                    np.asarray(state, dtype=np.float32).reshape(1, -1),
                    training=False,
                ).numpy()[0]
                action = int(np.argmax(q))
                exploitation += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += float(reward)
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                update = prioritized_double_dqn_update(
                    online_model=online_model,
                    target_model=target_model,
                    replay_buffer=buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    beta=beta,
                )
                losses.append(update["loss"])
                td_errors.append(update["mean_absolute_td_error"])
                gradient_update_count += 1
                if gradient_update_count % target_update_interval == 0:
                    synchronize_target_network(online_model, target_model)

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / episode_length,
            "Episode_Accuracy": correct / episode_length,
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Mean_Absolute_TD_Error": float(np.mean(td_errors)) if td_errors else np.nan,
            "Epsilon_Start": epsilon_at_start,
            "Epsilon_End": epsilon,
            "Beta": beta,
            "Exploration_Actions": exploration,
            "Exploitation_Actions": exploitation,
            "Episode_Time_Seconds": time.time() - episode_start,
            "Gradient_Updates": gradient_update_count,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | "
                f"Acc: {row['Episode_Accuracy']:.3f} | "
                f"Eps: {epsilon_at_start:.4f}->{epsilon:.4f} | "
                f"Beta: {beta:.3f} | "
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
                online_model=online_model,
                target_model=target_model,
                history_rows=history_rows,
                extra={"epsilon": epsilon, "gradient_updates": gradient_update_count},
            )

    return {
        "online_model": online_model,
        "target_model": target_model,
        "history_df": pd.DataFrame(history_rows),
        "epsilon": epsilon,
        "gradient_updates": gradient_update_count,
        "elapsed_seconds": time.time() - overall_start,
    }
