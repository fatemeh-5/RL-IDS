"""PER + Double/Dueling DQN trainer (B4 / B5 / B6)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.utils.checkpointing import load_training_checkpoint, save_training_checkpoint
from agent.models import build_double_dqn, build_dueling_dqn, synchronize_target_network
from agent.buffers.prioritized import PrioritizedReplayBuffer, calculate_per_beta
from agent.training.dueling_helpers import prioritized_double_dqn_update


def train_per(
    env,
    *,
    model_type: str = "double",  # double | dueling
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
    model_builder: Callable | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    n_actions = int(env.action_space.n)

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
                print(f"Resuming PER trainer from episode {start_episode}")
        elif mode == "resume":
            raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")

    def _default_builder(name: str):
        if model_builder is not None:
            return model_builder(name)
        if model_type == "dueling":
            return build_dueling_dqn(
                input_features=env.n_features,
                action_count=n_actions,
                learning_rate=learning_rate,
                model_name=name,
            )
        return build_double_dqn(
            n_features=env.n_features,
            learning_rate=learning_rate,
            name=name,
            n_actions=n_actions,
        )

    if online_model is None:
        online_model = _default_builder("PER_Online")
        target_model = _default_builder("PER_Target")
        synchronize_target_network(online_model, target_model)

    buffer = PrioritizedReplayBuffer(
        capacity=replay_capacity,
        alpha=per_alpha,
        priority_epsilon=priority_epsilon,
        random_state=seed,
    )
    epsilon = epsilon_start
    if history_rows:
        epsilon = float(
            history_rows[-1].get(
                "Epsilon_End", history_rows[-1].get("epsilon", epsilon_start)
            )
        )

    gradient_update_count = 0
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        episode_start = time.time()
        beta = calculate_per_beta(episode, episodes, beta_start, beta_end)
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses: list[float] = []
        td_errors: list[float] = []
        epsilon_at_start = epsilon

        for _ in range(episode_length):
            if rng.random() <= epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                q = online_model(
                    np.asarray(state, dtype=np.float32).reshape(1, -1),
                    training=False,
                ).numpy()[0]
                action = int(np.argmax(q))

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
            "Episode_Accuracy": correct / episode_length,
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Mean_Absolute_TD_Error": float(np.mean(td_errors)) if td_errors else np.nan,
            "Epsilon_Start": epsilon_at_start,
            "Epsilon_End": epsilon,
            "Beta": beta,
            "Episode_Time_Seconds": time.time() - episode_start,
            "Gradient_Updates": gradient_update_count,
        }
        if "episode_investigations" in info:
            row["Episode_Investigations"] = info["episode_investigations"]
        if "episode_attack_alerts" in info:
            row["Episode_Attack_Alerts"] = info["episode_attack_alerts"]
        history_rows.append(row)
        if verbose:
            extra = ""
            if "Episode_Investigations" in row:
                extra = f" | Investigations: {row['Episode_Investigations']}"
            print(
                f"Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | "
                f"Acc: {row['Episode_Accuracy']:.3f} | "
                f"Eps: {epsilon_at_start:.4f}->{epsilon:.4f} | "
                f"t: {row['Episode_Time_Seconds']:.1f}s{extra}"
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
