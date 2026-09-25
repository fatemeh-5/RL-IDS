"""REINFORCE and A2C trainers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from agent.training.advanced.common import (
    cfg_get,
    checkpoint,
    compute_gae,
    finish,
    maybe_resume,
    sample_policy_action,
)
from agent.training.advanced.networks import build_policy_network, build_value_network


def train_reinforce(
    env,
    *,
    episodes: int = 31,
    episode_length: int = 256,
    batch_size: int = 64,
    gamma: float = 0.97,
    learning_rate: float = 3e-4,
    seed: int = 42,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    cfg: dict[str, Any] | None = None,
    verbose: bool = True,
    **_: Any,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    use_baseline = bool(cfg_get(cfg, "use_baseline", True))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))

    start_episode, policy, value_net, history_rows, _ = maybe_resume(
        checkpoint_dir, mode, verbose
    )
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "REINFORCE_Policy")
        value_net = (
            build_value_network(env.n_features, learning_rate, "REINFORCE_Baseline")
            if use_baseline
            else None
        )

    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    v_opt = tf.keras.optimizers.Adam(learning_rate) if value_net is not None else None
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards = [], [], []
        total_reward = 0.0
        correct = 0

        for _ in range(episode_length):
            s = np.asarray(state, dtype=np.float32).reshape(1, -1)
            logits = policy.predict(s, verbose=0)[0]
            action = sample_policy_action(rng, logits)
            next_state, reward, terminated, truncated, info = env.step(action)
            states.append(s[0])
            actions.append(action)
            rewards.append(float(reward))
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if terminated or truncated:
                break

        # Monte-Carlo returns
        returns = np.zeros(len(rewards), dtype=np.float32)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma * g
            returns[t] = g
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int32)

        if value_net is not None:
            baseline = value_net.predict(states_np, verbose=0).reshape(-1)
            advantages = returns - baseline
        else:
            advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with tf.GradientTape() as tape:
            logits = policy(states_np, training=True)
            log_p = tf.nn.log_softmax(logits)
            chosen = tf.reduce_sum(log_p * tf.one_hot(actions_np, 2), axis=1)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * log_p, axis=1))
            loss = -tf.reduce_mean(chosen * advantages) - entropy_coef * entropy
        grads = tape.gradient(loss, policy.trainable_variables)
        pi_opt.apply_gradients(zip(grads, policy.trainable_variables))

        v_loss = np.nan
        if value_net is not None and v_opt is not None:
            with tf.GradientTape() as tape:
                v_pred = tf.squeeze(value_net(states_np, training=True), axis=-1)
                v_loss_t = tf.reduce_mean(tf.square(returns - v_pred))
            grads = tape.gradient(v_loss_t, value_net.trainable_variables)
            v_opt.apply_gradients(zip(grads, value_net.trainable_variables))
            v_loss = float(v_loss_t.numpy())

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(loss.numpy()),
            "Value_Loss": v_loss,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"[REINFORCE] Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}"
            )
        checkpoint(
            checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes,
            target_model=value_net,
        )

    return finish(policy, history_rows, overall_start, target_model=value_net)


def train_a2c(
    env,
    *,
    episodes: int = 31,
    episode_length: int = 256,
    batch_size: int = 64,
    gamma: float = 0.97,
    learning_rate: float = 3e-4,
    seed: int = 42,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    cfg: dict[str, Any] | None = None,
    verbose: bool = True,
    **_: Any,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    gae_lambda = float(cfg_get(cfg, "gae_lambda", 0.95))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))
    value_coef = float(cfg_get(cfg, "value_coef", 0.5))

    start_episode, policy, value_net, history_rows, _ = maybe_resume(
        checkpoint_dir, mode, verbose
    )
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "A2C_Policy")
        value_net = build_value_network(env.n_features, learning_rate, "A2C_Value")

    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    v_opt = tf.keras.optimizers.Adam(learning_rate)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards, dones, values = [], [], [], [], []
        total_reward = 0.0
        correct = 0

        for _ in range(episode_length):
            s = np.asarray(state, dtype=np.float32).reshape(1, -1)
            logits = policy.predict(s, verbose=0)[0]
            value = float(value_net.predict(s, verbose=0)[0, 0])
            action = sample_policy_action(rng, logits)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(s[0])
            actions.append(action)
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(value)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if done:
                break

        states_np = np.asarray(states, np.float32)
        actions_np = np.asarray(actions, np.int32)
        values_np = np.asarray(values, np.float32)
        rewards_np = np.asarray(rewards, np.float32)
        dones_np = np.asarray(dones, np.float32)
        next_v = 0.0
        if dones_np[-1] < 0.5:
            next_v = float(
                value_net.predict(np.asarray(state, np.float32).reshape(1, -1), verbose=0)[0, 0]
            )
        advantages, returns = compute_gae(
            rewards_np, values_np, dones_np, next_v, gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with tf.GradientTape() as tape:
            logits = policy(states_np, training=True)
            log_p = tf.nn.log_softmax(logits)
            chosen = tf.reduce_sum(log_p * tf.one_hot(actions_np, 2), axis=1)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * log_p, axis=1))
            loss_pi = -tf.reduce_mean(chosen * advantages) - entropy_coef * entropy
        grads = tape.gradient(loss_pi, policy.trainable_variables)
        pi_opt.apply_gradients(zip(grads, policy.trainable_variables))

        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(value_net(states_np, training=True), axis=-1)
            loss_v = value_coef * tf.reduce_mean(tf.square(returns - v_pred))
        grads = tape.gradient(loss_v, value_net.trainable_variables)
        v_opt.apply_gradients(zip(grads, value_net.trainable_variables))

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(loss_pi.numpy() + loss_v.numpy()),
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"[A2C] Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}"
            )
        checkpoint(
            checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes,
            target_model=value_net,
        )

    return finish(policy, history_rows, overall_start, target_model=value_net)
