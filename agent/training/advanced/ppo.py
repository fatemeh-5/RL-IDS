"""Proximal Policy Optimization (discrete) for IDS."""

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


def train_ppo(
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
    clip_range = float(cfg_get(cfg, "clip_range", 0.20))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))
    value_coef = float(cfg_get(cfg, "value_coef", 0.50))
    max_grad_norm = float(cfg_get(cfg, "max_grad_norm", 0.50))
    n_epochs = int(cfg_get(cfg, "n_epochs", 4))

    start_episode, policy, value_net, history_rows, _ = maybe_resume(
        checkpoint_dir, mode, verbose
    )
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, name="PPO_Policy")
        value_net = build_value_network(env.n_features, learning_rate, name="PPO_Value")

    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    v_opt = tf.keras.optimizers.Adam(learning_rate)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        total_reward = 0.0
        correct = 0

        for _ in range(episode_length):
            s = np.asarray(state, dtype=np.float32).reshape(1, -1)
            logits = policy.predict(s, verbose=0)[0]
            value = float(value_net.predict(s, verbose=0)[0, 0])
            action = sample_policy_action(rng, logits)
            # logprob from logits
            x = logits - np.max(logits)
            probs = np.exp(x) / np.exp(x).sum()
            log_prob = float(np.log(probs[action] + 1e-8))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(s[0])
            actions.append(action)
            rewards.append(float(reward))
            dones.append(float(done))
            log_probs.append(log_prob)
            values.append(value)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if done:
                break

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int32)
        old_log_probs = np.asarray(log_probs, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        next_v = 0.0
        if dones_np[-1] < 0.5:
            next_v = float(
                value_net.predict(
                    np.asarray(state, dtype=np.float32).reshape(1, -1), verbose=0
                )[0, 0]
            )
        advantages, returns = compute_gae(
            rewards_np, values_np, dones_np, next_v, gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states_np)
        idx = np.arange(n)
        losses = []
        for _ in range(n_epochs):
            rng.shuffle(idx)
            for start in range(0, n, batch_size):
                mb = idx[start : start + batch_size]
                if len(mb) < 2:
                    continue
                mb_s = tf.convert_to_tensor(states_np[mb])
                mb_a = tf.convert_to_tensor(actions_np[mb])
                mb_adv = tf.convert_to_tensor(advantages[mb])
                mb_ret = tf.convert_to_tensor(returns[mb])
                mb_old = tf.convert_to_tensor(old_log_probs[mb])

                with tf.GradientTape() as tape:
                    logits = policy(mb_s, training=True)
                    log_p_all = tf.nn.log_softmax(logits)
                    log_p = tf.reduce_sum(
                        log_p_all * tf.one_hot(mb_a, 2), axis=1
                    )
                    ratio = tf.exp(log_p - mb_old)
                    unclipped = ratio * mb_adv
                    clipped = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_adv
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * log_p_all, axis=1))
                    loss_pi = policy_loss - entropy_coef * entropy
                grads = tape.gradient(loss_pi, policy.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
                pi_opt.apply_gradients(zip(grads, policy.trainable_variables))

                with tf.GradientTape() as tape:
                    v_pred = tf.squeeze(value_net(mb_s, training=True), axis=-1)
                    loss_v = value_coef * tf.reduce_mean(tf.square(mb_ret - v_pred))
                grads = tape.gradient(loss_v, value_net.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
                v_opt.apply_gradients(zip(grads, value_net.trainable_variables))
                losses.append(float(loss_pi.numpy() + loss_v.numpy()))

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"[PPO] Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f} | "
                f"t: {row['Episode_Time_Seconds']:.1f}s"
            )
        checkpoint(
            checkpoint_dir,
            episode,
            policy,
            history_rows,
            save_every_episodes,
            episodes,
            target_model=value_net,
        )

    return finish(policy, history_rows, overall_start, target_model=value_net)
