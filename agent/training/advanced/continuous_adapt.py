"""Discrete adaptations of SAC, TD3, and DDPG for Binary IDS."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from agent.buffers import ReplayBuffer
from agent.training.advanced.common import (
    cfg_get,
    checkpoint,
    finish,
    hard_update,
    maybe_resume,
    sample_policy_action,
    soft_update,
)
from agent.training.advanced.networks import build_policy_network, build_q_network


def train_sac(env, **kwargs) -> dict[str, Any]:
    """Discrete Soft Actor-Critic (Haarnoja-style with categorical policy)."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    batch_size = int(kwargs.get("batch_size", 64))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    tau = float(cfg_get(cfg, "tau", 0.005))
    alpha = float(cfg_get(cfg, "alpha", 0.2))
    autotune_alpha = bool(cfg_get(cfg, "autotune_alpha", True))
    target_entropy = float(cfg_get(cfg, "target_entropy", -0.5))  # ~0.5 * log(|A|)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, policy, q1, history_rows, extra = maybe_resume(checkpoint_dir, mode, verbose)

    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "SAC_Policy", softmax=True)
        q1 = build_q_network(env.n_features, 2, learning_rate, "SAC_Q1")
    # Always ensure twin + targets exist
    q2 = build_q_network(env.n_features, 2, learning_rate, "SAC_Q2")
    q1_t = build_q_network(env.n_features, 2, learning_rate, "SAC_Q1T")
    q2_t = build_q_network(env.n_features, 2, learning_rate, "SAC_Q2T")
    hard_update(q1, q1_t)
    hard_update(q2, q2_t)

    log_alpha = tf.Variable(np.log(max(alpha, 1e-6)), dtype=tf.float32)
    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    q_opt = tf.keras.optimizers.Adam(learning_rate)
    alpha_opt = tf.keras.optimizers.Adam(learning_rate)
    buffer = ReplayBuffer(capacity=int(cfg_get(cfg, "replay_capacity", 10000)), random_state=seed)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses = []
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            probs = policy.predict(s, verbose=0)[0]
            action = sample_policy_action(rng, probs)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                alpha_v = tf.exp(log_alpha) if autotune_alpha else tf.constant(alpha, tf.float32)

                with tf.GradientTape() as tape:
                    next_pi = policy(next_states, training=True)
                    next_logp = tf.math.log(next_pi + 1e-8)
                    q1n = q1_t(next_states, training=False)
                    q2n = q2_t(next_states, training=False)
                    min_qn = tf.minimum(q1n, q2n)
                    v_next = tf.reduce_sum(next_pi * (min_qn - alpha_v * next_logp), axis=1)
                    target_q = rewards + (1.0 - dones) * gamma * v_next
                    q1_pred = tf.reduce_sum(q1(states, training=True) * tf.one_hot(actions, 2), axis=1)
                    q2_pred = tf.reduce_sum(q2(states, training=True) * tf.one_hot(actions, 2), axis=1)
                    loss_q = tf.reduce_mean(tf.square(q1_pred - tf.stop_gradient(target_q))) + tf.reduce_mean(
                        tf.square(q2_pred - tf.stop_gradient(target_q))
                    )
                q_vars = q1.trainable_variables + q2.trainable_variables
                q_opt.apply_gradients(zip(tape.gradient(loss_q, q_vars), q_vars))

                with tf.GradientTape() as tape:
                    pi = policy(states, training=True)
                    logp = tf.math.log(pi + 1e-8)
                    q_min = tf.minimum(q1(states, training=False), q2(states, training=False))
                    loss_pi = tf.reduce_mean(tf.reduce_sum(pi * (alpha_v * logp - q_min), axis=1))
                pi_opt.apply_gradients(zip(tape.gradient(loss_pi, policy.trainable_variables), policy.trainable_variables))

                if autotune_alpha:
                    with tf.GradientTape() as tape:
                        pi = policy(states, training=False)
                        logp = tf.math.log(pi + 1e-8)
                        entropy = -tf.reduce_sum(pi * logp, axis=1)
                        loss_alpha = -tf.reduce_mean(log_alpha * tf.stop_gradient(entropy - target_entropy))
                    alpha_opt.apply_gradients(zip(tape.gradient(loss_alpha, [log_alpha]), [log_alpha]))

                soft_update(q1, q1_t, tau)
                soft_update(q2, q2_t, tau)
                losses.append(float(loss_q.numpy() + loss_pi.numpy()))

            if done:
                break

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Alpha": float(tf.exp(log_alpha).numpy()) if autotune_alpha else alpha,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[SAC] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes, target_model=q1)

    return finish(policy, history_rows, overall_start, target_model=q1)


def train_td3(env, **kwargs) -> dict[str, Any]:
    """TD3 discrete adapt: twin delayed Q + deterministic argmax policy with noise."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    batch_size = int(kwargs.get("batch_size", 64))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    tau = float(cfg_get(cfg, "tau", 0.005))
    policy_delay = int(cfg_get(cfg, "policy_delay", 2))
    explore_eps = float(cfg_get(cfg, "epsilon_start", 0.3))
    eps_min = float(cfg_get(cfg, "epsilon_min", 0.05))
    eps_decay = float(cfg_get(cfg, "epsilon_decay", 0.95))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, actor, critic1, history_rows, extra = maybe_resume(checkpoint_dir, mode, verbose)
    if actor is None:
        actor = build_q_network(env.n_features, 2, learning_rate, "TD3_ActorQ")  # logits as preference
        critic1 = build_q_network(env.n_features, 2, learning_rate, "TD3_Q1")
    critic2 = build_q_network(env.n_features, 2, learning_rate, "TD3_Q2")
    critic1_t = build_q_network(env.n_features, 2, learning_rate, "TD3_Q1T")
    critic2_t = build_q_network(env.n_features, 2, learning_rate, "TD3_Q2T")
    actor_t = build_q_network(env.n_features, 2, learning_rate, "TD3_ActorT")
    hard_update(critic1, critic1_t)
    hard_update(critic2, critic2_t)
    hard_update(actor, actor_t)

    q_opt = tf.keras.optimizers.Adam(learning_rate)
    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    buffer = ReplayBuffer(capacity=int(cfg_get(cfg, "replay_capacity", 10000)), random_state=seed)
    epsilon = float(extra.get("epsilon", explore_eps)) if start_episode > 0 else explore_eps
    overall_start = time.time()
    step = 0

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses = []
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            q = actor.predict(s, verbose=0)[0]
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(q))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            step += 1

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                with tf.GradientTape() as tape:
                    next_act = tf.argmax(actor_t(next_states, training=False), axis=1)
                    # target policy smoothing: flip with small prob
                    flip = tf.cast(tf.random.uniform(tf.shape(next_act)) < 0.05, tf.int64)
                    next_act = (next_act + flip) % 2
                    q1n = tf.reduce_sum(critic1_t(next_states) * tf.one_hot(next_act, 2), axis=1)
                    q2n = tf.reduce_sum(critic2_t(next_states) * tf.one_hot(next_act, 2), axis=1)
                    target_q = rewards + (1.0 - dones) * gamma * tf.minimum(q1n, q2n)
                    q1_pred = tf.reduce_sum(critic1(states, training=True) * tf.one_hot(actions, 2), axis=1)
                    q2_pred = tf.reduce_sum(critic2(states, training=True) * tf.one_hot(actions, 2), axis=1)
                    loss_q = tf.reduce_mean(tf.square(q1_pred - tf.stop_gradient(target_q))) + tf.reduce_mean(
                        tf.square(q2_pred - tf.stop_gradient(target_q))
                    )
                q_vars = critic1.trainable_variables + critic2.trainable_variables
                q_opt.apply_gradients(zip(tape.gradient(loss_q, q_vars), q_vars))

                if step % policy_delay == 0:
                    with tf.GradientTape() as tape:
                        # Maximize Q1 of greedy action preference (via soft Q of actor logits)
                        logits = actor(states, training=True)
                        probs = tf.nn.softmax(logits)
                        q1v = critic1(states, training=False)
                        loss_pi = -tf.reduce_mean(tf.reduce_sum(probs * q1v, axis=1))
                    pi_opt.apply_gradients(zip(tape.gradient(loss_pi, actor.trainable_variables), actor.trainable_variables))
                    soft_update(actor, actor_t, tau)
                    soft_update(critic1, critic1_t, tau)
                    soft_update(critic2, critic2_t, tau)
                losses.append(float(loss_q.numpy()))

            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)
        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Epsilon": epsilon,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[TD3] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, actor, history_rows, save_every_episodes, episodes, target_model=critic1, extra={"epsilon": epsilon})

    return finish(actor, history_rows, overall_start, target_model=critic1, epsilon=epsilon)


def train_ddpg(env, **kwargs) -> dict[str, Any]:
    """DDPG discrete adapt: actor logits + critic Q, soft updates."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    batch_size = int(kwargs.get("batch_size", 64))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    tau = float(cfg_get(cfg, "tau", 0.005))
    explore_eps = float(cfg_get(cfg, "epsilon_start", 0.4))
    eps_min = float(cfg_get(cfg, "epsilon_min", 0.05))
    eps_decay = float(cfg_get(cfg, "epsilon_decay", 0.95))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, actor, critic, history_rows, extra = maybe_resume(checkpoint_dir, mode, verbose)
    if actor is None:
        actor = build_q_network(env.n_features, 2, learning_rate, "DDPG_Actor")
        critic = build_q_network(env.n_features, 2, learning_rate, "DDPG_Critic")
    actor_t = build_q_network(env.n_features, 2, learning_rate, "DDPG_ActorT")
    critic_t = build_q_network(env.n_features, 2, learning_rate, "DDPG_CriticT")
    hard_update(actor, actor_t)
    hard_update(critic, critic_t)

    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    q_opt = tf.keras.optimizers.Adam(learning_rate)
    buffer = ReplayBuffer(capacity=int(cfg_get(cfg, "replay_capacity", 10000)), random_state=seed)
    epsilon = float(extra.get("epsilon", explore_eps)) if start_episode > 0 else explore_eps
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses = []
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            q = actor.predict(s, verbose=0)[0]
            action = int(rng.integers(0, 2)) if rng.random() < epsilon else int(np.argmax(q))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                with tf.GradientTape() as tape:
                    next_a = tf.argmax(actor_t(next_states), axis=1)
                    target_q = rewards + (1.0 - dones) * gamma * tf.reduce_sum(
                        critic_t(next_states) * tf.one_hot(next_a, 2), axis=1
                    )
                    q_pred = tf.reduce_sum(critic(states, training=True) * tf.one_hot(actions, 2), axis=1)
                    loss_q = tf.reduce_mean(tf.square(q_pred - tf.stop_gradient(target_q)))
                q_opt.apply_gradients(zip(tape.gradient(loss_q, critic.trainable_variables), critic.trainable_variables))

                with tf.GradientTape() as tape:
                    probs = tf.nn.softmax(actor(states, training=True))
                    loss_pi = -tf.reduce_mean(tf.reduce_sum(probs * critic(states, training=False), axis=1))
                pi_opt.apply_gradients(zip(tape.gradient(loss_pi, actor.trainable_variables), actor.trainable_variables))

                soft_update(actor, actor_t, tau)
                soft_update(critic, critic_t, tau)
                losses.append(float(loss_q.numpy() + loss_pi.numpy()))

            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)
        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Epsilon": epsilon,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[DDPG] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, actor, history_rows, save_every_episodes, episodes, target_model=critic, extra={"epsilon": epsilon})

    return finish(actor, history_rows, overall_start, target_model=critic, epsilon=epsilon)
